//! Submitting a frame's command buffers in the render frame tail.

use alloc::{boxed::Box, vec::Vec};
use bevy_ecs::{resource::Resource, world::World};
use bevy_platform::sync::{Arc, PoisonError};
use std::{
    sync::{Condvar, Mutex, MutexGuard},
    thread::ThreadId,
};
use wgpu::SubmissionIndex;

use super::{FrameTailOnRenderThread, RenderFrameTail};

type SubmitWork = Box<dyn FnOnce() + Send>;
type DeferredSubmit = Box<dyn FnOnce() -> SubmissionIndex + Send>;

/// Opts the render world into submitting the render graph's command buffers in the
/// [`RenderFrameTail`], after the render app is handed back, so the next extraction overlaps
/// finishing and submitting them.
///
/// With this resource present, `RenderGraphSystems::Submit` can finish while the frame's command
/// buffers are still on the CPU, and they reach the queue only when the frame tail runs, which
/// overlaps the next extraction on the main thread. Any code that uses the
/// [`RenderQueue`](super::RenderQueue) while that submission is deferred, whether a render system
/// after `Submit`, an extraction system on the main thread, or another thread holding the queue,
/// must therefore follow this contract:
///
/// - Work that must follow the frame's submission, such as completion callbacks registered with
///   [`wgpu::Queue::on_submitted_work_done`] or [`wgpu::Buffer::map_async`] requests for buffers
///   the frame writes, goes through [`FrameSubmission::after_submit`].
/// - Command buffers submitted to the queue directly, rather than added to
///   [`PendingCommandBuffers`](super::PendingCommandBuffers), are submitted after
///   [`FrameSubmission::flush`]. Otherwise they reach the queue before the frame's own and can
///   read resources the frame has not drawn yet.
/// - Staged writes ([`wgpu::Queue::write_buffer`], [`wgpu::Queue::write_buffer_with`],
///   [`wgpu::Queue::write_texture`]) to a resource the deferred frame reads or writes are issued
///   after [`FrameSubmission::flush`]. A staged write executes before the command buffers of the
///   next `submit`, so issued while the frame is deferred, a write meant for the next frame would
///   land in this frame's draw. Writes to resources the deferred frame does not use, such as ones
///   allocated for later frames, need no flush.
///
/// Frames with screenshots, GPU readbacks or render diagnostics still submit immediately,
/// because those map buffers right after the submission. Bevy never inserts this resource; an
/// application that does takes on the contract for every system it adds after the graph.
#[derive(Resource, Default)]
pub struct SubmitInFrameTail;

#[derive(Default)]
struct FrameSubmissionState {
    /// Index of the latest submission made in the frame tail, until taken.
    index: Option<SubmissionIndex>,
    /// Work waiting for a submission deferred into the frame tail; `None` while none is.
    after: Option<Vec<SubmitWork>>,
    /// Counts deferred submissions, so only a frame's last one runs the waiting work.
    deferred: u64,
    /// Deferred submissions not yet made, oldest first, with their sequence numbers.
    queued: Vec<(u64, DeferredSubmit)>,
    /// The thread making a deferred submission taken off `queued`, until it has been submitted
    /// and its index recorded.
    in_flight: Option<ThreadId>,
}

impl FrameSubmissionState {
    /// Records the submission `index` for deferred submission `sequence` and returns the work
    /// waiting for it when no later submission was deferred this frame.
    fn record(&mut self, sequence: u64, index: SubmissionIndex) -> Option<Vec<SubmitWork>> {
        self.index = Some(index);
        if self.deferred == sequence {
            self.after.take()
        } else {
            None
        }
    }
}

#[derive(Default)]
struct FrameSubmissionShared {
    state: Mutex<FrameSubmissionState>,
    /// Signalled when an in-flight deferred submission has been submitted.
    submitted: Condvar,
}

/// Clears the submitting thread's in-flight mark if its submission unwinds, so other callers stop
/// waiting for it.
struct InFlight<'a> {
    shared: &'a FrameSubmissionShared,
    outer: Option<ThreadId>,
    armed: bool,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if self.armed {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            state.in_flight = self.outer;
            drop(state);
            self.shared.submitted.notify_all();
        }
    }
}

/// Shared state of the frame's submission made in the [`RenderFrameTail`]: its index and the
/// work that must follow it.
#[derive(Resource, Clone, Default)]
pub struct FrameSubmission(Arc<FrameSubmissionShared>);

impl FrameSubmission {
    fn state(&self) -> MutexGuard<'_, FrameSubmissionState> {
        self.0.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `work` after every submission made so far, including one deferred into the frame
    /// tail: immediately when no submission is deferred, otherwise in the tail right after the
    /// frame's last deferred submission.
    pub fn after_submit(&self, work: impl FnOnce() + Send + 'static) {
        let mut state = self.state();
        match state.after.as_mut() {
            Some(after) => after.push(Box::new(work)),
            None => {
                drop(state);
                work();
            }
        }
    }

    /// Takes the index of the latest submission made in the frame tail since the previous call.
    pub fn take_index(&self) -> Option<SubmissionIndex> {
        self.state().index.take()
    }

    /// Marks a submission as deferred into the tail and returns its sequence number. Work passed
    /// to [`Self::after_submit`] from now on waits for it.
    pub(super) fn begin_deferred(&self) -> u64 {
        let mut state = self.state();
        state.after.get_or_insert_with(Vec::new);
        state.deferred += 1;
        state.deferred
    }

    /// Records the tail's submission `index` for deferred submission `sequence`, then runs the
    /// waiting work if no later submission was deferred this frame.
    #[cfg(test)]
    fn finish_deferred(&self, sequence: u64, index: SubmissionIndex) {
        let waiting = self.state().record(sequence, index);
        for work in waiting.into_iter().flatten() {
            work();
        }
    }

    /// Queues `submit` as a deferred submission and returns its sequence number. The frame tail
    /// makes it with [`Self::run_deferred`], unless [`Self::flush`] made it earlier.
    pub(super) fn defer(&self, submit: DeferredSubmit) -> u64 {
        let sequence = self.begin_deferred();
        self.state().queued.push((sequence, submit));
        sequence
    }

    /// Makes deferred submission `sequence`, and every one queued before it, if they are still
    /// waiting; a [`Self::flush`] may already have made them.
    pub(super) fn run_deferred(&self, sequence: u64) {
        self.submit_due(|queued| queued <= sequence);
    }

    /// Makes every submission deferred into the frame tail now, in order, then runs the work
    /// waiting for them. Systems after `RenderGraphSystems::Submit` that submit to the
    /// [`RenderQueue`](super::RenderQueue) directly call this first, so the frame's own
    /// submission reaches the queue before theirs. Does nothing when no submission is deferred.
    pub fn flush(&self) {
        self.submit_due(|_| true);
    }

    /// Submits the queued deferred submissions whose sequence numbers `due` accepts, oldest first,
    /// then runs the work waiting for them.
    ///
    /// One submission is taken at a time and marked in flight while it is submitted, without
    /// holding a lock across the submit. A caller on another thread waits until no submission is
    /// in flight before it looks at the queue, so when it finds nothing left to submit, every
    /// earlier deferred submission has reached the queue.
    ///
    /// The submitting thread itself can call back in: `wgpu::Queue::submit` fires completion
    /// callbacks for earlier work, such as ones registered through [`Self::after_submit`] in a
    /// previous frame, before it returns. wgpu-core fires them only after the submission's
    /// command buffers were handed to the backend (`Queue::submit` in `device/queue.rs`), so the
    /// in-flight submission is already queued; a nested call does not wait for it and submits
    /// whatever is still queued, which keeps the order.
    fn submit_due(&self, due: impl Fn(u64) -> bool) {
        let current = std::thread::current().id();
        let mut waiting = Vec::new();
        let mut state = self.state();
        loop {
            while state.in_flight.is_some_and(|owner| owner != current) {
                state = self
                    .0
                    .submitted
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if !state
                .queued
                .first()
                .is_some_and(|(sequence, _)| due(*sequence))
            {
                break;
            }
            let (sequence, submit) = state.queued.remove(0);
            let mut in_flight = InFlight {
                shared: &self.0,
                outer: state.in_flight.replace(current),
                armed: true,
            };
            drop(state);
            let index = submit();
            state = self.state();
            waiting.extend(state.record(sequence, index).into_iter().flatten());
            state.in_flight = in_flight.outer;
            in_flight.armed = false;
            self.0.submitted.notify_all();
        }
        drop(state);
        for work in waiting {
            work();
        }
    }
}

/// The frame tail and submission state to submit this frame with, when the render world opted in
/// with [`SubmitInFrameTail`], the render thread runs the tail, and nothing this frame maps
/// buffers right after the submission.
pub fn frame_tail_submission(world: &World) -> Option<(RenderFrameTail, FrameSubmission)> {
    if !world.contains_resource::<SubmitInFrameTail>()
        || !world.contains_resource::<FrameTailOnRenderThread>()
        || world.contains_resource::<crate::diagnostic::internal::DiagnosticsRecorder>()
        || crate::view::screenshot::has_screenshot_commands(world)
        || crate::gpu_readback::has_readback_commands(world)
    {
        return None;
    }
    Some((
        world.get_resource::<RenderFrameTail>()?.clone(),
        world.get_resource::<FrameSubmission>()?.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn index() -> SubmissionIndex {
        let (device, queue) = wgpu::Device::noop(&Default::default());
        let _ = device;
        queue.submit([])
    }

    #[test]
    fn work_waits_only_while_a_submission_is_deferred() {
        let submission = FrameSubmission::default();
        let order = Arc::new(Mutex::new(Vec::new()));
        let record = |value| {
            let order = order.clone();
            move || order.lock().unwrap().push(value)
        };
        submission.after_submit(record(0));
        let first = submission.begin_deferred();
        submission.after_submit(record(1));
        let second = submission.begin_deferred();
        submission.after_submit(record(2));
        assert_eq!(*order.lock().unwrap(), vec![0]);
        submission.finish_deferred(first, index());
        assert_eq!(
            *order.lock().unwrap(),
            vec![0],
            "a later deferred submission is pending"
        );
        submission.finish_deferred(second, index());
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2]);
        assert!(submission.take_index().is_some());
        assert!(submission.take_index().is_none());
        submission.after_submit(record(3));
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3]);
    }

    /// A flush that finds the deferred submission already taken by another caller waits until
    /// that caller has submitted it, so its own queue work still follows the frame's.
    #[test]
    fn a_concurrent_flush_waits_for_a_submission_another_caller_took() {
        use core::{
            sync::atomic::{AtomicBool, Ordering},
            time::Duration,
        };
        use std::{sync::mpsc, thread};

        const TIMEOUT: Duration = Duration::from_secs(10);
        let submission = FrameSubmission::default();
        let submitted = Arc::new(AtomicBool::new(false));
        let (taken, was_taken) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let flag = submitted.clone();
        submission.defer(Box::new(move || {
            taken.send(()).unwrap();
            released.recv_timeout(TIMEOUT).unwrap();
            flag.store(true, Ordering::SeqCst);
            index()
        }));
        let first = {
            let submission = submission.clone();
            thread::spawn(move || submission.flush())
        };
        // The first caller has taken the submission and is submitting it.
        was_taken.recv_timeout(TIMEOUT).unwrap();

        let (done, returned) = mpsc::channel();
        let second = {
            let submission = submission.clone();
            let submitted = submitted.clone();
            thread::spawn(move || {
                submission.flush();
                done.send(submitted.load(Ordering::SeqCst)).unwrap();
            })
        };
        assert!(
            returned.recv_timeout(Duration::from_millis(200)).is_err(),
            "the second flush returned before the first caller submitted"
        );
        release.send(()).unwrap();
        assert!(
            returned.recv_timeout(TIMEOUT).unwrap(),
            "the second flush returned only after the submission"
        );
        first.join().unwrap();
        second.join().unwrap();
    }

    /// `Queue::submit` can fire completion callbacks before it returns. A callback that flushes
    /// from inside the submit must not wait for the submission it is called from; it submits what
    /// is still queued, so a later deferred submission still precedes the callback's own work.
    #[test]
    fn a_flush_from_inside_a_submit_submits_what_is_still_queued() {
        use core::time::Duration;
        use std::{sync::mpsc, thread};

        let submission = FrameSubmission::default();
        let order = Arc::new(Mutex::new(Vec::new()));
        let (first, nested) = (order.clone(), submission.clone());
        submission.defer(Box::new(move || {
            first.lock().unwrap().push("first submitted");
            // A completion callback for earlier work, fired by this submit.
            nested.flush();
            first.lock().unwrap().push("callback returned");
            index()
        }));
        let second = order.clone();
        submission.defer(Box::new(move || {
            second.lock().unwrap().push("second submitted");
            index()
        }));
        let (done, finished) = mpsc::channel();
        let worker = thread::spawn(move || {
            submission.flush();
            done.send(()).unwrap();
        });
        finished
            .recv_timeout(Duration::from_secs(10))
            .expect("a flush from inside a submit deadlocked");
        worker.join().unwrap();
        assert_eq!(
            *order.lock().unwrap(),
            vec!["first submitted", "second submitted", "callback returned"]
        );
    }
}
