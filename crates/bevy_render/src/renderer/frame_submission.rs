//! Submitting a frame's command buffers in the render frame tail.

use alloc::{boxed::Box, vec::Vec};
use bevy_ecs::{resource::Resource, world::World};
use bevy_platform::sync::{Arc, Mutex, PoisonError};
use wgpu::SubmissionIndex;

use super::{FrameTailOnRenderThread, RenderFrameTail};

type SubmitWork = Box<dyn FnOnce() + Send>;

/// Opts the render world into submitting the render graph's command buffers in the
/// [`RenderFrameTail`], after the render app is handed back, so the next extraction overlaps
/// finishing and submitting them.
///
/// Work that must follow the frame's submission, such as completion callbacks registered with
/// [`wgpu::Queue::on_submitted_work_done`] or [`wgpu::Buffer::map_async`] requests for buffers
/// the frame writes, must go through [`FrameSubmission::after_submit`] once the graph has run.
/// Frames with screenshots, GPU readbacks or render diagnostics still submit immediately,
/// because those map buffers right after the submission.
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
}

/// Shared state of the frame's submission made in the [`RenderFrameTail`]: its index and the
/// work that must follow it.
#[derive(Resource, Clone, Default)]
pub struct FrameSubmission(Arc<Mutex<FrameSubmissionState>>);

impl FrameSubmission {
    fn state(&self) -> bevy_platform::sync::MutexGuard<'_, FrameSubmissionState> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
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
    pub(super) fn finish_deferred(&self, sequence: u64, index: SubmissionIndex) {
        let after = {
            let mut state = self.state();
            state.index = Some(index);
            if state.deferred == sequence {
                state.after.take()
            } else {
                None
            }
        };
        for work in after.into_iter().flatten() {
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
}
