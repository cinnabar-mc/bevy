//! Presents window swap chain textures on a dedicated thread.
//!
//! A present call can spend a sizable fraction of a fast frame inside the graphics API, and
//! nothing in the frame depends on it returning. Handing it to a present thread lets it overlap
//! the render thread's next frame.

use super::{ExtractedWindow, SurfaceData};
use crate::{
    render_resource::SurfaceTexture,
    renderer::{RenderAdapterInfo, RenderQueue},
};
use alloc::sync::Arc;
use bevy_ecs::prelude::*;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

/// Environment variable that, set to `0`, `false` or `off`, makes the default
/// [`SurfacePresenter`] present on the render thread at the end of
/// [`render_system`](crate::renderer::render_system) instead of on a present thread.
pub const PRESENT_THREAD_ENV: &str = "BEVY_PRESENT_THREAD";

type PresentCallback = Box<dyn FnOnce() + Send>;

/// Decides where the frame's swap chain textures are presented.
///
/// When threaded, [`render_system`](crate::renderer::render_system) takes each presented window's
/// swap chain texture as before and stages it on the window's [`SurfaceData`], and
/// [`present_frames`] hands the staged textures to a dedicated present thread, which presents
/// them in order. A surface waits for its previous present to return before it acquires its next
/// texture, is reconfigured or is dropped; see [`SurfaceData::wait_for_present`]. The thread is
/// spawned by the first threaded present and joined when the presenter is dropped.
///
/// The default presenter is threaded unless [`PRESENT_THREAD_ENV`] disables it. The OpenGL
/// backend and the web always present inline, and so does any app that inserts
/// [`SurfacePresenter::inline`] into the render world.
#[derive(Resource)]
pub struct SurfacePresenter {
    threaded: bool,
    /// Whether `render_system` staged a texture that [`present_frames`] has not dispatched yet.
    staged: AtomicBool,
    callbacks: Mutex<Vec<PresentCallback>>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<PresentWorker>,
}

impl Default for SurfacePresenter {
    fn default() -> Self {
        let disabled = std::env::var(PRESENT_THREAD_ENV)
            .is_ok_and(|value| matches!(value.trim(), "0" | "false" | "off"));
        if disabled {
            Self::inline()
        } else {
            Self::threaded()
        }
    }
}

impl SurfacePresenter {
    /// Presents on a dedicated present thread where the platform allows it.
    pub fn threaded() -> Self {
        Self::new(cfg!(not(target_arch = "wasm32")))
    }

    /// Presents on the render thread at the end of [`render_system`](crate::renderer::render_system).
    pub fn inline() -> Self {
        Self::new(false)
    }

    fn new(threaded: bool) -> Self {
        Self {
            threaded,
            staged: AtomicBool::new(false),
            callbacks: Mutex::new(Vec::new()),
            #[cfg(not(target_arch = "wasm32"))]
            worker: None,
        }
    }

    /// Whether presents run on the present thread rather than the render thread.
    pub fn is_threaded(&self) -> bool {
        self.threaded
    }

    /// Runs `callback` once every present this frame has requested so far has returned.
    ///
    /// Register it after [`render_system`](crate::renderer::render_system) and before
    /// [`present_frames`]. When this frame staged textures for the present thread, `callback` runs
    /// there after the frame's last present returns. Otherwise this frame's presents, if any,
    /// returned inside `render_system`, and `callback` runs immediately on the caller's thread.
    pub fn after_presents(&self, callback: impl FnOnce() + Send + 'static) {
        if self.staged.load(Ordering::Relaxed) {
            lock(&self.callbacks).push(Box::new(callback));
        } else {
            callback();
        }
    }

    /// Moves the window's swap chain texture to its surface for [`present_frames`] to present.
    pub(crate) fn stage(&self, window: &mut ExtractedWindow, surface: &mut SurfaceData) {
        if let Some(texture) = window.swap_chain_texture.take() {
            surface.staged = Some(texture);
            self.staged.store(true, Ordering::Relaxed);
        }
    }

    /// Presents `texture` on the present thread, returning `false` if it presented inline instead.
    fn present(
        &mut self,
        texture: SurfaceTexture,
        queue: &RenderQueue,
        slot: &Arc<PresentSlot>,
        backend: wgpu::Backend,
    ) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(worker) = self.worker(backend) {
            slot.begin();
            let job = PresentJob::Present {
                texture,
                queue: queue.clone(),
                slot: slot.clone(),
            };
            return match worker.send(job) {
                Ok(()) => true,
                Err(job) => {
                    resume(job.run());
                    false
                }
            };
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (slot, backend);
        texture.present(queue);
        false
    }

    /// The present thread, spawned on first use, or `None` if presents run inline.
    #[cfg(not(target_arch = "wasm32"))]
    fn worker(&mut self, backend: wgpu::Backend) -> Option<&PresentWorker> {
        if !self.threaded {
            return None;
        }
        if self.worker.is_none() {
            // GL contexts are bound to threads, so keep GL presents with the rest of its work.
            if backend == wgpu::Backend::Gl {
                bevy_log::info!("Presenting on the render thread with the OpenGL backend");
                self.threaded = false;
                return None;
            }
            match PresentWorker::spawn() {
                Ok(worker) => self.worker = Some(worker),
                Err(error) => {
                    bevy_log::warn!(
                        "Couldn't spawn the present thread, presenting on the render thread: {error}"
                    );
                    self.threaded = false;
                    return None;
                }
            }
        }
        self.worker.as_ref()
    }

    /// Runs this frame's [`after_presents`](Self::after_presents) callbacks behind its presents.
    fn finish_frame(&mut self, dispatched: bool) {
        let callbacks = core::mem::take(
            self.callbacks
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
        );
        if callbacks.is_empty() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if dispatched && let Some(worker) = &self.worker {
            if let Err(job) = worker.send(PresentJob::Callbacks(callbacks)) {
                resume(job.run());
            }
            return;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = dispatched;
        for callback in callbacks {
            callback();
        }
    }
}

/// Hands the swap chain textures that [`render_system`](crate::renderer::render_system) staged
/// to the present thread, followed by this frame's
/// [`after_presents`](SurfacePresenter::after_presents) callbacks.
///
/// Runs after `render_system`. Systems that only need the frame's submission can run between
/// them, ahead of the present, without contending with it for the device's queue locks.
///
/// A panic on the present thread resurfaces here, on the render thread.
pub fn present_frames(
    mut presenter: ResMut<SurfacePresenter>,
    mut surfaces: Query<&mut SurfaceData>,
    queue: Res<RenderQueue>,
    adapter_info: Res<RenderAdapterInfo>,
) {
    let presenter = presenter.bypass_change_detection();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(worker) = &presenter.worker {
        worker.resume_panic();
    }
    if !core::mem::take(presenter.staged.get_mut()) {
        return;
    }
    let mut dispatched = false;
    for mut surface in &mut surfaces {
        let surface = surface.bypass_change_detection();
        if let Some(texture) = surface.staged.take() {
            dispatched |=
                presenter.present(texture, &queue, &surface.present, adapter_info.backend);
        }
    }
    presenter.finish_frame(dispatched);
}

/// Tracks whether a surface's last present is still running on the present thread.
#[derive(Default)]
pub(super) struct PresentSlot {
    in_flight: Mutex<bool>,
    returned: Condvar,
}

impl PresentSlot {
    #[cfg(not(target_arch = "wasm32"))]
    fn begin(&self) {
        *lock(&self.in_flight) = true;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish(&self) {
        *lock(&self.in_flight) = false;
        self.returned.notify_all();
    }

    /// Blocks until the surface's last present has returned.
    pub(super) fn wait(&self) {
        let mut in_flight = lock(&self.in_flight);
        if !*in_flight {
            return;
        }
        #[cfg(feature = "trace")]
        let _span = bevy_log::info_span!("wait_for_present").entered();
        while *in_flight {
            in_flight = self
                .returned
                .wait(in_flight)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(not(target_arch = "wasm32"))]
fn resume(result: std::thread::Result<()>) {
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum PresentJob {
    Present {
        texture: SurfaceTexture,
        queue: RenderQueue,
        slot: Arc<PresentSlot>,
    },
    Callbacks(Vec<PresentCallback>),
}

#[cfg(not(target_arch = "wasm32"))]
impl PresentJob {
    /// Runs the job, marking a present's surface as returned even if presenting panics.
    fn run(self) -> std::thread::Result<()> {
        use core::panic::AssertUnwindSafe;
        match self {
            PresentJob::Present {
                texture,
                queue,
                slot,
            } => {
                #[cfg(feature = "trace")]
                let _span = bevy_log::info_span!("present_frame").entered();
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| texture.present(&queue)));
                slot.finish();
                result
            }
            PresentJob::Callbacks(callbacks) => std::panic::catch_unwind(AssertUnwindSafe(|| {
                for callback in callbacks {
                    callback();
                }
            })),
        }
    }
}

/// The present thread, which runs jobs in the order they were sent.
#[cfg(not(target_arch = "wasm32"))]
struct PresentWorker {
    jobs: Option<std::sync::mpsc::Sender<PresentJob>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The first panic caught on the present thread, until the render thread resumes it.
    panic: Arc<Mutex<Option<Box<dyn core::any::Any + Send>>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl PresentWorker {
    fn spawn() -> std::io::Result<Self> {
        let (jobs, receiver) = std::sync::mpsc::channel::<PresentJob>();
        let panic = Arc::new(Mutex::new(None));
        let thread_panic = panic.clone();
        let thread = std::thread::Builder::new()
            .name("Present thread".into())
            .spawn(move || {
                for job in receiver {
                    if let Err(payload) = job.run() {
                        lock(&thread_panic).get_or_insert(payload);
                    }
                }
            })?;
        Ok(Self {
            jobs: Some(jobs),
            thread: Some(thread),
            panic,
        })
    }

    /// Queues `job`, returning it if the present thread is gone.
    fn send(&self, job: PresentJob) -> Result<(), PresentJob> {
        match &self.jobs {
            Some(jobs) => jobs.send(job).map_err(|error| error.0),
            None => Err(job),
        }
    }

    fn resume_panic(&self) {
        if let Some(payload) = lock(&self.panic).take() {
            std::panic::resume_unwind(payload);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for PresentWorker {
    fn drop(&mut self) {
        // Closing the channel lets the thread finish the queued jobs and exit.
        drop(self.jobs.take());
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            bevy_log::error!("The present thread panicked");
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use core::panic::AssertUnwindSafe;
    use std::sync::mpsc;

    #[test]
    fn present_slot_wait_returns_once_the_present_finishes() {
        let slot = Arc::new(PresentSlot::default());
        slot.wait();
        slot.begin();
        let (started, waiting) = mpsc::channel();
        let waiter = {
            let slot = slot.clone();
            std::thread::spawn(move || {
                started.send(()).unwrap();
                slot.wait();
                !*lock(&slot.in_flight)
            })
        };
        waiting.recv().unwrap();
        slot.finish();
        assert!(waiter.join().unwrap());
    }

    #[test]
    fn callbacks_run_immediately_unless_the_frame_staged_presents() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        for presenter in [SurfacePresenter::inline(), SurfacePresenter::threaded()] {
            let record = ran.clone();
            presenter.after_presents(move || lock(&record).push(1));
            assert_eq!(*lock(&ran), [1]);
            lock(&ran).clear();
        }

        let mut presenter = SurfacePresenter::threaded();
        presenter.staged.store(true, Ordering::Relaxed);
        let record = ran.clone();
        presenter.after_presents(move || lock(&record).push(2));
        assert!(lock(&ran).is_empty());
        // A frame whose staged textures all presented inline runs its callbacks right away.
        presenter.finish_frame(false);
        assert_eq!(*lock(&ran), [2]);
    }

    #[test]
    fn present_thread_runs_callbacks_in_order_and_is_joined_on_drop() {
        let mut presenter = SurfacePresenter::threaded();
        presenter.worker = Some(PresentWorker::spawn().unwrap());
        let ran = Arc::new(Mutex::new(Vec::new()));
        for frame in 0..3 {
            presenter.staged.store(true, Ordering::Relaxed);
            let record = ran.clone();
            presenter.after_presents(move || {
                let name = std::thread::current().name().map(str::to_owned);
                lock(&record).push((frame, name));
            });
            presenter.finish_frame(true);
            presenter.staged.store(false, Ordering::Relaxed);
        }
        drop(presenter);
        let thread = Some("Present thread".to_owned());
        assert_eq!(
            *lock(&ran),
            [(0, thread.clone()), (1, thread.clone()), (2, thread)]
        );
    }

    #[test]
    fn present_thread_panics_resurface_on_the_render_thread() {
        let worker = PresentWorker::spawn().unwrap();
        let failing: PresentCallback = Box::new(|| panic!("present failed"));
        assert!(worker.send(PresentJob::Callbacks(vec![failing])).is_ok());
        let (done, finished) = mpsc::channel();
        let signal: PresentCallback = Box::new(move || done.send(()).unwrap());
        assert!(worker.send(PresentJob::Callbacks(vec![signal])).is_ok());
        finished.recv().unwrap();
        let payload = std::panic::catch_unwind(AssertUnwindSafe(|| worker.resume_panic()))
            .expect_err("the present thread's panic resumes");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"present failed"));
        worker.resume_panic();
    }
}
