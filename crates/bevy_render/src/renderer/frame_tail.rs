//! Work the render thread finishes after it hands the render app back.

use alloc::{boxed::Box, vec::Vec};
use bevy_ecs::{resource::Resource, system::Res};
use bevy_platform::sync::{Arc, Mutex, PoisonError};

type TailWork = Box<dyn FnOnce() + Send>;

/// Queues this frame's work that needs no render world, such as presenting swap chains or polling
/// the device, to run after the render schedule.
///
/// With [`PipelinedRenderingPlugin`](crate::pipelined_rendering::PipelinedRenderingPlugin) the render
/// thread runs the queue right after it sends the render app back to the main app, so the next
/// extraction overlaps it. Without it, [`run_frame_tail`] runs the queue at the end of the render
/// schedule. Either way the queue runs in order, and before the next render schedule starts.
#[derive(Resource, Clone, Default)]
pub struct RenderFrameTail(Arc<Mutex<Vec<TailWork>>>);

impl RenderFrameTail {
    /// Runs `work` once this frame's render schedule has finished.
    pub fn defer(&self, work: impl FnOnce() + Send + 'static) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Box::new(work));
    }

    /// Runs and removes the queued work, in queue order.
    pub fn run(&self) {
        let work = core::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner));
        for work in work {
            work();
        }
    }
}

/// Marks a render world whose frame tail the render thread runs after handing the app back.
#[derive(Resource)]
pub struct FrameTailOnRenderThread;

/// Runs the frame tail at the end of the render schedule when no render thread runs it later.
pub fn run_frame_tail(tail: Res<RenderFrameTail>) {
    tail.run();
}

#[cfg(test)]
mod tests {
    use super::RenderFrameTail;
    use alloc::{sync::Arc, vec, vec::Vec};
    use bevy_platform::sync::Mutex;

    #[test]
    fn deferred_work_runs_once_in_queue_order() {
        let tail = RenderFrameTail::default();
        let order = Arc::new(Mutex::new(Vec::new()));
        for index in 0..3 {
            let order = order.clone();
            tail.defer(move || order.lock().unwrap().push(index));
        }
        tail.run();
        tail.run();
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn work_deferred_while_running_waits_for_the_next_run() {
        let tail = RenderFrameTail::default();
        let ran = Arc::new(Mutex::new(Vec::new()));
        let (inner_tail, inner_ran) = (tail.clone(), ran.clone());
        tail.defer(move || {
            inner_ran.lock().unwrap().push("first");
            let ran = inner_ran.clone();
            inner_tail.defer(move || ran.lock().unwrap().push("second"));
        });
        tail.run();
        assert_eq!(*ran.lock().unwrap(), vec!["first"]);
        tail.run();
        assert_eq!(*ran.lock().unwrap(), vec!["first", "second"]);
    }
}
