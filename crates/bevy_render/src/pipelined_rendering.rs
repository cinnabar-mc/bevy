use core::panic::AssertUnwindSafe;
use std::panic::{catch_unwind, resume_unwind};

use alloc::sync::Arc;
use async_channel::{Receiver, Sender, TryRecvError};
use bevy_platform::time::Instant;
use core::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use bevy_app::{App, AppExit, Plugin, SubApp};
use bevy_derive::AppLabel;
use bevy_ecs::{
    resource::Resource,
    schedule::MainThreadExecutor,
    world::{Mut, World},
};
use bevy_tasks::ComputeTaskPool;

use crate::RenderApp;

/// A Label for the sub app that runs the parts of pipelined rendering that need to run on the main thread.
///
/// The Main schedule of this app can be used to run logic after the render schedule starts, but
/// before I/O processing. This can be useful for something like frame pacing.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, AppLabel, Default)]
pub struct RenderExtractApp;

/// Channels used by the main app to send and receive the render app.
#[derive(Resource)]
pub struct RenderAppChannels {
    app_to_render_sender: Sender<SubApp>,
    render_to_app_receiver: Receiver<SubApp>,
    render_app_in_render_thread: bool,
    /// Pumped during shutdown so the render app's main-thread tasks can finish.
    main_thread_executor: MainThreadExecutor,
    /// Shared with the render thread: when its current update started and how long updates take.
    progress: Arc<RenderProgress>,
    /// How long the main thread may poll for a render app that is about to come back.
    spin_budget: Duration,
}

/// The render thread's progress, which the main thread reads to predict when the render app
/// comes back.
#[derive(Debug)]
struct RenderProgress {
    base: Instant,
    /// Nanoseconds after `base` at which the current render update started; 0 while the render
    /// thread holds no render app.
    started: AtomicU64,
    /// Running average of the time from the start of an update to the render app being sent back,
    /// in nanoseconds; 0 until the first update.
    average: AtomicU64,
}

impl RenderProgress {
    fn new() -> Self {
        Self {
            base: Instant::now(),
            started: AtomicU64::new(0),
            average: AtomicU64::new(0),
        }
    }

    fn now(&self) -> u64 {
        // Never 0, which marks "no update running".
        (self.base.elapsed().as_nanos() as u64).max(1)
    }

    /// The render thread starts an update.
    fn begin(&self) {
        self.started.store(self.now(), Ordering::Relaxed);
    }

    /// The render thread has sent the render app back.
    fn end(&self) {
        let started = self.started.swap(0, Ordering::Relaxed);
        if started == 0 {
            return;
        }
        let took = self.now().saturating_sub(started);
        let average = self.average.load(Ordering::Relaxed);
        let average = if average == 0 {
            took
        } else {
            average - average / 8 + took / 8
        };
        self.average.store(average, Ordering::Relaxed);
    }

    /// How long until the running update is expected to send the render app back, if one is
    /// running and the render thread has finished one before.
    fn due_in(&self) -> Option<Duration> {
        let started = self.started.load(Ordering::Relaxed);
        let average = self.average.load(Ordering::Relaxed);
        if started == 0 || average == 0 {
            return None;
        }
        Some(Duration::from_nanos(
            (started + average).saturating_sub(self.now()),
        ))
    }
}

/// How long the main thread polls for a render app that is due back: enough to cover the usual
/// spread of a render update's end, and zero where polling would take a hardware thread the render
/// thread or the task pools need.
fn main_spin_budget(hardware_threads: usize) -> Duration {
    if hardware_threads >= 4 {
        Duration::from_micros(20)
    } else {
        Duration::ZERO
    }
}

impl RenderAppChannels {
    /// Create a `RenderAppChannels` from a [`async_channel::Receiver`] and [`async_channel::Sender`].
    ///
    /// `main_thread_executor` must be the [`MainThreadExecutor`] shared with the render world.
    /// It is pumped during shutdown so the final render update can complete.
    pub fn new(
        app_to_render_sender: Sender<SubApp>,
        render_to_app_receiver: Receiver<SubApp>,
        main_thread_executor: MainThreadExecutor,
    ) -> Self {
        Self {
            app_to_render_sender,
            render_to_app_receiver,
            render_app_in_render_thread: false,
            main_thread_executor,
            progress: Arc::new(RenderProgress::new()),
            spin_budget: main_spin_budget(
                std::thread::available_parallelism().map_or(1, core::num::NonZero::get),
            ),
        }
    }

    /// Send the `render_app` to the rendering thread.
    pub fn send_blocking(&mut self, render_app: SubApp) {
        self.app_to_render_sender.send_blocking(render_app).unwrap();
        self.render_app_in_render_thread = true;
    }

    /// Polls for the render app when the render thread is predicted to send it back within the
    /// spin budget, so the main thread does not have to block and be woken for it. Returns `None`
    /// when it is not due that soon, does not arrive within the budget, or the render thread is
    /// gone; the caller then blocks on [`Self::recv`].
    fn try_recv_soon(&mut self) -> Option<SubApp> {
        let due = self.progress.due_in()?;
        if self.spin_budget.is_zero() || due > self.spin_budget {
            return None;
        }
        let deadline = Instant::now() + self.spin_budget;
        loop {
            match self.render_to_app_receiver.try_recv() {
                Ok(render_app) => {
                    self.render_app_in_render_thread = false;
                    return Some(render_app);
                }
                Err(TryRecvError::Closed) => return None,
                Err(TryRecvError::Empty) if Instant::now() >= deadline => return None,
                Err(TryRecvError::Empty) => {
                    for _ in 0..64 {
                        core::hint::spin_loop();
                    }
                }
            }
        }
    }

    /// Receive the `render_app` from the rendering thread.
    /// Return `None` if the render thread has panicked.
    pub async fn recv(&mut self) -> Option<SubApp> {
        let render_app = self.render_to_app_receiver.recv().await.ok()?;
        self.render_app_in_render_thread = false;
        Some(render_app)
    }

    /// Takes the render app if the render thread has already sent it back, without waiting.
    ///
    /// Returns `None` both when the render app is still in the render thread and when the
    /// render thread is gone; [`Self::recv`] tells the two apart.
    pub fn try_recv(&mut self) -> Option<SubApp> {
        let render_app = self.render_to_app_receiver.try_recv().ok()?;
        self.render_app_in_render_thread = false;
        Some(render_app)
    }
}

impl Drop for RenderAppChannels {
    fn drop(&mut self) {
        if self.render_app_in_render_thread {
            // The render world's non-send data was initialized on the main thread,
            // so wait for the render app to return and drop it here.
            //
            // A blocking receive would deadlock if the final render update has queued
            // main-thread tasks. Pump their executor while waiting, as renderer_extract does.
            let result = catch_unwind(AssertUnwindSafe(|| {
                ComputeTaskPool::get().scope_with_executor(
                    true,
                    Some(&self.main_thread_executor.0),
                    |scope| {
                        scope.spawn(async { self.render_to_app_receiver.recv().await.ok() });
                    },
                );
            }));

            if let Err(payload) = result {
                if std::thread::panicking() {
                    // The scope and the recovered render app's destructors can panic.
                    // Preserve an existing panic instead of aborting with a double panic.
                    // Even dropping the new panic's payload could panic again.
                    core::mem::forget(payload);
                } else {
                    resume_unwind(payload);
                }
            }
        }
    }
}

/// The [`PipelinedRenderingPlugin`] can be added to your application to enable pipelined rendering.
///
/// This moves rendering into a different thread, so that the Nth frame's rendering can
/// be run at the same time as the N + 1 frame's simulation.
///
/// ```text
/// |--------------------|--------------------|--------------------|--------------------|
/// | simulation thread  | frame 1 simulation | frame 2 simulation | frame 3 simulation |
/// |--------------------|--------------------|--------------------|--------------------|
/// | rendering thread   |                    | frame 1 rendering  | frame 2 rendering  |
/// |--------------------|--------------------|--------------------|--------------------|
/// ```
///
/// The plugin is dependent on the [`RenderApp`] added by [`crate::RenderPlugin`] and so must
/// be added after that plugin. If it is not added after, the plugin will do nothing.
///
/// A single frame of execution looks something like below
///
/// ```text
/// |---------------------------------------------------------------------------|
/// |      |         | RenderExtractApp schedule | winit events | main schedule |
/// | sync | extract |----------------------------------------------------------|
/// |      |         | extract commands | rendering schedule                    |
/// |---------------------------------------------------------------------------|
/// ```
///
/// - `sync` is the step where the entity-entity mapping between the main and render world is updated.
///   This is run on the main app's thread. For more information checkout [`SyncWorldPlugin`].
/// - `extract` is the step where data is copied from the main world to the render world.
///   This is run on the main app's thread.
/// - On the render thread, we first apply the `extract commands`. This is not run during extract, so the
///   main schedule can start sooner.
/// - Then the `rendering schedule` is run. See [`RenderSystems`](crate::RenderSystems) for the standard steps in this process.
/// - In parallel to the rendering thread the [`RenderExtractApp`] schedule runs. By
///   default, this schedule is empty. But it is useful if you need something to run before I/O processing.
/// - Next all the `winit events` are processed.
/// - And finally the `main app schedule` is run.
/// - Once both the `main app schedule` and the `render schedule` are finished running, `extract` is run again.
///
/// [`SyncWorldPlugin`]: bevy_extract::sync_world::SyncWorldPlugin
#[derive(Default)]
pub struct PipelinedRenderingPlugin;

impl Plugin for PipelinedRenderingPlugin {
    fn build(&self, app: &mut App) {
        // Don't add RenderExtractApp if RenderApp isn't initialized.
        if app.get_sub_app(RenderApp).is_none() {
            return;
        }
        app.insert_resource(MainThreadExecutor::new());

        let mut sub_app = SubApp::new();
        sub_app.set_extract(renderer_extract);
        app.insert_sub_app(RenderExtractApp, sub_app);
    }

    // Sets up the render thread and inserts resources into the main app used for controlling the render thread.
    fn cleanup(&self, app: &mut App) {
        // skip setting up when headless
        if app.get_sub_app(RenderExtractApp).is_none() {
            return;
        }

        let (app_to_render_sender, app_to_render_receiver) = async_channel::bounded::<SubApp>(1);
        let (render_to_app_sender, render_to_app_receiver) = async_channel::bounded::<SubApp>(1);

        let mut render_app = app
            .remove_sub_app(RenderApp)
            .expect("Unable to get RenderApp. Another plugin may have removed the RenderApp before PipelinedRenderingPlugin");

        // clone main thread executor to render world
        let executor = app.world().resource::<MainThreadExecutor>().clone();
        render_app.world_mut().insert_resource(executor.clone());
        render_app
            .world_mut()
            .insert_resource(crate::renderer::FrameTailOnRenderThread);

        render_to_app_sender.send_blocking(render_app).unwrap();

        let channels =
            RenderAppChannels::new(app_to_render_sender, render_to_app_receiver, executor);
        let progress = channels.progress.clone();
        app.insert_resource(channels);

        std::thread::Builder::new()
            .name("Render thread".into())
            .spawn(move || {
                #[cfg(feature = "trace")]
                let _span = bevy_log::info_span!("render thread").entered();

                let compute_task_pool = ComputeTaskPool::get();
                loop {
                    // Take a render app that is already waiting without a scope. Otherwise run a
                    // scope here to allow main world to use this thread while it's waiting for
                    // the render app. The receive runs on this thread's scope executor, so
                    // waiting does not wake a pool worker just to poll the channel.
                    let sent_app = match app_to_render_receiver.try_recv() {
                        Ok(render_app) => Some(Ok(render_app)),
                        Err(_) => compute_task_pool
                            .scope(|s| {
                                s.spawn_on_scope(async { app_to_render_receiver.recv().await });
                            })
                            .pop(),
                    };
                    let Some(Ok(render_app)) = sent_app else {
                        break;
                    };

                    if !update_and_return(render_app, &render_to_app_sender, &progress) {
                        break;
                    }
                }

                bevy_log::debug!("exiting pipelined rendering thread");
            })
            .expect("Failed to create render thread");
    }
}

/// Runs one render frame, sends the render app back to the main app, then runs the frame tail, so
/// the tail's work overlaps the next extraction. Returns `false` when the main app is gone.
///
/// `progress` times the update up to the send, which is when the main app gets the render app
/// back; the frame tail runs after that and is not part of the prediction.
fn update_and_return(
    mut render_app: SubApp,
    sender: &Sender<SubApp>,
    progress: &RenderProgress,
) -> bool {
    {
        #[cfg(feature = "trace")]
        let _sub_app_span = bevy_log::info_span!("sub app", name = ?RenderApp).entered();
        progress.begin();
        render_app.update();
    }
    let tail = render_app
        .world()
        .get_resource::<crate::renderer::RenderFrameTail>()
        .cloned();
    if sender.send_blocking(render_app).is_err() {
        return false;
    }
    progress.end();
    if let Some(tail) = tail {
        tail.run();
    }
    true
}

// This function waits for the rendering world to be received,
// runs extract, and then sends the rendering world back to the render thread.
fn renderer_extract(app_world: &mut World, _world: &mut World) {
    app_world.resource_scope(|world, main_thread_executor: Mut<MainThreadExecutor>| {
        world.resource_scope(|world, mut render_channels: Mut<RenderAppChannels>| {
            // we use a scope here to run any main thread tasks that the render world still needs to run
            // while we wait for the render world to be received.
            // A render app that is already back needs no main thread tasks, so skip the scope
            // and its task pool wake-ups.
            let render_app = render_channels
                .try_recv()
                .or_else(|| render_channels.try_recv_soon())
                .or_else(|| {
                    ComputeTaskPool::get()
                        .scope_with_executor(true, Some(&*main_thread_executor.0), |s| {
                            // Poll the receive on this thread: the scope ticks it while running
                            // main-thread tasks, and the render thread's send wakes this thread.
                            // Spawning it onto the pool would wake a worker every frame.
                            s.spawn_on_scope(async { render_channels.recv().await });
                        })
                        .pop()
                        .unwrap()
                });
            if let Some(mut render_app) = render_app {
                render_app.extract(world);

                render_channels.send_blocking(render_app);
            } else {
                // Renderer thread panicked
                world.write_message(AppExit::error());
            }
        });
    });
}

#[cfg(all(test, feature = "multi_threaded"))]
mod tests {
    use super::*;
    use bevy_tasks::{block_on, TaskPool};
    use core::time::Duration;
    use std::{
        sync::mpsc,
        thread::{self, JoinHandle, ThreadId},
    };

    const TIMEOUT: Duration = Duration::from_secs(10);

    // The executor must be created and ticked on the synthetic main thread.
    // Keep the test thread free to time out even if shutdown deadlocks.
    fn run_on_main_thread(test: impl FnOnce() + Send + 'static) {
        ComputeTaskPool::get_or_init(TaskPool::new);
        let (done_sender, done_receiver) = mpsc::channel();
        let main_thread = thread::spawn(move || {
            test();
            done_sender.send(()).unwrap();
        });

        done_receiver
            .recv_timeout(TIMEOUT)
            .expect("render shutdown did not finish");
        main_thread.join().unwrap();
    }

    fn start_render_thread(render_app: SubApp) -> (RenderAppChannels, JoinHandle<()>) {
        let main_thread_id = thread::current().id();
        let executor = MainThreadExecutor::new();
        let render_executor = executor.clone();
        let (app_to_render_sender, app_to_render_receiver) = async_channel::bounded::<SubApp>(1);
        let (render_to_app_sender, render_to_app_receiver) = async_channel::bounded::<SubApp>(1);
        let mut channels =
            RenderAppChannels::new(app_to_render_sender, render_to_app_receiver, executor);
        channels.send_blocking(render_app);

        let (queued_sender, queued_receiver) = mpsc::channel();
        let render_thread = thread::spawn(move || {
            let render_app = app_to_render_receiver.recv_blocking().unwrap();
            let task = render_executor.0.spawn(async move {
                assert_eq!(thread::current().id(), main_thread_id);
            });
            queued_sender.send(()).unwrap();
            block_on(task);
            render_to_app_sender.send_blocking(render_app).unwrap();
        });

        // Ensure shutdown encounters pending main-thread work, regardless of scheduling.
        queued_receiver.recv_timeout(TIMEOUT).unwrap();
        (channels, render_thread)
    }

    struct NotifyOnDrop(mpsc::Sender<ThreadId>);

    impl Drop for NotifyOnDrop {
        fn drop(&mut self) {
            self.0.send(thread::current().id()).unwrap();
        }
    }

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("render app drop panic");
        }
    }

    /// The main app gets the render app back while the frame tail's work is still running.
    #[test]
    fn the_frame_tail_runs_after_the_render_app_is_sent_back() {
        let (sender, receiver) = async_channel::bounded::<SubApp>(1);
        let tail = crate::renderer::RenderFrameTail::default();
        let mut render_app = SubApp::new();
        render_app.insert_resource(tail.clone());
        let (release, released) = mpsc::channel::<()>();
        let (ran, tail_ran) = mpsc::channel::<()>();
        // Blocks until the main side has the render app, so running it before the send times out.
        tail.defer(move || {
            released.recv_timeout(TIMEOUT).unwrap();
            ran.send(()).unwrap();
        });
        let render_thread =
            thread::spawn(move || update_and_return(render_app, &sender, &RenderProgress::new()));

        let returned = receiver.recv_blocking().unwrap();
        assert!(tail_ran.try_recv().is_err());
        release.send(()).unwrap();
        tail_ran.recv_timeout(TIMEOUT).unwrap();
        assert!(render_thread.join().unwrap());
        drop(returned);
    }

    #[test]
    fn try_recv_takes_only_a_render_app_that_was_sent_back() {
        run_on_main_thread(|| {
            let (app_to_render_sender, app_to_render_receiver) =
                async_channel::bounded::<SubApp>(1);
            let (render_to_app_sender, render_to_app_receiver) =
                async_channel::bounded::<SubApp>(1);
            let mut channels = RenderAppChannels::new(
                app_to_render_sender,
                render_to_app_receiver,
                MainThreadExecutor::new(),
            );
            channels.send_blocking(SubApp::new());
            assert!(channels.try_recv().is_none());

            let render_app = app_to_render_receiver.recv_blocking().unwrap();
            render_to_app_sender.send_blocking(render_app).unwrap();
            assert!(channels.try_recv().is_some());
            // The render app is back, so dropping the channels must not wait for it again.
            drop(channels);
        });
    }

    #[test]
    fn render_progress_predicts_only_after_a_full_update_and_while_one_runs() {
        // A clock started a second ago, so timestamps in the past stay above zero.
        let progress = RenderProgress {
            base: Instant::now() - Duration::from_secs(1),
            ..RenderProgress::new()
        };
        assert_eq!(progress.due_in(), None, "no update yet");
        progress.begin();
        assert_eq!(
            progress.due_in(),
            None,
            "no finished update to predict from"
        );
        progress.end();
        assert_eq!(progress.due_in(), None, "no update running");
        progress.begin();
        assert!(progress.due_in().is_some());
        progress.end();
        // The average follows the updates' lengths.
        progress.average.store(8_000, Ordering::Relaxed);
        progress
            .started
            .store(progress.now() - 16_000, Ordering::Relaxed);
        progress.end();
        assert!(progress.average.load(Ordering::Relaxed) >= 9_000);
    }

    #[test]
    fn the_main_thread_polls_only_with_hardware_threads_to_spare() {
        assert_eq!(main_spin_budget(2), Duration::ZERO);
        assert!(main_spin_budget(4) > Duration::ZERO);
    }

    fn channels_with_progress(
        started_ago: u64,
        average: u64,
        budget: Duration,
    ) -> (RenderAppChannels, Sender<SubApp>) {
        let (app_to_render_sender, _app_to_render_receiver) = async_channel::bounded::<SubApp>(1);
        let (render_to_app_sender, render_to_app_receiver) = async_channel::bounded::<SubApp>(1);
        let mut channels = RenderAppChannels::new(
            app_to_render_sender,
            render_to_app_receiver,
            MainThreadExecutor::new(),
        );
        channels.spin_budget = budget;
        let now = channels.progress.now();
        channels
            .progress
            .started
            .store(now.saturating_sub(started_ago).max(1), Ordering::Relaxed);
        channels.progress.average.store(average, Ordering::Relaxed);
        (channels, render_to_app_sender)
    }

    #[test]
    fn a_render_app_due_back_is_taken_when_it_arrives_during_the_poll() {
        // Due now, with a budget long enough that scheduling cannot make the test flaky.
        let (mut channels, sender) = channels_with_progress(1_000, 1_000, TIMEOUT);
        channels.render_app_in_render_thread = true;
        let render_thread = thread::spawn(move || {
            sender.send_blocking(SubApp::new()).unwrap();
            sender
        });
        assert!(channels.try_recv_soon().is_some());
        assert!(!channels.render_app_in_render_thread);
        drop(render_thread.join().unwrap());
    }

    #[test]
    fn a_render_app_not_due_soon_is_left_to_the_blocking_receive() {
        // Due in a second, well beyond the budget: no polling at all.
        let (mut channels, sender) =
            channels_with_progress(0, 1_000_000_000, Duration::from_micros(20));
        let start = Instant::now();
        assert!(channels.try_recv_soon().is_none());
        assert!(start.elapsed() < Duration::from_millis(500));
        // Without a budget it never polls either, even for an app that is already back.
        let (mut channels, sender2) = channels_with_progress(1_000, 1_000, Duration::ZERO);
        sender2.send_blocking(SubApp::new()).unwrap();
        assert!(channels.try_recv_soon().is_none());
        assert!(channels.try_recv().is_some());
        drop(sender);
    }

    #[test]
    fn renderer_extract_waits_for_the_render_app_while_running_main_thread_tasks() {
        #[derive(Resource)]
        struct Extracted;

        run_on_main_thread(|| {
            let main_thread_id = thread::current().id();
            let (app_to_render_sender, app_to_render_receiver) =
                async_channel::bounded::<SubApp>(1);
            let (render_to_app_sender, render_to_app_receiver) =
                async_channel::bounded::<SubApp>(1);
            let executor = MainThreadExecutor::new();
            let render_executor = executor.clone();
            let mut main_world = World::new();
            main_world.insert_resource(executor.clone());
            let mut channels =
                RenderAppChannels::new(app_to_render_sender, render_to_app_receiver, executor);
            let mut render_app = SubApp::new();
            render_app.set_extract(|_, render_world| render_world.insert_resource(Extracted));
            channels.send_blocking(render_app);
            main_world.insert_resource(channels);

            // The render thread keeps the app until a main-thread task has run, then takes the
            // extracted app back.
            let render_thread = thread::spawn(move || {
                let render_app = app_to_render_receiver.recv_blocking().unwrap();
                block_on(render_executor.0.spawn(async move {
                    assert_eq!(thread::current().id(), main_thread_id);
                }));
                render_to_app_sender.send_blocking(render_app).unwrap();
                let render_app = app_to_render_receiver.recv_blocking().unwrap();
                (
                    render_app.world().contains_resource::<Extracted>(),
                    render_to_app_sender,
                    render_app,
                )
            });

            renderer_extract(&mut main_world, &mut World::new());

            let (extracted, render_to_app_sender, render_app) = render_thread.join().unwrap();
            assert!(extracted);
            // Return the app so the channels' drop does not wait for it.
            render_to_app_sender.send_blocking(render_app).unwrap();
        });
    }

    #[test]
    fn renderer_extract_extracts_a_returned_render_app_and_sends_it_back() {
        #[derive(Resource)]
        struct Extracted;

        run_on_main_thread(|| {
            let (app_to_render_sender, app_to_render_receiver) =
                async_channel::bounded::<SubApp>(1);
            let (render_to_app_sender, render_to_app_receiver) =
                async_channel::bounded::<SubApp>(1);
            let executor = MainThreadExecutor::new();
            let mut main_world = World::new();
            main_world.insert_resource(executor.clone());
            main_world.insert_resource(RenderAppChannels::new(
                app_to_render_sender,
                render_to_app_receiver,
                executor,
            ));
            let mut render_app = SubApp::new();
            render_app.set_extract(|_, render_world| render_world.insert_resource(Extracted));
            render_to_app_sender.send_blocking(render_app).unwrap();

            renderer_extract(&mut main_world, &mut World::new());

            let render_app = app_to_render_receiver.try_recv().unwrap();
            assert!(render_app.world().contains_resource::<Extracted>());
            // Return the app so the channels' drop does not wait for it.
            render_to_app_sender.send_blocking(render_app).unwrap();
        });
    }

    #[test]
    fn drop_pumps_main_thread_executor_to_avoid_shutdown_deadlock() {
        run_on_main_thread(|| {
            let main_thread_id = thread::current().id();
            let (dropped_sender, dropped_receiver) = mpsc::channel();
            let mut render_app = SubApp::new();
            render_app
                .world_mut()
                .insert_non_send(NotifyOnDrop(dropped_sender));
            let (channels, render_thread) = start_render_thread(render_app);

            drop(channels);

            render_thread.join().unwrap();
            assert_eq!(
                dropped_receiver.recv_timeout(TIMEOUT).unwrap(),
                main_thread_id
            );
        });
    }

    #[test]
    fn drop_suppresses_render_app_drop_panic_during_unwind() {
        run_on_main_thread(|| {
            let mut render_app = SubApp::new();
            render_app.world_mut().insert_non_send(PanicOnDrop);
            let (channels, render_thread) = start_render_thread(render_app);

            let result = catch_unwind(AssertUnwindSafe(move || {
                let _channels = channels;
                panic!("main app panic");
            }));

            assert_eq!(
                result.unwrap_err().downcast_ref::<&str>(),
                Some(&"main app panic")
            );
            render_thread.join().unwrap();
        });
    }

    #[test]
    fn drop_propagates_render_app_drop_panic_without_existing_panic() {
        run_on_main_thread(|| {
            let mut render_app = SubApp::new();
            render_app.world_mut().insert_non_send(PanicOnDrop);
            let (channels, render_thread) = start_render_thread(render_app);

            let result = catch_unwind(AssertUnwindSafe(|| drop(channels)));

            assert_eq!(
                result.unwrap_err().downcast_ref::<&str>(),
                Some(&"render app drop panic")
            );
            render_thread.join().unwrap();
        });
    }
}
