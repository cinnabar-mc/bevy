use super::TaskPool;
use bevy_platform::sync::OnceLock;
use core::ops::Deref;

macro_rules! taskpool {
    ($(#[$attr:meta])* ($static:ident, $type:ident)) => {
        static $static: OnceLock<$type> = OnceLock::new();

        $(#[$attr])*
        #[derive(Debug)]
        pub struct $type(TaskPool);

        impl $type {
            #[doc = concat!(" Gets the global [`", stringify!($type), "`] instance, or initializes it with `f`.")]
            pub fn get_or_init(f: impl FnOnce() -> TaskPool) -> &'static Self {
                $static.get_or_init(|| Self(f()))
            }

            #[doc = concat!(" Attempts to get the global [`", stringify!($type), "`] instance, \
                or returns `None` if it is not initialized.")]
            pub fn try_get() -> Option<&'static Self> {
                $static.get()
            }

            #[doc = concat!(" Gets the global [`", stringify!($type), "`] instance.")]
            #[doc = ""]
            #[doc = " # Panics"]
            #[doc = " Panics if the global instance has not been initialized yet."]
            pub fn get() -> &'static Self {
                $static.get().expect(
                    concat!(
                        "The ",
                        stringify!($type),
                        " has not been initialized yet. Please call ",
                        stringify!($type),
                        "::get_or_init beforehand."
                    )
                )
            }
        }

        impl Deref for $type {
            type Target = TaskPool;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

taskpool! {
    /// A newtype for a task pool for CPU-intensive work that must be completed to
    /// deliver the next frame
    ///
    /// See [`TaskPool`] documentation for details on Bevy tasks.
    /// [`AsyncComputeTaskPool`] should be preferred if the work does not have to be
    /// completed before the next frame.
    (COMPUTE_TASK_POOL, ComputeTaskPool)
}

taskpool! {
    /// A newtype for a task pool for CPU-intensive work that may span across multiple frames
    ///
    /// See [`TaskPool`] documentation for details on Bevy tasks.
    /// Use [`ComputeTaskPool`] if the work must be complete before advancing to the next frame.
    (ASYNC_COMPUTE_TASK_POOL, AsyncComputeTaskPool)
}

taskpool! {
    /// A newtype for a task pool for IO-intensive work (i.e. tasks that spend very little time in a
    /// "woken" state)
    ///
    /// See [`TaskPool`] documentation for details on Bevy tasks.
    (IO_TASK_POOL, IoTaskPool)
}

crate::cfg::web! {
    if {} else {
        /// A function used by `bevy_app` to tick the global tasks pools on the main thread.
        /// This will run a maximum of 100 local tasks per executor per call to this function,
        /// one per executor per round, and stops after a round in which none of them had a
        /// runnable task: tasks woken by the tasks that ran are already queued by then. A task
        /// woken from another thread after that round runs on the next call, as one woken after
        /// the last round always did.
        ///
        /// # Warning
        ///
        /// This function *must* be called on the main thread, or the task pools will not be updated appropriately.
        pub fn tick_global_task_pools_on_main_thread() {
            COMPUTE_TASK_POOL
                .get()
                .unwrap()
                .with_local_executor(|compute_local_executor| {
                    ASYNC_COMPUTE_TASK_POOL
                        .get()
                        .unwrap()
                        .with_local_executor(|async_local_executor| {
                            IO_TASK_POOL
                                .get()
                                .unwrap()
                                .with_local_executor(|io_local_executor| {
                                    for _ in 0..100 {
                                        // Non-short-circuiting, so every executor ticks once
                                        // per round as before.
                                        let ran = compute_local_executor.try_tick()
                                            | async_local_executor.try_tick()
                                            | io_local_executor.try_tick();
                                        if !ran {
                                            break;
                                        }
                                    }
                                });
                        });
                });
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::{
        future::Future,
        pin::Pin,
        sync::atomic::{AtomicUsize, Ordering},
        task::{Context, Poll},
    };

    /// Wakes itself on every poll, counting polls, until it has been polled `limit` times.
    struct Spin {
        polls: Arc<AtomicUsize>,
        limit: usize,
    }

    impl Future for Spin {
        type Output = ();

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.polls.fetch_add(1, Ordering::Relaxed) + 1 >= self.limit {
                return Poll::Ready(());
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    #[test]
    fn main_thread_tick_runs_queued_local_work_and_later_wakes() {
        ComputeTaskPool::get_or_init(TaskPool::default);
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        IoTaskPool::get_or_init(TaskPool::default);

        // A task that keeps waking itself runs until it finishes.
        let polls = Arc::new(AtomicUsize::new(0));
        ComputeTaskPool::get()
            .spawn_local(Spin {
                polls: Arc::clone(&polls),
                limit: 50,
            })
            .detach();
        tick_global_task_pools_on_main_thread();
        assert_eq!(polls.load(Ordering::Relaxed), 50);

        // With nothing queued the call returns, and work spawned afterwards runs on the next one.
        tick_global_task_pools_on_main_thread();
        let later = Arc::new(AtomicUsize::new(0));
        IoTaskPool::get()
            .spawn_local(Spin {
                polls: Arc::clone(&later),
                limit: 5,
            })
            .detach();
        tick_global_task_pools_on_main_thread();
        assert_eq!(later.load(Ordering::Relaxed), 5);

        // A task woken from another thread runs on the next call after the wake.
        let (sender, receiver) = async_channel::bounded::<()>(1);
        let woken = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&woken);
        ComputeTaskPool::get()
            .spawn_local(async move {
                receiver.recv().await.unwrap();
                seen.fetch_add(1, Ordering::Relaxed);
            })
            .detach();
        tick_global_task_pools_on_main_thread();
        assert_eq!(woken.load(Ordering::Relaxed), 0);
        std::thread::spawn(move || sender.send_blocking(()).unwrap())
            .join()
            .unwrap();
        tick_global_task_pools_on_main_thread();
        assert_eq!(woken.load(Ordering::Relaxed), 1);
    }
}
