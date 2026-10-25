//! Selects the pipeline executor while preserving upstream platform defaults.

use super::{CachedPipelineState, Pipeline, ShaderCacheError};
use core::future::Future;

/// Starts a worker compile only where the application and platform allow it.
pub(super) fn create_pipeline_task(
    task: impl Future<Output = Result<Pipeline, ShaderCacheError>> + Send + 'static,
    synchronous: bool,
    async_on_macos: bool,
) -> CachedPipelineState {
    #[cfg(all(not(target_arch = "wasm32"), feature = "multi_threaded"))]
    if !synchronous && (!cfg!(target_os = "macos") || async_on_macos) {
        return CachedPipelineState::Creating(bevy_tasks::AsyncComputeTaskPool::get().spawn(
            async move {
                #[cfg(target_os = "macos")]
                return compile(task);
                #[cfg(not(target_os = "macos"))]
                task.await
            },
        ));
    }
    let _ = (synchronous, async_on_macos);
    match compile(task) {
        Ok(pipeline) => CachedPipelineState::Ok(pipeline),
        Err(err) => CachedPipelineState::Err(err),
    }
}

/// Drains Objective-C temporaries before a native worker returns to its executor.
fn compile(
    task: impl Future<Output = Result<Pipeline, ShaderCacheError>>,
) -> Result<Pipeline, ShaderCacheError> {
    #[cfg(target_os = "macos")]
    return objc2::rc::autoreleasepool(|_| bevy_tasks::block_on(task));
    #[cfg(not(target_os = "macos"))]
    bevy_tasks::block_on(task)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::AssetId;
    use bevy_shader::Shader;

    /// Returns a shader error after reporting the executing thread, without touching a GPU.
    async fn fail(
        sender: std::sync::mpsc::Sender<std::thread::ThreadId>,
    ) -> Result<Pipeline, ShaderCacheError> {
        sender.send(std::thread::current().id()).unwrap();
        Err(ShaderCacheError::ShaderNotLoaded(
            AssetId::<Shader>::default(),
        ))
    }

    #[test]
    fn explicit_synchronous_compilation_stays_on_the_caller() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let state = create_pipeline_task(fail(sender), true, true);
        assert!(matches!(state, CachedPipelineState::Err(_)));
        assert_eq!(receiver.try_recv().unwrap(), std::thread::current().id());
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "multi_threaded"))]
    #[test]
    fn opted_in_compilation_runs_on_a_worker_and_preserves_errors() {
        bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::new);
        let (sender, receiver) = std::sync::mpsc::channel();
        let CachedPipelineState::Creating(task) = create_pipeline_task(fail(sender), false, true)
        else {
            panic!("opted-in native compilation must return a task");
        };
        assert!(matches!(
            bevy_tasks::block_on(task),
            Err(ShaderCacheError::ShaderNotLoaded(_))
        ));
        assert_ne!(receiver.try_recv().unwrap(), std::thread::current().id());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_keeps_synchronous_compilation_without_the_opt_in() {
        let (sender, receiver) = std::sync::mpsc::channel();
        assert!(matches!(
            create_pipeline_task(fail(sender), false, false),
            CachedPipelineState::Err(_)
        ));
        assert_eq!(receiver.try_recv().unwrap(), std::thread::current().id());
    }
}
