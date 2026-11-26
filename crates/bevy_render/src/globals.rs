use crate::{
    extract_resource::ExtractResource,
    render_resource::{ShaderType, UniformBuffer},
    renderer::{RenderDevice, RenderQueue},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
};
use bevy_app::{App, Plugin};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{component::Mutable, prelude::*};
use bevy_reflect::prelude::*;
use bevy_shader::load_shader_library;
use bevy_time::Time;

pub struct GlobalsPlugin;

impl Plugin for GlobalsPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "globals.wesl");
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_gpu_resource::<GlobalsBuffer>()
                .init_resource::<Time>()
                .add_systems(ExtractSchedule, (extract_frame_count, extract_time))
                .add_systems(
                    Render,
                    prepare_globals_buffer.in_set(RenderSystems::PrepareResources),
                );
        }
    }
}

fn extract_frame_count(mut commands: Commands, frame_count: Extract<Res<FrameCount>>) {
    commands.queue(write_resource(**frame_count));
}

/// Copies the main world's [`Time`] into the render world when the extraction commands apply.
fn extract_time(mut commands: Commands, time: Extract<Res<Time>>) {
    commands.queue(write_resource(**time));
}

/// Stores `value` as the render world's `R`. Once the render world has one, it is assigned in
/// place: the same value, marked changed at the same tick as a reinsertion, without reinserting
/// the resource. Neither [`Time`] nor [`FrameCount`] has hooks or observers that a reinsertion
/// would run.
fn write_resource<R: Resource<Mutability = Mutable>>(value: R) -> impl FnOnce(&mut World) {
    move |world: &mut World| match world.get_resource_mut::<R>() {
        Some(mut current) => *current = value,
        None => world.insert_resource(value),
    }
}

/// Contains global values useful when writing shaders.
/// Currently only contains values related to time.
#[derive(Default, Clone, Resource, ExtractResource, Reflect, ShaderType)]
#[reflect(Resource, Default, Clone)]
#[extract_app(RenderApp)]
pub struct GlobalsUniform {
    /// The time since startup in seconds.
    /// Wraps to 0 after 1 hour.
    time: f32,
    /// The delta time since the previous frame in seconds
    delta_time: f32,
    /// Frame count since the start of the app.
    /// It wraps to zero when it reaches `u32::MAX`.
    frame_count: u32,
    /// WebGL2 structs must be 16 byte aligned.
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding: f32,
}

/// The buffer containing the [`GlobalsUniform`]
#[derive(Resource, Default)]
pub struct GlobalsBuffer {
    pub buffer: UniformBuffer<GlobalsUniform>,
}

fn prepare_globals_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut globals_buffer: ResMut<GlobalsBuffer>,
    time: Res<Time>,
    frame_count: Res<FrameCount>,
) {
    let buffer = globals_buffer.buffer.get_mut();
    buffer.time = time.elapsed_secs_wrapped();
    buffer.delta_time = time.delta_secs();
    buffer.frame_count = frame_count.0;

    globals_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::{change_detection::DetectChanges, world::CommandQueue};
    use core::time::Duration;

    #[test]
    fn a_resource_written_in_place_matches_a_reinsertion() {
        let ticked = |seconds| {
            let mut time = Time::<()>::default();
            time.advance_by(Duration::from_secs(seconds));
            time
        };
        let mut assigned = World::new();
        let mut reinserted = World::new();
        write_resource(ticked(1))(&mut assigned);
        reinserted.insert_resource(ticked(1));
        for world in [&mut assigned, &mut reinserted] {
            world.increment_change_tick();
        }

        let mut queue = CommandQueue::default();
        queue.push(write_resource(ticked(2)));
        queue.apply(&mut assigned);
        reinserted.insert_resource(ticked(2));

        let state = |world: &World| {
            let time = world.resource_ref::<Time>();
            (time.elapsed(), time.added(), time.last_changed())
        };
        assert_eq!(state(&assigned), state(&reinserted));
        assert_eq!(state(&assigned).0, Duration::from_secs(2));
    }
}
