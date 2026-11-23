//! The core rendering pipelines schedules. These schedules define the "default" render graph
//! for 2D and 3D rendering in Bevy.
//!
//! Rendering in Bevy is "camera driven", meaning that for each camera in the world, its
//! associated rendering schedule is executed. This allows different cameras to have different
//! rendering pipelines, for example a 3D camera with post-processing effects and a 2D camera
//! with a simple clear and sprite rendering.
//!
//! The [`camera_driver`] system is responsible for iterating over all cameras in the world
//! and executing their associated schedules. In this way, the schedule for each camera is a
//! sub-schedule or sub-graph of the root render graph schedule.
use core::fmt::{self, Display, Formatter};

use bevy_camera::{ClearColor, NormalizedRenderTarget};
use bevy_ecs::{
    entity::EntityHashSet,
    prelude::*,
    schedule::{InternedScheduleLabel, IntoScheduleConfigs, Schedule, ScheduleLabel, SystemSet},
    system::SystemState,
};
use bevy_reflect::Reflect;
use bevy_render::{
    camera::{ExtractedCamera, SortedCameras},
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
    render_resource::{
        CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
        RenderPassDescriptor, StoreOp,
    },
    renderer::{
        run_render_schedule, CurrentView, PendingCommandBuffers, RenderDevice, RenderQueue,
        SerialRenderExecutor,
    },
    sync_world::MainEntity,
    view::ExtractedWindow,
};

/// Schedule label for the Core 3D rendering pipeline.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Core3d;

/// System sets for the Core 3D rendering pipeline, defining the main stages of rendering.
/// These stages include and run in the following order:
/// - `Prepass`: Initial rendering operations, such as depth pre-pass.
/// - `MainPass`: The primary rendering operations, including drawing opaque and transparent objects.
/// - `EarlyPostProcess`: Initial post processing effects.
/// - `PostProcess`: Final rendering operations, such as post-processing effects.
///
/// Additional systems can be added to these sets to customize the rendering pipeline, or additional
/// sets can be created relative to these core sets.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Core3dSystems {
    Prepass,
    MainPass,
    EarlyPostProcess,
    PostProcess,
}

impl Core3d {
    pub fn base_schedule() -> Schedule {
        use bevy_ecs::schedule::ScheduleBuildSettings;
        use Core3dSystems::*;

        let mut schedule = Schedule::new(Self);
        // Systems run one at a time and share the frame's command encoder.
        schedule.set_executor(SerialRenderExecutor::new());

        schedule.set_build_settings(ScheduleBuildSettings {
            auto_insert_apply_deferred: false,
            ..Default::default()
        });

        schedule.configure_sets((Prepass, MainPass, EarlyPostProcess, PostProcess).chain_weak());

        schedule
    }
}

/// Schedule label for the Core 2D rendering pipeline.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Core2d;

/// System sets for the Core 2D rendering pipeline, defining the main stages of rendering.
/// These stages include and run in the following order:
/// - `Prepass`: Initial rendering operations, such as depth pre-pass.
/// - `MainPass`: The primary rendering operations, including drawing 2D sprites and meshes.
/// - `EarlyPostProcess`: Initial post processing effects.
/// - `PostProcess`: Final rendering operations, such as post-processing effects.
///
/// Additional systems can be added to these sets to customize the rendering pipeline, or additional
/// sets can be created relative to these core sets.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Core2dSystems {
    Prepass,
    MainPass,
    EarlyPostProcess,
    PostProcess,
}

impl Core2d {
    pub fn base_schedule() -> Schedule {
        use bevy_ecs::schedule::ScheduleBuildSettings;
        use Core2dSystems::*;

        let mut schedule = Schedule::new(Self);
        // Systems run one at a time and share the frame's command encoder.
        schedule.set_executor(SerialRenderExecutor::new());

        schedule.set_build_settings(ScheduleBuildSettings {
            auto_insert_apply_deferred: false,
            ..Default::default()
        });

        schedule.configure_sets((Prepass, MainPass, EarlyPostProcess, PostProcess).chain_weak());

        schedule
    }
}

/// The windows that are a render target for a camera, from [`camera_driver`] until
/// [`handle_uncovered_swap_chains`] takes them.
///
/// The resource stays in the render world between frames, so handing the set over each frame
/// spawns and despawns nothing.
#[derive(Resource, Default)]
struct CameraWindows(Option<EntityHashSet>);

/// The windows and the auxiliary views that [`camera_driver`] reads, with query states kept
/// between frames.
type CameraDriverQueries = (
    Query<'static, 'static, (MainEntity, &'static ExtractedWindow)>,
    Query<'static, 'static, Entity, With<RootNonCameraView>>,
);

/// A render-world marker component for a view that corresponds to neither a
/// camera nor a camera-associated shadow map.
///
/// This is used for point light and spot light shadow maps, since these aren't
/// associated with views.
#[derive(Clone, Copy, Component, Debug, Reflect)]
#[reflect(Clone, Component)]
#[reflect(from_reflect = false)]
pub struct RootNonCameraView(#[reflect(ignore)] pub InternedScheduleLabel);

/// The default entry point for camera driven rendering added to the root [`bevy_render::renderer::RenderGraph`]
/// schedule. This system iterates over all cameras in the world, executing their associated
/// rendering schedules defined by the [`bevy_render::camera::CameraRenderGraph`] component.
///
/// After all camera schedules have run, swap chains that were not covered by a camera are cleared
/// and every pending command buffer is submitted to the GPU at once. Users can order any additional
/// operations (e.g. one-off compute passes) before or after this system in the root render
/// graph schedule.
pub fn camera_driver(world: &mut World, state: &mut SystemState<CameraDriverQueries>) {
    // Gather up all cameras and auxiliary views not associated with a camera.
    let root_views: Vec<_> = {
        let auxiliary_views = state
            .get(world)
            .map(|(_, auxiliary_views)| auxiliary_views.iter().collect::<Vec<_>>())
            .unwrap_or_default();
        let sorted = world.resource::<SortedCameras>();
        auxiliary_views
            .into_iter()
            .map(RootView::Auxiliary)
            .chain(sorted.0.iter().map(|c| RootView::Camera {
                entity: c.entity,
                order: c.order,
            }))
            .collect()
    };

    let mut camera_windows = EntityHashSet::default();

    let diagnostics_enabled = world.contains_resource::<DiagnosticsRecorder>();
    let mut reached_root_camera = false;
    if diagnostics_enabled {
        let device = world.resource::<RenderDevice>().clone();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("camera_driver_diagnostic_span"),
        });
        world
            .resource::<DiagnosticsRecorder>()
            .begin_time_span(&mut encoder, "auxiliary_views".into());
        world
            .resource_mut::<PendingCommandBuffers>()
            .push_encoder(encoder, "camera_driver_span_begin");
    }
    let end_time_span = |world: &mut World| {
        let device = world.resource::<RenderDevice>().clone();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("camera_driver_diagnostic_span"),
        });
        world
            .resource::<DiagnosticsRecorder>()
            .end_time_span(&mut encoder);
        world
            .resource_mut::<PendingCommandBuffers>()
            .push_encoder(encoder, "camera_driver_span_end");
    };

    for root_view in root_views {
        let mut run_schedule = true;
        let (schedule, view_entity);

        match root_view {
            RootView::Camera {
                entity: camera_entity,
                ..
            } => {
                if diagnostics_enabled && !reached_root_camera {
                    reached_root_camera = true;
                    end_time_span(world);
                }
                let Some(camera) = world.get::<ExtractedCamera>(camera_entity) else {
                    continue;
                };

                schedule = camera.schedule;
                let target = camera.target.clone();

                if let Some(NormalizedRenderTarget::Window(window_ref)) = &target {
                    let window_entity = window_ref.entity();

                    if let Ok((windows, _)) = state.get(world)
                        && windows
                            .iter()
                            .find(|(e, _)| *e == window_entity)
                            .is_some_and(|(_, w)| w.physical_width > 0 && w.physical_height > 0)
                    {
                        camera_windows.insert(window_entity);
                    } else {
                        run_schedule = false;
                    }
                }

                view_entity = camera_entity;
            }

            RootView::Auxiliary(auxiliary_view_entity) => {
                let Some(root_view) = world.get::<RootNonCameraView>(auxiliary_view_entity) else {
                    continue;
                };

                view_entity = auxiliary_view_entity;
                schedule = root_view.0;
            }
        }

        if run_schedule {
            world.insert_resource(CurrentView(view_entity));

            #[cfg(feature = "trace")]
            let _span =
                bevy_log::info_span!("camera_schedule", camera = root_view.to_string()).entered();

            run_render_schedule(world, schedule);
        }
    }
    if diagnostics_enabled && !reached_root_camera {
        end_time_span(world);
    }

    world.remove_resource::<CurrentView>();

    world.get_resource_or_init::<CameraWindows>().0 = Some(camera_windows);
}

/// A view not associated with any other camera.
enum RootView {
    /// A camera.
    Camera { entity: Entity, order: isize },

    /// An auxiliary view not associated with a camera.
    ///
    /// This is currently used for point and spot light shadow maps.
    Auxiliary(Entity),
}

impl Display for RootView {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            RootView::Camera { entity, order } => write!(f, "Camera {} ({:?})", order, entity),
            RootView::Auxiliary(entity) => write!(f, "Auxiliary View {:?}", entity),
        }
    }
}

pub(crate) fn submit_pending_command_buffers(
    world: &mut World,
    state: &mut SystemState<(ResMut<PendingCommandBuffers>, Res<RenderQueue>)>,
) {
    let (mut pending, queue) = state.get_mut(world).unwrap();
    pending.submit(&queue);
}

pub(crate) fn handle_uncovered_swap_chains(
    world: &mut World,
    windows: &mut SystemState<Query<(MainEntity, &ExtractedWindow)>>,
) {
    let windows_to_clear: Vec<_> = {
        let clear_color = world.resource::<ClearColor>().0.to_linear();
        let Some(camera_windows) = world
            .get_resource_mut::<CameraWindows>()
            .and_then(|mut camera_windows| camera_windows.0.take())
        else {
            return;
        };
        let Ok(windows) = windows.get(world) else {
            return;
        };

        windows
            .iter()
            .filter_map(|(window_entity, window)| {
                if camera_windows.contains(&window_entity.entity()) {
                    return None;
                }
                let swap_chain_texture = window.swap_chain_texture_view.as_ref()?;
                Some((swap_chain_texture.clone(), clear_color))
            })
            .collect()
    };

    if windows_to_clear.is_empty() {
        return;
    }

    world.resource_scope(|world, mut pending: Mut<PendingCommandBuffers>| {
        let encoder = pending.command_encoder(world.resource::<RenderDevice>());
        for (swap_chain_texture, clear_color) in &windows_to_clear {
            #[cfg(feature = "trace")]
            let _span = bevy_log::info_span!("no_camera_clear_pass").entered();

            let pass_descriptor = RenderPassDescriptor {
                label: Some("no_camera_clear_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: swap_chain_texture,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear((*clear_color).into()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            encoder.begin_render_pass(&pass_descriptor);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::schedule::Schedule;

    #[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
    struct AuxiliaryView;

    #[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
    struct Frame;

    #[derive(Resource, Default)]
    struct Runs(usize);

    #[derive(Component)]
    struct Marker;

    fn frame_world() -> (World, Schedule) {
        let mut world = World::new();
        world.init_resource::<SortedCameras>();
        world.init_resource::<ClearColor>();
        world.init_resource::<Runs>();
        let mut auxiliary = Schedule::new(AuxiliaryView);
        auxiliary.add_systems(|mut runs: ResMut<Runs>| runs.0 += 1);
        world.add_schedule(auxiliary);
        let mut frame = Schedule::new(Frame);
        frame.add_systems((camera_driver, handle_uncovered_swap_chains).chain());
        (world, frame)
    }

    #[test]
    fn auxiliary_views_spawned_between_frames_are_driven() {
        let (mut world, mut frame) = frame_world();
        frame.run(&mut world);
        assert_eq!(world.resource::<Runs>().0, 0);
        world.spawn(RootNonCameraView(AuxiliaryView.intern()));
        frame.run(&mut world);
        assert_eq!(world.resource::<Runs>().0, 1);
        // A view in a new archetype is found too.
        world.spawn((RootNonCameraView(AuxiliaryView.intern()), Marker));
        frame.run(&mut world);
        assert_eq!(world.resource::<Runs>().0, 3);
    }

    #[test]
    fn camera_windows_are_handed_over_once_per_frame() {
        let (mut world, mut frame) = frame_world();
        frame.run(&mut world);
        assert!(world.resource::<CameraWindows>().0.is_none());
        let entities = world.entities().count_spawned();
        for _ in 0..3 {
            frame.run(&mut world);
            assert!(world.resource::<CameraWindows>().0.is_none());
        }
        assert_eq!(
            world.entities().count_spawned(),
            entities,
            "the hand-off leaves the render world's entities alone"
        );
    }
}
