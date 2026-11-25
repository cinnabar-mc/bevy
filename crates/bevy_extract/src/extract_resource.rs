use alloc::vec::Vec;
use core::marker::PhantomData;

use bevy_app::{App, AppLabel, Plugin};
#[expect(
    deprecated,
    reason = "`FilteredResourcesMutParamBuilder` is the builder for the `FilteredResourcesMut` used here."
)]
use bevy_ecs::system::FilteredResourcesMutParamBuilder;
#[expect(
    deprecated,
    reason = "`FilteredResourcesMut` gives one system write access to a set of resources chosen at run time."
)]
use bevy_ecs::world::{FilteredResourcesMut, FilteredResourcesMutBuilder};
use bevy_ecs::{
    change_detection::Tick,
    component::Mutable,
    prelude::*,
    system::{LocalBuilder, ParamBuilder, SystemParamBuilder, SystemState},
};
pub use bevy_extract_macros::ExtractResource;
use bevy_utils::once;

use crate::{Extract, ExtractSchedule, MainWorld};

/// Describes how a resource gets extracted for processing.
///
/// Therefore the resource is transferred from the "main world" into the "sub world"
/// in the [`ExtractSchedule`] step.
///
/// The marker type `F` is only used as a way to bypass the orphan rules. To
/// implement the trait for a foreign type you can use a local type as the
/// marker, e.g. the type of the plugin that calls [`ExtractResourcePlugin`].
pub trait ExtractResource<L: AppLabel, F = ()>: Resource {
    type Source: Resource;

    /// Defines how the resource is transferred into the "sub world".
    fn extract_resource(source: &Self::Source) -> Self;
}

/// This plugin extracts the resources into the "sub world".
///
/// Therefore it sets up the[`ExtractSchedule`] step
/// for the specified [`Resource`].
///
/// The marker type `F` is only used as a way to bypass the orphan rules. To
/// implement the trait for a foreign type you can use a local type as the
/// marker, e.g. the type of the plugin that calls [`ExtractResourcePlugin`].
pub struct ExtractResourcePlugin<R: ExtractResource<L, F>, L: AppLabel, F = ()>(
    PhantomData<(R, L, F)>,
);

impl<R: ExtractResource<L, F>, L: AppLabel, F> Default for ExtractResourcePlugin<R, L, F> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<
        R: ExtractResource<L, F, Mutability = Mutable>,
        L: AppLabel + Default,
        F: 'static + Send + Sync,
    > Plugin for ExtractResourcePlugin<R, L, F>
{
    fn build(&self, app: &mut App) {
        if let Some(sub_app) = app.get_sub_app_mut(L::default()) {
            let world = sub_app.world_mut();
            let registry = world.get_resource_or_init::<ExtractResourceRegistry<L>>();
            if registry.built {
                // Registered after the batched system was built: extract it on its own.
                sub_app.add_systems(ExtractSchedule, extract_resource::<R, L, F>);
            } else {
                world
                    .resource_mut::<ExtractResourceRegistry<L>>()
                    .extractors
                    .push(ResourceExtractor {
                        access: |builder| {
                            builder.add_write::<R>();
                        },
                        extract: extract_registered::<R, L, F>,
                    });
            }
        } else {
            once!(bevy_log::error!(
                "Sub app did not exist when trying to add `extract_resource` for <{}>.",
                core::any::type_name::<R>()
            ));
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(sub_app) = app.get_sub_app_mut(L::default()) {
            add_batched_extraction::<L>(sub_app);
        }
    }
}

/// The set holding the system that extracts every resource registered with
/// [`ExtractResourcePlugin`] in sub app `L`. Order against this set to run before or after those
/// extractions.
#[derive(SystemSet)]
pub struct ExtractResourceSystems<L>(PhantomData<fn() -> L>);

impl<L> Default for ExtractResourceSystems<L> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<L> Clone for ExtractResourceSystems<L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for ExtractResourceSystems<L> {}

impl<L> PartialEq for ExtractResourceSystems<L> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<L> Eq for ExtractResourceSystems<L> {}

impl<L> core::hash::Hash for ExtractResourceSystems<L> {
    fn hash<H: core::hash::Hasher>(&self, _state: &mut H) {}
}

impl<L> core::fmt::Debug for ExtractResourceSystems<L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ExtractResourceSystems<{}>", core::any::type_name::<L>())
    }
}

/// The resources registered with [`ExtractResourcePlugin`] for sub app `L`. One system extracts
/// them all, so a frame pays one system's scheduling, fetch and apply overhead for all of them
/// instead of one per resource.
#[derive(Resource)]
struct ExtractResourceRegistry<L> {
    extractors: Vec<ResourceExtractor>,
    /// Whether the batched system was built; later registrations get their own system.
    built: bool,
    marker: PhantomData<fn() -> L>,
}

impl<L> Default for ExtractResourceRegistry<L> {
    fn default() -> Self {
        Self {
            extractors: Vec::new(),
            built: false,
            marker: PhantomData,
        }
    }
}

#[expect(
    deprecated,
    reason = "`FilteredResourcesMut` gives one system write access to a set of resources chosen at run time."
)]
struct ResourceExtractor {
    /// Declares write access to the extracted resource in the sub world.
    access: fn(&mut FilteredResourcesMutBuilder),
    /// Extracts it, as [`extract_resource`] does, with the batch's main-world ticks.
    extract: fn(&mut Commands, &World, Tick, Tick, &mut FilteredResourcesMut),
}

/// Adds the system that extracts every registered resource, once, after all plugins have built.
#[expect(
    deprecated,
    reason = "`FilteredResourcesMut` gives one system write access to a set of resources chosen at run time."
)]
fn add_batched_extraction<L: AppLabel + Default>(sub_app: &mut bevy_app::SubApp) {
    let world = sub_app.world_mut();
    let Some(mut registry) = world.get_resource_mut::<ExtractResourceRegistry<L>>() else {
        return;
    };
    if registry.built {
        return;
    }
    registry.built = true;
    let extractors = core::mem::take(&mut registry.extractors);
    if extractors.is_empty() {
        return;
    }
    let accesses: Vec<_> = extractors
        .iter()
        .map(|extractor| extractor.access)
        .collect();
    let resources =
        FilteredResourcesMutParamBuilder::new(move |builder: &mut FilteredResourcesMutBuilder| {
            for access in accesses {
                access(builder);
            }
        });
    let state: SystemState<(
        Commands<'static, 'static>,
        Res<'static, MainWorld>,
        Local<'static, Option<Tick>>,
        FilteredResourcesMut<'static, 'static>,
    )> = (
        ParamBuilder,
        ParamBuilder,
        LocalBuilder(None::<Tick>),
        resources,
    )
        .build_state(world);
    let system = state.build_system(
        move |mut commands: Commands,
              main_world: Res<MainWorld>,
              mut last_run: Local<Option<Tick>>,
              mut resources: FilteredResourcesMut| {
            let main_world: &World = &main_world;
            // The same ticks an `Extract` parameter takes from the main world: the previous
            // run's tick, or on the first run the tick a new `SystemState` starts from (one
            // past the current tick, so every existing change counts).
            let previous = last_run
                .unwrap_or_else(|| Tick::new(main_world.read_change_tick().get().wrapping_add(1)));
            let this_run = main_world
                .as_unsafe_world_cell_readonly()
                .increment_change_tick();
            for extractor in &extractors {
                (extractor.extract)(
                    &mut commands,
                    main_world,
                    previous,
                    this_run,
                    &mut resources,
                );
            }
            *last_run = Some(this_run);
        },
    );
    sub_app.add_systems(
        ExtractSchedule,
        system.in_set(ExtractResourceSystems::<L>::default()),
    );
}

/// [`extract_resource`] for one registered resource, inside the batched system.
#[expect(
    deprecated,
    reason = "`FilteredResourcesMut` gives one system write access to a set of resources chosen at run time."
)]
fn extract_registered<R: ExtractResource<L, F, Mutability = Mutable>, L: AppLabel, F>(
    commands: &mut Commands,
    main_world: &World,
    last_run: Tick,
    this_run: Tick,
    resources: &mut FilteredResourcesMut,
) {
    let (Some(main_resource), Some(ticks)) = (
        main_world.get_resource::<R::Source>(),
        main_world.get_resource_change_ticks::<R::Source>(),
    ) else {
        return;
    };
    if let Ok(mut target_resource) = resources.get_mut::<R>() {
        if ticks.is_changed(last_run, this_run) {
            *target_resource = R::extract_resource(main_resource);
        }
    } else {
        #[cfg(debug_assertions)]
        if !ticks.is_added(last_run, this_run) {
            once!(bevy_log::warn!(
                "Removing resource {} from sub world not expected, adding using `Commands`.
                This may decrease performance",
                core::any::type_name::<R>()
            ));
        }

        commands.insert_resource(R::extract_resource(main_resource));
    }
}

/// This system extracts the resource of the corresponding [`Resource`] type
pub fn extract_resource<R: ExtractResource<L, F, Mutability = Mutable>, L: AppLabel, F>(
    mut commands: Commands,
    main_resource: Extract<Option<Res<R::Source>>>,
    target_resource: Option<ResMut<R>>,
) {
    if let Some(main_resource) = main_resource.as_ref() {
        if let Some(mut target_resource) = target_resource {
            if main_resource.is_changed() {
                *target_resource = R::extract_resource(main_resource);
            }
        } else {
            #[cfg(debug_assertions)]
            if !main_resource.is_added() {
                once!(bevy_log::warn!(
                    "Removing resource {} from sub world not expected, adding using `Commands`.
                This may decrease performance",
                    core::any::type_name::<R>()
                ));
            }

            commands.insert_resource(R::extract_resource(main_resource));
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_app::{App, AppLabel};
    use bevy_ecs::{prelude::*, schedule::ScheduleLabel};

    use super::{ExtractResource, ExtractResourcePlugin};
    use crate::extract_plugin::ExtractPlugin;

    #[derive(AppLabel, Debug, Hash, PartialEq, Eq, Clone, Default, Copy)]
    struct SubApp;

    #[derive(ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone, Default)]
    struct SubSchedule;

    #[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
    enum SubSystems {
        ExtractCommands,
        PostCleanup,
    }

    #[derive(Resource, Clone, Debug, PartialEq)]
    struct Value(u32);

    impl ExtractResource<SubApp> for Value {
        type Source = Value;
        fn extract_resource(source: &Value) -> Self {
            source.clone()
        }
    }

    #[derive(Resource, Clone, Debug, PartialEq)]
    struct Late(u32);

    impl ExtractResource<SubApp> for Late {
        type Source = Late;
        fn extract_resource(source: &Late) -> Self {
            source.clone()
        }
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(ExtractPlugin::<SubApp>::new(
            |_, _| {},
            || {
                let mut schedule = Schedule::new(SubSchedule);
                schedule
                    .configure_sets((SubSystems::ExtractCommands, SubSystems::PostCleanup).chain());
                schedule
            },
            SubSchedule.intern(),
            SubSystems::ExtractCommands.intern(),
            SubSystems::PostCleanup.intern(),
        ));
        app.add_plugins(ExtractResourcePlugin::<Value, SubApp>::default());
        app.get_sub_app_mut(SubApp).unwrap().update_schedule = Some(SubSchedule.intern());
        app
    }

    fn sub_value(app: &mut App) -> (Option<Value>, bool) {
        let world = app.get_sub_app_mut(SubApp).unwrap().world_mut();
        world
            .run_system_cached(|value: Option<Res<Value>>| {
                (
                    value.as_deref().cloned(),
                    value.is_some_and(|v| v.is_changed()),
                )
            })
            .unwrap()
    }

    #[test]
    fn registered_resources_are_inserted_then_updated_only_when_changed() {
        let mut app = app();
        app.finish();
        app.cleanup();
        app.insert_resource(Value(1));
        app.update();
        assert_eq!(
            sub_value(&mut app).0,
            Some(Value(1)),
            "inserted on first extraction"
        );

        app.update();
        assert_eq!(
            sub_value(&mut app),
            (Some(Value(1)), false),
            "unchanged source"
        );

        app.world_mut().resource_mut::<Value>().0 = 2;
        app.update();
        assert_eq!(
            sub_value(&mut app),
            (Some(Value(2)), true),
            "changed source"
        );
    }

    #[derive(Resource, Default)]
    struct SeenAfter(Vec<(Option<Value>, Option<Late>)>);

    #[test]
    fn systems_after_the_extraction_set_see_this_frames_values() {
        let mut app = app();
        app.add_plugins(ExtractResourcePlugin::<Late, SubApp>::default());
        let sub_app = app.get_sub_app_mut(SubApp).unwrap();
        sub_app.world_mut().init_resource::<SeenAfter>();
        sub_app.add_systems(
            crate::ExtractSchedule,
            (|value: Option<Res<Value>>, late: Option<Res<Late>>, mut seen: ResMut<SeenAfter>| {
                seen.0
                    .push((value.as_deref().cloned(), late.as_deref().cloned()));
            })
            .after(super::ExtractResourceSystems::<SubApp>::default()),
        );
        app.finish();
        app.cleanup();
        app.insert_resource(Value(1));
        app.insert_resource(Late(7));
        // The first extraction inserts through commands, applied after the extract schedule.
        app.update();
        app.world_mut().resource_mut::<Value>().0 = 2;
        app.update();
        let world = app.get_sub_app_mut(SubApp).unwrap().world_mut();
        assert_eq!(
            world.resource::<SeenAfter>().0,
            vec![(None, None), (Some(Value(2)), Some(Late(7)))]
        );
    }
}
