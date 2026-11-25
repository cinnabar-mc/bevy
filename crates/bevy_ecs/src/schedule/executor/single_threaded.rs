#[cfg(feature = "std")]
use core::panic::AssertUnwindSafe;

use fixedbitset::FixedBitSet;

#[cfg(feature = "trace")]
use alloc::string::ToString as _;
#[cfg(feature = "trace")]
use tracing::info_span;

#[cfg(feature = "std")]
use crate::{error::BevyError, system::BoxedSystem};
use crate::{
    error::{ErrorContext, ErrorHandler},
    schedule::{
        is_apply_deferred, BoxedCondition, ConditionWithAccess, SystemExecutor, SystemSchedule,
    },
    system::{RunSystemError, ScheduleSystem},
    world::World,
};

#[cfg(feature = "hotpatching")]
use crate::{change_detection::DetectChanges, HotPatchChanges};

use super::__rust_begin_short_backtrace;

/// Runs the schedule using a single thread.
///
/// Useful if you're dealing with a single-threaded environment, saving your threads for
/// other things, or just trying minimize overhead.
#[derive(Default)]
pub struct SingleThreadedExecutor {
    /// System sets whose conditions have been evaluated.
    evaluated_sets: FixedBitSet,
    /// Systems that have run or been skipped.
    completed_systems: FixedBitSet,
    /// Systems that have run but have not had their buffers applied.
    unapplied_systems: FixedBitSet,
    /// Systems whose [`System::may_apply_deferred`] is true. Applying any other system does
    /// nothing, so only these are queued for application. The answer holds for the system's whole
    /// lifetime, so it is read once here. It is a property of the parameter types rather than
    /// [`System::has_deferred`], which a custom parameter could leave unset while still applying
    /// something.
    ///
    /// [`System::may_apply_deferred`]: crate::system::System::may_apply_deferred
    /// [`System::has_deferred`]: crate::system::System::has_deferred
    deferred_systems: FixedBitSet,
    /// Systems that are [`ApplyDeferred`](crate::schedule::ApplyDeferred) sync points. A system's
    /// type never changes, so this is recorded once instead of asking each system for its type on
    /// every run.
    sync_points: FixedBitSet,
    /// Systems in at least one set with conditions. The others skip the walk over their sets,
    /// which would scan a bitset sized to every set in the schedule and find nothing.
    in_conditioned_sets: FixedBitSet,
    /// Setting when true applies deferred system buffers after all systems have run
    apply_final_deferred: bool,
}

impl SystemExecutor for SingleThreadedExecutor {
    fn init(&mut self, schedule: &SystemSchedule) {
        // pre-allocate space
        let sys_count = schedule.system_ids.len();
        let set_count = schedule.set_ids.len();
        self.evaluated_sets = FixedBitSet::with_capacity(set_count);
        self.completed_systems = FixedBitSet::with_capacity(sys_count);
        self.unapplied_systems = FixedBitSet::with_capacity(sys_count);
        self.deferred_systems = FixedBitSet::with_capacity(sys_count);
        self.sync_points = FixedBitSet::with_capacity(sys_count);
        self.in_conditioned_sets = FixedBitSet::with_capacity(sys_count);
        // The schedule sizes this list by the schedule graph's systems, which can outnumber the
        // systems it runs; only the first `sys_count` entries belong to them.
        for (index, sets) in schedule
            .sets_with_conditions_of_systems
            .iter()
            .take(sys_count)
            .enumerate()
        {
            self.in_conditioned_sets.set(index, !sets.is_clear());
        }
        for (index, system) in schedule.systems.iter().enumerate() {
            self.deferred_systems
                .set(index, system.system.may_apply_deferred());
            self.sync_points
                .set(index, is_apply_deferred(&*system.system));
        }
    }

    fn run(
        &mut self,
        schedule: &mut SystemSchedule,
        world: &mut World,
        _skip_systems: Option<&FixedBitSet>,
        error_handler: ErrorHandler,
    ) {
        // If stepping is enabled, make sure we skip those systems that should
        // not be run.
        #[cfg(feature = "bevy_debug_stepping")]
        if let Some(skipped_systems) = _skip_systems {
            // mark skipped systems as completed
            self.completed_systems |= skipped_systems;
        }

        #[cfg(feature = "hotpatching")]
        let hotpatch_tick = world
            .get_resource_ref::<HotPatchChanges>()
            .map(|r| r.last_changed())
            .unwrap_or_default();

        for system_index in 0..schedule.systems.len() {
            let system = &mut schedule.systems[system_index].system;

            #[cfg(feature = "trace")]
            let name = system.name();
            #[cfg(feature = "trace")]
            let should_run_span = info_span!("check_conditions", name = name.to_string()).entered();

            let mut should_run = !self.completed_systems.contains(system_index);
            let conditioned_sets = self
                .in_conditioned_sets
                .contains(system_index)
                .then(|| &schedule.sets_with_conditions_of_systems[system_index]);
            for set_idx in conditioned_sets.into_iter().flat_map(FixedBitSet::ones) {
                if self.evaluated_sets.contains(set_idx) {
                    continue;
                }

                // evaluate system set's conditions
                let set_conditions_met = evaluate_and_fold_conditions(
                    &mut schedule.set_conditions[set_idx],
                    world,
                    error_handler,
                    system,
                    true,
                );

                if !set_conditions_met {
                    self.completed_systems
                        .union_with(&schedule.systems_in_sets_with_conditions[set_idx]);
                }

                should_run &= set_conditions_met;
                self.evaluated_sets.insert(set_idx);
            }

            // evaluate system's conditions
            let system_conditions_met = evaluate_and_fold_conditions(
                &mut schedule.system_conditions[system_index],
                world,
                error_handler,
                system,
                false,
            );

            should_run &= system_conditions_met;

            #[cfg(feature = "trace")]
            should_run_span.exit();

            #[cfg(feature = "hotpatching")]
            if hotpatch_tick.is_newer_than(system.get_last_run(), world.change_tick()) {
                system.refresh_hotpatch();
            }

            // system has either been skipped or will run
            self.completed_systems.insert(system_index);

            if !should_run {
                continue;
            }

            if self.sync_points.contains(system_index) {
                self.apply_deferred(schedule, world, error_handler);
                continue;
            }

            let f = |system: &mut _| {
                if let Err(RunSystemError::Failed(err)) =
                    __rust_begin_short_backtrace::run_without_applying_deferred(system, world)
                {
                    error_handler(
                        err,
                        ErrorContext::System {
                            name: system.name(),
                            last_run: system.get_last_run(),
                        },
                    );
                }
            };

            #[cfg(feature = "std")]
            {
                handle_unwind(f, system, error_handler, "System panicked");
            }

            #[cfg(not(feature = "std"))]
            {
                let mut f = f;
                (f)(system);
            }

            if self.deferred_systems.contains(system_index) {
                self.unapplied_systems.insert(system_index);
            }
        }

        if self.apply_final_deferred {
            self.apply_deferred(schedule, world, error_handler);
        }
        self.evaluated_sets.clear();
        self.completed_systems.clear();
    }

    fn set_apply_final_deferred(&mut self, apply_final_deferred: bool) {
        self.apply_final_deferred = apply_final_deferred;
    }
}

impl SingleThreadedExecutor {
    /// Creates a new single-threaded executor for use in a [`Schedule`].
    ///
    /// [`Schedule`]: crate::schedule::Schedule
    pub const fn new() -> Self {
        Self {
            evaluated_sets: FixedBitSet::new(),
            completed_systems: FixedBitSet::new(),
            unapplied_systems: FixedBitSet::new(),
            deferred_systems: FixedBitSet::new(),
            sync_points: FixedBitSet::new(),
            in_conditioned_sets: FixedBitSet::new(),
            apply_final_deferred: true,
        }
    }

    fn apply_deferred(
        &mut self,
        schedule: &mut SystemSchedule,
        world: &mut World,
        error_handler: ErrorHandler,
    ) {
        for system_index in self.unapplied_systems.ones() {
            let system = &mut schedule.systems[system_index].system;
            #[cfg(not(feature = "std"))]
            {
                system.apply_deferred(world);
                let _ = error_handler;
            }

            #[cfg(feature = "std")]
            {
                handle_unwind(
                    |system| system.apply_deferred(world),
                    system,
                    error_handler,
                    "Encountered a panic while applying system buffers",
                );
            }
        }

        self.unapplied_systems.clear();
    }
}

fn evaluate_and_fold_conditions(
    conditions: &mut [ConditionWithAccess],
    world: &mut World,
    error_handler: ErrorHandler,
    for_system: &ScheduleSystem,
    on_set: bool,
) -> bool {
    #[cfg(feature = "hotpatching")]
    let hotpatch_tick = world
        .get_resource_ref::<HotPatchChanges>()
        .map(|r| r.last_changed())
        .unwrap_or_default();

    #[expect(
        clippy::unnecessary_fold,
        reason = "Short-circuiting here would prevent conditions from mutating their own state as needed."
    )]
    conditions
        .iter_mut()
        .map(|ConditionWithAccess { condition, .. }| {
            #[cfg(feature = "hotpatching")]
            if hotpatch_tick.is_newer_than(condition.get_last_run(), world.change_tick()) {
                condition.refresh_hotpatch();
            }
            let f = |condition: &mut BoxedCondition| {
                __rust_begin_short_backtrace::readonly_run(&mut **condition, world).unwrap_or_else(
                    |err| {
                        if let RunSystemError::Failed(err) = err {
                            error_handler(
                                err,
                                ErrorContext::RunCondition {
                                    name: condition.name(),
                                    last_run: condition.get_last_run(),
                                    system: for_system.name(),
                                    on_set,
                                },
                            );
                        };
                        false
                    },
                )
            };
            #[cfg(not(feature = "std"))]
            let result = {
                let mut f = f;
                f(condition)
            };
            #[cfg(feature = "std")]
            let result =
                handle_unwind_in_run_condition(f, condition, for_system, on_set, error_handler);
            result
        })
        .fold(true, |acc, res| acc && res)
}

/// Handle a potential panic by invoking the error handler
#[cfg(feature = "std")]
fn handle_unwind(
    f: impl FnOnce(&mut BoxedSystem),
    system: &mut BoxedSystem,
    error_handler: ErrorHandler,
    error_message: &str,
) {
    let potential_unwind = std::panic::catch_unwind(AssertUnwindSafe(|| f(system)));
    if let Err(payload) = potential_unwind {
        __rust_begin_short_backtrace::error_handler(
            error_handler,
            BevyError::panic(error_message, payload),
            ErrorContext::System {
                name: system.name(),
                last_run: system.get_last_run(),
            },
        );
    }
}

/// Handle a potential panic by invoking the error handler
#[cfg(feature = "std")]
fn handle_unwind_in_run_condition(
    f: impl FnOnce(&mut BoxedCondition) -> bool,
    condition: &mut BoxedCondition,
    for_system: &ScheduleSystem,
    on_set: bool,
    error_handler: ErrorHandler,
) -> bool {
    let potential_unwind = std::panic::catch_unwind(AssertUnwindSafe(|| f(condition)));
    match potential_unwind {
        Ok(r) => r,
        Err(payload) => {
            let err = BevyError::panic("Encountered panic", payload);
            __rust_begin_short_backtrace::error_handler(
                error_handler,
                err,
                ErrorContext::RunCondition {
                    name: condition.name(),
                    last_run: condition.get_last_run(),
                    system: for_system.name(),
                    on_set,
                },
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        change_detection::Tick,
        prelude::*,
        schedule::{ApplyDeferred, SingleThreadedExecutor},
        system::{SystemAccess, SystemMeta, SystemParam, SystemParamValidationError},
        world::unsafe_world_cell::UnsafeWorldCell,
    };
    use alloc::{vec, vec::Vec};

    #[derive(Component)]
    struct Marker;

    #[derive(Resource, Default)]
    struct Seen(Vec<usize>);

    fn spawn_one(mut commands: Commands) {
        commands.spawn(Marker);
    }

    fn count(markers: Query<&Marker>, mut seen: ResMut<Seen>) {
        seen.0.push(markers.iter().count());
    }

    #[test]
    fn deferred_buffers_apply_at_sync_points_and_at_the_end_among_systems_without_them() {
        let mut world = World::new();
        world.init_resource::<Seen>();
        let mut schedule = Schedule::default();
        schedule.set_executor(SingleThreadedExecutor::new());
        schedule.add_systems(
            (
                count,
                spawn_one,
                count,
                ApplyDeferred,
                count,
                spawn_one,
                count,
            )
                .chain_ignore_deferred(),
        );
        schedule.run(&mut world);
        // The second spawn is applied only by the schedule's final application.
        assert_eq!(world.resource::<Seen>().0, vec![0, 0, 1, 1]);
        assert_eq!(world.query::<&Marker>().iter(&world).count(), 2);
        schedule.run(&mut world);
        assert_eq!(world.resource::<Seen>().0, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[derive(crate::schedule::SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
    enum Gated {
        Open,
        Closed,
    }

    fn push(n: usize) -> impl FnMut(ResMut<Seen>) {
        move |mut seen: ResMut<Seen>| seen.0.push(n)
    }

    #[test]
    fn init_accepts_more_conditioned_set_entries_than_systems() {
        use crate::schedule::{executor::SystemSchedule, SystemExecutor};
        use fixedbitset::FixedBitSet;

        let mut schedule = SystemSchedule::new();
        let mut sets = FixedBitSet::with_capacity(1);
        sets.insert(0);
        schedule.sets_with_conditions_of_systems.push(sets);
        SingleThreadedExecutor::new().init(&schedule);
    }

    #[test]
    fn set_conditions_gate_only_the_systems_in_their_sets() {
        let mut world = World::new();
        world.init_resource::<Seen>();
        let mut schedule = Schedule::default();
        schedule.set_executor(SingleThreadedExecutor::new());
        schedule.configure_sets((Gated::Open.run_if(|| true), Gated::Closed.run_if(|| false)));
        schedule.add_systems(
            (
                push(1),
                push(2).in_set(Gated::Closed),
                push(3).in_set(Gated::Open),
                push(4),
                push(5).in_set(Gated::Open).in_set(Gated::Closed),
            )
                .chain(),
        );
        schedule.run(&mut world);
        assert_eq!(world.resource::<Seen>().0, vec![1, 3, 4]);
    }

    /// Counts spawn requests and spawns them in `apply`, without ever marking its system as having
    /// deferred buffers. Its buffer is only created on the system's first run, after the executor
    /// was built.
    struct LazySpawner<'s>(&'s mut Option<u32>);

    // SAFETY: The parameter accesses no world data.
    unsafe impl SystemParam for LazySpawner<'_> {
        type State = Option<u32>;
        type Item<'w, 's> = LazySpawner<'s>;

        fn init_state(_world: &mut World) -> Self::State {
            None
        }

        fn init_access(
            _state: &Self::State,
            _system_meta: &mut SystemMeta,
            _system_access: &mut SystemAccess,
            _world: &mut World,
        ) {
        }

        fn apply(state: &mut Self::State, _system_meta: &SystemMeta, world: &mut World) {
            for _ in 0..state.take().unwrap_or(0) {
                world.spawn(Marker);
            }
        }

        unsafe fn get_param<'w, 's>(
            state: &'s mut Self::State,
            _system_meta: &SystemMeta,
            _world: UnsafeWorldCell<'w>,
            _change_tick: Tick,
        ) -> Result<Self::Item<'w, 's>, SystemParamValidationError> {
            Ok(LazySpawner(state))
        }
    }

    fn lazy_spawn(spawner: LazySpawner) {
        *spawner.0.get_or_insert(0) += 1;
    }

    #[test]
    fn a_parameter_that_applies_without_the_deferred_flag_is_applied_at_the_sync_point() {
        let mut world = World::new();
        world.init_resource::<Seen>();
        let mut schedule = Schedule::default();
        schedule.set_executor(SingleThreadedExecutor::new());
        schedule.add_systems((count, lazy_spawn, ApplyDeferred, count).chain_ignore_deferred());
        schedule.run(&mut world);
        schedule.run(&mut world);
        assert_eq!(world.resource::<Seen>().0, vec![0, 1, 1, 2]);
    }

    #[derive(Resource)]
    struct Value(usize);

    fn write_one(mut commands: Commands) {
        commands.insert_resource(Value(1));
    }

    fn write_two(mut commands: Commands) {
        commands.insert_resource(Value(2));
    }

    fn read_value(value: Option<Res<Value>>, mut seen: ResMut<Seen>) {
        seen.0.push(value.map_or(0, |value| value.0));
    }

    #[test]
    fn buffers_meeting_at_one_sync_point_apply_in_schedule_order() {
        for (first_writes_two, expected) in [(false, 2), (true, 1)] {
            let mut world = World::new();
            world.init_resource::<Seen>();
            let mut schedule = Schedule::default();
            schedule.set_executor(SingleThreadedExecutor::new());
            if first_writes_two {
                schedule.add_systems(
                    (write_two, count, write_one, ApplyDeferred, read_value)
                        .chain_ignore_deferred(),
                );
            } else {
                schedule.add_systems(
                    (write_one, count, write_two, ApplyDeferred, read_value)
                        .chain_ignore_deferred(),
                );
            }
            schedule.run(&mut world);
            assert_eq!(world.resource::<Seen>().0, vec![0, expected]);
        }
    }

    #[derive(SystemParam)]
    #[expect(dead_code, reason = "Only the parameter type is inspected.")]
    struct Observing<'w, 's> {
        markers: Query<'w, 's, &'static Marker>,
        seen: Res<'w, Seen>,
        next: Local<'s, u32>,
    }

    #[derive(SystemParam)]
    #[expect(dead_code, reason = "Only the parameter type is inspected.")]
    struct Spawning<'w, 's> {
        commands: Commands<'w, 's>,
        seen: Res<'w, Seen>,
    }

    const _: () = {
        assert!(!<Observing as SystemParam>::MAY_APPLY_DEFERRED);
        assert!(
            !<(Query<&Marker>, ResMut<Seen>, Option<Res<Seen>>) as SystemParam>::MAY_APPLY_DEFERRED
        );
        assert!(<Spawning as SystemParam>::MAY_APPLY_DEFERRED);
        assert!(<Option<Commands> as SystemParam>::MAY_APPLY_DEFERRED);
        assert!(<LazySpawner as SystemParam>::MAY_APPLY_DEFERRED);
    };

    #[test]
    fn only_systems_whose_parameters_may_apply_report_it() {
        assert!(!IntoSystem::into_system(count).may_apply_deferred());
        assert!(IntoSystem::into_system(spawn_one).may_apply_deferred());
        assert!(IntoSystem::into_system(lazy_spawn).may_apply_deferred());
        assert!(!IntoSystem::into_system(count.pipe(|| {})).may_apply_deferred());
        assert!(IntoSystem::into_system(count.pipe(lazy_spawn_after)).may_apply_deferred());
    }

    fn lazy_spawn_after(spawner: LazySpawner) {
        lazy_spawn(spawner);
    }
}
