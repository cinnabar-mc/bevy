use crate::diagnostic::internal::DiagnosticsRecorder;
use crate::render_phase::TrackedRenderPass;
use crate::render_resource::{CommandEncoder, RenderPassDescriptor};
use crate::renderer::{wgpu_wrapper, RenderDevice};
use alloc::borrow::Cow;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::change_detection::Tick;
use bevy_ecs::component::ComponentId;
use bevy_ecs::error::{BevyError, ErrorContext};
use bevy_ecs::prelude::*;
use bevy_ecs::query::{QueryData, QueryFilter, QueryState};
use bevy_ecs::schedule::{
    FixedBitSet, ScheduleLabel, SingleThreadedExecutor, SystemExecutor, SystemSchedule,
};
use bevy_ecs::system::{
    Deferred, SystemAccess, SystemBuffer, SystemMeta, SystemName, SystemParam,
    SystemParamValidationError,
};
use bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell;
use bevy_ecs::world::DeferredWorld;
#[cfg(feature = "trace")]
use bevy_log::info_span;
#[cfg(not(target_arch = "wasm32"))]
use bevy_tasks::ComputeTaskPool;
use core::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError};
use wgpu::CommandBuffer;

/// Label of the command encoder shared by render systems that run serially.
const FRAME_ENCODER_LABEL: &str = "frame_encoder";

/// Up to this many unfinished encoders are finished one after another on the calling thread,
/// because spawning and joining a task for each costs more than finishing them.
#[cfg(not(target_arch = "wasm32"))]
const MAX_SEQUENTIAL_FINISHES: usize = 4;

#[derive(Default)]
struct PendingCommandBuffersInner {
    commands: Vec<PendingCommandBuffer>,
    /// The frame's open command encoder, whose commands follow every entry of `commands`.
    frame_encoder: Option<CommandEncoder>,
    /// Whether a [`RenderContext`] holds the frame encoder for the system it is running.
    frame_encoder_leased: bool,
    /// Whether render systems currently run one at a time in topological order, which lets
    /// [`RenderContext`]s record into the frame encoder.
    serial: bool,
}

impl PendingCommandBuffersInner {
    /// Queues the open frame encoder, so that anything queued next follows its commands.
    fn close_frame_encoder(&mut self) {
        if let Some(encoder) = self.frame_encoder.take() {
            self.push_frame_encoder(encoder);
        }
    }

    fn push_frame_encoder(&mut self, encoder: CommandEncoder) {
        self.commands.push(PendingCommandBuffer::Encoder {
            encoder,
            #[cfg(feature = "trace")]
            name: FRAME_ENCODER_LABEL.into(),
        });
    }
}

enum PendingCommandBuffer {
    Buffer(CommandBuffer),
    Encoder {
        encoder: CommandEncoder,
        #[cfg(feature = "trace")]
        name: Cow<'static, str>,
    },
}

wgpu_wrapper!(struct WgpuPendingCommandBuffersInner(PendingCommandBuffersInner));

/// A resource that holds command buffers and encoders that are pending submission to the render queue.
///
/// Render systems run by a [`SerialRenderExecutor`] record into one frame command encoder kept
/// open after the pending commands. Queuing anything else closes that encoder first, so command
/// buffers are always submitted in the order their commands were recorded.
#[derive(Resource)]
pub struct PendingCommandBuffers(Mutex<WgpuPendingCommandBuffersInner>);

impl Default for PendingCommandBuffers {
    fn default() -> Self {
        Self(Mutex::new(WgpuPendingCommandBuffersInner::new(
            PendingCommandBuffersInner::default(),
        )))
    }
}

impl PendingCommandBuffers {
    fn inner(&mut self) -> &mut PendingCommandBuffersInner {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    fn lock(&self) -> MutexGuard<'_, WgpuPendingCommandBuffersInner> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn push(&mut self, buffers: impl IntoIterator<Item = CommandBuffer>) {
        let mut buffers = buffers.into_iter().peekable();
        if buffers.peek().is_none() {
            return;
        }
        let inner = self.inner();
        inner.close_frame_encoder();
        inner
            .commands
            .extend(buffers.map(PendingCommandBuffer::Buffer));
    }

    fn append(&mut self, commands: &mut Vec<PendingCommandBuffer>) {
        let inner = self.inner();
        inner.close_frame_encoder();
        inner.commands.append(commands);
    }

    pub fn push_encoder(&mut self, encoder: CommandEncoder, name: impl Into<Cow<'static, str>>) {
        #[cfg(not(feature = "trace"))]
        let _ = name;

        let inner = self.inner();
        inner.close_frame_encoder();
        inner.commands.push(PendingCommandBuffer::Encoder {
            encoder,
            #[cfg(feature = "trace")]
            name: name.into(),
        });
    }

    /// Returns the frame's open command encoder, creating it if needed.
    ///
    /// Commands recorded into it are submitted after everything already pending and before
    /// anything queued later, without the cost of another command buffer.
    pub fn command_encoder(&mut self, render_device: &RenderDevice) -> &mut CommandEncoder {
        self.inner().frame_encoder.get_or_insert_with(|| {
            render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(FRAME_ENCODER_LABEL),
            })
        })
    }

    /// Finishes every pending encoder, returning all pending command buffers in submission order.
    ///
    /// A few encoders are finished in order as the returned iterator advances; more are finished
    /// in parallel up front.
    pub fn finish(&mut self) -> impl Iterator<Item = CommandBuffer> {
        #[cfg(feature = "trace")]
        let _finish_command_buffers_span = info_span!("finish_command_buffers").entered();

        let inner = self.inner();
        inner.close_frame_encoder();
        let commands = inner.commands.drain(..);

        #[cfg(target_arch = "wasm32")]
        {
            finish_sequential(commands)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let encoders = commands
                .as_slice()
                .iter()
                .filter(|command| matches!(command, PendingCommandBuffer::Encoder { .. }))
                .count();
            if encoders <= MAX_SEQUENTIAL_FINISHES {
                FinishedCommandBuffers::Sequential(finish_sequential(commands))
            } else {
                FinishedCommandBuffers::Parallel(finish_parallel(commands))
            }
        }
    }

    /// Returns whether no command buffer is pending submission.
    pub fn is_empty(&self) -> bool {
        let inner = self.lock();
        inner.commands.is_empty() && inner.frame_encoder.is_none()
    }

    /// Returns how many command buffers [`Self::finish`] would return.
    pub fn len(&self) -> usize {
        let inner = self.lock();
        inner.commands.len() + usize::from(inner.frame_encoder.is_some())
    }

    /// Sets whether render systems run serially and may share the frame encoder, returning the
    /// previous setting.
    fn set_serial(&mut self, serial: bool) -> bool {
        core::mem::replace(&mut self.inner().serial, serial)
    }

    /// Lends the frame encoder to a [`RenderContext`] while render systems run serially. The lent
    /// encoder is `None` when the frame has no open encoder yet.
    fn lease_frame_encoder(&self) -> Option<Option<CommandEncoder>> {
        let mut inner = self.lock();
        if !inner.serial || inner.frame_encoder_leased {
            return None;
        }
        inner.frame_encoder_leased = true;
        Some(inner.frame_encoder.take())
    }

    /// Ends a lease, keeping `encoder` open as the frame encoder.
    fn release_frame_encoder(&self, encoder: Option<CommandEncoder>) {
        let mut inner = self.lock();
        inner.frame_encoder_leased = false;
        if encoder.is_some() {
            inner.close_frame_encoder();
            inner.frame_encoder = encoder;
        }
    }

    /// Queues the leased `encoder`, if any, followed by `buffer`.
    fn push_leased(&self, encoder: Option<CommandEncoder>, buffer: CommandBuffer) {
        let mut inner = self.lock();
        inner.close_frame_encoder();
        if let Some(encoder) = encoder {
            inner.push_frame_encoder(encoder);
        }
        inner.commands.push(PendingCommandBuffer::Buffer(buffer));
    }
}

/// Finishes pending command buffers sequentially, preserving their order.
fn finish_sequential(
    commands: impl Iterator<Item = PendingCommandBuffer>,
) -> impl Iterator<Item = CommandBuffer> {
    commands.map(|command| match command {
        PendingCommandBuffer::Buffer(command_buffer) => command_buffer,
        PendingCommandBuffer::Encoder {
            encoder,
            #[cfg(feature = "trace")]
            name,
        } => {
            #[cfg(feature = "trace")]
            let _span = info_span!("finish_command_buffer", system = name.as_ref()).entered();
            encoder.finish()
        }
    })
}

/// Finishes pending command encoders in parallel on the [`ComputeTaskPool`],
/// then reassembles the command buffers in their original order.
#[cfg(not(target_arch = "wasm32"))]
fn finish_parallel(
    commands: impl Iterator<Item = PendingCommandBuffer>,
) -> impl Iterator<Item = CommandBuffer> {
    let mut command_buffers = Vec::with_capacity(commands.size_hint().0);
    let mut finished_encoders = ComputeTaskPool::get().scope(|scope| {
        for (index, command) in commands.into_iter().enumerate() {
            match command {
                PendingCommandBuffer::Buffer(command_buffer) => {
                    command_buffers.push((index, command_buffer));
                }
                PendingCommandBuffer::Encoder {
                    encoder,
                    #[cfg(feature = "trace")]
                    name,
                } => {
                    scope.spawn(async move {
                        #[cfg(feature = "trace")]
                        let _span =
                            info_span!("finish_command_buffer", system = name.as_ref()).entered();
                        (index, encoder.finish())
                    });
                }
            }
        }
    });

    command_buffers.append(&mut finished_encoders);
    command_buffers.sort_unstable_by_key(|(index, _)| *index);
    command_buffers
        .into_iter()
        .map(|(_, command_buffer)| command_buffer)
}

/// Command buffers finished either in order as they are iterated or up front in parallel.
#[cfg(not(target_arch = "wasm32"))]
enum FinishedCommandBuffers<S, P> {
    Sequential(S),
    Parallel(P),
}

#[cfg(not(target_arch = "wasm32"))]
impl<S, P> Iterator for FinishedCommandBuffers<S, P>
where
    S: Iterator<Item = CommandBuffer>,
    P: Iterator<Item = CommandBuffer>,
{
    type Item = CommandBuffer;

    fn next(&mut self) -> Option<CommandBuffer> {
        match self {
            Self::Sequential(buffers) => buffers.next(),
            Self::Parallel(buffers) => buffers.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Sequential(buffers) => buffers.size_hint(),
            Self::Parallel(buffers) => buffers.size_hint(),
        }
    }
}

/// Runs a render schedule's systems one at a time in topological order on the calling thread,
/// letting their [`RenderContext`]s record into the frame's shared command encoder instead of
/// creating one encoder each.
///
/// [`super::RenderGraph`] and the core 2D and 3D camera schedules use this executor. Render systems
/// of schedules with another executor record into their own encoders, so they can record in
/// parallel; their command buffers are still queued in topological system order.
pub struct SerialRenderExecutor(SingleThreadedExecutor);

impl SerialRenderExecutor {
    /// Creates a new serial render executor.
    pub const fn new() -> Self {
        Self(SingleThreadedExecutor::new())
    }
}

impl Default for SerialRenderExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemExecutor for SerialRenderExecutor {
    fn init(&mut self, schedule: &SystemSchedule) {
        self.0.init(schedule);
    }

    fn run(
        &mut self,
        schedule: &mut SystemSchedule,
        world: &mut World,
        skip_systems: Option<&FixedBitSet>,
        error_handler: fn(BevyError, ErrorContext),
    ) {
        let serial = set_serial(world, true);
        self.0.run(schedule, world, skip_systems, error_handler);
        if let Some(serial) = serial {
            set_serial(world, serial);
        }
    }

    fn set_apply_final_deferred(&mut self, value: bool) {
        self.0.set_apply_final_deferred(value);
    }
}

/// Runs a render schedule from within another one, such as a camera's schedule from the root
/// render graph.
///
/// Whether the schedule's [`RenderContext`]s share the frame encoder then depends only on its own
/// executor, not on the executor of the schedule running it.
pub fn run_render_schedule(world: &mut World, label: impl ScheduleLabel) {
    let serial = set_serial(world, false);
    world.run_schedule(label);
    if let Some(serial) = serial {
        set_serial(world, serial);
    }
}

/// Sets whether render systems may share the frame encoder, returning the previous setting if the
/// world has [`PendingCommandBuffers`].
fn set_serial(world: &mut World, serial: bool) -> Option<bool> {
    let mut pending = world.get_resource_mut::<PendingCommandBuffers>()?;
    Some(pending.bypass_change_detection().set_serial(serial))
}

/// How a [`RenderContext`] records during the current system run.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum FrameLease {
    /// Decided on the first use of the context in a system run.
    #[default]
    Undecided,
    /// The run holds the frame encoder of [`PendingCommandBuffers`], records into it and queues
    /// command buffers directly.
    Leased,
    /// The run records into its own encoder, queued when the system's buffers are applied.
    Own,
}

#[derive(Default)]
struct RenderContextStateInner {
    command_encoder: Option<CommandEncoder>,
    commands: Vec<PendingCommandBuffer>,
    lease: FrameLease,
}

impl RenderContextStateInner {
    fn flush_encoder(&mut self) {
        if let Some(encoder) = self.command_encoder.take() {
            self.commands.push(PendingCommandBuffer::Encoder {
                encoder,
                #[cfg(feature = "trace")]
                name: "RenderContextState::flush_encoder".into(),
            });
        }
    }

    /// On the first use in a system run, borrows the frame encoder if render systems run
    /// serially and nothing of an earlier run is still waiting to be applied.
    fn decide_lease(&mut self, pending: Option<&PendingCommandBuffers>) {
        if self.lease != FrameLease::Undecided {
            return;
        }
        self.lease = FrameLease::Own;
        if self.command_encoder.is_some() || !self.commands.is_empty() {
            return;
        }
        if let Some(encoder) = pending.and_then(PendingCommandBuffers::lease_frame_encoder) {
            self.command_encoder = encoder;
            self.lease = FrameLease::Leased;
        }
    }

    fn command_encoder(
        &mut self,
        pending: Option<&PendingCommandBuffers>,
        render_device: &RenderDevice,
        label: &str,
    ) -> &mut CommandEncoder {
        self.decide_lease(pending);
        self.command_encoder.get_or_insert_with(|| {
            render_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        })
    }
}

wgpu_wrapper!(struct WgpuRenderContextStateInner(RenderContextStateInner));

/// A resource that holds the current render context state, including command encoder and command buffers.
/// This is used internally by the [`RenderContext`] system parameter. Implements [`SystemBuffer`] to
/// append command buffers and unfinished encoders in topological system order. Pending encoders are
/// finished immediately before submission.
pub struct RenderContextState(WgpuRenderContextStateInner);

impl Default for RenderContextState {
    fn default() -> Self {
        Self(WgpuRenderContextStateInner::new(
            RenderContextStateInner::default(),
        ))
    }
}

impl SystemBuffer for RenderContextState {
    fn queue(&mut self, _system_meta: &SystemMeta, mut world: DeferredWorld) {
        #[cfg(feature = "trace")]
        let _span =
            info_span!("RenderContextState::apply", system = %_system_meta.name()).entered();

        let inner = &mut *self.0;

        inner.flush_encoder();

        #[cfg(feature = "trace")]
        for command in &mut inner.commands {
            if let PendingCommandBuffer::Encoder { name, .. } = command {
                *name = _system_meta.name().clone().into();
            }
        }

        if !inner.commands.is_empty() {
            let mut pending = world.resource_mut::<PendingCommandBuffers>();
            pending.append(&mut inner.commands);
        }
    }
}

/// A system parameter that provides access to a command encoder and render device for issuing
/// rendering commands inside any system running beneath the root [`super::RenderGraph`] schedule in the
/// [`super::render_system`] system.
///
/// Systems run by a [`SerialRenderExecutor`] record into the frame's shared command encoder and
/// queue added command buffers right away. Systems of other schedules record into their own
/// encoder, queued with their added command buffers in topological system order when the
/// system's buffers are applied.
#[derive(SystemParam)]
pub struct RenderContext<'w, 's> {
    state: Deferred<'s, RenderContextState>,
    system_name: SystemName,
    render_device: Res<'w, RenderDevice>,
    diagnostics_recorder: Option<Res<'w, DiagnosticsRecorder>>,
    pending: Option<Res<'w, PendingCommandBuffers>>,
}

impl<'w, 's> RenderContext<'w, 's> {
    /// Returns the render device associated with this render context.
    pub fn render_device(&self) -> &RenderDevice {
        &self.render_device
    }

    /// Returns the diagnostics recorder, if available.
    pub fn diagnostic_recorder(&self) -> Option<Res<'w, DiagnosticsRecorder>> {
        self.diagnostics_recorder.as_ref().map(Res::clone)
    }

    /// Returns the current command encoder, creating one if it does not already exist.
    pub fn command_encoder(&mut self) -> &mut CommandEncoder {
        self.state.0.command_encoder(
            self.pending.as_deref(),
            &self.render_device,
            self.system_name.as_str(),
        )
    }

    /// Begins a tracked render pass with the given descriptor.
    pub fn begin_tracked_render_pass<'a>(
        &'a mut self,
        descriptor: RenderPassDescriptor<'_>,
    ) -> TrackedRenderPass<'a> {
        let command_encoder = self.state.0.command_encoder(
            self.pending.as_deref(),
            &self.render_device,
            self.system_name.as_str(),
        );
        let render_pass = command_encoder.begin_render_pass(&descriptor);
        TrackedRenderPass::new(&self.render_device, render_pass)
    }

    /// Adds a finished command buffer to be submitted after the commands recorded so far.
    pub fn add_command_buffer(&mut self, command_buffer: CommandBuffer) {
        let pending = self.pending.as_deref();
        let state = &mut *self.state.0;
        state.decide_lease(pending);
        match (state.lease, pending) {
            (FrameLease::Leased, Some(pending)) => {
                pending.push_leased(state.command_encoder.take(), command_buffer);
            }
            _ => {
                state.flush_encoder();
                state
                    .commands
                    .push(PendingCommandBuffer::Buffer(command_buffer));
            }
        }
    }
}

impl Drop for RenderContext<'_, '_> {
    fn drop(&mut self) {
        let state = &mut *self.state.0;
        if state.lease == FrameLease::Leased
            && let Some(pending) = self.pending.as_deref()
        {
            pending.release_frame_encoder(state.command_encoder.take());
        }
        state.lease = FrameLease::Undecided;
    }
}

/// A system parameter that can be used to explicitly flush pending command buffers to the render queue.
/// This is typically not necessary, as command buffers are automatically flushed at the end of each
/// render system. However, in some cases it may be useful to flush command buffers earlier.
#[derive(SystemParam)]
pub struct FlushCommands<'w> {
    pending: ResMut<'w, PendingCommandBuffers>,
    queue: Res<'w, super::RenderQueue>,
}

impl<'w> FlushCommands<'w> {
    /// Flushes all pending command buffers to the render queue.
    pub fn flush(&mut self) {
        let mut buffers = self.pending.finish().peekable();
        if buffers.peek().is_some() {
            self.queue.submit(buffers);
        }
    }
}

/// The entity corresponding to the current view being rendered.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Deref, DerefMut)]
pub struct CurrentView(pub Entity);

/// A query that fetches components for the entity corresponding to the current view being rendered,
/// as defined by the [`CurrentView`] resource, equivalent to `query.get(current_view.entity())`.
pub struct ViewQuery<'w, 's, D: QueryData, F: QueryFilter = ()> {
    entity: Entity,
    item: D::Item<'w, 's>,
    _filter: PhantomData<F>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> ViewQuery<'w, 's, D, F> {
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    pub fn into_inner(self) -> D::Item<'w, 's> {
        self.item
    }
}

pub struct ViewQueryState<D: QueryData, F: QueryFilter> {
    resource_id: ComponentId,
    query_state: QueryState<D, F>,
}

// SAFETY: ViewQuery accesses the CurrentView resource (read) and query components.
// Access is properly registered in init_access.
unsafe impl<'a, D: QueryData + 'static, F: QueryFilter + 'static> SystemParam
    for ViewQuery<'a, '_, D, F>
{
    type State = ViewQueryState<D, F>;
    type Item<'w, 's> = ViewQuery<'w, 's, D, F>;

    fn init_state(world: &mut World) -> Self::State {
        ViewQueryState {
            resource_id: world
                .components_registrator()
                .register_component::<CurrentView>(),
            query_state: QueryState::new(world),
        }
    }

    fn init_access(
        state: &Self::State,
        system_meta: &mut SystemMeta,
        system_access: &mut SystemAccess,
        world: &mut World,
    ) {
        let component_access_set = system_access.require_shared_access::<Self>(system_meta);
        component_access_set.add_resource_read(state.resource_id);

        <Query<'_, '_, D, F> as SystemParam>::init_access(
            &state.query_state,
            system_meta,
            system_access,
            world,
        );
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        _system_meta: &SystemMeta,
        world: UnsafeWorldCell<'w>,
        _change_tick: Tick,
    ) -> Result<Self::Item<'w, 's>, SystemParamValidationError> {
        // SAFETY: We have registered resource read access in init_access
        let current_view = unsafe { world.get_resource::<CurrentView>() };

        let Some(current_view) = current_view else {
            return Err(SystemParamValidationError::skipped::<Self>(
                "CurrentView resource not present",
            ));
        };

        let entity = current_view.entity();

        // SAFETY: Query state access is properly registered in init_access.
        // The caller ensures the world matches the one used in init_state.
        let item = unsafe { state.query_state.get_unchecked(world, entity) }.map_err(|_| {
            SystemParamValidationError::skipped::<Self>("Current view entity does not match query")
        })?;

        Ok(ViewQuery {
            entity,
            item,
            _filter: PhantomData,
        })
    }
}

// SAFETY: ViewQuery with ReadOnlyQueryData only reads from the world.
unsafe impl<'w, 's, D: bevy_ecs::query::ReadOnlyQueryData + 'static, F: QueryFilter + 'static>
    bevy_ecs::system::ReadOnlySystemParam for ViewQuery<'w, 's, D, F>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::RenderQueue;
    use crate::test_utils::create_dummy_device;
    use bevy_ecs::schedule::Schedule;
    use bevy_tasks::TaskPool;

    #[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
    struct Outer;

    #[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
    struct Inner;

    fn world() -> World {
        ComputeTaskPool::get_or_init(TaskPool::default);
        let (device, queue) = create_dummy_device();
        let mut world = World::new();
        world.insert_resource(device);
        world.insert_resource(queue);
        world.init_resource::<PendingCommandBuffers>();
        world
    }

    fn schedule(label: impl ScheduleLabel, serial: bool) -> Schedule {
        let mut schedule = Schedule::new(label);
        if serial {
            schedule.set_executor(SerialRenderExecutor::new());
        } else {
            schedule.set_executor(SingleThreadedExecutor::new());
        }
        schedule
    }

    /// Pending entries in submission order: `E` for a closed encoder, `B` for a finished command
    /// buffer and `F` for the open frame encoder.
    fn entries(world: &mut World) -> String {
        let mut pending = world.resource_mut::<PendingCommandBuffers>();
        let inner = pending.inner();
        assert!(!inner.frame_encoder_leased, "leases end with their system");
        inner
            .commands
            .iter()
            .map(|command| match command {
                PendingCommandBuffer::Buffer(_) => 'B',
                PendingCommandBuffer::Encoder { .. } => 'E',
            })
            .chain(inner.frame_encoder.as_ref().map(|_| 'F'))
            .collect()
    }

    /// Finishes and submits everything pending, returning the number of command buffers.
    fn submit(world: &mut World) -> usize {
        let queue = world.resource::<RenderQueue>().clone();
        let buffers: Vec<_> = world
            .resource_mut::<PendingCommandBuffers>()
            .finish()
            .collect();
        let count = buffers.len();
        queue.submit(buffers);
        assert!(world.resource::<PendingCommandBuffers>().is_empty());
        count
    }

    fn encode(mut context: RenderContext) {
        context.command_encoder().insert_debug_marker("encode");
    }

    fn add_buffer(mut context: RenderContext) {
        let buffer = context
            .render_device()
            .create_command_encoder(&Default::default())
            .finish();
        context.add_command_buffer(buffer);
    }

    fn encode_and_add_buffer(mut context: RenderContext) {
        context.command_encoder().insert_debug_marker("encode");
        let buffer = context
            .render_device()
            .create_command_encoder(&Default::default())
            .finish();
        context.add_command_buffer(buffer);
    }

    fn push_buffer(world: &mut World) {
        let buffer = world
            .resource::<RenderDevice>()
            .create_command_encoder(&Default::default())
            .finish();
        world.resource_mut::<PendingCommandBuffers>().push([buffer]);
    }

    #[test]
    fn serial_render_systems_share_one_frame_encoder() {
        let mut world = world();
        let mut schedule = schedule(Outer, true);
        schedule.add_systems((encode, encode, encode).chain());
        schedule.run(&mut world);
        assert_eq!(entries(&mut world), "F");
        assert_eq!(submit(&mut world), 1);
    }

    #[test]
    fn added_and_pushed_buffers_split_the_frame_encoder_in_recording_order() {
        let mut world = world();
        let mut schedule = schedule(Outer, true);
        schedule.add_systems((encode, add_buffer, encode, push_buffer, encode, encode).chain());
        schedule.run(&mut world);
        assert_eq!(entries(&mut world), "EBEBF");
        assert_eq!(submit(&mut world), 5);
    }

    #[test]
    fn systems_of_other_executors_keep_their_own_encoders() {
        let mut world = world();
        let mut schedule = schedule(Outer, false);
        schedule.add_systems((encode, add_buffer, encode, encode).chain());
        schedule.run(&mut world);
        assert_eq!(entries(&mut world), "EBEE");
        assert_eq!(submit(&mut world), 4);
    }

    #[test]
    fn many_encoders_finish_in_parallel() {
        let mut world = world();
        let mut schedule = schedule(Outer, false);
        schedule.add_systems((encode, encode, encode, add_buffer, encode, encode, encode).chain());
        schedule.run(&mut world);
        assert_eq!(entries(&mut world), "EEEBEEE");
        assert_eq!(submit(&mut world), 7);
    }

    fn run_inner(world: &mut World) {
        run_render_schedule(world, Inner);
    }

    #[test]
    fn nested_render_schedules_share_only_with_a_serial_executor() {
        for (inner_serial, expected) in [(false, "EEBF"), (true, "EBF")] {
            let mut world = world();
            let mut inner = schedule(Inner, inner_serial);
            inner.add_systems(encode_and_add_buffer);
            world.add_schedule(inner);
            let mut outer = schedule(Outer, true);
            outer.add_systems((encode, run_inner, encode).chain());
            outer.run(&mut world);
            assert_eq!(
                entries(&mut world),
                expected,
                "inner serial: {inner_serial}"
            );
            assert_eq!(submit(&mut world), expected.len());
        }
    }

    #[test]
    fn exclusive_systems_record_into_the_frame_encoder() {
        let mut world = world();
        let mut schedule = schedule(Outer, true);
        schedule.add_systems(
            (
                encode,
                |world: &mut World| {
                    world.resource_scope(|world, mut pending: Mut<PendingCommandBuffers>| {
                        pending
                            .command_encoder(world.resource::<RenderDevice>())
                            .insert_debug_marker("exclusive");
                    });
                },
                encode,
            )
                .chain(),
        );
        schedule.run(&mut world);
        assert_eq!(entries(&mut world), "F");
        assert_eq!(submit(&mut world), 1);
    }
}
