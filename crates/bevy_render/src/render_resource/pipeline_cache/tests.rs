//! Exercises real descriptors through the optional driver cache and worker executor, and the
//! terminal state of a pipeline the driver failed to build.

use super::*;

const SOURCE: &str = "
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let positions = array(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
    return vec4f(positions[index], 0., 1.);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(0.25, 0.5, 0.75, 1.); }
@compute @workgroup_size(1) fn compute() {}
";

/// Creates and renders a worker-compiled pipeline on Metal, or exercises cache forwarding on Vulkan.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "A missing native adapter must have a visible fixture diagnostic."
)]
fn native_driver_cache_and_worker_compilation() {
    let backend = if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Ok(adapter) = bevy_tasks::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("missing-fixture: native Metal/Vulkan adapter for pipeline compilation");
        return;
    };
    let features = adapter.features() & wgpu::Features::PIPELINE_CACHE;
    let (device, queue) = bevy_tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        ..Default::default()
    }))
    .unwrap();
    let driver_cache = features.contains(wgpu::Features::PIPELINE_CACHE).then(|| {
        // SAFETY: No initial bytes are provided.
        unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: None,
                data: None,
                fallback: true,
            })
        }
    });
    bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::new);
    let mut cache = PipelineCache::new(RenderDevice::from(device.clone()), false);
    cache.set_async_pipeline_compilation_on_macos(true);
    cache.set_pipeline_cache(driver_cache.clone());
    let shader = Handle::<Shader>::default();
    cache.set_shader(
        shader.id(),
        Shader::from_wgsl(SOURCE, "pipeline-worker-test.wgsl"),
    );
    let render = cache.queue_render_pipeline(RenderPipelineDescriptor {
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..Default::default()
        },
        fragment: Some(FragmentState {
            shader: shader.clone(),
            entry_point: Some("fragment".into()),
            targets: vec![Some(TextureFormat::Rgba8Unorm.into())],
            ..Default::default()
        }),
        ..Default::default()
    });
    let compute = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        shader,
        entry_point: Some("compute".into()),
        ..Default::default()
    });
    cache.process_queue();
    cache.block_on_render_pipeline(render);
    let pipeline = cache
        .get_render_pipeline(render)
        .expect("worker render pipeline");
    let target = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let readback = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 256,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    device.poll(PollType::wait_indefinitely()).unwrap();
    receiver.try_recv().unwrap().unwrap();
    assert_eq!(
        &readback.get_mapped_range(..).unwrap()[..4],
        &[64, 128, 191, 255]
    );
    readback.unmap();
    // Join the compute task directly so this test needs no clock, sleeps or polling deadline.
    let state = &mut cache.pipelines[compute.id()].state;
    if let CachedPipelineState::Creating(task) = state {
        *state = match bevy_tasks::block_on(task) {
            Ok(pipeline) => CachedPipelineState::Ok(pipeline),
            Err(error) => CachedPipelineState::Err(error),
        };
    }
    assert!(cache.get_compute_pipeline(compute).is_some());
    if let Some(driver_cache) = driver_cache {
        assert!(
            driver_cache.get_data().is_some(),
            "Vulkan cache must be serializable"
        );
    }
    cache.set_pipeline_cache(None);
}

/// A cache on the noop backend that compiles on the caller, with one queued render pipeline.
fn noop_cache() -> (wgpu::Device, PipelineCache, CachedRenderPipelineId) {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let mut cache = PipelineCache::new(RenderDevice::from(device.clone()), true);
    let shader = Handle::<Shader>::default();
    cache.set_shader(
        shader.id(),
        Shader::from_wgsl(SOURCE, "pipeline-failure-test.wgsl"),
    );
    let render = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("failure test".into()),
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..Default::default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fragment".into()),
            targets: vec![Some(TextureFormat::Rgba8Unorm.into())],
            ..Default::default()
        }),
        ..Default::default()
    });
    (device, cache, render)
}

/// A driver failure stays an error without being polled or waited for again, and compiles
/// again only when the application asks.
#[test]
fn a_driver_failure_is_terminal_until_the_application_retries() {
    let (_device, mut cache, render) = noop_cache();
    cache.process_queue();
    assert!(cache.get_render_pipeline(render).is_some());
    assert!(
        !cache.retry_failed_pipeline(render.id()),
        "a ready pipeline must not be recompiled"
    );
    assert!(!cache.retry_failed_pipeline(render.id() + 1));

    cache.pipelines[render.id()].state =
        CachedPipelineState::Err(ShaderCacheError::CreatePipeline("driver refused".into()));
    cache.waiting_pipelines.insert(render.id());
    for _ in 0..3 {
        cache.process_queue();
        cache.block_on_render_pipeline(render);
    }
    assert!(matches!(
        cache.get_render_pipeline_state(render),
        CachedPipelineState::Err(ShaderCacheError::CreatePipeline(description))
            if description == "driver refused"
    ));
    assert!(cache.get_render_pipeline(render).is_none());
    assert_eq!(cache.waiting_pipelines().count(), 0);

    assert!(cache.retry_failed_pipeline(render.id()));
    cache.process_queue();
    assert!(cache.get_render_pipeline(render).is_some());
}

/// Only an internal error the scope already resolved counts; anything else leaves the
/// created object in use.
#[test]
fn only_a_resolved_internal_error_is_a_driver_failure() {
    let internal = wgpu::Error::Internal {
        source: "driver".into(),
        description: "vkCreateGraphicsPipelines failed".into(),
    };
    assert_eq!(
        driver_failure(Some(Some(internal))).as_deref(),
        Some("vkCreateGraphicsPipelines failed")
    );
    let validation = wgpu::Error::Validation {
        source: "caller".into(),
        description: "bad descriptor".into(),
    };
    assert_eq!(driver_failure(Some(Some(validation))), None);
    let out_of_memory = wgpu::Error::OutOfMemory {
        source: "allocator".into(),
    };
    assert_eq!(driver_failure(Some(Some(out_of_memory))), None);
    assert_eq!(driver_failure(Some(None)), None);
    assert_eq!(driver_failure(None), None);
}

/// On a worker thread the internal scope returns the created value, leaves a validation error
/// to the caller's own scope, and is popped before that scope.
#[test]
fn the_internal_scope_passes_other_errors_to_the_enclosing_scope() {
    let (device, _cache, _render) = noop_cache();
    let render_device = RenderDevice::from(device);
    std::thread::spawn(move || {
        assert_eq!(catch_driver_failure(&render_device, || 7), Ok(7));
        let validation = render_device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let created = catch_driver_failure(&render_device, || {
            render_device
                .wgpu_device()
                .create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: 0,
                        height: 0,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
        });
        assert!(created.is_ok());
        assert!(matches!(
            bevy_tasks::block_on(validation.pop()),
            Some(wgpu::Error::Validation { .. })
        ));
    })
    .join()
    .unwrap();
}
