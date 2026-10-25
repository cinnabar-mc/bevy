//! Exercises real descriptors through the optional driver cache and worker executor.

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
            targets: vec![Some(wgpu::TextureFormat::Rgba8Unorm.into())],
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
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.try_recv().unwrap().unwrap();
    assert_eq!(
        &readback.slice(..).get_mapped_range()[..4],
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
