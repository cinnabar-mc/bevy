//! Deferred surface repair and admission of window-sized render targets.

/// Accepts only nonempty dimensions that the device can allocate.
pub fn window_size_is_valid(width: u32, height: u32, max_dimension: u32) -> bool {
    width != 0 && height != 0 && width <= max_dimension && height <= max_dimension
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Repair {
    #[default]
    None,
    Configure,
    Recreate,
}

#[derive(Default)]
pub(super) struct SurfaceRecovery {
    repair: Repair,
    validation_failed: bool,
}

impl SurfaceRecovery {
    /// Defers repair until all references to the previous frame have been released.
    pub(super) fn observe(&mut self, result: &wgpu::CurrentSurfaceTexture) {
        match result {
            wgpu::CurrentSurfaceTexture::Success(_)
            | wgpu::CurrentSurfaceTexture::Suboptimal(_) => self.acquired(),
            wgpu::CurrentSurfaceTexture::Lost => self.repair = Repair::Recreate,
            wgpu::CurrentSurfaceTexture::Outdated => self.repair = Repair::Configure,
            wgpu::CurrentSurfaceTexture::Validation => {
                self.repair = if self.validation_failed {
                    Repair::Recreate
                } else {
                    Repair::Configure
                };
                self.validation_failed = true;
            }
            _ => {}
        }
    }

    /// Clears failure history only after a frame was acquired successfully.
    pub(super) fn acquired(&mut self) {
        *self = Self::default();
    }

    /// Reports whether the next acquisition must wait for surface repair.
    pub(super) fn needs_configuration(&self) -> bool {
        self.repair != Repair::None
    }

    /// Reports whether configuring the existing surface is insufficient.
    pub(super) fn needs_recreation(&self) -> bool {
        self.repair == Repair::Recreate
    }

    /// Allows acquisition without forgetting a validation failure until a frame succeeds.
    pub(super) fn configured(&mut self) {
        self.repair = Repair::None;
    }
}

/// Drops the old surface before attaching a replacement to the same native window.
/// A failed creation leaves the slot empty so the next frame can retry safely.
pub(super) fn replace_surface<T, E>(
    surface: &mut Option<T>,
    create: impl FnOnce() -> Result<T, E>,
) -> Result<(), E> {
    drop(surface.take());
    *surface = Some(create()?);
    Ok(())
}

/// Consumes the initial-present requirement only when a drawable is actually submitted.
pub(super) fn present_frame<T>(
    frame: Option<T>,
    needs_initial_present: &mut bool,
    present: impl FnOnce(T),
) {
    if let Some(frame) = frame {
        present(frame);
        *needs_initial_present = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;

    #[test]
    fn initial_present_survives_suspension_until_a_drawable_is_submitted() {
        let mut initial = true;
        present_frame::<()>(None, &mut initial, |_| {
            panic!("no drawable while suspended")
        });
        assert!(initial);
        let mut submitted = false;
        present_frame(Some(()), &mut initial, |_| submitted = true);
        assert!(submitted);
        assert!(!initial);
    }

    #[test]
    fn rejects_impossible_sizes_and_accepts_the_next_legal_resize() {
        let limit = 8192;
        for (width, height) in [
            (0, 1),
            (1, 0),
            (limit + 1, 1),
            (1, limit + 1),
            (u32::MAX, u32::MAX),
        ] {
            assert!(!window_size_is_valid(width, height, limit));
            assert!(window_size_is_valid(limit, limit, limit));
            assert!(window_size_is_valid(1, 1, limit));
        }
        assert!(!window_size_is_valid(1, 1, 0));
    }

    #[test]
    fn lost_surface_is_replaced_after_releasing_its_ownership() {
        let old = Arc::new(());
        let weak = Arc::downgrade(&old);
        let mut surface = Some(old);
        let mut recovery = SurfaceRecovery::default();
        recovery.observe(&wgpu::CurrentSurfaceTexture::Lost);
        assert!(recovery.needs_recreation());
        replace_surface(&mut surface, || {
            assert!(weak.upgrade().is_none());
            Ok::<_, ()>(Arc::new(()))
        })
        .unwrap();
        recovery.configured();
        assert!(surface.is_some());
        assert!(!recovery.needs_configuration());
    }

    #[test]
    fn repeated_validation_escalates_to_replacement_and_failed_creation_can_retry() {
        let mut recovery = SurfaceRecovery::default();
        recovery.observe(&wgpu::CurrentSurfaceTexture::Validation);
        assert!(recovery.needs_configuration());
        assert!(!recovery.needs_recreation());
        recovery.configured();
        recovery.observe(&wgpu::CurrentSurfaceTexture::Validation);
        assert!(recovery.needs_recreation());
        let mut surface = Some(());
        assert!(replace_surface(&mut surface, || Err::<(), _>(())).is_err());
        assert!(surface.is_none());
        assert!(recovery.needs_recreation());
        replace_surface(&mut surface, || Ok::<_, ()>(())).unwrap();
        recovery.configured();
        assert!(surface.is_some());
        assert!(!recovery.needs_configuration());
        recovery.acquired();
        recovery.observe(&wgpu::CurrentSurfaceTexture::Validation);
        assert!(!recovery.needs_recreation());
    }

    #[test]
    fn temporary_unavailability_does_not_repair_the_surface() {
        let mut recovery = SurfaceRecovery::default();
        recovery.observe(&wgpu::CurrentSurfaceTexture::Occluded);
        recovery.observe(&wgpu::CurrentSurfaceTexture::Timeout);
        assert!(!recovery.needs_configuration());
        recovery.observe(&wgpu::CurrentSurfaceTexture::Outdated);
        assert!(recovery.needs_configuration());
        assert!(!recovery.needs_recreation());
    }
}
