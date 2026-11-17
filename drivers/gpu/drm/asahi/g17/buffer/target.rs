// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Rotating tile-map/heap metadata and the retained tail-pointer cache.
//! Larger targets replace only the objects that need more capacity. Every old
//! backing keeps the exact commands that still reference it until retirement.

use super::compact_alias;
use crate::g17::{
    fw::render::{RegisterPrograms, RenderPass},
    object::{CpuMap, KernelObject},
};
use crate::{
    driver::AsahiDrmDevice,
    hw::t8140::{queues, resources as cfg},
    mmu,
};
use kernel::prelude::*;

const DEPTH: usize = queues::RENDER_DEPTH as usize;

fn aligned(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or(EOVERFLOW)
}

/// The 25G83 tiler parameter block: the TPC and seven sub-buffers the tiling registers name, at
/// the default parameter-buffer sizes. Sizes are
/// [TPC, tile map × 64, 0x4000 for layered targets, 0, 0xc00, 0xa00, 0x600, 0x80].
const GTP_ALIGNMENTS: [usize; 8] = [0x40, 0x40, 0x100, 1, 0x20, 0x20, 0x80, 0x40];
const GTP_FIXED_SIZES: [usize; 4] = [0xc00, 0xa00, GTP6_SIZE, 0x80];
/// Room for the tiler's records across a partial render. macOS uses 0x1c00, so this is a superset,
/// and the size every partial-render test ran with.
const GTP_SLOTS: usize = 64;
const GTP6_SIZE: usize = GTP_SLOTS
    * (((cfg::TVB_MAX_BLOCKS * cfg::TVB_BLOCK_SIZE).div_ceil(0x8000) + 0x17ff) / 24 * 8 + 8);

/// Per-command geometry, independent of the capacity already allocated.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Layout {
    pub(crate) tilemap_size: usize,
    pub(crate) layer_offset: usize,
    pub(crate) heap_offset: usize,
    pub(crate) used: usize,
    pub(crate) tpc_size: usize,
    pub(crate) tvb_blocks: usize,
    /// Offsets of the tiler parameter sub-buffers in the TPC object (25G83), `None` for an
    /// empty one; the TPC is entry 0.
    gtp: [Option<usize>; 8],
    /// Bytes of the TPC object the command uses.
    tpc_end: usize,
    /// The M5 tiler slices of the TPC object and their cluster controls.
    m5: Option<(crate::g17::fw::m5::target::TargetSlices, [u64; 3])>,
}
impl Layout {
    pub(crate) fn new(
        pass: &RenderPass,
        clusters: u32,
        programs: RegisterPrograms,
    ) -> Result<Self> {
        let width = pass.width as usize;
        let height = pass.height as usize;
        let layers = pass.layers as usize;
        let (uw, uh) = (pass.utile_width as usize, pass.utile_height as usize);
        if width == 0
            || height == 0
            || layers == 0
            || clusters == 0
            || uw == 0
            || uh == 0
            || 32 % uw != 0
            || 32 % uh != 0
        {
            return Err(EINVAL);
        }
        let utiles = (32 / uw) * (32 / uh);
        let tiles_x = width.div_ceil(32);
        let tiles_y = height.div_ceil(32);
        let macro_x = aligned(tiles_x.div_ceil(4), 4)?;
        let macro_y = aligned(tiles_y.div_ceil(4), 4)?;
        let macro_tiles = macro_x.checked_mul(macro_y).ok_or(EOVERFLOW)?;
        let tilemap_size = aligned(
            5usize
                .checked_mul(macro_tiles)
                .and_then(|v| v.checked_mul(utiles))
                .ok_or(EOVERFLOW)?,
            4,
        )?
        .checked_mul(16)
        .and_then(|v| v.checked_mul(layers))
        .ok_or(EOVERFLOW)?;
        let tpc_size = 8usize
            .checked_mul(utiles)
            .and_then(|v| v.checked_mul(macro_tiles))
            .map(|v| v / 4 * 4)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(layers))
            .and_then(|v| v.checked_mul(clusters as usize))
            .ok_or(EOVERFLOW)?;
        let layer_offset = aligned(tilemap_size, 0x1000)?;
        let heap_offset = layer_offset
            .checked_add(if layers > 1 { 0x100 } else { 0 })
            .ok_or(EOVERFLOW)?;
        let used = heap_offset.checked_add(0x200).ok_or(EOVERFLOW)?;
        let tiles = tiles_x.checked_mul(tiles_y).ok_or(EOVERFLOW)?;
        let mut tvb_blocks = aligned(tiles.div_ceil(128), 8)?.max(8);
        if clusters > 1 {
            tvb_blocks = tvb_blocks.max(
                7usize
                    .checked_add(2usize.checked_mul(layers).ok_or(EOVERFLOW)?)
                    .ok_or(EOVERFLOW)?,
            );
        }
        tvb_blocks = tvb_blocks.max(cfg::TVB_INITIAL_PAGES.len());
        let mut gtp = [None; 8];
        let mut tpc_end = tpc_size;
        if programs == RegisterPrograms::V25G83 {
            let sizes = [
                tpc_size,
                tilemap_size.checked_mul(64).ok_or(EOVERFLOW)?,
                if layers > 1 { 0x4000 } else { 0 },
                0,
                GTP_FIXED_SIZES[0],
                GTP_FIXED_SIZES[1],
                GTP_FIXED_SIZES[2],
                GTP_FIXED_SIZES[3],
            ];
            let mut position = 0;
            for ((offset, size), alignment) in gtp.iter_mut().zip(sizes).zip(GTP_ALIGNMENTS) {
                position = aligned(position, alignment)?;
                if size != 0 {
                    *offset = Some(position);
                }
                position = position.checked_add(size).ok_or(EOVERFLOW)?;
            }
            tpc_end = position;
        }
        Ok(Self {
            tilemap_size,
            layer_offset,
            heap_offset,
            used,
            tpc_size,
            tvb_blocks,
            gtp,
            tpc_end,
            m5: None,
        })
    }
    pub(crate) fn with_m5_slices(mut self, pass: &RenderPass, clusters: u32, groups: u32,
        segments: u32, parameter_bytes: usize) -> Result<Self> {
        let slices = crate::g17::fw::m5::target::TargetSlices::new(
            self.tilemap_size, self.tpc_size, pass.layers, groups, segments, parameter_bytes)
            .ok_or(EINVAL)?;
        // The last word is the part's minimum parameter-buffer size, whatever
        // the group count.
        let control = crate::g17::fw::m5::target::cluster_control(groups,
            crate::g17::fw::m5::target::min_parameter_buffer_pages(clusters))
            .ok_or(EINVAL)?;
        self.m5 = Some((slices, control));
        Ok(self)
    }
    pub(crate) fn is_m5(&self) -> bool { self.m5.is_some() }
    fn check_capacity(&self, stride: usize) -> Result {
        if self.m5.is_none() && stride > cfg::TARGET_RECORD_MAX {
            return Err(ERANGE);
        }
        stride.checked_mul(DEPTH).ok_or(EOVERFLOW)?;
        Ok(())
    }
    pub(crate) fn tpc_capacity(&self) -> Result<usize> {
        if let Some((slices, _)) = self.m5 {
            let bytes = aligned(slices.bytes, mmu::UAT_PGSZ)?;
            return Ok(if tpc_slot_mode(self) { bytes.max(TPC_SLOT_STRIDE * DEPTH) } else { bytes });
        }
        // TPC_MAX bounds the tail-pointer cache itself. On 25G83 this object
        // also contains GTP sub-buffers, notably 64 times the tile map. Applying
        // the cache limit to their combined extent rejects valid large targets.
        // Keep the original cache and tile-record limits; they also bound the
        // derived auxiliary extent before any backing allocation or replacement.
        if aligned(self.tpc_size, mmu::UAT_PGSZ)?.max(cfg::TPC_MIN) > cfg::TPC_MAX
            || self.record_stride()? > cfg::TARGET_RECORD_MAX
        {
            return Err(ERANGE);
        }
        Ok(aligned(self.tpc_end, mmu::UAT_PGSZ)?.max(cfg::TPC_MIN))
    }
    pub(crate) fn record_stride(&self) -> Result<usize> {
        let bytes = aligned(self.used, mmu::UAT_PGSZ)?;
        Ok(if self.m5.is_some() { bytes } else { bytes.max(cfg::TARGET_RECORD_MIN) })
    }
}

struct Object {
    address: u64,
    _mapping: mmu::KernelMapping,
    backing: KernelObject,
    /// Diagnostics: allocation sequence number (all pairs' target objects).
    seq: u64,
}

/// Diagnostics: prepares refused because the slot's previous owner was still active,
/// prepares of an ordinal that already owned its slot, and slots whose word +0 was nonzero
/// before the clear.
static DIAG_PREPARE: [core::sync::atomic::AtomicU64; 3] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 3];
pub(crate) fn diag_prepare_counts() -> [u64; 3] {
    DIAG_PREPARE.each_ref().map(|w| w.load(core::sync::atomic::Ordering::Relaxed))
}

/// Diagnostics: target objects created, and the last eight dropped as (address, size, seq,
/// GPU stamp time), to see whether a live record's GPU address reuses a dropped one's.
static DIAG_OBJECT_SEQ: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static DIAG_DROPPED: [[core::sync::atomic::AtomicU64; 4]; 8] =
    [const { [const { core::sync::atomic::AtomicU64::new(0) }; 4] }; 8];
static DIAG_DROP_NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

impl Drop for Object {
    fn drop(&mut self) {
        use core::sync::atomic::Ordering::Relaxed;
        let entry = &DIAG_DROPPED[(DIAG_DROP_NEXT.fetch_add(1, Relaxed) % 8) as usize];
        entry[0].store(self.address, Relaxed);
        entry[1].store(self.backing.size() as u64, Relaxed);
        entry[2].store(self.seq, Relaxed);
        entry[3].store(crate::g17::queue::render::diag_gpu_now(), Relaxed);
    }
}

/// Diagnostics: target objects dropped so far, and among the last eight the most recent one
/// whose GPU range contains `va`, as (address, size, seq, GPU stamp time).
pub(crate) fn diag_dropped_containing(va: u64) -> (u64, Option<[u64; 4]>) {
    use core::sync::atomic::Ordering::Relaxed;
    let total = DIAG_DROP_NEXT.load(Relaxed);
    let mut best: Option<[u64; 4]> = None;
    for entry in DIAG_DROPPED.iter() {
        let words = [0, 1, 2, 3].map(|index| entry[index].load(Relaxed));
        if words[1] != 0 && va >= words[0] && va - words[0] < words[1]
            && best.is_none_or(|known| words[3] > known[3])
        {
            best = Some(words);
        }
    }
    (total, best)
}

impl Object {
    fn new(dev: &AsahiDrmDevice, vm: &mmu::Vm, size: usize, m5: bool) -> Result<Self> {
        let backing = KernelObject::backing(dev, size, CpuMap::WriteCombined)?;
        let prot = mmu::PROT_GPU_FW_SHARED_RW;
        let mapping = if m5 {
            backing.map_range(vm, 0..size, super::BlockGeometry::M5_COMPACT,
                cfg::TVB_PAGE, prot, mmu::UAT_PGSZ)?
        } else {
            compact_alias(&backing, vm, 0..size, cfg::TVB_PAGE, prot, mmu::UAT_PGSZ)?
        };
        Ok(Self {
            address: mapping.iova(),
            _mapping: mapping,
            backing,
            seq: DIAG_OBJECT_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed) + 1,
        })
    }
}
struct Retired {
    _object: Object,
    owners: [Option<u64>; DEPTH],
    /// Diagnostics: the record stride of a replaced record object, 0 for a replaced TPC.
    stride: usize,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Addresses {
    pub(crate) tilemap: u64,
    pub(crate) layer: u64,
    pub(crate) heap: u64,
    /// The TPC the TA uses: the TPC object, or its slot slice with `m5_tpc_slot`.
    pub(crate) tpc: u64,
}

/// Experiment (`m5_tpc_slot`, GPU_J704_TIMEOUT.md round 7): per record slot a TPC slice of this
/// stride, zeroed at prepare. Only for M5 layouts whose TPC object holds the TPC alone.
const TPC_SLOT_STRIDE: usize = 0x8000;
fn tpc_slot_mode(layout: &Layout) -> bool {
    crate::module_parameters::m5_tpc_slot.value() != 0
        && layout.m5.is_some_and(|(slices, _)| slices.sizes[1..].iter().all(|&size| size == 0)
            && slices.offsets[0] == 0 && slices.sizes[0] <= TPC_SLOT_STRIDE)
}

pub(crate) struct Target {
    records: Object,
    tpc: Object,
    stride: usize,
    slots: [Option<u64>; DEPTH],
    retired: KVec<Retired>,
    m5: bool,
    /// Diagnostics: per slot, the ordinal and GPU stamp time of its last clear by `prepare`.
    cleared: [(u64, u64); DEPTH],
    /// Diagnostics: per slot, word +0 before the last clear and the ordinal cleared before it.
    clear_old: [(u64, u64); DEPTH],
    /// Per slot: the span the last prepare cleared, and (experiment 3) the span already cleared
    /// at its owner's retirement.
    spans: [usize; DEPTH],
    precleared: [usize; DEPTH],
    /// Diagnostics: record-object replacements (growth) and the newest ordinal owning a slot
    /// when the last one happened (u64::MAX: none).
    replaced: u64,
    replaced_at: u64,
}
impl Target {
    /// Keep this allocation frame separate from the rest of the render graph.
    #[inline(never)]
    pub(crate) fn new(dev: &AsahiDrmDevice, vm: &mmu::Vm, layout: &Layout) -> Result<KBox<Self>> {
        let tpc = layout.tpc_capacity()?;
        let stride = layout.record_stride()?;
        layout.check_capacity(stride)?;
        let m5 = layout.m5.is_some();
        let records = Object::new(dev, vm, stride * DEPTH, m5)?;
        let tpc = Object::new(dev, vm, tpc, m5)?;
        KBox::new(
            Self {
                records,
                tpc,
                stride,
                slots: [None; DEPTH],
                retired: KVec::new(),
                m5,
                cleared: [(u64::MAX, 0); DEPTH],
                clear_old: [(0, u64::MAX); DEPTH],
                spans: [0; DEPTH],
                precleared: [0; DEPTH],
                replaced: 0,
                replaced_at: u64::MAX,
            },
            GFP_KERNEL,
        )
        .map_err(Into::into)
    }
    pub(crate) fn tpc_va(&self) -> u64 {
        self.tpc.address
    }

    /// Diagnostics: the CPU view of up to `len` bytes at GPU address `va` in the current or a
    /// retained record or TPC object. Torn reads of firmware-written memory; never maps anything.
    pub(crate) fn diag_view(&self, va: u64, len: usize) -> Option<&[u8]> {
        let objects = Iterator::chain([&self.records, &self.tpc].into_iter(),
            self.retired.iter().map(|retired| &retired._object));
        for object in objects {
            let Some(start) = va.checked_sub(object.address) else { continue };
            let Ok(start) = usize::try_from(start) else { continue };
            let bytes = object.backing.bytes();
            if start < bytes.len() {
                return bytes.get(start..start.saturating_add(len).min(bytes.len()));
            }
        }
        None
    }
    /// Diagnostics: the record object's seq, GPU address and size, the replacement count and the
    /// ordinal of the last one, the TPC object's seq, and the dropped-object total with the most
    /// recent dropped object whose range contains `va` (seq, GPU stamp time; 0 when none).
    pub(crate) fn diag_identity(&self, va: u64) -> [u64; 9] {
        let (dropped, containing) = diag_dropped_containing(va);
        let containing = containing.unwrap_or([0; 4]);
        [self.records.seq, self.records.address, self.records.backing.size() as u64,
            self.replaced, self.replaced_at, self.tpc.seq, dropped, containing[2], containing[3]]
    }
    /// Diagnostics: for each retained replaced record object, its seq and address and the first
    /// 0x10 bytes of `ordinal`'s slot in it.
    pub(crate) fn diag_old_records(&self, ordinal: u64, mut each: impl FnMut(usize, u64, u64, &[u8])) {
        for (index, retired) in self.retired.iter().enumerate().filter(|(_, r)| r.stride != 0) {
            let start = (ordinal as usize % DEPTH) * retired.stride;
            let bytes = retired._object.backing.bytes();
            each(index, retired._object.seq, retired._object.address + start as u64,
                bytes.get(start..start + 0x10).unwrap_or(&[]));
        }
    }
    /// Diagnostics (round 7): every slot of the current record object: (slot, owner ordinal or
    /// u64::MAX, ordinal of its last clear, word +0, word +0x1000). Torn reads.
    pub(crate) fn diag_all_slots(&self, mut each: impl FnMut(usize, u64, u64, u64, u64)) {
        let bytes = self.records.backing.bytes();
        let word = |at: usize| bytes.get(at..at + 8)
            .map_or(u64::MAX, |b| u64::from_le_bytes(b.try_into().unwrap_or([0xff; 8])));
        for slot in 0..DEPTH {
            let base = slot * self.stride;
            each(slot, self.slots[slot].unwrap_or(u64::MAX), self.cleared[slot].0, word(base),
                word(base + 0x1000));
        }
    }
    /// Diagnostics: the owner ordinal of `ordinal`'s record slot (u64::MAX: none), the record
    /// stride, the number of retained replaced objects, and the ordinal and GPU stamp time of
    /// the slot's last clear.
    pub(crate) fn diag_slot(&self, ordinal: u64) -> [u64; 7] {
        let slot = ordinal as usize % DEPTH;
        [self.slots[slot].unwrap_or(u64::MAX), self.stride as u64, self.retired.len() as u64,
            self.cleared[slot].0, self.cleared[slot].1, self.clear_old[slot].0,
            self.clear_old[slot].1]
    }
    /// The ordinal whose clear preceded the one about to be recorded for `slot`.
    fn cleared_prev_owner(&self, slot: usize) -> u64 {
        self.cleared[slot].0
    }
    /// Experiment: completes the clear's stores (`dsb sy`) and reads the first and last words back.
    fn diag_barrier_readback(bytes: &mut [u8]) {
        // SAFETY: A barrier has no other effect.
        unsafe { core::arch::asm!("dsb sy", options(nostack, preserves_flags)) };
        let len = bytes.len();
        if len >= 8 {
            // SAFETY: Both reads are in bounds of the slice; volatile keeps them.
            unsafe {
                core::ptr::read_volatile(bytes.as_ptr().cast::<u64>());
                core::ptr::read_volatile(bytes.as_ptr().add(len - 8).cast::<u64>());
            }
        }
    }

    /// The M5 tiling words naming the TPC object's slices of `layout`.
    pub(crate) fn m5_extra(&self, layout: &Layout)
        -> Result<crate::g17::fw::m5::render_registers::TaExtra> {
        if !self.m5 { return Err(EINVAL); }
        let (slices, control) = layout.m5.ok_or(EINVAL)?;
        if slices.bytes > self.tpc.backing.size() { return Err(ERANGE); }
        let base = self.tpc.address.checked_sub(super::BlockGeometry::M5_COMPACT.start)
            .ok_or(EINVAL)?;
        let mut pointers = [0; 8];
        for index in 0..8 {
            if slices.sizes[index] != 0 {
                pointers[index] = base.checked_add(slices.offsets[index] as u64).ok_or(EOVERFLOW)?;
            }
        }
        Ok(crate::g17::fw::m5::render_registers::TaExtra {
            gtp1: pointers[1], gtp2: pointers[2], gtp4: pointers[4], gtp5: pointers[5],
            gtp6: pointers[6], gtp7: pointers[7], segment_control: control[0], segment_index: control[1],
            min_free_tvb_pages: control[2], scs_bracket: 0, matrix_bracket: 0, j813: false,
        })
    }

    /// GPU addresses of the tiler parameter sub-buffers of `layout` (25G83), zero for an empty
    /// or absent one. The TPC object holds them after the TPC.
    pub(crate) fn gtp_va(&self, layout: &Layout) -> Result<[u64; 8]> {
        if layout.tpc_end > self.tpc.backing.size() {
            return Err(ERANGE);
        }
        let mut addresses = [0; 8];
        for (address, offset) in addresses.iter_mut().zip(layout.gtp) {
            if let Some(offset) = offset {
                *address = self.tpc.address.checked_add(offset as u64).ok_or(EOVERFLOW)?;
            }
        }
        Ok(addresses)
    }

    /// Only the span consumed by this render is cleared. A small render does
    /// not clear a previous large target's complete retained record capacity.
    pub(crate) fn prepare(&mut self, ordinal: u64, layout: &Layout) -> Result<Addresses> {
        use core::sync::atomic::Ordering::Relaxed;
        let slot = ordinal as usize % DEPTH;
        if self.m5 != layout.m5.is_some() { return Err(EINVAL); }
        if self.slots[slot].is_some_and(|owner| owner != ordinal) {
            DIAG_PREPARE[0].fetch_add(1, Relaxed);
            return Err(EBUSY);
        }
        if self.slots[slot] == Some(ordinal) {
            DIAG_PREPARE[1].fetch_add(1, Relaxed);
        }
        if layout.used > self.stride {
            return Err(ERANGE);
        }
        let offset = slot * self.stride;
        let span = aligned(layout.used, mmu::UAT_PGSZ)?.min(self.stride);
        let bytes = self.records.backing.bytes_mut().get_mut(offset..offset + span).ok_or(ERANGE)?;
        // Diagnostics: the slot's word +0 before this clear.
        let old = bytes.get(..8).map_or(u64::MAX, |b| u64::from_le_bytes(b.try_into().unwrap_or([0xff; 8])));
        if old != 0 {
            DIAG_PREPARE[2].fetch_add(1, Relaxed);
        }
        // Experiment (`m5_target_clear`, off by default; GPU_J704_TIMEOUT.md round 7).
        let mode = crate::module_parameters::m5_target_clear.value();
        if self.m5 && mode != 0 {
            static LOGGED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
            crate::mmu::diag_log_once(&LOGGED, format_args!("m5_target_clear {}: {}", mode, match mode {
                1 => "target record clear followed by dsb sy and a readback",
                2 => "mode 2 removed (it preserved stale tile pointers): cleared as usual",
                _ => "target record cleared at its owner's retirement, with dsb sy",
            }));
        }
        match mode {
            3 if self.m5 && self.precleared[slot] >= span => {}
            _ => bytes.fill(0),
        }
        if self.m5 && (mode == 1 || mode == 3) {
            Self::diag_barrier_readback(bytes);
        }
        self.precleared[slot] = 0;
        let previous_clear = self.cleared_prev_owner(slot);
        self.cleared[slot] = (ordinal, crate::g17::queue::render::diag_gpu_now());
        self.clear_old[slot] = (old, previous_clear);
        self.spans[slot] = span;
        self.slots[slot] = Some(ordinal);
        let tilemap = self
            .records
            .address
            .checked_add(offset as u64)
            .ok_or(EOVERFLOW)?;
        let mut tpc = self.tpc.address;
        if tpc_slot_mode(layout) && self.tpc.backing.size() >= TPC_SLOT_STRIDE * DEPTH {
            static LOGGED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
            crate::mmu::diag_log_once(&LOGGED, format_args!(
                "m5_tpc_slot: every record slot has its own TPC slice, zeroed at prepare"));
            let start = slot * TPC_SLOT_STRIDE;
            let size = layout.m5.map_or(0, |(slices, _)| slices.sizes[0]);
            self.tpc.backing.bytes_mut().get_mut(start..start + size).ok_or(ERANGE)?.fill(0);
            Self::diag_barrier_readback(&mut self.tpc.backing.bytes_mut()[start..start + size]);
            tpc = tpc.checked_add(start as u64).ok_or(EOVERFLOW)?;
        }
        Ok(Addresses {
            tpc,
            tilemap,
            layer: tilemap
                .checked_add(layout.layer_offset as u64)
                .ok_or(EOVERFLOW)?,
            heap: tilemap
                .checked_add(layout.heap_offset as u64)
                .ok_or(EOVERFLOW)?,
        })
    }

    /// Growth never shrinks a live allocation. A successfully replaced TPC
    /// stays enlarged even if allocating the following record object fails.
    pub(crate) fn ensure_capacity(
        &mut self,
        dev: &AsahiDrmDevice,
        vm: &mmu::Vm,
        layout: &Layout,
        owners: &[Option<u64>; DEPTH],
    ) -> Result<bool> {
        let tpc = layout.tpc_capacity()?;
        let stride = layout.record_stride()?;
        if self.m5 != layout.m5.is_some() { return Err(EINVAL); }
        layout.check_capacity(stride)?;
        let mut grew = false;
        if tpc > self.tpc.backing.size() {
            Self::replace(&mut self.retired, &mut self.tpc, dev, vm, tpc, owners, self.m5, 0)?;
            grew = true;
        }
        if stride > self.stride {
            Self::replace(
                &mut self.retired,
                &mut self.records,
                dev,
                vm,
                stride * DEPTH,
                owners,
                self.m5,
                self.stride,
            )?;
            self.replaced += 1;
            self.replaced_at = owners.iter().flatten().copied().max().unwrap_or(u64::MAX);
            self.stride = stride;
            self.slots = [None; DEPTH];
            self.precleared = [0; DEPTH];
            grew = true;
        }
        Ok(grew)
    }
    fn replace(
        retired: &mut KVec<Retired>,
        old: &mut Object,
        dev: &AsahiDrmDevice,
        vm: &mmu::Vm,
        size: usize,
        owners: &[Option<u64>; DEPTH],
        m5: bool,
        stride: usize,
    ) -> Result {
        let retain = owners.iter().any(Option::is_some);
        if retain {
            retired.reserve(1, GFP_KERNEL)?;
        }
        let new = Object::new(dev, vm, size, m5)?;
        let previous = core::mem::replace(old, new);
        if retain {
            super::append_reserved(
                retired,
                Retired {
                    _object: previous,
                    owners: *owners,
                    stride,
                },
            );
        }
        Ok(())
    }
    pub(crate) fn retire(&mut self, ordinal: u64) -> Result {
        let index = ordinal as usize % DEPTH;
        // Experiment (`m5_target_clear` 3): clear the record now that its owner completed.
        if self.m5 && crate::module_parameters::m5_target_clear.value() == 3
            && self.slots[index] == Some(ordinal)
        {
            let (offset, span) = (index * self.stride, self.spans[index]);
            if let Some(bytes) = self.records.backing.bytes_mut().get_mut(offset..offset + span) {
                bytes.fill(0);
                Self::diag_barrier_readback(bytes);
                self.precleared[index] = span;
            }
        }
        let slot = &mut self.slots[index];
        if *slot == Some(ordinal) {
            *slot = None;
        }
        let mut index = 0;
        while index < self.retired.len() {
            for owner in &mut self.retired[index].owners {
                if *owner == Some(ordinal) {
                    *owner = None;
                }
            }
            if self.retired[index].owners.iter().all(Option::is_none) {
                drop(self.retired.remove(index)?);
            } else {
                index += 1;
            }
        }
        Ok(())
    }
    /// Cancels only an unpublished record; older referenced backing remains
    /// owned until normal paired retirement or the firmware-stop boundary.
    pub(crate) fn cancel(&mut self, ordinal: u64) {
        let slot = &mut self.slots[ordinal as usize % DEPTH];
        if *slot == Some(ordinal) {
            *slot = None;
        }
    }
}


#[kernel::macros::kunit_tests(asahi_g17_target_capacity)]
mod tests {
    use super::*;

    fn pass(size: u32, utile: u8) -> RenderPass {
        // SAFETY: RenderPass and its nested surfaces/programs contain only
        // integers and bools, for which the all-zero representation is valid.
        let mut pass: RenderPass = unsafe { core::mem::zeroed() };
        pass.width = size;
        pass.height = size;
        pass.layers = 1;
        pass.utile_width = utile;
        pass.utile_height = utile;
        pass
    }

    #[test]
    fn t6040_auxiliary_tiler_storage_is_not_tail_pointer_cache() {
        // CTS point_size_10000 renders to 10240 squared. Its GTP sub-buffer
        // is required by the existing 25G83 program, not a larger cache.
        for utile in [16, 32] {
            let layout = Layout::new(&pass(10240, utile), 2, RegisterPrograms::V25G83).unwrap();
            assert!(layout.tpc_size < cfg::TPC_MAX);
            assert!(layout.record_stride().unwrap() < cfg::TARGET_RECORD_MAX);
            assert!(layout.tpc_end > cfg::TPC_MAX);
            assert!(layout.tpc_capacity().unwrap() >= layout.tpc_end);
            assert_eq!(layout.gtp[0], Some(0));
            assert!(layout.gtp[1].unwrap() >= layout.tpc_size);
        }
    }

    #[test]
    fn core_and_tile_record_limits_still_reject_oversized_targets() {
        let mut layout = Layout::new(&pass(10240, 32), 2, RegisterPrograms::V25G83).unwrap();
        layout.tpc_size = cfg::TPC_MAX + 1;
        assert_eq!(layout.tpc_capacity(), Err(ERANGE));
        let mut layered = pass(16384, 16);
        layered.layers = 2;
        let layout = Layout::new(&layered, 2, RegisterPrograms::V25G83).unwrap();
        assert!(layout.record_stride().unwrap() > cfg::TARGET_RECORD_MAX);
        assert_eq!(layout.tpc_capacity(), Err(ERANGE));
    }

    #[test]
    fn neo_keeps_the_same_cache_extent_and_limit() {
        let mut layout = Layout::new(&pass(10240, 32), 1, RegisterPrograms::G17P).unwrap();
        assert!(layout.gtp.iter().all(Option::is_none));
        assert_eq!(layout.tpc_capacity().unwrap(),
            aligned(layout.tpc_size, mmu::UAT_PGSZ).unwrap().max(cfg::TPC_MIN));
        layout.tpc_size = cfg::TPC_MAX + 1;
        layout.tpc_end = layout.tpc_size;
        assert_eq!(layout.tpc_capacity(), Err(ERANGE));
    }
}
