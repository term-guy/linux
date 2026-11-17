// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Diagnostics (GPU_J704_TIMEOUT.md round 7): the pass shape of an M5 render, captured when its
//! commands are encoded and kept with its active record, then (with `m5_diag_shape`) in a small
//! ring of retired renders. Printed only at a recovery: the culprit render, matched by the queue
//! id and kick the hardware names in 0xc078, and up to four healthy renders for comparison.
//! Every line starts with "G17: shape ".

use crate::g17::{
    buffer::{self, target},
    fw::{
        kick::{RegisterWrite, WriteRange},
        m5::{
            render::{M5RenderCommandInputs, M5_3D_PROGRAM_OFFSETS, M5_TA_PROGRAM_OFFSET},
            render_registers::{M5RenderControls, Workspace},
        },
        render as abi,
    },
};
use kernel::fmt;
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
use kernel::prelude::*;

/// Register capacities of the four 3D arrays: main, partial store, partial resume, SPM load.
const CAPACITY: [usize; 4] = [94, 17, 24, 12];
/// Byte offset of a program's register count from its array start (`render.rs` place_program).
const COUNT_AT: usize = 0x700 + 8;
const REGISTER_BYTES: usize = 12;
/// Retired shapes kept per pair.
pub(crate) const RING: usize = 8;
/// The kick bits of the 0xc078 word (qid << 40 | kick).
pub(crate) const KICK_MASK: u64 = (1 << 40) - 1;

/// 0xc078 as the last recovery's fault-register read found it (u64::MAX: unread).
pub(crate) static FAULT_KICK: AtomicU64 = AtomicU64::new(u64::MAX);
/// Healthy retirements by process_empty (x4), no attachments (x2) and target word +0 zero (x1).
static REGION_COUNTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Prints the region-entry counters in one line.
pub(crate) fn report_counts() {
    let c = |index: usize| REGION_COUNTS[index].load(Relaxed);
    let (mut zero, mut nonzero) = (0, 0);
    for index in 0..8 {
        if index & 1 != 0 { zero += c(index) } else { nonzero += c(index) }
    }
    let t = crate::mmu::diag_tlb_counts();
    let p = target::diag_prepare_counts();
    pr_warn!("G17: shape counts: retired word0 zero={} nonzero={}; zero/nonzero by (process_empty,no_attachments): (0,0)={}/{} (0,1)={}/{} (1,0)={}/{} (1,1)={}/{}; ctx assignments={} reuses={} tlbi requests={} per_ctx_issued={} skipped_no_ctx={} asid_assign_release={} experiment_flushes={}; prepare refused_active_owner={} same_ordinal={} word0_nonzero_before_clear={}; doorbell readbacks={} word0_nonzero={}\n",
        zero, nonzero, c(1), c(0), c(3), c(2), c(5), c(4), c(7), c(6), t[0], t[1], t[2], t[3], t[4],
        t[5], t[6], p[0], p[1], p[2], READBACKS.load(Relaxed), READBACK_NONZERO.load(Relaxed));
    let e = |index: usize| EMPTY_CLASS[index].load(Relaxed);
    pr_warn!("G17: shape empty classes (1,1): executed scene10!=0={} executed scene10=0={} unexecuted scene10!=0={} unexecuted scene10=0={}\n",
        e(3), e(2), e(1), e(0));
}

/// Attachmentless process_empty retirements with an empty region entry, by (stage stamps all
/// set and 3D end after 3D start) x2 and (scene +0x10 nonzero) x1 (round 7).
static EMPTY_CLASS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Doorbell readbacks of target word +0, and those that were nonzero.
static READBACKS: AtomicU64 = AtomicU64::new(0);
static READBACK_NONZERO: AtomicU64 = AtomicU64::new(0);

/// Prints the doorbell readback and TA state words (+0..0xbf context store, +0xc0.. status page).
pub(crate) fn report_tastat(tag: &str, readback: [u64; 2], tastat: &[u8]) {
    pr_err!("G17: shape {} tastat: doorbell readback tgt+0={:#x} tgt+0x1000={:#x} (u64::MAX: not read); words below: +0..0xbf TA context store, +0xc0.. stage status page (TA +0xc0, 3D +0xc8)\n",
        tag, readback[0], readback[1]);
    dump_words(tag, "tastat", tastat);
}

/// Retirement order of shapes across all pairs.
static RETIRED: AtomicU64 = AtomicU64::new(0);

/// The pass shape of one render. Integer-only: the zero value is valid.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    /// Retirement order (0 while active).
    pub(crate) retired: u64,
    pub(crate) ordinal: u64,
    pub(crate) qids: [u8; 2],
    pub(crate) kicks: [u64; 2],
    pub(crate) pid: u32,
    pub(crate) context: u32,
    /// The context's identity when the commands were encoded (`mmu::diag_context_identity`).
    vm_ident: [u64; 9],
    // Render pass, as userspace submitted it.
    width: u32,
    height: u32,
    layers: u16,
    utile: [u8; 2],
    samples_log2: u8,
    sample_size: u8,
    /// process_empty_tiles, few_primitives, whole_attachments, shader_writes_memory,
    /// depth_bias_is_int, rsrc_spec_hi, fragment_barriers, stage_prefixes (bits 0..7).
    pass_flags: u8,
    msaa_control: u64,
    ppp_control: u64,
    zls_control: u64,
    depth_dimensions: u64,
    /// va, stride, compression va, compression stride.
    depth: [u64; 4],
    stencil: [u64; 4],
    clears: [u32; 2],
    vdm_va: u64,
    scissor_va: u64,
    depth_bias_va: u64,
    occlusion_query_va: u64,
    sampler_count: u32,
    usc_base: u64,
    /// Addresses of the background, partial background, end-of-tile and partial end-of-tile
    /// programs (usc_base + offset without its tag bits), and their resource specifiers.
    usc: [u64; 4],
    rsrc: [u64; 4],
    /// Write ranges the TA and 3D stages declare (count), and the first four 3D ones.
    attachment_counts: [u8; 2],
    attachments: [[u64; 2]; 4],
    // Command fields the driver computes.
    tile_count: u32,
    utile_grid: u32,
    raster_control: u64,
    tile_flags: u64,
    event_control: u64,
    ta_tile_control: u64,
    ta_dispatch_control: u64,
    partial_tile_flags: u64,
    aux_fb: [u64; 2],
    store_pipeline_2: [u64; 2],
    partial_store_pipeline_2: [u64; 2],
    isp_sample_control: u64,
    /// TA then 3D: resource flags, resource control, PTC maximum, scratch enabled.
    resources: [[u64; 4]; 2],
    empty_tile_flags: [u8; 4],
    short_partial: u8,
    /// fragment_pass_flag, fragment_flag_a, fragment_flag_b, fragment_flag_c, tail_flag.
    command_flags: [u8; 5],
    /// Main array rce count, mid count.
    main_counts: [u16; 2],
    // Pointers.
    tilemap_va: u64,
    layermeta_va: u64,
    heapmeta_va: u64,
    tpc: [u64; 2],
    gtp_va: [u64; 8],
    pb_va: u64,
    scene_va: u64,
    scene_end_va: u64,
    aux_fb_va: u64,
    deflake_va: u64,
    mesh_storage: u64,
    /// tilemap size, layer offset, heap offset, used, TVB blocks.
    layout: [u64; 5],
    // The four 3D register arrays as encoded, in order.
    counts: [u16; 4],
    main: [RegisterWrite; 94],
    store: [RegisterWrite; 17],
    resume: [RegisterWrite; 24],
    load: [RegisterWrite; 12],
    // Render memory, copied at retirement (healthy ring entries only).
    content_captured: u8,
    /// `m5_empty_tile_mode` applied to this render's encoding (0: as submitted).
    empty_mode: u8,
    /// Target words +0 and +0x1000 read back just before the doorbell (u64::MAX: not read).
    readback: [u64; 2],
    /// TA context-store head (0xc0) and stage status page head (0x40), copied at retirement.
    tastat: [u8; 0x100],
    /// The TA's and 3D's pointer words as encoded ([`PTR_SOURCES`]).
    ptrs: [u64; PTRS],
    content: ContentInfo,
    tgt: [u8; TARGET_STORED],
    scene: [u8; SCENE_BYTES],
    stream: [u8; STREAM_BYTES],
}

/// Target-record bytes kept (layout `used`, at most this).
pub(crate) const TARGET_BYTES: usize = 0x1200;
/// Target-record bytes copied for a retired render (the live culprit read covers
/// [`TARGET_BYTES`]); 972-974 showed no nonzero word past +0x1008 in this record.
pub(crate) const TARGET_STORED: usize = 0x1010;
/// Scene-entry bytes kept from `scene`: this scene's entry (0x80).
pub(crate) const SCENE_BYTES: usize = 0x80;
/// Bytes of the tile control stream kept from the first TVB address in the target record.
pub(crate) const STREAM_BYTES: usize = 0x200;
/// Nonzero words printed per dump.
const DUMP_WORDS: usize = 256;
const _: () = assert!(core::mem::size_of::<Shape>() <= 8192);

/// What the driver's own mappings show of a render's memory.
#[derive(Clone, Copy, Default)]
pub(crate) struct ContentInfo {
    /// TA start, TA end, 3D start, 3D end (GPU stamp ticks).
    pub(crate) stamps: [u64; 4],
    /// Manager::diag_tvb_info.
    pub(crate) tvb: [u64; 11],
    /// Target::diag_slot.
    pub(crate) slot: [u64; 7],
    /// Target::diag_identity.
    pub(crate) identity: [u64; 9],
    /// Bytes read of the target record, scene and stream.
    pub(crate) lengths: [usize; 3],
    /// Stream source: target-record byte offset, raw word, resolved address, TVB block, form
    /// (0 none, 1 low 48 bits, 2 compact base + low 32 bits, 3 low 48 bits & !0x7f).
    pub(crate) stream: [u64; 5],
}

/// Borrowed views of a render's memory through the driver's mappings.
pub(crate) struct Views<'a> {
    pub(crate) info: ContentInfo,
    pub(crate) tgt: &'a [u8],
    pub(crate) scene: &'a [u8],
    pub(crate) stream: &'a [u8],
}

/// The first word of `tgt` that names an address inside a TVB block the driver maps.
fn resolve_stream<'a>(tgt: &[u8], manager: &'a buffer::Manager) -> ([u64; 5], &'a [u8]) {
    for offset in (0..tgt.len() / 8).map(|index| index * 8) {
        let word = read(tgt, offset, 8);
        if word == 0 {
            continue;
        }
        let low = word & ((1 << 48) - 1);
        let candidates = [(1u64, low),
            (2, buffer::BlockGeometry::M5_COMPACT.start + (word & 0xffff_ffff)),
            (3, low & !0x7f)];
        for (form, va) in candidates {
            if let Some((block, view)) = manager.diag_tvb_view(va, STREAM_BYTES) {
                return ([offset as u64, word, va, block as u64, form], view);
            }
        }
    }
    ([0; 5], &[])
}

/// The live views of `shape`'s render memory. Never allocates or maps.
pub(crate) fn views<'a>(
    shape: &Shape,
    target: &'a target::Target,
    manager: &'a buffer::Manager,
    stamps: [u64; 4],
) -> Views<'a> {
    let used = (shape.layout[3] as usize).min(TARGET_BYTES);
    let tgt = target.diag_view(shape.tilemap_va, used).unwrap_or(&[]);
    let scene_len = (shape.scene_end_va.saturating_sub(shape.scene_va) as usize).min(SCENE_BYTES);
    let scene = manager.diag_graph_view(shape.scene_va, scene_len).unwrap_or(&[]);
    let (source, stream) = resolve_stream(tgt, manager);
    Views {
        info: ContentInfo {
            stamps,
            tvb: manager.diag_tvb_info(shape.ordinal),
            slot: target.diag_slot(shape.ordinal),
            identity: target.diag_identity(shape.tilemap_va),
            lengths: [tgt.len(), scene.len(), stream.len()],
            stream: source,
        },
        tgt,
        scene,
        stream,
    }
}

/// Prints the views: stamps, tvb bookkeeping, then the nonzero words of each dump.
pub(crate) fn report_views(tag: &str, views: &Views<'_>) {
    let info = &views.info;
    let s = info.stamps;
    pr_err!("G17: shape {} stamps: ta {:#x}..{:#x} ({} ticks) 3d {:#x}..{:#x} ({} ticks) ta_end->3d_start {} ticks\n",
        tag, s[0], s[1], s[1].wrapping_sub(s[0]) as i64, s[2], s[3], s[3].wrapping_sub(s[2]) as i64,
        s[2].wrapping_sub(s[1]) as i64);
    let t = info.tvb;
    pr_err!("G17: shape {} tvb: id={} gen={:#x} blocks={} extensions={} pages={:#x} initialized={} scene_index={:#x} block_va first={:#x} last={:#x} base={:#x} block_bytes={:#x} target slot_owner={:#x} stride={:#x} retained_objects={} last_clear ordinal={:#x} ticks={:#x} word0_before_clear={:#x} prev_cleared_ordinal={:#x} read tgt={:#x} scene={:#x} stream={:#x} stream_src(off,word,va,block,form)={:x?}\n",
        tag, t[0], t[1], t[2], t[3], t[4], t[5], t[6] as i64, t[7], t[8], t[9], t[10],
        info.slot[0], info.slot[1], info.slot[2], info.slot[3] as i64, info.slot[4], info.slot[5], info.slot[6] as i64, info.lengths[0], info.lengths[1],
        info.lengths[2], info.stream);
    let id = info.identity;
    pr_err!("G17: shape {} tvb: record seq={} va={:#x} size={:#x} replaced={} last_replace_ordinal={:#x} tpc seq={} dropped_objects={}\n",
        tag, id[0], id[1], id[2], id[3], id[4] as i64, id[5], id[6]);
    dump_words(tag, "tgt", views.tgt);
    dump_words(tag, "scene", views.scene);
    if views.stream.is_empty() {
        pr_err!("G17: shape {} stream[0..]: no target-record word resolves into a TVB block the driver maps\n", tag);
    } else {
        dump_words(tag, "stream", views.stream);
    }
}

/// Nonzero little-endian 64-bit words of `bytes` with their byte offsets, at most
/// [`DUMP_WORDS`], 24 per line; zero words are counted, not printed.
pub(crate) fn dump_words(tag: &str, name: &str, bytes: &[u8]) {
    let mut line = [(0u32, 0u64); 24];
    let (mut n, mut printed, mut zeros, mut dropped) = (0usize, 0usize, 0usize, 0usize);
    for index in 0..bytes.len() / 8 {
        let word = read(bytes, index * 8, 8);
        if word == 0 {
            zeros += 1;
            continue;
        }
        if printed == DUMP_WORDS {
            dropped += 1;
            continue;
        }
        line[n] = ((index * 8) as u32, word);
        n += 1;
        printed += 1;
        if n == line.len() {
            pr_err!("G17: shape {} {}[{:#x}..]: {}\n", tag, name, line[0].0, Words(&line[..n]));
            n = 0;
        }
    }
    if n != 0 {
        pr_err!("G17: shape {} {}[{:#x}..]: {}\n", tag, name, line[0].0, Words(&line[..n]));
    }
    pr_err!("G17: shape {} {}[end]: {:#x} bytes, {} nonzero words printed, {} more not printed, {} zero words\n",
        tag, name, bytes.len(), printed, dropped, zeros);
}

/// (offset, word) pairs as "+offset=word" in hex.
struct Words<'a>(&'a [(u32, u64)]);
impl fmt::Display for Words<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (offset, word) in self.0 {
            write!(f, "+{:x}={:x} ", offset, word)?;
        }
        Ok(())
    }
}

impl Shape {
    /// A zeroed shape on the heap, or `None` (diagnostics only; never fails the caller).
    pub(crate) fn new_boxed() -> Option<KBox<Self>> {
        let mut shape = KBox::<Self>::new_uninit(GFP_KERNEL).ok()?;
        // SAFETY: Shape is integer-only (RegisterWrite included); all zeroes is a valid value.
        unsafe { core::ptr::write_bytes(shape.as_mut_ptr(), 0, 1) };
        // SAFETY: Initialized above.
        Some(unsafe { shape.assume_init() })
    }

    /// Fills the shape from the encoding just done by `encode_into` for `args`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture(
        &mut self,
        args: &abi::RenderArgs<'_>,
        layout: &target::Layout,
        controls: &M5RenderControls,
        inputs: &M5RenderCommandInputs,
        workspace: &Workspace,
        attachments: (usize, &[WriteRange]),
        pid: u32,
    ) {
        let p = args.pass;
        let r = args.resources;
        let f = &inputs.fragment;
        self.retired = 0;
        self.content_captured = 0;
        self.empty_mode = 0;
        self.readback = [u64::MAX; 2];
        self.ordinal = args.ordinal;
        self.qids = [args.ta.qid, args.fragment.qid];
        self.pid = pid;
        self.context = u32::from(r.context_id);
        self.vm_ident = crate::mmu::diag_context_identity(usize::from(r.context_id));
        self.width = p.width;
        self.height = p.height;
        self.layers = p.layers;
        self.utile = [p.utile_width, p.utile_height];
        self.samples_log2 = p.samples_log2;
        self.sample_size = p.sample_size;
        self.pass_flags = [p.process_empty_tiles, p.few_primitives, p.whole_attachments,
            p.shader_writes_memory, p.depth_bias_is_int, p.rsrc_spec_hi, p.fragment_barriers,
            p.stage_prefixes].iter().enumerate().fold(0, |bits, (bit, set)| bits | (u8::from(*set) << bit));
        self.msaa_control = p.msaa_control;
        self.ppp_control = p.ppp_control;
        self.zls_control = p.zls_control;
        self.depth_dimensions = p.depth_dimensions;
        self.depth = [p.depth.va, p.depth.stride, p.depth.compression_va, p.depth.compression_stride];
        self.stencil = [p.stencil.va, p.stencil.stride, p.stencil.compression_va,
            p.stencil.compression_stride];
        self.clears = [p.depth_clear, p.stencil_clear];
        self.vdm_va = p.vdm_va;
        self.scissor_va = p.scissor_va;
        self.depth_bias_va = p.depth_bias_va;
        self.occlusion_query_va = p.occlusion_query_va;
        self.sampler_count = p.sampler_count;
        self.usc_base = p.usc_base;
        let usc = |offset: u32| p.usc_base.wrapping_add(u64::from(offset & !7));
        self.usc = [usc(p.bg.offset), usc(p.partial_bg.offset), usc(p.eot_offset),
            usc(p.partial_eot_offset)];
        self.rsrc = [p.bg.rsrc_spec, p.partial_bg.rsrc_spec, p.eot_rsrc_spec, p.partial_eot_rsrc_spec];
        self.attachment_counts = [attachments.0.min(255) as u8, attachments.1.len().min(255) as u8];
        self.attachments = [[0; 2]; 4];
        for (slot, range) in self.attachments.iter_mut().zip(attachments.1) {
            *slot = [range.va, range.size];
        }
        self.tile_count = f.tile_count;
        self.utile_grid = f.utile_grid;
        self.raster_control = controls.fragment_raster_control;
        self.tile_flags = workspace.diag_fragment_tile_flags;
        self.event_control = controls.fragment_event_control;
        self.ta_tile_control = controls.ta_tile_control;
        self.ta_dispatch_control = controls.ta_dispatch_control;
        self.partial_tile_flags = controls.partial_tile_flags;
        self.aux_fb = [controls.aux_fb_flags, controls.aux_fb_control];
        self.store_pipeline_2 = [controls.store_pipeline_2_bind, controls.store_pipeline_2];
        self.partial_store_pipeline_2 = [controls.partial_store_pipeline_2_bind,
            controls.partial_store_pipeline_2];
        self.isp_sample_control = f.sample_aux_flags[0];
        for (out, stage) in self.resources.iter_mut().zip([&controls.ta, &controls.fragment]) {
            *out = [stage.resource_flags, stage.resource_control, u64::from(stage.ptc_max),
                u64::from(stage.scs.scratch.enabled())];
        }
        self.empty_tile_flags = f.empty_tile_flags;
        self.short_partial = f.short_partial_programs;
        self.command_flags = [f.fragment_pass_flag, u8::from(f.fragment_flag_a), f.fragment_flag_b,
            f.fragment_flag_c, f.tail_flag];
        let count = workspace.fragment_count as u16;
        self.main_counts = [count.wrapping_sub(4), count.wrapping_sub(2)];
        self.tilemap_va = r.tilemap_va;
        self.layermeta_va = r.layermeta_va;
        self.heapmeta_va = r.heapmeta_va;
        self.tpc = [r.tpc_va, r.tpc_size];
        self.gtp_va = r.gtp_va;
        self.pb_va = r.pb_va;
        self.scene_va = r.scene_va;
        self.scene_end_va = r.scene_end_va;
        self.aux_fb_va = r.aux_fb_va;
        self.deflake_va = r.deflake_va;
        self.mesh_storage = r.mesh_storage;
        self.layout = [layout.tilemap_size as u64, layout.layer_offset as u64,
            layout.heap_offset as u64, layout.used as u64, layout.tvb_blocks as u64];
        // The arrays exactly as encoded in the (cached) fragment command copy.
        let bytes = &workspace.encoded_fragment[..];
        let arrays: [&mut [RegisterWrite]; 4] =
            [&mut self.main, &mut self.store, &mut self.resume, &mut self.load];
        for (index, (array, offset)) in arrays.into_iter().zip(M5_3D_PROGRAM_OFFSETS).enumerate() {
            let offset = offset as usize;
            let count = read(bytes, offset + COUNT_AT, 2) as usize;
            self.counts[index] = count as u16;
            for (slot, entry) in array.iter_mut().enumerate() {
                *entry = if slot < count.min(CAPACITY[index]) {
                    let at = offset + slot * REGISTER_BYTES;
                    RegisterWrite::new(read(bytes, at, 4) as u32, read(bytes, at + 4, 8))
                } else {
                    RegisterWrite::new(0, 0)
                };
            }
        }
        self.ptrs = ptr_words(&workspace.encoded_ta, &workspace.encoded_fragment);
    }

    /// Prints the TA's against the 3D's pointer words, and which differ from `live` (the
    /// words read back from the command storage now), if given.
    pub(crate) fn report_ptrs(&self, tag: &str, live: Option<&[u64; PTRS]>) {
        report_ptrs(tag, &self.ptrs, live, self.tilemap_va);
    }

    /// The `id:` line: render identity and its context's identity at encoding.
    pub(crate) fn report_id(&self, tag: &str) {
        let v = self.vm_ident;
        pr_err!("G17: shape {} id: ordinal={} qids={:?} kicks=[{:#x}, {:#x}] pid={} ctx={} retired#={} vm: slot={} gen={} vm_id={} assigned={:#x} prev_released={:#x} tlbi_since_assign={} prev_vm_id={} prev_pid={} (at encoding)\n",
            tag, self.ordinal, self.qids, self.kicks[0], self.kicks[1], self.pid, self.context,
            self.retired, self.context, v[0], v[4], v[1], v[2], v[3], v[5], v[7]);
    }

    /// Records the doorbell readback of target words +0 and +0x1000.
    pub(crate) fn note_readback(&mut self, words: [u64; 2]) {
        self.readback = words;
        if words[0] != 0 {
            READBACK_NONZERO.fetch_add(1, Relaxed);
        }
        READBACKS.fetch_add(1, Relaxed);
    }
    /// Copies the TA state words at retirement.
    pub(crate) fn store_tastat(&mut self, state: Option<(&[u8], &[u8])>) {
        self.tastat.fill(0);
        if let Some((store, status)) = state {
            let n = store.len().min(0xc0);
            self.tastat[..n].copy_from_slice(&store[..n]);
            let m = status.len().min(0x40);
            self.tastat[0xc0..0xc0 + m].copy_from_slice(&status[..m]);
        }
    }
    /// Prints the doorbell readback and the stored TA state words.
    pub(crate) fn report_tastat(&self, tag: &str) {
        report_tastat(tag, self.readback, &self.tastat);
    }

    /// Prints the context's identity now (`vm_live:`), next to the one at encoding.
    pub(crate) fn report_vm_live(&self, tag: &str) {
        let v = crate::mmu::diag_context_identity(self.context as usize);
        let same = v[0] == self.vm_ident[0] && v[4] == self.vm_ident[4] && v[1] == self.vm_ident[1];
        pr_err!("G17: shape {} vm_live: slot={} gen={} vm_id={} assigned={:#x} prev_released={:#x} tlbi_since_assign={} prev_vm_id={} prev_pid={} pid={} last_released={:#x} same_owner_as_encoding={}\n",
            tag, self.context, v[0], v[4], v[1], v[2], v[3], v[5], v[7], v[6], v[8], same);
    }
    pub(crate) fn tilemap_va(&self) -> u64 {
        self.tilemap_va
    }
    pub(crate) fn scene_va(&self) -> u64 {
        self.scene_va
    }

    /// Copies the views of the retired render's memory (no allocation).
    pub(crate) fn store_content(&mut self, views: &Views<'_>) {
        let copy = |out: &mut [u8], view: &[u8]| {
            let n = view.len().min(out.len());
            out[..n].copy_from_slice(&view[..n]);
            out[n..].fill(0);
        };
        copy(&mut self.tgt, views.tgt);
        copy(&mut self.scene, views.scene);
        copy(&mut self.stream, views.stream);
        self.content = views.info;
        self.content.lengths = [views.tgt.len().min(TARGET_STORED), views.scene.len().min(SCENE_BYTES),
            views.stream.len().min(STREAM_BYTES)];
        self.content_captured = 1;
    }

    /// Prints the content copied at retirement, in the format of [`report_views`].
    pub(crate) fn report_content(&self, tag: &str) {
        self.report_ptrs(tag, None);
        self.report_tastat(tag);
        if self.content_captured == 0 {
            pr_err!("G17: shape {} tvb: no content captured\n", tag);
            return;
        }
        let l = self.content.lengths;
        report_views(tag, &Views {
            info: self.content,
            tgt: &self.tgt[..l[0]],
            scene: &self.scene[..l[1]],
            stream: &self.stream[..l[2]],
        });
    }

    /// Records that `m5_empty_tile_mode` `mode` cleared process_empty for this encoding.
    pub(crate) fn note_empty_override(&mut self, applied: bool, mode: u32) {
        self.empty_mode = if applied { mode.min(255) as u8 } else { 0 };
    }

    /// Whether word +0 of the copied target record (the first tile's region entry) is zero.
    /// Whether the TA and the 3D both ran (all four stage stamps set, 3D end after 3D start).
    fn executed(&self) -> bool {
        let st = self.content.stamps;
        st.iter().all(|&stamp| stamp != 0) && st[3] > st[2]
    }
    /// The scene entry's word +0x10 (firmware TA accounting), 0 when not captured.
    fn scene10(&self) -> u64 {
        if self.content.lengths[1] >= 0x18 { read(&self.scene, 0x10, 8) } else { 0 }
    }
    fn region_empty(&self) -> bool {
        self.content.lengths[0] >= 8 && read(&self.tgt, 0, 8) == 0
    }

    /// Counts a healthy retirement by its region entry, after `store_content`.
    pub(crate) fn count_retired(&self) {
        let index = usize::from(self.pass_flags & 1) * 4
            + usize::from(self.attachment_counts[1] == 0) * 2 + usize::from(self.region_empty());
        if self.content.lengths[0] >= 8 {
            REGION_COUNTS[index].fetch_add(1, Relaxed);
            if index == 7 {
                EMPTY_CLASS[usize::from(self.executed()) * 2 + usize::from(self.scene10() != 0)]
                    .fetch_add(1, Relaxed);
            }
        }
    }

    /// Records the published kicks.
    pub(crate) fn publish(&mut self, qids: [u8; 2], kicks: [u64; 2]) {
        self.qids = qids;
        self.kicks = kicks;
    }

    /// Stamps the retirement order before the shape enters a pair's ring.
    pub(crate) fn retire(&mut self) {
        self.retired = RETIRED.fetch_add(1, Relaxed) + 1;
    }

    /// Whether `kick` (0xc078: qid << 40 | kick) names one of this render's kicks.
    pub(crate) fn names(&self, word: u64) -> Option<usize> {
        let qid = ((word >> 40) & 0xff) as u8;
        (0..2).find(|&stage| self.qids[stage] == qid && self.kicks[stage] & KICK_MASK == word & KICK_MASK)
    }

    /// Same size, layers, samples, tile-buffer bytes, attachment count and empty-tile intent.
    pub(crate) fn same_kind(&self, other: &Self) -> bool {
        (self.width, self.height, self.layers, self.samples_log2, self.sample_size,
            self.attachment_counts, self.pass_flags & 1, self.depth[0] != 0, self.stencil[0] != 0)
            == (other.width, other.height, other.layers, other.samples_log2, other.sample_size,
                other.attachment_counts, other.pass_flags & 1, other.depth[0] != 0,
                other.stencil[0] != 0)
    }

    fn arrays(&self) -> [(&'static str, &[RegisterWrite]); 4] {
        let n = |index: usize| usize::from(self.counts[index]).min(CAPACITY[index]);
        [("main", &self.main[..n(0)]), ("store", &self.store[..n(1)]),
            ("resume", &self.resume[..n(2)]), ("load", &self.load[..n(3)])]
    }

    /// The first 0x10039 (tile control / action bits) value of each array.
    fn actions(&self) -> [u64; 4] {
        self.arrays().map(|(_, array)| {
            array.iter().map(|w| { let w = *w; (w.reg, w.value) })
                .find(|(reg, _)| *reg == 0x10039).map_or(u64::MAX, |(_, value)| value)
        })
    }

    /// The first value of `reg` in the main array (u64::MAX: absent).
    fn main_value(&self, reg: u32) -> u64 {
        self.arrays()[0].1.iter().map(|w| { let w = *w; (w.reg, w.value) })
            .find(|(r, _)| *r == reg).map_or(u64::MAX, |(_, value)| value)
    }

    /// Prints the shape in lines under ~900 characters, each starting with "G17: shape {tag}".
    pub(crate) fn report(&self, tag: &str) {
        let zls = self.zls_control;
        self.report_id(tag);
        pr_err!("G17: shape {} pass: {}x{} layers={} samples={} utile={}x{} sample_size={} tiles={}x{} tile_count={} utile_grid={:#x} process_empty={} few_prims={} whole_att={} shader_writes={} bias_int={} spec_hi={} frag_barriers={} stage_prefixes={} attachments ta={} 3d={} ranges(va,size)={:x?}\n",
            tag, self.width, self.height, self.layers, 1u32 << self.samples_log2, self.utile[0],
            self.utile[1], self.sample_size, self.width.div_ceil(32), self.height.div_ceil(32),
            self.tile_count, self.utile_grid, self.pass_flags & 1, (self.pass_flags >> 1) & 1,
            (self.pass_flags >> 2) & 1, (self.pass_flags >> 3) & 1, (self.pass_flags >> 4) & 1,
            (self.pass_flags >> 5) & 1, (self.pass_flags >> 6) & 1, (self.pass_flags >> 7) & 1,
            self.attachment_counts[0], self.attachment_counts[1],
            &self.attachments[..usize::from(self.attachment_counts[1]).min(4)]);
        pr_err!("G17: shape {} zs: zls={:#x} (load d={} s={} store d={} s={}) depth(va,stride,cva,cstride)={:x?} stencil={:x?} dims={:#x} clear d={:#x} s={:#x} msaa={:#x} ppp={:#x} bg_clear_intent(rsrc bit6)={} pbg bit6={}\n",
            tag, zls, (zls >> 15) & 1, (zls >> 14) & 1, (zls >> 19) & 1, (zls >> 18) & 1,
            self.depth, self.stencil, self.depth_dimensions, self.clears[0], self.clears[1],
            self.msaa_control, self.ppp_control, (self.rsrc[0] >> 6) & 1, (self.rsrc[1] >> 6) & 1);
        let actions = self.actions();
        let action_bits = if actions[0] == u64::MAX { 0 } else { (actions[0] >> 17) & 7 };
        pr_err!("G17: shape {} ctl: raster={:#x} (0x10791 main={:#x}) tile_flags={:#x} (0x16068 main={:#x}) event={:#x} ta_tile={:#x} ta_dispatch={:#x} partial_tile_flags={:#x} aux_fb={:x?} store2={:x?} pstore2={:x?} isp_sample={:#x} (0x10051 main={:#x})\n",
            tag, self.raster_control, self.main_value(0x10791), self.tile_flags,
            self.main_value(0x16068), self.event_control, self.ta_tile_control,
            self.ta_dispatch_control, self.partial_tile_flags, self.aux_fb, self.store_pipeline_2,
            self.partial_store_pipeline_2, self.isp_sample_control, self.main_value(0x10051));
        pr_err!("G17: shape {} res: ta(flags,ctl,ptc,scs)={:x?} 3d={:x?} 0x10039 main/store/resume/load={:x?} main bits17..19={:#x}\n",
            tag, self.resources[0], self.resources[1], actions, action_bits);
        pr_err!("G17: shape {} cmd: empty_mode={} empty_tile_flags(main,store,load,dispatch)={:?} short_partial={} pass_flag={} flag_a={} flag_b={} flag_c={} tail={} main rce={} mid={} counts(main,store,resume,load)={:?}\n",
            tag, self.empty_mode, self.empty_tile_flags, self.short_partial, self.command_flags[0],
            self.command_flags[1], self.command_flags[2], self.command_flags[3],
            self.command_flags[4], self.main_counts[0], self.main_counts[1], self.counts);
        pr_err!("G17: shape {} ptr: tilemap={:#x} layermeta={:#x} heapmeta={:#x} tpc={:#x}+{:#x} gtp={:x?} pb={:#x} scene={:#x} scene_end={:#x} aux_fb={:#x} deflake={:#x} mesh={:#x} layout(tilemap_size,layer_off,heap_off,used,tvb_blocks)={:x?}\n",
            tag, self.tilemap_va, self.layermeta_va, self.heapmeta_va, self.tpc[0], self.tpc[1],
            self.gtp_va, self.pb_va, self.scene_va, self.scene_end_va, self.aux_fb_va,
            self.deflake_va, self.mesh_storage, self.layout);
        pr_err!("G17: shape {} addr: scissor={:#x} depth_bias={:#x} visibility={:#x} vdm={:#x} samplers={} usc_base={:#x} bg={:#x} partial_bg={:#x} eot={:#x} partial_eot={:#x} rsrc(bg,pbg,eot,peot)={:x?}\n",
            tag, self.scissor_va, self.depth_bias_va, self.occlusion_query_va, self.vdm_va,
            self.sampler_count, self.usc_base, self.usc[0], self.usc[1], self.usc[2], self.usc[3],
            self.rsrc);
        for (name, array) in self.arrays() {
            for (chunk, registers) in array.chunks(28).enumerate() {
                pr_err!("G17: shape {} {}[{}..]: {}\n", tag, name, chunk * 28, Registers(registers));
            }
        }
    }
}

/// Little-endian read of `n` bytes at `at` (u64::MAX when out of range).
pub(crate) fn read(bytes: &[u8], at: usize, n: usize) -> u64 {
    bytes.get(at..at + n).map_or(u64::MAX, |b| b.iter().rev().fold(0u64, |v, x| v << 8 | u64::from(*x)))
}

/// Ordered (register, value) pairs as "reg=value" in hex.
struct Registers<'a>(&'a [RegisterWrite]);
impl fmt::Display for Registers<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for write in self.0 {
            let write = *write;
            let (reg, value) = (write.reg, write.value);
            write!(f, "{:x}={:x} ", reg, value)?;
        }
        Ok(())
    }
}

/// Pointer words compared between the TA and 3D commands: (name, source, register or byte
/// offset, bytes). Sources: 0 TA register, 1 TA command, 2 3D register, 3 3D command.
pub(crate) const PTRS: usize = 27;
const PTR_SOURCES: [(&str, u8, u32, usize); PTRS] = [
    ("ta1c039", 0, 0x1c039, 8), ("ta1c9c8", 0, 0x1c9c8, 8), ("ta1c0a1", 0, 0x1c0a1, 8),
    ("ta1c031", 0, 0x1c031, 8), ("ta1c9c0", 0, 0x1c9c0, 8), ("ta1c079", 0, 0x1c079, 8),
    ("ta1c9d8", 0, 0x1c9d8, 8), ("ta1ca30", 0, 0x1ca30, 8), ("ta16c39", 0, 0x16c39, 8),
    ("ta1c910", 0, 0x1c910, 8), ("ta1c830", 0, 0x1c830, 8),
    ("tacmd_ctx", 1, 0xc, 4), ("tacmd_pbslot", 1, 0x18, 4), ("tacmd_pbstate", 1, 0x20, 8),
    ("tacmd_scene", 1, 0x28, 8), ("tacmd_scene_end", 1, 0x30, 8), ("tacmd_tpc", 1, 0x780, 8),
    ("f16429", 2, 0x16429, 8), ("f16060", 2, 0x16060, 8), ("f16098", 2, 0x16098, 8),
    ("f1ca28", 2, 0x1ca28, 8), ("f1c838", 2, 0x1c838, 8),
    ("fcmd_ctx", 3, 0xc, 4), ("fcmd_pmstate", 3, 0x28, 8), ("fcmd_scene", 3, 0x30, 8),
    ("fcmd_scene_end", 3, 0x38, 8), ("fcmd_tilemap", 3, 0x40, 8),
];

/// The first value of register `reg` in the program array at `offset` (u64::MAX: absent).
fn program_value(bytes: &[u8], offset: usize, reg: u32) -> u64 {
    let count = (read(bytes, offset + COUNT_AT, 2) as usize).min(0x700 / REGISTER_BYTES);
    (0..count).map(|slot| offset + slot * REGISTER_BYTES)
        .find(|&at| read(bytes, at, 4) as u32 & 0x3ffff == reg)
        .map_or(u64::MAX, |at| read(bytes, at + 4, 8))
}

/// The [`PTR_SOURCES`] words of an encoded TA and 3D command.
pub(crate) fn ptr_words(ta: &[u8], fragment: &[u8]) -> [u64; PTRS] {
    PTR_SOURCES.map(|(_, source, at, bytes)| match source {
        0 => program_value(ta, M5_TA_PROGRAM_OFFSET as usize, at),
        1 => read(ta, at as usize, bytes),
        2 => program_value(fragment, M5_3D_PROGRAM_OFFSETS[0] as usize, at),
        _ => read(fragment, at as usize, bytes),
    })
}

struct Names<'a>(&'a [&'a str]);
impl fmt::Display for Names<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for name in self.0 {
            write!(f, "{} ", name)?;
        }
        Ok(())
    }
}

fn report_ptrs(tag: &str, p: &[u64; PTRS], live: Option<&[u64; PTRS]>, tilemap: u64) {
    const B: u64 = 0x10_0000_0000;
    let off = |v: u64| if v == u64::MAX { v } else { v.wrapping_add(B) };
    let meta = |v: u64| if v == u64::MAX { v } else { (v & !(1 << 63)).wrapping_add(B) };
    // Groups that must agree: (name, values normalized to full GPU addresses or plain words).
    let groups: [(&str, [u64; 4]); 10] = [
        ("tilemap", [off(p[0]), off(p[1]), p[17], p[26]]),
        ("heapmeta", [meta(p[3]), meta(p[4]), p[19], p[19]]),
        ("layermeta", [off(p[5]), off(p[6]), p[18], p[18]]),
        ("tpc", [off(p[2]), p[16], p[16], p[16]]),
        ("pm_scratch", [p[7], p[8], p[20], p[20]]),
        ("pb_slot", [p[10], p[12], p[21], p[21]]),
        ("pb_state", [p[13], p[23], p[23], p[23]]),
        ("scene", [p[14], p[24], p[24], p[24]]),
        ("scene_end", [p[15], p[25], p[25], p[25]]),
        ("ctx", [p[11], p[22], p[22], p[22]]),
    ];
    let mut mismatch: [&str; 11] = [""; 11];
    let mut count = 0;
    for (name, values) in groups.iter() {
        if values.iter().any(|v| *v != values[0]) && count < mismatch.len() {
            mismatch[count] = name;
            count += 1;
        }
    }
    if p[17] != tilemap && count < mismatch.len() {
        mismatch[count] = "tilemap_vs_resources";
        count += 1;
    }
    let mut changed: [&str; PTRS] = [""; PTRS];
    let mut changes = 0;
    if let Some(live) = live {
        for (index, (name, ..)) in PTR_SOURCES.iter().enumerate() {
            if live[index] != p[index] {
                changed[changes] = name;
                changes += 1;
            }
        }
    }
    pr_err!("G17: shape {} ta_ptrs: tilemap ta={:#x}/{:#x} 3d={:#x}/{:#x} heapmeta ta={:#x}/{:#x} 3d={:#x} layermeta ta={:#x}/{:#x} 3d={:#x} tpc ta={:#x} cmd={:#x} pm_scratch ta={:#x}/{:#x} 3d={:#x} metric ta={:#x} pb_slot ta={:#x}/{:#x} 3d={:#x} pb_state ta={:#x} 3d={:#x} scene ta={:#x} 3d={:#x} scene_end ta={:#x} 3d={:#x} ctx ta={:#x} 3d={:#x} mismatch=[{}] live_changed={}[{}]\n",
        tag, p[0], p[1], p[17], p[26], p[3], p[4], p[19], p[5], p[6], p[18], p[2], p[16],
        p[7], p[8], p[20], p[9], p[10], p[12], p[21], p[13], p[23], p[14], p[24], p[15], p[25],
        p[11], p[22], Names(&mismatch[..count]), if live.is_some() { "" } else { "n/a" },
        Names(&changed[..changes]));
}

/// The pair's ring of retired shapes.
pub(crate) struct Ring {
    shapes: [Option<KBox<Shape>>; RING],
    next: usize,
    /// An evicted or unused shape, reused by the next preparation instead of allocating.
    spare: Option<KBox<Shape>>,
    /// The most recent healthy retirement with target word +0 zero (preallocated; valid
    /// when `empty_valid`).
    empty: Option<KBox<Shape>>,
    empty_valid: bool,
}
impl Ring {
    pub(crate) const fn new() -> Self {
        Self { shapes: [const { None }; RING], next: 0, spare: None, empty: None, empty_valid: false }
    }
    /// Preallocates the slot for the last empty-region render (allocating context only).
    pub(crate) fn reserve_empty(&mut self) {
        if self.empty.is_none() {
            self.empty = Shape::new_boxed();
        }
    }
    /// The last healthy render with an empty region entry, if any.
    pub(crate) fn empty(&self) -> Option<&Shape> {
        self.empty.as_deref().filter(|_| self.empty_valid)
    }
    /// Keeps a retired render's shape; the oldest becomes the spare. Never allocates. Counts
    /// it by its region entry and copies an empty-region one into the preallocated slot.
    pub(crate) fn push(&mut self, mut shape: KBox<Shape>) {
        shape.retire();
        shape.count_retired();
        // Prefer the informative kind: executed with normal TA accounting (round 7).
        let keep = shape.region_empty() && (!self.empty_valid
            || (shape.executed() && shape.scene10() != 0)
            || !self.empty.as_deref().is_some_and(|kept| kept.executed() && kept.scene10() != 0));
        if keep {
            if let Some(slot) = self.empty.as_deref_mut() {
                // SAFETY: Two distinct boxes of the same integer-only type.
                unsafe { core::ptr::copy_nonoverlapping(&*shape, slot, 1) };
                self.empty_valid = true;
            }
        }
        let evicted = self.shapes[self.next % RING].replace(shape);
        self.next = self.next.wrapping_add(1);
        if self.spare.is_none() {
            self.spare = evicted;
        }
    }
    /// Keeps a shape that is not entered in the ring for reuse.
    pub(crate) fn recycle(&mut self, shape: KBox<Shape>) {
        if self.spare.is_none() {
            self.spare = Some(shape);
        }
    }
    /// A shape for the next render: the spare, else a new one.
    pub(crate) fn take(&mut self) -> Option<KBox<Shape>> {
        self.spare.take().or_else(Shape::new_boxed)
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Shape> + Clone {
        self.shapes.iter().filter_map(|shape| shape.as_deref())
    }
}

/// Prints the culprit and then, with `m5_diag_shape`, up to four healthy shapes from `rings`:
/// first those of the culprit's kind, then the most recently retired.
pub(crate) fn report<'a>(
    culprit: Option<(&Shape, &str)>,
    blamed: u8,
    rings: impl Iterator<Item = &'a Shape> + Clone,
    empties: impl Iterator<Item = &'a Shape>,
    culprit_content: impl FnOnce(&Shape),
) {
    let word = FAULT_KICK.load(Relaxed);
    match culprit {
        Some((shape, how)) => {
            pr_err!("G17: shape culprit blamed queue {} c078={:#x}: matched by {}\n", blamed, word, how);
            shape.report("culprit");
            if crate::module_parameters::m5_diag_shape.value() != 0 {
                culprit_content(shape);
            }
        }
        None => pr_err!("G17: shape culprit blamed queue {} c078={:#x}: no captured render matched\n",
            blamed, word),
    }
    if crate::module_parameters::m5_diag_shape.value() == 0 {
        return;
    }
    // The four best by (same kind as the culprit, retirement order).
    let mut best: [Option<(bool, u64, &Shape)>; 4] = [None; 4];
    for shape in rings {
        let kind = culprit.is_some_and(|(c, _)| c.same_kind(shape));
        let key = (kind, shape.retired);
        // An empty slot (None) sorts first.
        let worst = best.iter().enumerate()
            .min_by_key(|(_, entry)| entry.map(|(k, r, _)| (k, r)))
            .map_or(0, |(index, _)| index);
        let replace = match best[worst] {
            None => true,
            Some((k, r, _)) => key > (k, r),
        };
        if replace {
            best[worst] = Some((kind, shape.retired, shape));
        }
    }
    best.sort_unstable_by_key(|entry| core::cmp::Reverse(entry.map(|(k, r, _)| (k, r))));
    for (index, (kind, _, shape)) in best.iter().flatten().enumerate() {
        pr_err!("G17: shape healthy{} same_kind={} (of the {} most recent retired per pair)\n",
            index, kind, RING);
        let tag: &str = ["healthy0", "healthy1", "healthy2", "healthy3"][index];
        shape.report(tag);
        shape.report_content(tag);
    }
    match empties.max_by_key(|shape| shape.retired) {
        Some(shape) => {
            pr_err!("G17: shape healthy_empty: the most recent healthy retirement with target word +0 zero\n");
            shape.report("healthy_empty");
            shape.report_content("healthy_empty");
        }
        None => pr_err!("G17: shape healthy_empty: no healthy retirement with target word +0 zero kept\n"),
    }
    report_counts();
}
