// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! Recovery targets for retained render graphs while firmware is halted.

use crate::g17::{runtime::Registry, status::VmStatus};
use kernel::{prelude::*, sync::Arc};

impl Registry {
    pub(in crate::g17::runtime) fn report_render_recovery(&self, qid: u8, port: &crate::g17::port::Port) {
        pr_warn!("G17: recovery grow pending={} active={:?} reply={:?}\n",
            self.render.grow.requests.len(), self.render.grow.active, self.render.grow.reply);
        for entry in self.render.entries.iter().flatten().filter(|entry| {
            entry.reservation.ids.iter().any(|id| id.qid() == qid)
        }) {
            if let Some(pair) = &entry.pair {
                let qids = pair.qids();
                pair.report_recovery((|| Ok([port.stamp(qids[0])?, port.stamp(qids[1])?]))());
                pair.report_recovery_work();
            } else {
                pr_warn!("G17: recovery render queue={} unavailable borrowed={} constructing={}\n",
                    qid, entry.borrowed_active, entry.constructing);
            }
        }
        // Diagnostics: every other pair, and the stamps of every QID they wait on.
        for entry in self.render.entries.iter().flatten().filter(|entry| {
            !entry.reservation.ids.iter().any(|id| id.qid() == qid)
        }) {
            if let Some(pair) = &entry.pair {
                if pair.active_count() == 0 { continue; }
                let qids = pair.qids();
                pr_warn!("G17: recovery other pair:\n");
                pair.report_recovery((|| Ok([port.stamp(qids[0])?, port.stamp(qids[1])?]))());
            }
        }
        for pair in self.render.entries.iter().flatten().filter_map(|entry| entry.pair.as_ref()) {
            pair.report_retired();
        }
        self.report_render_shapes(qid);
        let mut stamps = [0u32; 128];
        for (q, stamp) in stamps.iter_mut().enumerate() {
            *stamp = port.stamp(q as u8).unwrap_or(u32::MAX);
        }
        for queue in self.compute.iter().flatten().filter_map(|entry| entry.queue.as_deref()) {
            queue.report_recovery(port.stamp(queue.qid()));
        }
        for chunk in 0..8 {
            pr_warn!("G17: recovery stamps q{}..: {:x?}\n", chunk * 16, &stamps[chunk * 16..chunk * 16 + 16]);
        }
    }

    /// Diagnostics (GPU_J704_TIMEOUT.md round 7): the pass shape of the culprit render, found by
    /// the queue id and kick in 0xc078 among every pair's active renders, else the oldest active
    /// render of the blamed queue's pair; then healthy retired shapes for comparison.
    fn report_render_shapes(&self, blamed: u8) {
        use crate::g17::queue::render::shape;
        let pairs = || self.render.entries.iter().flatten().filter_map(|entry| entry.pair.as_ref());
        let word = shape::FAULT_KICK.load(core::sync::atomic::Ordering::Relaxed);
        let named = (word != u64::MAX)
            .then(|| pairs().find_map(|pair| pair.shape_named(word).map(|found| (pair, found))))
            .flatten()
            .map(|(pair, (found, stage))| (pair, found, if stage == 0 { "c078 qid+kick (TA kick)" }
                else { "c078 qid+kick (3D kick)" }));
        let culprit = named.or_else(|| {
            let pair = pairs().find(|pair| pair.qids().contains(&blamed))?;
            pair.shape_oldest().map(|found| (pair, found,
                "fallback: oldest active render of the blamed queue (no qid+kick match)"))
        });
        shape::report(culprit.map(|(_, found, how)| (found, how)), blamed,
            pairs().flat_map(|pair| pair.shapes_retired()),
            pairs().filter_map(|pair| pair.shape_empty()),
            |found| if let Some((pair, _, _)) = culprit { pair.report_shape_views(found, "culprit") });
        if crate::module_parameters::m5_diag_shape.value() != 0 {
            let exclude = culprit.map(|(pair, found, _)| (pair.qids()[1], found.ordinal));
            let mut budget = 3;
            for pair in pairs() {
                pair.report_inflight(exclude, &mut budget);
            }
        }
    }

    pub(in crate::g17::runtime) fn render_pass_started(&self, qid: u8, stamp: u64) -> Option<bool> {
        self.render.entries.iter().flatten()
            .filter_map(|entry| entry.pair.as_ref())
            .find(|pair| pair.qids().contains(&qid))?
            .pass_started(stamp)
    }

    /// See `recovery::Queues::render_culprit`.
    pub(in crate::g17::runtime) fn render_culprit(&self, blamed: u8) -> Option<u128> {
        let pairs = || self.render.entries.iter().flatten().filter_map(|entry| entry.pair.as_ref());
        let untouched = pairs().find(|pair| pair.qids().contains(&blamed))?.front_untouched() == Some(true);
        let running = pairs().filter_map(|pair| {
            let qids = pair.qids().into_iter().fold(0, |mask, qid| mask | 1u128 << qid);
            Some((qids, pair.running_owner()?))
        });
        culprit(running, untouched, |culprit| {
            let mut foreign = false;
            for queue in self.compute.iter().flatten().filter_map(|entry| entry.queue.as_deref()) {
                queue.visit_owners(&mut |owner| foreign |= !Arc::ptr_eq(owner, culprit));
            }
            foreign
        })
    }

    pub(in crate::g17::runtime) fn publish_render_recovery_targets(&self) -> Option<Error> {
        let mut failure = None;
        for pair in self.render.entries.iter().flatten()
            .filter_map(|entry| entry.pair.as_ref())
        {
            if let Err(error) = pair.publish_recovery_targets() {
                failure = failure.or(Some(error));
            }
        }
        failure
    }
}

/// Picks the QIDs to blame from each pair's running owner. A VM an earlier recovery found guilty
/// comes first, even when the blamed render had started: its groups queued before that keep
/// hanging the GPU one per restart, while bystanders' renders whose tiler started wait behind
/// them, and that recovery failed its interrupted work. Otherwise the blamed render must be
/// `untouched`, exactly one VM may be running, and `foreign` must find no other VM's compute work.
fn culprit<'a>(
    running: impl Iterator<Item = (u128, &'a Arc<VmStatus>)>,
    untouched: bool,
    foreign: impl FnOnce(&Arc<VmStatus>) -> bool,
) -> Option<u128> {
    let (mut guilty, mut qids, mut several) = (0u128, 0u128, false);
    let mut culprit: Option<&Arc<VmStatus>> = None;
    for (pair, owner) in running {
        if owner.recovery_guilty() {
            guilty |= pair;
        }
        several |= culprit.is_some_and(|known| !Arc::ptr_eq(known, owner));
        culprit = Some(owner);
        qids |= pair;
    }
    if guilty != 0 {
        return Some(guilty);
    }
    let culprit = culprit.filter(|_| untouched && !several)?;
    (!foreign(culprit)).then_some(qids)
}

#[kernel::macros::kunit_tests(asahi_g17_render_culprit)]
mod tests {
    use super::*;

    fn vm() -> Arc<VmStatus> {
        Arc::pin_init(VmStatus::new(), GFP_KERNEL).unwrap()
    }

    #[test]
    fn a_guilty_vm_still_running_is_blamed_before_running_bystanders() {
        let (spinner, bystander) = (vm(), vm());
        let running = || [(0b11, &bystander), (0b1100, &spinner)].into_iter();
        // Two VMs running and no earlier guilt: the firmware's blame stands, even when one of
        // them failed only for its own client.
        bystander.record(EIO);
        assert_eq!(culprit(running(), true, |_| false), None);
        spinner.note_recovery_guilty();
        assert_eq!(culprit(running(), true, |_| panic!("compute is not consulted")), Some(0b1100));
        // The bystander's blamed render whose tiler started is starved too.
        assert_eq!(culprit(running(), false, |_| panic!("compute is not consulted")), Some(0b1100));
    }

    #[test]
    fn one_running_vm_is_blamed_only_without_foreign_compute() {
        let (a, b) = (vm(), vm());
        assert_eq!(culprit([(0b11, &a), (0b1100, &a)].into_iter(), true, |_| false), Some(0b1111));
        // A blamed render that had started may be the hang itself.
        assert_eq!(culprit([(0b11, &a), (0b1100, &a)].into_iter(), false, |_| false), None);
        assert_eq!(culprit([(0b11, &a)].into_iter(), true, |_| true), None);
        assert_eq!(culprit([(0b11, &a), (0b1100, &b)].into_iter(), true, |_| false), None);
        assert_eq!(culprit(core::iter::empty(), true, |_| false), None);
    }
}
