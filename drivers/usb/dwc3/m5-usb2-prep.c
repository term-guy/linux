// SPDX-License-Identifier: GPL-2.0
/*
 * M5 (T8142) bring-up debug only: there is no ATC PHY driver for this SoC
 * yet, so redo m1n1's USB2-only port bring-up (usb.c usb_phy_bringup) before
 * the plain snps,dwc3 nodes probe: USB2 PHY out of reset and running,
 * pipehandler muxed to the dummy PHY, DWC3 reset released and unclamped.
 * Register state is printed before and after for each port, and a DWC3
 * device soft reset (DCTL.CSFTRST) is tried on the inherited state and again
 * after an Apple-style DWC3 reset cycle (as atcphy's dwc3 reset control does).
 */

#include <linux/atomic.h>
#include <linux/delay.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/printk.h>

#define USB2PHY_USBCTL		0x00
#define USB2PHY_USBCTL_RUN	2
#define USB2PHY_CTL		0x04
#define USB2PHY_CTL_RESET	BIT(0)
#define USB2PHY_CTL_PORT_RESET	BIT(1)
#define USB2PHY_CTL_APB_RESET_N	BIT(2)
#define USB2PHY_CTL_SIDDQ	BIT(3)
#define USB2PHY_USBCTL_ISOLATION 4
#define USB2PHY_MISCTUNE_GATE_OFF (BIT(29) | BIT(30))
#define USB2PHY_SIG		0x08
#define USB2PHY_MISCTUNE	0x1c

#define PIPEHANDLER_MUX_CTRL		0x0c
#define PIPEHANDLER_MUX_CTRL_DUMMY	0x22
#define PIPEHANDLER_AON_GEN		0x1c
#define PIPEHANDLER_AON_GEN_DWC3_RESET_N BIT(0)
#define PIPEHANDLER_AON_GEN_DWC3_FORCE_CLAMP_EN BIT(4)
#define PIPEHANDLER_NONSELECTED_OVERRIDE 0x20

#define DWC3_GCTL	0xc110
#define DWC3_GSNPSID	0xc120
#define DWC3_DCTL	0xc704
#define DWC3_DCTL_CSFTRST BIT(30)
#define DWC3_DEVTEN	0xc708
#define DWC3_GEVNTADRLO	0xc400
#define DWC3_GEVNTADRHI	0xc404
#define DWC3_GEVNTSIZ	0xc408
#define DWC3_GEVNTSIZ_INTMASK BIT(31)
#define DWC3_GEVNTCOUNT	0xc40c

/* Physical bases from the ADT (identity-mapped /arm-io ranges) */
static const struct {
	int idx;
	phys_addr_t drd, usb2phy, pipehandler;
} ports[] = {
	{ 0, 0x402280000, 0x402a90000, 0x402a84000 },
	{ 1, 0x40a280000, 0x40aa90000, 0x40aa84000 },
	{ 3, 0x41a280000, 0x41aa90000, 0x41aa84000 },
};

static void dump(int idx, const char *when, void __iomem *drd,
		 void __iomem *phy, void __iomem *ph)
{
	pr_info("m5-usb2-prep: port%d %s: phy usbctl=%08x ctl=%08x sig=%08x misctune=%08x\n",
		idx, when, readl(phy + USB2PHY_USBCTL), readl(phy + USB2PHY_CTL),
		readl(phy + USB2PHY_SIG), readl(phy + USB2PHY_MISCTUNE));
	pr_info("m5-usb2-prep: port%d %s: pipe mux=%08x aon_gen=%08x nonsel=%08x dwc3 snpsid=%08x gctl=%08x dctl=%08x\n",
		idx, when, readl(ph + PIPEHANDLER_MUX_CTRL), readl(ph + PIPEHANDLER_AON_GEN),
		readl(ph + PIPEHANDLER_NONSELECTED_OVERRIDE), readl(drd + DWC3_GSNPSID),
		readl(drd + DWC3_GCTL), readl(drd + DWC3_DCTL));
}

static void rmw(void __iomem *reg, u32 clr, u32 set)
{
	writel((readl(reg) & ~clr) | set, reg);
	udelay(10);
}

static void csftrst_test(int idx, const char *when, void __iomem *drd)
{
	int us;

	rmw(drd + DWC3_DCTL, 0, DWC3_DCTL_CSFTRST);
	for (us = 0; us < 200000; us += 10) {
		if (!(readl(drd + DWC3_DCTL) & DWC3_DCTL_CSFTRST)) {
			pr_info("m5-usb2-prep: port%d %s: CSFTRST cleared after ~%d us\n",
				idx, when, us);
			return;
		}
		udelay(10);
	}
	pr_info("m5-usb2-prep: port%d %s: CSFTRST STUCK (dctl=%08x)\n",
		idx, when, readl(drd + DWC3_DCTL));
}

/*
 * Exported for the debug hooks in apple-dart.c and dwc3 core.c: dump the
 * DWC3 PHY config registers and try a device soft reset on the controller
 * at physical address @drd.
 */
/* Serialises every sleepable-context CSFTRST test (initcall thread vs deferred probe). */
static DEFINE_MUTEX(m5dbg_lock);

/*
 * #120: reading port3 DART0 ERROR (+0x100) hangs by gadget init but works
 * under m1n1. Read it (read-only) at each hook to find the step that breaks it.
 */
void m5dbg_dart_err(phys_addr_t drd, const char *when);
void m5dbg_dart_err(phys_addr_t drd, const char *when)
{
	void __iomem *d;

	if (drd != 0x41a280000)
		return;
	d = ioremap(0x41af00000, 0x4000);
	if (!d)
		return;
	pr_info("m5dart %s: reading dart0 err\n", when);
	pr_info("m5dart %s: dart0 err=%08x\n", when, readl(d + 0x100));
	iounmap(d);
}

void m5dbg_csftrst_phys(phys_addr_t drd, const char *when)
{
	void __iomem *r;
	int us;

	m5dbg_dart_err(drd, when);
	r = ioremap(drd, 0xd000);
	if (!r)
		return;
	mutex_lock(&m5dbg_lock);
	pr_info("m5dbg %pa %s: gusb2phycfg=%08x gusb3pipectl=%08x gctl=%08x dctl=%08x\n",
		&drd, when, readl(r + 0xc200), readl(r + 0xc2c0), readl(r + DWC3_GCTL),
		readl(r + DWC3_DCTL));
	rmw(r + DWC3_DCTL, BIT(31), DWC3_DCTL_CSFTRST);
	for (us = 0; us < 200000; us += 10) {
		if (!(readl(r + DWC3_DCTL) & DWC3_DCTL_CSFTRST)) {
			pr_info("m5dbg %pa %s: CSFTRST cleared after ~%d us\n", &drd, when, us);
			goto out;
		}
		udelay(10);
	}
	pr_info("m5dbg %pa %s: CSFTRST STUCK\n", &drd, when);
out:
	mutex_unlock(&m5dbg_lock);
	iounmap(r);
}

/* Apple-style DWC3 reset cycle: assert+clamp, USB2 PHY off, PHY on, deassert */
static void m5_usb2_cycle(void __iomem *phy, void __iomem *ph)
{
	/* Assert DWC3 reset + clamp, USB2 PHY off (atcphy reset assert) */
	rmw(ph + PIPEHANDLER_AON_GEN, PIPEHANDLER_AON_GEN_DWC3_RESET_N,
	    PIPEHANDLER_AON_GEN_DWC3_FORCE_CLAMP_EN);
	writel((readl(phy + USB2PHY_USBCTL) & ~7) | USB2PHY_USBCTL_ISOLATION,
	       phy + USB2PHY_USBCTL);
	udelay(10);
	rmw(phy + USB2PHY_CTL, 0, USB2PHY_CTL_SIDDQ);
	rmw(phy + USB2PHY_CTL, 0, USB2PHY_CTL_PORT_RESET);
	rmw(phy + USB2PHY_CTL, 0, USB2PHY_CTL_RESET);
	rmw(phy + USB2PHY_CTL, USB2PHY_CTL_APB_RESET_N, 0);
	rmw(phy + USB2PHY_MISCTUNE, 0, USB2PHY_MISCTUNE_GATE_OFF);

	/* USB2 PHY on, device mode (m1n1's SIG value, no SIG_HOST) */
	writel(0x01c1000f, phy + USB2PHY_SIG);
	udelay(10);
	rmw(phy + USB2PHY_CTL, USB2PHY_CTL_SIDDQ, 0);
	rmw(phy + USB2PHY_CTL, USB2PHY_CTL_RESET, 0);
	rmw(phy + USB2PHY_CTL, USB2PHY_CTL_PORT_RESET, 0);
	rmw(phy + USB2PHY_CTL, 0, USB2PHY_CTL_APB_RESET_N);
	rmw(phy + USB2PHY_MISCTUNE, USB2PHY_MISCTUNE_GATE_OFF, 0);
	writel(USB2PHY_USBCTL_RUN, phy + USB2PHY_USBCTL);

	/* PIPE stays on the dummy PHY (USB2 only) */
	writel(PIPEHANDLER_MUX_CTRL_DUMMY, ph + PIPEHANDLER_MUX_CTRL);
	writel(0x9332, ph + PIPEHANDLER_NONSELECTED_OVERRIDE);

	/* Deassert DWC3 reset: unclamp, then release RESET_N */
	rmw(ph + PIPEHANDLER_AON_GEN, PIPEHANDLER_AON_GEN_DWC3_FORCE_CLAMP_EN, 0);
	rmw(ph + PIPEHANDLER_AON_GEN, 0, PIPEHANDLER_AON_GEN_DWC3_RESET_N);
	udelay(100);
}

/*
 * Read-only dump of this port's pmgr domains: atcN_usb / atcN_usb_aon (NUB
 * pmgr 0x388280000) and atcN_common (main pmgr 0x380700000).
 */
#define PMGR_AUTO_ENABLE BIT(28)
#define PMGR_FLAGS (BIT(8) | BIT(9))

void m5_usb2_pmgr_dump(phys_addr_t drd, const char *when)
{
	static const struct { phys_addr_t drd; u32 usb, aon, common; } map[] = {
		{ 0x402280000, 0xd0, 0xc8, 0x2c8 },
		{ 0x40a280000, 0xe0, 0xd8, 0x2d0 },
		{ 0x41a280000, 0x100, 0xf8, 0x2e0 },
	};
	void __iomem *nub = ioremap(0x388280000, 0x4000);
	void __iomem *pm = ioremap(0x380700000, 0x4000);
	void __iomem *r[3];
	int i;

	if (!nub || !pm)
		goto out;
	for (i = 0; i < ARRAY_SIZE(map); i++) {
		if (map[i].drd != drd)
			continue;
		r[0] = nub + map[i].usb;
		r[1] = nub + map[i].aon;
		r[2] = pm + map[i].common;
		pr_info("m5-usb2-prep: %pa pmgr %s: usb=%08x aon=%08x common=%08x\n",
			&drd, when, readl(r[0]), readl(r[1]), readl(r[2]));
	}
out:
	if (nub)
		iounmap(nub);
	if (pm)
		iounmap(pm);
}

/* Called from dwc3_probe: full reset cycle right before dwc3_core_init */
void m5_usb2_release(phys_addr_t drd)
{
	void __iomem *phy = ioremap(drd + 0x810000, 0x4000);
	void __iomem *ph = ioremap(drd + 0x804000, 0x4000);

	m5_usb2_pmgr_dump(drd, "at dwc3_probe");
	m5dbg_dart_err(drd, "before dwc3_probe reset cycle");
	if (phy && ph) {
		m5_usb2_cycle(phy, ph);
		pr_info("m5-usb2-prep: %pa reset cycle at dwc3_probe done, aon_gen=%08x phy ctl=%08x usbctl=%08x\n",
			&drd, readl(ph + PIPEHANDLER_AON_GEN), readl(phy + USB2PHY_CTL),
			readl(phy + USB2PHY_USBCTL));
	}
	if (phy)
		iounmap(phy);
	if (ph)
		iounmap(ph);
}

static int __init m5_usb2_prep(void)
{
	int i;

	if (!of_machine_is_compatible("apple,j704"))
		return 0;

	pr_info("m5-usb2-prep: delaying 2000 ms before early CSFTRST tests\n");
	mdelay(2000);

	for (i = 0; i < ARRAY_SIZE(ports); i++) {
		void __iomem *drd = ioremap(ports[i].drd, 0xd000);
		void __iomem *phy = ioremap(ports[i].usb2phy, 0x4000);
		void __iomem *ph = ioremap(ports[i].pipehandler, 0x4000);

		if (!drd || !phy || !ph) {
			pr_err("m5-usb2-prep: port%d: ioremap failed\n", ports[i].idx);
			goto unmap;
		}

		dump(ports[i].idx, "before", drd, phy, ph);
		m5_usb2_pmgr_dump(ports[i].drd, "early_initcall");
		m5dbg_csftrst_phys(ports[i].drd, "early_initcall");

		csftrst_test(ports[i].idx, "inherited", drd);

		m5_usb2_cycle(phy, ph);

		csftrst_test(ports[i].idx, "after reset cycle", drd);

		/*
		 * m1n1's event buffer still points at an IOVA in m1n1's DART page
		 * tables, which the Linux DART driver is about to reset. Stop all
		 * DWC3 event DMA now; dwc3 core reprograms this after its own reset.
		 */
		pr_info("m5-usb2-prep: port%d inherited evnt adr=%08x:%08x siz=%08x cnt=%08x devten=%08x\n",
			ports[i].idx, readl(drd + DWC3_GEVNTADRHI), readl(drd + DWC3_GEVNTADRLO),
			readl(drd + DWC3_GEVNTSIZ), readl(drd + DWC3_GEVNTCOUNT),
			readl(drd + DWC3_DEVTEN));
		writel(0, drd + DWC3_DEVTEN);
		writel(DWC3_GEVNTSIZ_INTMASK, drd + DWC3_GEVNTSIZ);
		writel(readl(drd + DWC3_GEVNTCOUNT), drd + DWC3_GEVNTCOUNT);
		writel(0, drd + DWC3_GEVNTADRLO);
		writel(0, drd + DWC3_GEVNTADRHI);
		m5dbg_csftrst_phys(ports[i].drd, "after evnt quiesce");

		dump(ports[i].idx, "after", drd, phy, ph);
unmap:
		if (drd)
			iounmap(drd);
		if (phy)
			iounmap(phy);
		if (ph)
			iounmap(ph);
	}
	return 0;
}
early_initcall(m5_usb2_prep);

/* Port0 reset checkpoints at the end of successive initcall levels. */
#define M5_USB_STAGE_TEST(level) \
static int __init m5_usb_stage_##level(void) \
{ \
	if (of_machine_is_compatible("apple,j704")) \
		m5dbg_csftrst_phys(ports[0].drd, #level); \
	return 0; \
} \
level(m5_usb_stage_##level)

M5_USB_STAGE_TEST(pure_initcall);
M5_USB_STAGE_TEST(core_initcall_sync);
M5_USB_STAGE_TEST(postcore_initcall_sync);
M5_USB_STAGE_TEST(arch_initcall_sync);
M5_USB_STAGE_TEST(subsys_initcall_sync);
M5_USB_STAGE_TEST(fs_initcall_sync);
M5_USB_STAGE_TEST(rootfs_initcall);

/* Stop after the first failing device-initcall checkpoint; no log per success. */
void __init m5_usb2_device_initcall_checkpoint(initcall_t fn);
void __init m5_usb2_device_initcall_checkpoint(initcall_t fn)
{
	static bool stopped;
	static initcall_t last_good;
	static unsigned int checked;
	phys_addr_t drd = ports[0].drd;
	void __iomem *r;
	int us;

	if (stopped || !of_machine_is_compatible("apple,j704"))
		return;
	r = ioremap(drd, 0xd000);
	if (!r) {
		stopped = true;
		pr_err("m5dev %pa CSFTRST checkpoint mapping failed\n", &drd);
		return;
	}
	mutex_lock(&m5dbg_lock);
	/* Same reset operation as the existing m5dbg_csftrst_phys test. */
	rmw(r + DWC3_DCTL, BIT(31), DWC3_DCTL_CSFTRST);
	for (us = 0; us < 200000; us += 10) {
		if (!(readl(r + DWC3_DCTL) & DWC3_DCTL_CSFTRST)) {
			if (!fn)
				pr_info("m5dev %pa CSFTRST device-level entry clear ~%d us\n", &drd, us);
			else {
				last_good = fn;
				checked++;
			}
			goto out;
		}
		udelay(10);
	}
	stopped = true;
	pr_info("m5dev %pa CSFTRST FIRST STUCK after %ps; last clear %ps; checked=%u\n",
		&drd, fn, last_good, checked);
out:
	mutex_unlock(&m5dbg_lock);
	iounmap(r);
}

/*
 * pmgr-pwrstate localization (#113). m5pm_record() only stores a pmgr write
 * in a small lock-free ring; it is called with the IRQ-safe genpd spinlock
 * held and does no MMIO. m5pm_checkpoint() runs the port0 reset test and is
 * for sleepable apple_pmgr_ps_probe() context only. Output lines carry
 * "CSFTRST" and the port0 address so the diagnostic /init keeps them.
 */
#define M5PM_RING 16
static struct { const char *name; u32 off, before, after; unsigned int seq; char tag; }
	m5pm_ring[M5PM_RING];
static atomic_t m5pm_seq = ATOMIC_INIT(0);

void m5pm_record(const char *name, u32 off, char tag, u32 before, u32 after)
{
	unsigned int seq = atomic_inc_return(&m5pm_seq);
	unsigned int i = seq % M5PM_RING;

	m5pm_ring[i].name = name;
	m5pm_ring[i].off = off;
	m5pm_ring[i].tag = tag;
	m5pm_ring[i].before = before;
	m5pm_ring[i].after = after;
	smp_wmb();
	m5pm_ring[i].seq = seq;
}

void m5pm_checkpoint(const char *name, u32 off, const char *stage)
{
	static bool stopped;
	static void __iomem *r;
	static const char *good_name = "-", *good_stage = "-";
	static u32 good_off;
	static unsigned int checked, good_seq;
	phys_addr_t drd = ports[0].drd;
	unsigned int seq, s;
	int us;

	if (READ_ONCE(stopped) || !of_machine_is_compatible("apple,j704"))
		return;
	mutex_lock(&m5dbg_lock);
	if (stopped)
		goto out;
	if (!r)
		r = ioremap(drd, 0xd000);	/* kept mapped for the whole boot */
	if (!r) {
		stopped = true;
		pr_err("m5pm %pa CSFTRST checkpoint mapping failed\n", &drd);
		goto out;
	}
	rmw(r + DWC3_DCTL, BIT(31), DWC3_DCTL_CSFTRST);
	for (us = 0; us < 200000; us += 10) {
		if (!(readl(r + DWC3_DCTL) & DWC3_DCTL_CSFTRST)) {
			if (!checked)
				pr_info("m5pm %pa CSFTRST first pmgr check clear ~%d us at %s %s@0x%x\n",
					&drd, us, stage, name, off);
			checked++;
			good_name = name;
			good_off = off;
			good_stage = stage;
			good_seq = atomic_read(&m5pm_seq);
			goto out;
		}
		udelay(10);
	}
	stopped = true;
	seq = atomic_read(&m5pm_seq);
	pr_info("m5pm %pa CSFTRST FIRST STUCK at %s %s@0x%x; last clear %s %s@0x%x; checked=%u writes=%u/%u\n",
		&drd, stage, name, off, good_stage, good_name, good_off, checked,
		seq - good_seq, seq);
	/* Writes since the last clear check (* = after it), plus a little context. */
	s = max3(1, (int)seq - (M5PM_RING - 1), (int)good_seq - 2);
	for (; s <= seq; s++) {
		unsigned int i = s % M5PM_RING;

		if (m5pm_ring[i].seq != s)
			continue;
		pr_info("m5pm %pa CSFTRST w%u%s %s@0x%x %c %08x->%08x\n", &drd, s,
			s > good_seq ? "*" : "", m5pm_ring[i].name, m5pm_ring[i].off,
			m5pm_ring[i].tag, m5pm_ring[i].before, m5pm_ring[i].after);
	}
out:
	mutex_unlock(&m5dbg_lock);
}
