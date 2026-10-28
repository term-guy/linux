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

#include <linux/delay.h>
#include <linux/init.h>
#include <linux/io.h>
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

static int __init m5_usb2_prep(void)
{
	int i;

	if (!of_machine_is_compatible("apple,j704"))
		return 0;

	for (i = 0; i < ARRAY_SIZE(ports); i++) {
		void __iomem *drd = ioremap(ports[i].drd, 0xd000);
		void __iomem *phy = ioremap(ports[i].usb2phy, 0x4000);
		void __iomem *ph = ioremap(ports[i].pipehandler, 0x4000);

		if (!drd || !phy || !ph) {
			pr_err("m5-usb2-prep: port%d: ioremap failed\n", ports[i].idx);
			goto unmap;
		}

		dump(ports[i].idx, "before", drd, phy, ph);

		csftrst_test(ports[i].idx, "inherited", drd);

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

		csftrst_test(ports[i].idx, "after reset cycle", drd);
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
