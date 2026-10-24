"""Quality-control figures of the sac, EAT and calcium for scripts/01_acrin_eat_cac_run.py.

Needs deepcac_torch (not distributed, see README) for the CAC threshold.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import deepcac_torch as dc  # noqa: E402

EAT_CMAP = ListedColormap(["yellow"])
SAC_MARGIN_ITER = 2      # sac margin for the restricted CAC: 2 voxels (1.4 mm in plane, 5 mm in z)


def render_cac(pid, a, s3, sac_dil, ag, ag_sac, path):
    """Scorable CAC slices: red inside the sac (counted in cac_sac), cyan outside (original DeepCAC only)."""
    from scipy import ndimage
    b = s3["bbox"]
    full = np.zeros(a.shape, bool)
    full[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = s3["prob"] >= dc.CAC_THR
    keep = np.zeros_like(full)
    for z in np.where(full.any((1, 2)))[0]:
        lab, n = ndimage.label(full[z], structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() > dc.NR_CON_PX and a[z][m].max() >= 130:
                keep[z] |= m
    zs = np.where(keep.any((1, 2)))[0]
    if not len(zs):
        return
    show = zs if len(zs) <= 8 else zs[np.linspace(0, len(zs) - 1, 8).astype(int)]
    ys, xs = np.where(sac_dil.any(0))
    y0, y1, x0, x1 = max(0, ys.min() - 25), ys.max() + 25, max(0, xs.min() - 25), xs.max() + 25
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for i, axx in enumerate(ax.flat):
        axx.axis("off")
        if i >= len(show):
            continue
        z = show[i]
        axx.imshow(np.clip(a[z, y0:y1, x0:x1], -200, 600), cmap="gray")
        axx.contour(sac_dil[z, y0:y1, x0:x1], levels=[0.5], colors="lime", linewidths=0.6)
        axx.contour(keep[z, y0:y1, x0:x1] & sac_dil[z, y0:y1, x0:x1], levels=[0.5], colors="red", linewidths=1.2)
        axx.contour(keep[z, y0:y1, x0:x1] & ~sac_dil[z, y0:y1, x0:x1], levels=[0.5], colors="cyan", linewidths=1.2)
        axx.set_title(f"z={z}", fontsize=9)
    fig.suptitle(f"{pid}  DeepCAC original {ag:.1f}  |  dentro del saco {ag_sac:.1f}  (rojo dentro, cian fuera)")
    plt.tight_layout()
    plt.savefig(path, dpi=55)
    plt.close(fig)


def render(pid, m, res, path):
    a, sac, eat = m["arr"], m["sac"], m["eat"]
    zs = np.where(sac.any((1, 2)))[0]
    if not len(zs):
        return
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    for i, q in enumerate((25, 50, 75)):
        z = int(np.percentile(zs, q))
        ax[i].imshow(np.clip(a[z], -250, 250), cmap="gray")
        ax[i].contour(sac[z], levels=[0.5], colors="red", linewidths=0.8)
        ax[i].imshow(np.ma.masked_equal(eat[z].astype(float), 0), cmap=EAT_CMAP, alpha=0.85, vmin=0, vmax=1)
        ax[i].set_title(f"{pid} axial z={z}")
        ax[i].axis("off")
    yc = int(np.median(np.where(sac)[1]))
    kw = dict(origin="lower", aspect=2.5 / 0.68)
    ax[3].imshow(np.clip(a[:, yc, :], -250, 250), cmap="gray", **kw)
    ax[3].contour(sac[:, yc, :], levels=[0.5], colors="red", linewidths=0.8, origin="lower")
    ax[3].imshow(np.ma.masked_equal(eat[:, yc, :].astype(float), 0), cmap=EAT_CMAP, alpha=0.85, vmin=0, vmax=1, **kw)
    ax[3].set_title(f"coronal | saco {res['sac_ml']:.0f} mL, EAT {res['eat_ml']:.0f} mL, {res['eat_density_hu']:.1f} HU | sangre+miocardio {res['sac_core_median_hu']:.0f} HU")
    ax[3].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=55)
    plt.close(fig)
