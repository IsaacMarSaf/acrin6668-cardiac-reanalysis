"""Subgroups and interactions (analysis plan, section 10.5). EXPLORATORY: added after the primary analysis, with
the list of modifiers and cut points fixed before running. All 24 tests are reported, with Benjamini-Hochberg FDR.

Model per primary marker and modifier: the main model (base + baseline + change, complete cases) plus the modifier and
change x modifier. HR per SD of the change at each level (coefficient and its sum with the interaction) and Wald p of
the interaction.

Outputs: results/tables/acrin_subgroups_EXPLORATORIO.csv and figures/acrin_fig_subgroups_EXPLORATORIO.png.
Usage: python scripts/09_acrin_subgroups.py
"""
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("s", os.path.join(ROOT, "scripts", "05_acrin_survival.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)            # same data, same base model and same Cox model as the main analysis

MODIFIERS = {
    "edad >= 65": lambda d: (d.age >= 65).astype(float),
    "hombre": lambda d: d.male,
    "Zubrod >= 1": lambda d: d.zubrod1,
    "estadio IIIB": lambda d: d.stage3b,
    "tumor izquierdo": lambda d: d.left_tumor,
    "lobulo inferior o hilio": lambda d: d.lower_or_hilar,
    "SUVpeak tumoral post > mediana": lambda d: (d.tumor_suvpeak_post > d.tumor_suvpeak_post.median()).astype(float)
    .where(d.tumor_suvpeak_post.notna()),
    "mismo equipo": lambda d: d.same_model.astype(float),
}
IN_BASE = {"hombre": "male", "Zubrod >= 1": "zubrod1", "estadio IIIB": "stage3b"}


def main():
    d = s.load()
    rows = []
    for k in s.PRIMARY:
        for name, fn in MODIFIERS.items():
            x = d.copy()
            x["mod"] = fn(x)
            x["inter"] = x[f"{k}_dz"] * x["mod"]
            base = [c for c in s.BASE if c != IN_BASE.get(name)]
            covs = base + [f"{k}_prez", f"{k}_dz", "mod", "inter"]
            x = x.dropna(subset=covs + ["os_years", "death"])
            f = s.cox(x, covs)
            b, V = f.params_, f.variance_matrix_
            b0 = b[f"{k}_dz"]
            se0 = np.sqrt(V.loc[f"{k}_dz", f"{k}_dz"])
            b1 = b0 + b["inter"]
            se1 = np.sqrt(V.loc[f"{k}_dz", f"{k}_dz"] + V.loc["inter", "inter"] + 2 * V.loc[f"{k}_dz", "inter"])
            g0, g1 = x[x["mod"] == 0], x[x["mod"] == 1]
            rows.append(dict(marker=k, label=s.MARKERS[k][1], modifier=name,
                             n_no=len(g0), ev_no=int(g0.death.sum()), n_si=len(g1), ev_si=int(g1.death.sum()),
                             hr_no=np.exp(b0), lo_no=np.exp(b0 - 1.96 * se0), hi_no=np.exp(b0 + 1.96 * se0),
                             hr_si=np.exp(b1), lo_si=np.exp(b1 - 1.96 * se1), hi_si=np.exp(b1 + 1.96 * se1),
                             hr_ratio=np.exp(b["inter"]), p_interaction=float(f.summary.loc["inter", "p"])))
    t = pd.DataFrame(rows)
    t["p_fdr"] = multipletests(t.p_interaction, method="fdr_bh")[1]
    t["rotulo"] = "EXPLORATORIO"
    t.to_csv(os.path.join(ROOT, "results", "tables", "acrin_subgroups_EXPLORATORIO.csv"), index=False)

    fig, ax = s.plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for a, k in zip(ax, s.PRIMARY):
        tk = t[t.marker == k].reset_index(drop=True)
        for i, r in tk.iterrows():
            for off, lvl, col in ((-0.15, "no", "#7f7f7f"), (0.15, "si", "#1f4e79")):
                a.plot([r[f"lo_{lvl}"], r[f"hi_{lvl}"]], [i + off] * 2, color=col, lw=1)
                a.plot(r[f"hr_{lvl}"], i + off, "s", color=col, ms=4)
            a.text(3.6, i, f"p int {r.p_interaction:.2f}", fontsize=7, va="center")
        a.axvline(1, color="grey", lw=.8)
        a.set_xscale("log")
        a.set_xlim(0.25, 6)
        a.set_xticks([0.25, 0.5, 1, 2, 4])
        a.set_xticklabels(["0,25", "0,5", "1", "2", "4"])
        a.minorticks_off()
        a.set_title(f"Cambio en {s.MARKERS[k][1].lower()}", fontsize=9)
        a.set_xlabel("HR por DE (gris: no; azul: si)")
    ax[0].set_yticks(range(len(MODIFIERS)))
    ax[0].set_yticklabels(list(MODIFIERS), fontsize=8)
    ax[0].invert_yaxis()
    fig.suptitle("Subgrupos: EXPLORATORIO, 24 pruebas (se espera ~1 falso positivo con p < 0,05)", fontsize=10)
    s.plt.tight_layout()
    s.plt.savefig(os.path.join(ROOT, "figures", "acrin_fig_subgroups_EXPLORATORIO.png"), dpi=150)
    s.plt.close(fig)

    pd.set_option("display.width", 220)
    for _, r in t.iterrows():
        print(f"{r.label:<24} | {r.modifier:<31} no: {r.hr_no:.2f} ({r.lo_no:.2f}-{r.hi_no:.2f}) n={r.n_no}/{r.ev_no} | "
              f"si: {r.hr_si:.2f} ({r.lo_si:.2f}-{r.hi_si:.2f}) n={r.n_si}/{r.ev_si} | p int {r.p_interaction:.3f} "
              f"FDR {r.p_fdr:.2f}")
    print(f"\ninteracciones con p < 0,05 sin corregir: {(t.p_interaction < 0.05).sum()} de {len(t)} "
          f"(esperadas por azar: {0.05 * len(t):.1f}); con FDR < 0,05: {(t.p_fdr < 0.05).sum()}")


if __name__ == "__main__":
    main()
