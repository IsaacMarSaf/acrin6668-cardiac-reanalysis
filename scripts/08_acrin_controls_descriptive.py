"""Positive controls and description of the changes (analysis plan, sections 10.2 and 10.3). No outcome data.

Positive controls: linear regression of the change of each marker on a factor, adjusted for the baseline value of the
marker; difference in means (or slope) with 95% CI. Directional hypothesis: more cardiac dose -> more positive change
of the sac, the rim volume and the EAT density. Indirect dose indicators: left-sided tumor, lower-lobe or hilar tumor,
and recorded maximum heart dose (T1). Context without a hypothesis: tumor dose, cisplatin in the regimen and days from
the end of radiotherapy to the restaging PET.

Description: proportions beyond the thresholds fixed in 10.3 (whole cohort and same scanner) and Spearman correlations
between structural and metabolic changes.

Outputs: results/tables/acrin_positive_controls.csv, acrin_change_proportions.csv, acrin_struct_vs_suv.csv.
Usage: python scripts/08_acrin_controls_descriptive.py
"""
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
spec = importlib.util.spec_from_file_location("s", os.path.join(ROOT, "scripts", "05_acrin_survival.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

OUTCOMES = {"sac": "saco (mL)", "fluid": "liquido 0-25 HU en la cascara (mL)", "eatd": "densidad del EAT (HU)",
            "eat": "volumen del EAT (mL)"}
FACTORS = {                                   # name: (column, directional hypothesis)
    "tumor izquierdo": ("left_tumor", True),
    "lobulo inferior o hilio": ("lower_or_hilar", True),
    "dosis maxima al corazon (por 10 Gy)": ("heart_max_10gy", True),
    "dosis total al GTV (por 5 Gy)": ("gtv_5gy", False),
    "cisplatino en el esquema": ("cisplatin", False),
    "dias del fin de RT al PET post (por 30 d)": ("rt_to_pet_30d", False),
}


def main():
    d = s.load()
    d["heart_max_10gy"] = d.heart_max_dose_gy / 10
    d["gtv_5gy"] = d.gtv_dose_gy / 5
    d["cisplatin"] = d.chemo_agents.fillna("").str.split(";").apply(lambda a: float("1" in a)).where(
        d.chemo_agents.notna())
    d["rt_to_pet_30d"] = d.rt_end_to_landmark_days / 30

    rows = []
    for k, lab in OUTCOMES.items():
        for fname, (col, directional) in FACTORS.items():
            x = d[[f"{k}_d", f"{k}_pre", col]].dropna().rename(columns={f"{k}_d": "y", f"{k}_pre": "pre", col: "x"})
            if len(x) < 15 or x.x.nunique() < 2:
                continue
            m = smf.ols("y ~ x + pre", data=x).fit()
            lo, hi = m.conf_int().loc["x"]
            rows.append(dict(cambio=lab, factor=fname, hipotesis="mas dosis -> cambio mas positivo" if directional
                             else "contexto", n=len(x), efecto=m.params["x"], lo=lo, hi=hi, p=m.pvalues["x"],
                             n_factor_1=int((x.x == 1).sum()) if set(x.x.unique()) <= {0, 1} else np.nan))
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(TAB, "acrin_positive_controls.csv"), index=False)
    pd.set_option("display.width", 230)
    print("CONTROLES POSITIVOS (diferencia o pendiente del cambio, ajustada por el valor basal)")
    for _, r in pc.iterrows():
        print(f"  {r.cambio:<36} ~ {r.factor:<42} {r.efecto:+7.2f} ({r.lo:+.2f} a {r.hi:+.2f}) p={r.p:.3f} n={r.n}"
              + (f" (con factor: {int(r.n_factor_1)})" if not np.isnan(r.n_factor_1) else ""))

    props = []
    for name, sub in (("toda la cohorte", d), ("mismo equipo", d[d.same_model])):
        def pr(mask, base):
            v = mask[base.notna()]
            return f"{int(v.sum())}/{len(v)} ({100 * v.mean():.0f}%)"
        props.append(dict(grupo=name,
                          saco_aumento_10pct=pr(sub.sac_rel >= 10, sub.sac_rel),
                          saco_descenso_10pct=pr(sub.sac_rel <= -10, sub.sac_rel),
                          eat_descenso_10pct=pr(sub.eat_rel <= -10, sub.eat_rel),
                          eat_aumento_10pct=pr(sub.eat_rel >= 10, sub.eat_rel),
                          densidad_eat_aumento_5hu=pr(sub.eatd_d >= 5, sub.eatd_d),
                          densidad_eat_cualquier_aumento=pr(sub.eatd_d > 0, sub.eatd_d),
                          liquido_aumento_25ml=pr(sub.fluid_d >= 25, sub.fluid_d)))
    pt = pd.DataFrame(props)
    pt.to_csv(os.path.join(TAB, "acrin_change_proportions.csv"), index=False)
    print("\nPROPORCIONES (umbrales de 10.3)")
    print(pt.set_index("grupo").T.to_string())

    corr = []
    for a in ("sac", "eat", "eatd", "fluid"):
        for b in ("suvmean_full", "suvmax_full"):
            x = d[[f"{a}_d", f"{b}_d"]].dropna()
            if len(x) < 15:
                continue
            rho, p = stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1])
            corr.append(dict(estructural=OUTCOMES[a], metabolico=s.MARKERS[b][1], rho=rho, p=p, n=len(x)))
    ct = pd.DataFrame(corr)
    ct.to_csv(os.path.join(TAB, "acrin_struct_vs_suv.csv"), index=False)
    print("\nCAMBIO ESTRUCTURAL vs CAMBIO METABOLICO (Spearman)")
    print(ct.round(3).to_string(index=False) if len(ct) else "  sin datos de SUV todavia")


if __name__ == "__main__":
    main()
