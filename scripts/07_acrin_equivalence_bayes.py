"""Equivalence and Bayesian bounds (analysis plan, section 10.1; exploratory): TOST equivalence and Bayesian analysis
with a skeptical prior for the HR per SD of change of each primary marker, using the coefficients of the main analysis
(results/acrin_survival_results.json, multiple imputation pooled with Rubin's rules).

- TOST: margins 1/1.35 and 1.35 (HR detectable with 80% power with the observed events); 90% CI and one-sided p
  against harm.
- Bayes, normal approximation of the log HR: prior N(0, 0.182^2) with P(HR > 1.35) = 5%, and a nearly flat prior
  N(0, 1).

Output: results/tables/acrin_null_formal.csv and screen.
Usage: python scripts/07_acrin_equivalence_bayes.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARGIN = 1.35
PRIOR_SD = {"esceptico": np.log(MARGIN) / stats.norm.ppf(0.95), "casi_plano": 1.0}


def main():
    r = json.load(open(os.path.join(ROOT, "results", "acrin_survival_results.json"), encoding="utf-8"))
    rows = []
    for x in r["cox"]:
        if x["kind"] != "d" or x.get("skipped") or x["marker"] not in ("sac", "eat", "eatd", "cac"):
            continue
        b, se, m = x["coef"], x["se"], np.log(MARGIN)
        p_harm = stats.norm.cdf((b - m) / se)            # H0: HR >= 1.35
        p_benefit = stats.norm.sf((b + m) / se)          # H0: HR <= 1/1.35
        row = dict(marker=x["marker"], label=x["label"], hr=np.exp(b), ci90_lo=np.exp(b - 1.645 * se),
                   ci90_hi=np.exp(b + 1.645 * se), p_tost=max(p_harm, p_benefit), p_vs_harm_1_35=p_harm,
                   sd=x["sd"], unit=x["unit"], hr_margin_per_unit=MARGIN ** (x["unit"] / x["sd"]))
        for name, s0 in PRIOR_SD.items():
            prec = 1 / s0 ** 2 + 1 / se ** 2
            mu, sd = (b / se ** 2) / prec, np.sqrt(1 / prec)
            row[f"post_hr_{name}"] = np.exp(mu)
            row[f"p_hr_gt_1_2_{name}"] = stats.norm.sf((np.log(1.2) - mu) / sd)
            row[f"p_hr_gt_1_35_{name}"] = stats.norm.sf((np.log(MARGIN) - mu) / sd)
            row[f"p_hr_lt_0_83_{name}"] = stats.norm.cdf((np.log(1 / 1.2) - mu) / sd)
        rows.append(row)
    t = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "tables", "acrin_null_formal.csv")
    t.to_csv(out, index=False)
    print(f"prior esceptico: DE {PRIOR_SD['esceptico']:.3f} en log HR (P(HR > {MARGIN}) = 5%)")
    for _, x in t.iterrows():
        print(f"{x.label:<26} HR {x.hr:.2f} IC90 {x.ci90_lo:.2f}-{x.ci90_hi:.2f} | TOST p={x.p_tost:.3f} | "
              f"contra HR>=1,35 p={x.p_vs_harm_1_35:.4f} | margen por unidad clinica ({x.unit:g}) "
              f"{x.hr_margin_per_unit:.2f} | P(HR>1,2): esceptico {x.p_hr_gt_1_2_esceptico:.3f}, "
              f"plano {x.p_hr_gt_1_2_casi_plano:.3f} | P(HR>1,35) esceptico {x.p_hr_gt_1_35_esceptico:.4f} | "
              f"P(HR<0,83) esceptico {x.p_hr_lt_0_83_esceptico:.3f}")
    print("->", out)


if __name__ == "__main__":
    main()
