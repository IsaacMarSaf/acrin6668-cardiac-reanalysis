"""Two checks requested after the positive controls (analysis plan, section 10.6; exploratory).

1. Interval from the end of radiotherapy to the restaging PET and sac change:
   - the main model adjusted for that interval and, separately, for the days between PET scans; and with the change
     per month;
   - robustness: robust (Huber) regression, median (0.5 quantile) regression, without |change|
     > 150 mL, Spearman, and the same analysis for the rim volume;
   - who had a late PET: interruption and stopping of radiotherapy, completed chemotherapy, Zubrod, stage, age, weight
     change, radiotherapy duration and tumor SUVpeak at restaging, by tertile of the interval.
2. EAT density change by technique: same scanner model, kVp and kernel at both time points versus the rest (fat
   attenuation depends on kVp).

Outputs: results/tables/acrin_interval_technique.json and screen.
Usage: python scripts/10_acrin_interval_technique.py
"""
import importlib.util
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("s", os.path.join(ROOT, "scripts", "05_acrin_survival.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
FORMS = importlib.util.spec_from_file_location("c", os.path.join(ROOT, "scripts", "03_acrin_clinical_cohort.py"))
c = importlib.util.module_from_spec(FORMS)
FORMS.loader.exec_module(c)


def main():
    d = s.load()
    out = {}
    t1 = c.form("T1")[["cn", "t1e43", "t1e44"]].rename(columns={"t1e43": "rt_int_tox", "t1e44": "rt_int_other"})
    d = d.merge(t1, on="cn", how="left")
    d["rt_interrupt_days"] = d[["rt_int_tox", "rt_int_other"]].sum(axis=1, min_count=1)
    d["rt_duration_days"] = d.rt_end_day - d.rt_start_day
    d["rt_incomplete"] = (d.rt_stop_reason.fillna(0) > 0).astype(float)
    d["chemo_complete"] = (d.chemo_completed_code == 3).astype(float).where(d.chemo_completed_code.notna())
    d["rt_to_pet"] = d.rt_end_to_landmark_days

    # 1a. main model adjusted for the interval and with the change per month
    sens = {}
    for k in s.PRIMARY:
        for name, extra in (("ajuste por dias fin RT -> PET post", ["rt_to_pet"]),
                            ("ajuste por dias entre PET", ["interval_days"])):
            r = s.marker_model(d, k, "d", extra=extra, impute=False)
            sens[f"{k} | {name}"] = {x: r[x] for x in ("hr", "lo", "hi", "p", "n", "events")}
        x = d.copy()
        x[f"{k}_rate"] = x[f"{k}_d"] / (x.interval_days / 30.44)
        x[f"{k}_ratez"] = (x[f"{k}_rate"] - x[f"{k}_rate"].mean()) / x[f"{k}_rate"].std(ddof=1)
        f = s.cox(x, s.BASE + [f"{k}_prez", f"{k}_ratez"])
        ci = f.confidence_intervals_.loc[f"{k}_ratez"]
        sens[f"{k} | cambio por mes"] = dict(hr=float(np.exp(f.params_[f"{k}_ratez"])), lo=float(np.exp(ci.iloc[0])),
                                             hi=float(np.exp(ci.iloc[1])), p=float(f.summary.loc[f"{k}_ratez", "p"]),
                                             n=len(x), events=int(x.death.sum()))
    out["supervivencia_con_intervalo"] = sens
    print("MODELO PRINCIPAL CON EL INTERVALO (HR por DE del cambio)")
    for kk, r in sens.items():
        print(f"  {kk:<52} {r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.2f} n={r['n']}/{r['events']}")

    # 1b. robustness of sac change ~ interval (per 30 days), adjusted for the baseline sac
    x = d[["sac_d", "sac_pre", "fluid_d", "fluid_pre", "fluid50_d", "fluid50_pre", "rt_to_pet"]].dropna().copy()
    x["m30"] = x.rt_to_pet / 30
    rob = {}
    for y, pre in (("sac_d", "sac_pre"), ("fluid_d", "fluid_pre"), ("fluid50_d", "fluid50_pre")):
        ols = smf.ols(f"{y} ~ m30 + {pre}", data=x).fit()
        hub = smf.rlm(f"{y} ~ m30 + {pre}", data=x, M=sm.robust.norms.HuberT()).fit()
        med = smf.quantreg(f"{y} ~ m30 + {pre}", data=x).fit(q=0.5)
        trim = x[x.sac_d.abs() <= 150]
        olt = smf.ols(f"{y} ~ m30 + {pre}", data=trim).fit()
        rho, p = stats.spearmanr(x.rt_to_pet, x[y])
        rob[y] = {
            "ols": [ols.params["m30"], *ols.conf_int().loc["m30"], ols.pvalues["m30"]],
            "huber": [hub.params["m30"], *hub.conf_int().loc["m30"], hub.pvalues["m30"]],
            "mediana": [med.params["m30"], *med.conf_int().loc["m30"], med.pvalues["m30"]],
            "sin_|cambio_saco|>150": [olt.params["m30"], *olt.conf_int().loc["m30"], olt.pvalues["m30"], len(trim)],
            "spearman_crudo": [rho, p, len(x)]}
    out["robustez_intervalo"] = rob
    q = x.rt_to_pet.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).tolist()
    out["intervalo_percentiles_10_25_50_75_90"] = q
    print(f"\nINTERVALO fin RT -> PET post (dias), p10-p25-p50-p75-p90: {q}")
    for y, r in rob.items():
        print(f"  {y}: " + " | ".join(f"{k} {v[0]:+.1f} ({v[1]:+.1f} a {v[2]:+.1f}) p={v[3]:.3f}" for k, v in r.items()
                                      if k != "spearman_crudo") + f" | Spearman {r['spearman_crudo'][0]:+.2f} p={r['spearman_crudo'][1]:.3f}")

    # 1c. who had a late PET (tertiles of the interval)
    d["tercil_intervalo"] = pd.qcut(d.rt_to_pet, 3, labels=["temprano", "medio", "tardio"])
    cols = {"edad": "age", "Zubrod >= 1": "zubrod1", "estadio IIIB": "stage3b", "dias de interrupcion de RT":
            "rt_interrupt_days", "RT no completada": "rt_incomplete", "quimio completa": "chemo_complete",
            "duracion de la RT (d)": "rt_duration_days", "cambio de peso (kg)": "weight_change_kg",
            "SUVpeak tumoral post": "tumor_suvpeak_post", "saco basal (mL)": "sac_pre"}
    tab, tests = {}, {}
    for lab, col in cols.items():
        g = d.groupby("tercil_intervalo", observed=True)[col]
        tab[lab] = g.median().round(2).to_dict() if d[col].nunique() > 2 else g.mean().round(2).to_dict()
        vals = [v.dropna() for _, v in g]
        tests[lab] = float(stats.kruskal(*vals).pvalue) if all(len(v) > 2 for v in vals) else None
    out["quien_tuvo_pet_tardio"] = {"por_tercil": tab, "p_kruskal": tests,
                                    "rango_dias_por_tercil": d.groupby("tercil_intervalo", observed=True).rt_to_pet
                                    .agg(["min", "max"]).to_dict("index")}
    print("\nPET TARDIO: medianas (o proporciones) por tercil del intervalo, p de Kruskal-Wallis")
    for lab in cols:
        print(f"  {lab:<28} " + "  ".join(f"{k}: {v}" for k, v in tab[lab].items()) + f"  p={tests[lab]:.3f}")

    # 2. EAT density by technique
    ct = pd.read_csv(os.path.join(ROOT, "results", "acrin_ct_eat_cac.csv"), dtype={"StudyDate": str})
    ct["cn"] = ct.PatientID.str[-3:].astype(int)
    ct = ct.sort_values(["cn", "StudyDate"])
    ct["tp"] = np.where(ct.groupby("cn").cumcount() == 0, "pre", "post")
    w = ct.pivot(index="cn", columns="tp", values=["ManufacturerModelName", "KVP", "ConvolutionKernel",
                                                     "SliceThickness"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w["same_model"] = w.ManufacturerModelName_pre == w.ManufacturerModelName_post
    w["same_kvp"] = w.KVP_pre.astype(float) == w.KVP_post.astype(float)
    w["same_kernel"] = w.ConvolutionKernel_pre == w.ConvolutionKernel_post
    w["same_thk"] = w.SliceThickness_pre.astype(float) == w.SliceThickness_post.astype(float)
    w["strict"] = w.same_model & w.same_kvp & w.same_kernel
    x = d.merge(w[["same_kvp", "same_kernel", "same_thk", "strict"]].reset_index(), on="cn", how="left")
    tech = {}
    for name, m in (("toda la cohorte", np.ones(len(x), bool)), ("mismo modelo, kVp y kernel", x.strict.values),
                    ("tecnica distinta", ~x.strict.values), ("mismo modelo, kVp, kernel y espesor",
                                                              (x.strict & x.same_thk).values)):
        v = x.loc[m, "eatd_d"].dropna()
        tech[name] = dict(n=len(v), mediana=float(v.median()), media=float(v.mean()),
                          p25=float(v.quantile(.25)), p75=float(v.quantile(.75)),
                          pct_aumento=float(100 * (v > 0).mean()), pct_aumento_5hu=float(100 * (v >= 5).mean()),
                          wilcoxon_p=float(stats.wilcoxon(v).pvalue) if len(v) > 10 else None)
    mw = stats.mannwhitneyu(x.loc[x.strict, "eatd_d"].dropna(), x.loc[~x.strict, "eatd_d"].dropna()).pvalue
    tech["p_estricta_vs_distinta"] = float(mw)
    kv = x.assign(dkvp=w.reindex(x.cn).eval("KVP_post.astype('float') - KVP_pre.astype('float')").values)
    tech["cambio_kvp_conteo"] = kv.dkvp.value_counts().to_dict()
    out["densidad_eat_por_tecnica"] = tech
    print("\nCAMBIO DE LA DENSIDAD DEL EAT SEGUN LA TECNICA")
    for name, r in tech.items():
        if isinstance(r, dict) and "n" in r:
            print(f"  {name:<38} n={r['n']:<4} mediana {r['mediana']:+.1f} HU (RIC {r['p25']:+.1f} a {r['p75']:+.1f}) | "
                  f"media {r['media']:+.1f} | aumento {r['pct_aumento']:.0f}% | >=5 HU {r['pct_aumento_5hu']:.0f}% | "
                  f"Wilcoxon p={r['wilcoxon_p']:.2g}")
    print(f"  estricta vs distinta: Mann-Whitney p={mw:.3f} | cambios de kVp (post - pre): {tech['cambio_kvp_conteo']}")
    with open(os.path.join(ROOT, "results", "tables", "acrin_interval_technique.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
