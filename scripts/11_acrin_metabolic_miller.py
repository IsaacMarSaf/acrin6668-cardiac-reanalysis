"""Re-analysis of Miller et al. 2022 (analysis plan, section 10.7; exploratory): metabolic state, HR scales, Miller's
specification and selection of the PET subsample.

1. Metabolic state: how much of the change in cardiac SUVmean is explained by the changes in glucose, fasting and
   uptake time (TA form and DICOM), and Miller's model (age + stage + SUVmean change) adjusted for glucose.
2. Scales: Miller et al. do not report the unit of their HRs (absolute SUVmean 0.954 [0.92-0.99]; relative SUVmean and
   absolute SUVmax 0.811 [0.68-0.96]; relative SUVmax 0.996 [0.99-1.00]). Our HR with the same adjustment is expressed
   on each plausible unit and compared with theirs: whether our 95% CI includes their estimate, and a heterogeneity
   test (z on the log HR, Miller's SE from their CI).
3. Selection: patients with a valid SUV at both time points versus the rest of the 118 (standardized differences and
   survival).
4. Miller's specification: time from the baseline PET (DICOM interval + survival from the landmark), age and stage,
   absolute and relative change in SUVmean; Kaplan-Meier by sign of the change from the baseline PET; same scanner.
5. Unit compatible with the published precision: with 77 patients the SE of the log HR per SD is at least ~1/sqrt(77);
   the SE implied by the published CI fixes the minimum SD that the exposure would need in Miller's unit. This is
   compared with the range of change allowed by their SUVmean ranges (0.9-5.6 baseline; 1.0-6.1 restaging).
6. Stage: Miller describes 35 patients with stage II and 42 with stage III of 77. The stage variables of the forms are
   tabulated (I1 at enrollment, A0 eligibility, O1 after PET, SS nodal stage of the core lab under AJCC 7) and Miller's
   model is refitted with six stage codings (no stage, IIIB, III versus IIB, core-lab N3 and N2-3, post-PET IIIB or IV).

Output: results/tables/acrin_metabolic_miller.json and screen.
Usage: python scripts/11_acrin_metabolic_miller.py
"""
import importlib.util
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("s", os.path.join(ROOT, "scripts", "05_acrin_survival.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

MILLER = {   # (HR, lo, hi) as printed in Miller et al., Radiother Oncol 2022;171:22-24
    "SUVmean absoluto": (0.954, 0.92, 0.99),
    "SUVmean relativo": (0.811, 0.68, 0.96),
    "SUVmax absoluto": (0.811, 0.68, 0.96),
    "SUVmax relativo": (0.996, 0.99, 1.00),
}


def cox_coef(d, covs, var):
    f = s.cox(d, covs)
    return float(f.params_[var]), float(f.standard_errors_[var]), float(f.summary.loc[var, "p"]), len(d), \
        int(d.death.sum())


def main():
    d = s.load()
    out = {}
    pet = d[d.pet_both].copy()
    for t in ("pre", "post"):
        pet[f"glu_{t}"] = pet[f"glucose_{t}"]
        pet[f"fast_{t}"] = pet[f"fasting_h_{t}"]
        pet[f"upt_{t}"] = pd.to_numeric(pet[f"uptake_min_{t}"], errors="coerce")
    for v in ("glu", "fast", "upt"):
        pet[f"{v}_d"] = pet[f"{v}_post"] - pet[f"{v}_pre"]

    # 1. metabolic state
    met = {}
    for name, col in (("glucemia (mg/dL)", "glu_d"), ("ayuno (h)", "fast_d"), ("captacion (min)", "upt_d"),
                      ("saco (mL)", "sac_d")):
        x = pet[["suvmean_full_d", col]].dropna()
        rho, p = stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1])
        met[f"rho cambio SUVmean ~ cambio {name}"] = dict(rho=rho, p=p, n=len(x))
    long = pd.concat([pet[[f"suvmean_full_{t}", f"glu_{t}"]].set_axis(["suv", "glu"], axis=1) for t in ("pre", "post")])
    long = long.dropna()
    rho, p = stats.spearmanr(long.suv, long.glu)
    met["rho SUVmean ~ glucemia (ambos tiempos)"] = dict(rho=rho, p=p, n=len(long))
    x = pet[["suvmean_full_d", "suvmean_full_pre", "glu_d", "fast_d", "upt_d", "sac_d", "same_model"]].dropna().copy()
    x["same"] = x.same_model.astype(float)
    m0 = smf.ols("suvmean_full_d ~ suvmean_full_pre", data=x).fit()
    m1 = smf.ols("suvmean_full_d ~ suvmean_full_pre + glu_d + fast_d + upt_d + same", data=x).fit()
    m2 = smf.ols("suvmean_full_d ~ suvmean_full_pre + glu_d + fast_d + upt_d + same + sac_d", data=x).fit()
    met["R2 solo basal"] = m0.rsquared
    met["R2 + glucemia, ayuno, captacion, equipo"] = m1.rsquared
    met["R2 + todo lo anterior + cambio del saco"] = m2.rsquared
    met["R2 parcial metabolico-tecnico"] = (m1.rsquared - m0.rsquared) / (1 - m0.rsquared)
    met["coef glucemia por 10 mg/dL"] = [10 * m1.params["glu_d"], *(10 * m1.conf_int().loc["glu_d"]),
                                         m1.pvalues["glu_d"]]
    met["coef captacion por 10 min"] = [10 * m1.params["upt_d"], *(10 * m1.conf_int().loc["upt_d"]),
                                        m1.pvalues["upt_d"]]
    met["coef ayuno por h"] = [m1.params["fast_d"], *m1.conf_int().loc["fast_d"], m1.pvalues["fast_d"]]
    met["n_modelo"] = int(m1.nobs)
    # Miller adjusted for glucose
    adj = {}
    for lab, extra in (("edad + estadio", []), ("+ cambio de glucemia", ["glu_d"]),
                       ("+ glucemia basal y post", ["glu_pre", "glu_post"]),
                       ("+ cambio de glucemia y de captacion", ["glu_d", "upt_d"])):
        sub = pet.dropna(subset=["age10", "stage3b", "suvmean_full_d"] + extra)
        b, se, p, n, ev = cox_coef(sub, ["age10", "stage3b", "suvmean_full_d"] + extra, "suvmean_full_d")
        adj[lab] = dict(hr_por_0_1=np.exp(0.1 * b), lo=np.exp(0.1 * (b - 1.96 * se)), hi=np.exp(0.1 * (b + 1.96 * se)),
                        p=p, n=n, events=ev)
    met["Miller ajustado (HR por 0,1 de aumento del SUVmean)"] = adj
    out["estado_metabolico"] = met

    # 2. scales versus Miller
    sc = []
    sub = pet.dropna(subset=["age10", "stage3b"])
    for meas, col in (("SUVmean", "suvmean_full"), ("SUVmax", "suvmax_full")):
        for kind, var in (("absoluto", f"{col}_d"), ("relativo", f"{col}_rel")):
            x = sub.dropna(subset=[var]).copy()
            if kind == "relativo":
                x[var] = x[var] / 100                        # fraction: 1.0 = 100% change
            b, se, p, n, ev = cox_coef(x, ["age10", "stage3b", var], var)
            sd = float(x[var].std(ddof=1))
            units = ({"por 1 unidad de SUV": 1.0, "por 0,1 de SUV": 0.1, "por DE": sd} if kind == "absoluto" else
                     {"por 100% (fraccion 1,0)": 1.0, "por 10%": 0.1, "por 1%": 0.01, "por DE": sd})
            hm, lm, um = MILLER[f"{meas} {kind}"]
            sem = (np.log(um) - np.log(lm)) / (2 * 1.96)
            for uname, k in units.items():
                lo, hi = np.exp(k * (b - 1.96 * se)), np.exp(k * (b + 1.96 * se))
                z = (k * b - np.log(hm)) / np.sqrt((k * se) ** 2 + sem ** 2)
                sc.append(dict(medida=f"{meas} {kind}", unidad=uname, hr=np.exp(k * b), lo=lo, hi=hi, p=p, n=n,
                               events=ev, miller=f"{hm} ({lm}-{um})", ic_incluye_miller=bool(lo <= hm <= hi),
                               p_heterogeneidad=2 * stats.norm.sf(abs(z))))
    out["escalas_vs_miller"] = sc

    # 3. selection of the PET subsample
    d["con_pet"] = d.pet_both.astype(int)
    sel = {}
    for lab, col in (("edad", "age"), ("hombre", "male"), ("Zubrod >= 1", "zubrod1"), ("estadio IIIB", "stage3b"),
                     ("SUVpeak tumoral post", "tumor_suvpeak_post"), ("saco basal", "sac_pre"),
                     ("cambio del saco", "sac_d"), ("mismo equipo", "same_model")):
        a, b_ = d.loc[d.con_pet == 1, col].astype(float), d.loc[d.con_pet == 0, col].astype(float)
        smd = (a.mean() - b_.mean()) / np.sqrt((a.var(ddof=1) + b_.var(ddof=1)) / 2)
        sel[lab] = dict(con_pet=round(a.mean(), 2), sin_pet=round(b_.mean(), 2), smd=round(float(smd), 2))
    lr = logrank_test(d.os_years[d.con_pet == 1], d.os_years[d.con_pet == 0], d.death[d.con_pet == 1],
                      d.death[d.con_pet == 0])
    sel["supervivencia log-rank p"] = float(lr.p_value)
    sel["n"] = dict(con_pet=int(d.con_pet.sum()), sin_pet=int((d.con_pet == 0).sum()))
    out["seleccion_pet"] = sel


    # 4. Miller's specification (time from the baseline PET)
    pet["os_base_years"] = (pet.os_days + pet.interval_days) / 365.25
    spec_m = {}
    for lab, var, scale in (("SUVmean absoluto, HR por 0,1", "suvmean_full_d", 0.1),
                            ("SUVmean relativo, HR por 10%", "suvmean_full_rel", 10.0)):
        for orig, tcol in (("landmark", "os_years"), ("PET basal", "os_base_years")):
            x = pet.dropna(subset=["age10", "stage3b", var])
            f = s.cox(x, ["age10", "stage3b", var], time=tcol)
            b, se = float(f.params_[var]), float(f.standard_errors_[var])
            spec_m[f"{lab}, origen {orig}"] = dict(hr=np.exp(scale * b), lo=np.exp(scale * (b - 1.96 * se)),
                                                  hi=np.exp(scale * (b + 1.96 * se)),
                                                  p=float(f.summary.loc[var, "p"]), n=len(x), events=int(x.death.sum()))
        x = pet[pet.same_model].dropna(subset=["age10", "stage3b", var])
        f = s.cox(x, ["age10", "stage3b", var])
        b, se = float(f.params_[var]), float(f.standard_errors_[var])
        spec_m[f"{lab}, mismo equipo"] = dict(hr=np.exp(scale * b), lo=np.exp(scale * (b - 1.96 * se)),
                                              hi=np.exp(scale * (b + 1.96 * se)), p=float(f.summary.loc[var, "p"]),
                                              n=len(x), events=int(x.death.sum()))
    up = pet.suvmean_full_d > 0
    for orig, tcol in (("landmark", "os_years"), ("PET basal", "os_base_years")):
        lr = logrank_test(pet.loc[up, tcol], pet.loc[~up, tcol], pet.death[up], pet.death[~up])
        med = {}
        for g, m_ in (("sube", up), ("baja", ~up)):
            km = KaplanMeierFitter().fit(pet.loc[m_, tcol], pet.death[m_])
            med[g] = float(km.median_survival_time_) * 365.25
        spec_m[f"KM por signo, origen {orig}"] = dict(n_sube=int(up.sum()), n_baja=int((~up).sum()),
                                                     mediana_dias_sube=med["sube"], mediana_dias_baja=med["baja"],
                                                     logrank_p=float(lr.p_value))
    for name, col in (("intervalo PET basal-post (dias)", "interval_days"),
                      ("fin de RT a PET post (dias)", "rt_end_to_landmark_days")):
        x = pet[["suvmean_full_d", col]].dropna()
        rho, p = stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1])
        spec_m[f"rho cambio SUVmean ~ {name}"] = dict(rho=rho, p=p, n=len(x))
    out["especificacion_miller"] = spec_m

    # 5. unit compatible with the published precision
    se_min_sd = 1 / np.sqrt(77)
    sds = {"SUVmean absoluto": float(pet.suvmean_full_d.std(ddof=1)),
           "SUVmean relativo": float((pet.suvmean_full_rel / 100).std(ddof=1)),
           "SUVmax absoluto": float(pet.suvmax_full_d.std(ddof=1)),
           "SUVmax relativo": float((pet.suvmax_full_rel / 100).std(ddof=1))}
    prec = {}
    for k, (hm, lm, um) in MILLER.items():
        se_m = (np.log(um) - np.log(lm)) / (2 * 1.96)
        prec[k] = dict(ee_log_hr_por_unidad=se_m, de_minima_en_unidad_de_miller=se_min_sd / se_m,
                       nuestra_de=sds[k], nota="relativo como fraccion (1,0 = 100%)" if "relativo" in k else "SUV")
    prec["rango_maximo_cambio_SUVmean_Miller"] = [1.0 - 5.6, 6.1 - 0.9]
    out["precision_miller"] = prec

    # 6. stage variables of the forms and Miller's model with each coding
    sp3 = importlib.util.spec_from_file_location("c03", os.path.join(ROOT, "scripts", "03_acrin_clinical_cohort.py"))
    c03 = importlib.util.module_from_spec(sp3)
    sp3.loader.exec_module(c03)
    i1 = c03.form("I1").drop_duplicates("cn")
    o1 = c03.form("O1").drop_duplicates("cn")[["cn", "o1e2"]]
    ssf = c03.form("SS").drop_duplicates("cn")[["cn", "sse6"]]
    a0 = c03.form("A0").drop_duplicates("cn")[["cn", "a0e36"]]
    an = d[["cn"]].merge(o1, on="cn", how="left").merge(ssf, on="cn", how="left").merge(a0, on="cn", how="left")

    def counts(x):
        return {str(int(k)): int(v) for k, v in x.dropna().value_counts().sort_index().items()}
    st_vars = dict(ingreso_I1_todos=counts(i1.i1e6), ingreso_I1_118=counts(d.stage_code),
                   elegibilidad_A0_IIB_o_III_118=counts(an.a0e36), post_pet_O1_118=counts(an.o1e2),
                   n_core_lab_SS_ajcc7_118=counts(an.sse6),
                   nota="I1 y O1: 1 IIB, 2 IIIA, 3 IIIB, 4 otro (I1) o IV (O1); A0: 2 = si; SS: N0 a N3")
    pc = pet.merge(o1, on="cn", how="left").merge(ssf, on="cn", how="left")
    pc["stage3_any"] = (pc.stage_code >= 2).astype(float)
    pc["n3"] = (pc.sse6 == 3).astype(float).where(pc.sse6.notna())
    pc["n23"] = (pc.sse6 >= 2).astype(float).where(pc.sse6.notna())
    pc["post_pet_3b4"] = (pc.o1e2 >= 3).astype(float).where(pc.o1e2.notna())
    cod = {}
    for lab, extra in (("sin estadio", []), ("estadio IIIB (principal)", ["stage3b"]),
                       ("estadio III frente a IIB", ["stage3_any"]), ("N3 del core lab", ["n3"]),
                       ("N2-3 del core lab", ["n23"]), ("estadio post-PET IIIB o IV", ["post_pet_3b4"])):
        for var, k, u in (("suvmean_full_d", 0.1, "por 0,1"), ("suvmean_full_rel", 10.0, "por 10%")):
            x = pc.dropna(subset=["age10", var] + extra)
            f = s.cox(x, ["age10"] + extra + [var])
            b, se = float(f.params_[var]), float(f.standard_errors_[var])
            cod[f"{lab} | {u}"] = dict(hr=np.exp(k * b), lo=np.exp(k * (b - 1.96 * se)), hi=np.exp(k * (b + 1.96 * se)),
                                      p=float(f.summary.loc[var, "p"]), n=len(x), events=int(x.death.sum()))
    out["estadio"] = dict(variables=st_vars, modelo_miller_por_codificacion=cod)

    with open(os.path.join(ROOT, "results", "tables", "acrin_metabolic_miller.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("1. ESTADO METABOLICO")
    for k, v in met.items():
        print(f"  {k}: {v}")
    print("\n2. ESCALAS FRENTE A MILLER (edad + estadio)")
    for r in sc:
        print(f"  {r['medida']:<17} {r['unidad']:<24} HR {r['hr']:.3f} ({r['lo']:.3f}-{r['hi']:.3f}) | Miller "
              f"{r['miller']} | nuestro IC incluye a Miller: {r['ic_incluye_miller']} | p heterog "
              f"{r['p_heterogeneidad']:.3f}")
    print("\n3. SELECCION (con SUV valido vs sin)")
    for k, v in sel.items():
        print(f"  {k}: {v}")
    print("\n4. ESPECIFICACION DE MILLER")
    for k, v in spec_m.items():
        print(f"  {k}: {v}")
    print("\n5. UNIDAD COMPATIBLE CON LA PRECISION PUBLICADA")
    for k, v in prec.items():
        print(f"  {k}: {v}")
    print("\n6. ESTADIO")
    for k, v in out["estadio"]["variables"].items():
        print(f"  {k}: {v}")
    for k, v in out["estadio"]["modelo_miller_por_codificacion"].items():
        print(f"  {k:<40} HR {v['hr']:.3f} ({v['lo']:.3f}-{v['hi']:.3f}) p={v['p']:.2f} n={v['n']}/{v['events']}")


if __name__ == "__main__":
    main()
