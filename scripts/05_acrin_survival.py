"""Survival analysis of ACRIN 6668 following the analysis plan: structural cardiac changes (CT of the PET/CT) and
overall survival from the landmark (restaging PET).

Inputs: results/acrin_ct_eat_cac.csv (01), results/acrin_effusion_pet.csv (04), results/acrin_clinical_cohort.csv (03).
Outputs: results/acrin_analysis_dataset.csv, results/acrin_survival_results.json, results/tables/acrin_*.csv and
figures/acrin_*.png. --permute shuffles the outcome (only to test the code without looking at the results).

Usage: python scripts/05_acrin_survival.py [--boot 2000] [--permute]
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter  # noqa: E402
from lifelines.statistics import proportional_hazard_test  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.impute import IterativeImputer  # noqa: E402
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc  # noqa: E402
from sksurv.util import Surv  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
TAB = os.path.join(R, "tables")
FIG = os.path.join(ROOT, "figures")
SEED, M_IMP, TAU, T_AUC = 2026, 20, 3.0, 2.0
BASE = ["age10", "male", "zubrod1", "stage3b", "log_suvpeak_post"]
EXTENDED = BASE + ["weight_change_kg", "diabetes", "left_tumor", "lower_or_hilar", "gtv_dose_gy", "interval_days"]
# key: (column per series, label, clinical unit for the HR)
MARKERS = {
    "sac": ("sac_ml", "Volumen del saco (mL)", 50),
    "eat": ("eat_ml", "Volumen de EAT (mL)", 10),
    "eatd": ("eat_density_hu", "Densidad del EAT (HU)", 5),
    "cac": ("cac_log", "log(1 + CAC en el saco)", 1),
    "fluid": ("fluid25_shell_ml", "Liquido 0-25 HU en la cascara (mL)", 10),
    "fluid50": ("fluid_shell_ml", "Liquido 0-50 HU en la cascara (mL)", 10),
    "fluid50sac": ("fluid_sac_ml", "Liquido 0-50 HU en el saco (mL)", 50),
    "fluid4": ("fluid25_shell4_ml", "Liquido 0-25 HU en la cascara de 4 mm (mL)", 10),
    "suvmean_full": ("heart_suvmean_full", "SUVmean cardiaco (saco completo)", 0.5),
    "suvmax_full": ("heart_suvmax_full", "SUVmax cardiaco (saco completo)", 1),
    "suvmean": ("heart_suvmean", "SUVmean cardiaco (saco erosionado)", 0.5),
    "suvmax": ("heart_suvmax", "SUVmax cardiaco (saco erosionado)", 1),
}
PRIMARY = ["sac", "eat", "eatd"]
SECONDARY = ["cac", "fluid", "suvmean_full", "suvmax_full"]
SENS_MARKERS = ["fluid50", "fluid50sac", "fluid4", "suvmean", "suvmax"]
DS_FLAGS = ("step1_empty", "fallback_full")          # DeepHeartSeg flags that point to a possible sac failure


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------

def load(permute=False, keep_visual_fail=False):
    """One row per patient. Series that failed the visual review (results/acrin_visual_qc.csv) count as
    segmentation errors unless keep_visual_fail=True (sensitivity analysis)."""
    ct = pd.read_csv(os.path.join(R, "acrin_ct_eat_cac.csv"), dtype={"StudyDate": str})
    vq = pd.read_csv(os.path.join(R, "acrin_visual_qc.csv"), dtype={"StudyDate": str})
    fail = set(zip(vq.PatientID[vq.falla == 1], vq.StudyDate[vq.falla == 1]))
    ct["visual_fail"] = [(p, s) in fail for p, s in zip(ct.PatientID, ct.StudyDate)]
    ep = pd.read_csv(os.path.join(R, "acrin_effusion_pet.csv"), dtype={"StudyDate": str})
    ep = ep.rename(columns={"error": "pet_error", "weight_kg": "pet_weight_kg"})
    # SUV invalid when the uptake time falls outside 30-180 min (misrecorded injection time: 121 baseline, 816 min)
    bad = ~ep.uptake_min.between(30, 180) & ep.pet_units.eq("BQML")
    ep.loc[bad, [c for c in ep.columns if c.startswith("heart_")]] = np.nan
    m = ct.merge(ep.drop(columns=["PatientID", "StudyDate", "secs", "run_at"]), on="SeriesInstanceUID", how="left")
    m["cn"] = m.PatientID.str[-3:].astype(int)
    m = m.sort_values(["cn", "StudyDate"])
    m["tp"] = np.where(m.groupby("cn").cumcount() == 0, "pre", "post")
    m["cac_log"] = np.log1p(m.cac_sac)
    m["ct_ok"] = (m.error.fillna("") == "") & (keep_visual_fail | ~m.visual_fail)
    m["pet_ok"] = (m.pet_error.fillna("") == "") & m.heart_suvmean_full.notna()
    m["qc_flag"] = (m.eat_flags.fillna("").str.contains("|".join(DS_FLAGS)) | (m.contrast_by_image == "yes") |
                    (m.sac_touches_z_edge.astype(str) == "True") | ~m.sac_ml.between(400, 1300))
    m["large_kept"] = m.eat_flags.fillna("").str.contains("large_kept")
    cols = [v[0] for v in MARKERS.values()] + ["ct_ok", "pet_ok", "qc_flag", "large_kept", "ManufacturerModelName",
                                               "contrast_by_image", "sac_core_median_hu", "pet_weight_kg", "uptake_min"]
    w = m.pivot(index="cn", columns="tp", values=cols)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    for k, (col, _, _) in MARKERS.items():
        w[f"{k}_pre"], w[f"{k}_post"] = w[f"{col}_pre"].astype(float), w[f"{col}_post"].astype(float)
        w[f"{k}_d"] = w[f"{k}_post"] - w[f"{k}_pre"]
        w[f"{k}_rel"] = 100 * w[f"{k}_d"] / w[f"{k}_pre"].where(w[f"{k}_pre"].abs() > 1e-6)
    c = pd.read_csv(os.path.join(R, "acrin_clinical_cohort.csv"))
    d = c.merge(w, on="cn", how="inner").copy()
    d["age10"] = d.age / 10
    d["zubrod1"] = (d.zubrod >= 1).astype(float).where(d.zubrod.notna())
    d["stage3b"] = (d.stage_code == 3).astype(float).where(d.stage_code.notna())
    d["log_suvpeak_post"] = np.log(d.tumor_suvpeak_post.clip(lower=0.5))
    d["left_tumor"] = (d.tumor_side == 1).astype(float).where(d.tumor_side.isin([1, 2]))
    d["lower_or_hilar"] = d[["loc_rll", "loc_lll", "loc_rhilum_ml"]].max(axis=1)
    d["interval_days"] = d.dicom_interval
    # weight of the analyzed PET (DICOM); if missing, from the TA form
    wpre = d.pet_weight_kg_pre.astype(float).fillna(d.weight_kg_pre)
    wpost = d.pet_weight_kg_post.astype(float).fillna(d.weight_kg_post)
    d["weight_change_kg"] = wpost - wpre
    d["same_model"] = d.ManufacturerModelName_pre == d.ManufacturerModelName_post
    d["iv_contrast_any"] = ((d.iv_contrast_pre == 2) | (d.iv_contrast_post == 2) |
                            (d.contrast_by_image_pre == "yes") | (d.contrast_by_image_post == "yes"))
    d["ct_both"] = d.ct_ok_pre.astype(bool) & d.ct_ok_post.astype(bool)
    d["pet_both"] = d.pet_ok_pre.astype(bool) & d.pet_ok_post.astype(bool)
    d["qc_any"] = d.qc_flag_pre.astype(bool) | d.qc_flag_post.astype(bool)
    d["large_any"] = d.large_kept_pre.astype(bool) | d.large_kept_post.astype(bool)
    d["nsclc_death"] = (d.death_cause == "nsclc").astype(int)
    d["other_death"] = (d.death_cause == "other").astype(int)
    d["other_unknown_death"] = d.death_cause.isin(["other", "unknown"]).astype(int)
    d = d[(d.in_analysis == 1) & d.ct_both].copy()
    if permute:
        p = np.random.default_rng(SEED).permutation(len(d))
        for col in ("os_years", "death", "nsclc_death", "other_death", "other_unknown_death", "death_cause"):
            d[col] = d[col].values[p]
    for k in MARKERS:
        for s in ("pre", "d", "post", "rel"):
            x = d[f"{k}_{s}"]
            d[f"{k}_{s}z"] = (x - x.mean()) / x.std(ddof=1)
    return d


# ---------------------------------------------------------------------------------------------------------------
# Cox, multiple imputation and metrics
# ---------------------------------------------------------------------------------------------------------------

def cox(d, covs, time="os_years", event="death"):
    f = CoxPHFitter()
    f.fit(d[covs + [time, event]].dropna(), duration_col=time, event_col=event)
    return f


def imputations(d, cols, m=M_IMP):
    """m imputed sets (chained equations with posterior sampling) of the columns with missing values; the imputation
    model includes the event and the Nelson-Aalen estimate. 0/1 variables are rounded. Without missing values it
    returns [d]."""
    need = [c for c in cols if d[c].isna().any()]
    if not need:
        return [d]
    X = d[cols].copy()
    X["_H"] = NelsonAalenFitter().fit(d.os_years, d.death).cumulative_hazard_at_times(d.os_years).values
    X["_E"] = d.death.values
    binary = [c for c in need if set(d[c].dropna().unique()) <= {0, 1}]
    out = []
    for i in range(m):
        Xi = pd.DataFrame(IterativeImputer(sample_posterior=True, random_state=SEED + i, max_iter=25)
                          .fit_transform(X), columns=X.columns, index=X.index)
        di = d.copy()
        for c in need:
            di[c] = Xi[c].clip(0, 1).round() if c in binary else Xi[c]
        out.append(di)
    return out


def pooled(sets, covs, var, time="os_years", event="death"):
    """Rubin's rules: coefficient, SE, HR, 95% CI and p of 'var' in the Cox model of covs over the imputed sets."""
    b, v = [], []
    for di in sets:
        f = cox(di, covs, time, event)
        b.append(f.params_[var])
        v.append(f.standard_errors_[var] ** 2)
    b, v = np.array(b), np.array(v)
    q = b.mean()
    t = v.mean() + (1 + 1 / len(b)) * (b.var(ddof=1) if len(b) > 1 else 0)
    se = np.sqrt(t)
    return dict(coef=q, se=se, hr=np.exp(q), lo=np.exp(q - 1.96 * se), hi=np.exp(q + 1.96 * se),
                p=2 * stats.norm.sf(abs(q / se)), n=int(len(sets[0].dropna(subset=covs + [time, event]))),
                events=int(sets[0].dropna(subset=covs + [time, event])[event].sum()), m=len(sets))


def lrt(f0, f1):
    stat = 2 * (f1.log_likelihood_ - f0.log_likelihood_)
    df = len(f1.params_) - len(f0.params_)
    return dict(chi2=stat, df=df, p=stats.chi2.sf(stat, df))


def rcs_basis(x, knots):
    """Harrell's restricted cubic spline: x plus K-2 nonlinear terms."""
    k = np.asarray(knots, float)
    s = (k[-1] - k[0]) ** 2
    cols = [x]
    for j in range(len(k) - 2):
        cols.append((np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])) / s)
    return np.column_stack(cols)


def metrics(f, train, test, covs):
    """Harrell's C, Uno's C (tau) and time-dependent AUC (T_AUC) of model f on test. The linear predictor is used
    (same order as the risk, without exp overflow in fits that diverge in a bootstrap sample)."""
    risk = f.predict_log_partial_hazard(test[covs]).values.ravel()
    if not np.all(np.isfinite(risk)):
        raise ValueError("predictor lineal no finito")
    ytr = Surv.from_arrays(train.death.astype(bool).values, train.os_years.values)
    yte = Surv.from_arrays(test.death.astype(bool).values, test.os_years.values)
    tau = min(TAU, test.os_years.max() - 1e-6, train.os_years.max() - 1e-6)
    out = dict(harrell=concordance_index(test.os_years, -risk, test.death),
               uno=concordance_index_ipcw(ytr, yte, risk, tau=tau)[0])
    try:
        out["auc2"] = float(cumulative_dynamic_auc(ytr, yte, risk, [T_AUC])[0][0])
    except ValueError:
        out["auc2"] = np.nan
    return out


def incremental(d, covs_a, covs_b, n_boot, rng):
    """A versus B: likelihood-ratio test, apparent and optimism-corrected metrics (Harrell's bootstrap), percentile CI
    of the apparent difference."""
    d = d.dropna(subset=covs_b + ["os_years", "death"])
    if len(d) < 30 or d.death.sum() < 15:
        return dict(n=len(d), events=int(d.death.sum()), skipped="muestra insuficiente")
    fa, fb = cox(d, covs_a), cox(d, covs_b)
    app = {m: metrics(f, d, d, c) for m, f, c in (("A", fa, covs_a), ("B", fb, covs_b))}
    opt = {m: {k: [] for k in ("harrell", "uno", "auc2")} for m in ("A", "B")}
    diffs = {k: [] for k in ("harrell", "uno", "auc2")}
    skipped = 0
    for _ in range(n_boot):
        bs = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        if bs.death.sum() < 10:
            continue
        try:
            fab, fbb = cox(bs, covs_a), cox(bs, covs_b)
            got = {m: (metrics(f, bs, bs, c), metrics(f, bs, d, c))
                   for m, f, c in (("A", fab, covs_a), ("B", fbb, covs_b))}
        except Exception:
            skipped += 1
            continue
        for m in ("A", "B"):
            for k in opt[m]:
                opt[m][k].append(got[m][0][k] - got[m][1][k])
        for k in diffs:
            diffs[k].append(got["B"][0][k] - got["A"][0][k])
    res = dict(n=len(d), events=int(d.death.sum()), lrt=lrt(fa, fb), n_boot=len(diffs["harrell"]),
               boot_skipped=skipped)
    for k in ("harrell", "uno", "auc2"):
        ca = app["A"][k] - np.nanmean(opt["A"][k])
        cb = app["B"][k] - np.nanmean(opt["B"][k])
        res[k] = dict(A_apparent=app["A"][k], B_apparent=app["B"][k], A_corrected=ca, B_corrected=cb,
                      delta_apparent=app["B"][k] - app["A"][k], delta_corrected=cb - ca,
                      delta_ci=list(np.nanpercentile(diffs[k], [2.5, 97.5])))
    return res


# ---------------------------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------------------------

def marker_model(d, k, kind="d", extra=None, time="os_years", event="death", impute=True):
    """HR per SD of {k}_{kind} adjusted for the base model and the standardized baseline value of the same marker."""
    covs = BASE + (extra or []) + ([f"{k}_prez"] if kind != "pre" else []) + [f"{k}_{kind}z"]
    sub = d.dropna(subset=[f"{k}_{kind}z", f"{k}_prez", time, event])
    if len(sub) < 30 or sub[event].sum() < 10:
        return dict(n=len(sub), events=int(sub[event].sum()), hr=np.nan, lo=np.nan, hi=np.nan, p=np.nan, coef=np.nan,
                    se=np.nan, sd=np.nan, unit=np.nan, hr_unit=np.nan, lo_unit=np.nan, hi_unit=np.nan,
                    skipped="muestra insuficiente")
    sets = imputations(sub, covs) if impute else [sub.dropna(subset=covs)]
    r = pooled(sets, covs, f"{k}_{kind}z", time, event)
    sd = sub[f"{k}_{kind}"].std(ddof=1)
    unit = MARKERS[k][2] if kind in ("d", "pre", "post") else 10
    r.update(sd=sd, unit=unit, hr_unit=np.exp(r["coef"] * unit / sd), lo_unit=np.exp((r["coef"] - 1.96 * r["se"]) * unit / sd),
             hi_unit=np.exp((r["coef"] + 1.96 * r["se"]) * unit / sd))
    return r


def table1(d, markers):
    rows = []
    def add(name, x, kind="cont"):
        x = x.dropna()
        if kind == "cont":
            rows.append((name, f"{x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]", len(x)))
        else:
            rows.append((name, f"{int(x.sum())} ({100 * x.mean():.0f}%)", len(x)))
    add("Edad (anos)", d.age)
    add("Hombres", d.male, "bin")
    add("Zubrod >= 1", d.zubrod1, "bin")
    add("Estadio IIIB", d.stage3b, "bin")
    add("Tumor izquierdo", d.left_tumor, "bin")
    add("Lobulo inferior o hilio", d.lower_or_hilar, "bin")
    add("Diabetes", d.diabetes, "bin")
    add("IMC basal (kg/m2)", d.bmi_pre)
    add("Dosis al GTV (Gy)", d.gtv_dose_gy)
    add("SUVpeak del primario post (lectura central)", d.tumor_suvpeak_post)
    add("Dias entre PET basal y post", d.interval_days)
    add("Dias del fin de RT al landmark", d.rt_end_to_landmark_days)
    add("Cambio de peso entre PET (kg)", d.weight_change_kg)
    add("Mismo modelo de equipo", d.same_model.astype(float), "bin")
    for k in markers:
        lab = MARKERS[k][1]
        pre, post, dd = d[f"{k}_pre"], d[f"{k}_post"], d[f"{k}_d"]
        ok = dd.notna()
        p = stats.wilcoxon(pre[ok], post[ok]).pvalue if ok.sum() > 10 else np.nan
        rows.append((f"{lab}: basal / post / cambio (Wilcoxon p={p:.3g})",
                     f"{pre.median():.1f} / {post.median():.1f} / {dd.median():.1f} [{dd.quantile(.25):.1f}, "
                     f"{dd.quantile(.75):.1f}]", int(ok.sum())))
    return pd.DataFrame(rows, columns=["variable", "mediana [RIC] o n (%)", "n"])


def spline_figure(d, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    out = {}
    for a, k in zip(ax, PRIMARY):
        sub = d.dropna(subset=BASE + [f"{k}_d", f"{k}_prez"]).copy()
        x = sub[f"{k}_d"].values
        knots = np.percentile(x, [10, 50, 90])
        B = rcs_basis(x, knots)
        sub["s1"], sub["s2"] = B[:, 0], B[:, 1]
        covs = BASE + [f"{k}_prez", "s1", "s2"]
        f = cox(sub, covs)
        lin = cox(sub, BASE + [f"{k}_prez", f"{k}_d"])
        out[k] = dict(knots=knots.tolist(), p_nonlinear=lrt(lin, f)["p"])
        g = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 100)
        G, G0 = rcs_basis(g, knots), rcs_basis(np.zeros(1), knots)
        beta = f.params_[["s1", "s2"]].values
        V = f.variance_matrix_.loc[["s1", "s2"], ["s1", "s2"]].values
        diff = G - G0
        lh = diff @ beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", diff, V, diff))
        a.plot(g, np.exp(lh), color="#1f4e79")
        a.fill_between(g, np.exp(lh - 1.96 * se), np.exp(lh + 1.96 * se), color="#1f4e79", alpha=.15)
        a.axhline(1, color="grey", lw=.8)
        a.axvline(0, color="grey", lw=.8, ls=":")
        a.set_yscale("log")
        a.set_xlabel(f"Cambio en {MARKERS[k][1].lower()}")
        a.set_ylabel("HR frente a cambio 0")
        a.set_title(f"p no lineal = {out[k]['p_nonlinear']:.2f}", fontsize=9)
        a.plot(x, np.full_like(x, a.get_ylim()[0] * 1.05), "|", color="k", alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return out


def forest_figure(rows, path):
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1))
    for i, r in enumerate(rows):
        ax.plot([r["lo"], r["hi"]], [i, i], color="k", lw=1)
        ax.plot(r["hr"], i, "s", color="#1f4e79" if r["group"] == "primario" else "#888888")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.axvline(1, color="grey", lw=.8)
    ax.set_xscale("log")
    ax.set_xlabel("HR de muerte por DE del cambio (ajustado)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def km_figure(d, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, k in zip(ax, PRIMARY):
        x = d[f"{k}_d"]
        t = pd.qcut(x, 3, labels=["tercil inferior", "tercil medio", "tercil superior"])
        for lab, col in zip(t.cat.categories, ("#1f4e79", "#7f7f7f", "#c00000")):
            s = d[t == lab]
            KaplanMeierFitter().fit(s.os_years, s.death, label=f"{lab} (n={len(s)})").plot_survival_function(
                ax=a, ci_show=False, color=col)
        a.set_title(f"Cambio en {MARKERS[k][1].lower()}", fontsize=9)
        a.set_xlabel("Anos desde el PET post")
        a.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def paired_figure(d, path, ks):
    fig, ax = plt.subplots(1, len(ks), figsize=(3 * len(ks), 3.6))
    for a, k in zip(ax, ks):
        s = d[[f"{k}_pre", f"{k}_post"]].dropna()
        for _, r in s.iterrows():
            a.plot([0, 1], [r.iloc[0], r.iloc[1]], color="grey", alpha=.25, lw=.7)
        a.boxplot([s.iloc[:, 0], s.iloc[:, 1]], positions=[0, 1], widths=.35, showfliers=False)
        a.set_xticks([0, 1])
        a.set_xticklabels(["basal", "post"])
        a.set_title(MARKERS[k][1], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def cif_figure(d, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    code = np.select([d.death_cause == "nsclc", d.death_cause.isin(["other", "unknown"])], [1, 2], 0)
    for a, ev, lab in ((ax[0], 1, "Muerte por CPNM"), (ax[1], 2, "Muerte por otras causas o desconocida")):
        t = pd.qcut(d.sac_d, 3, labels=["inferior", "medio", "superior"])
        for g, col in zip(t.cat.categories, ("#1f4e79", "#7f7f7f", "#c00000")):
            s = (t == g).values
            AalenJohansenFitter(calculate_variance=False).fit(d.os_years.values[s], code[s], event_of_interest=ev,
                                                              label=f"tercil {g} de cambio del saco").plot(
                ax=a, color=col, ci_show=False)
        a.set_title(lab, fontsize=9)
        a.set_xlabel("Anos desde el PET post")
        a.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--primary-only", action="store_true",
                    help="solo marcadores de la TC (primarios y CAC); sin derrame, PET ni 'mas alla del FDG'")
    args = ap.parse_args()
    for p in (TAB, FIG):
        os.makedirs(p, exist_ok=True)
    tag = "_PERMUTADO" if args.permute else ""
    rng = np.random.default_rng(SEED)
    d = load(args.permute)
    d.to_csv(os.path.join(R, f"acrin_analysis_dataset{tag}.csv"), index=False)
    res = dict(n=len(d), deaths=int(d.death.sum()), nsclc_deaths=int(d.nsclc_death.sum()),
               other_deaths=int(d.other_death.sum()), unknown_cause=int((d.death_cause == "unknown").sum()),
               median_fu_alive_years=float(d[d.death == 0].os_years.median()), permuted=args.permute,
               primary_only=args.primary_only)
    markers = PRIMARY + ["cac"] if args.primary_only else PRIMARY + SECONDARY + SENS_MARKERS
    km = KaplanMeierFitter().fit(d.os_years, d.death)
    res["os_median_years"] = float(km.median_survival_time_)
    res["os_1y"], res["os_2y"] = (float(km.survival_function_at_times(t).iloc[0]) for t in (1, 2))
    print(f"n={res['n']}, muertes {res['deaths']} (CPNM {res['nsclc_deaths']}, otras {res['other_deaths']}, "
          f"desconocida {res['unknown_cause']}); mediana SG {res['os_median_years']:.2f} a{tag}")
    table1(d, markers).to_csv(os.path.join(TAB, f"acrin_table1{tag}.csv"), index=False)

    # models per marker (change adjusted for baseline), with multiple imputation of the base model
    rows = []
    for k in markers:
        for kind in ("d", "pre", "post", "rel"):
            r = marker_model(d, k, kind)
            rows.append(dict(marker=k, label=MARKERS[k][1], kind=kind, **r))
    t2 = pd.DataFrame(rows)
    prim = (t2.marker.isin(PRIMARY)) & (t2.kind == "d")
    t2.loc[prim, "p_holm"] = multipletests(t2.loc[prim, "p"], method="holm")[1]
    sec = (t2.marker.isin(SECONDARY)) & (t2.kind == "d")
    if not args.primary_only:
        t2.loc[sec, "p_fdr"] = multipletests(t2.loc[sec, "p"], method="fdr_bh")[1]
    t2.to_csv(os.path.join(TAB, f"acrin_table2_cox{tag}.csv"), index=False)
    res["cox"] = t2.to_dict("records")

    # assumptions (complete cases): proportional hazards and nonlinearity
    ph = {}
    for k in PRIMARY:
        sub = d.dropna(subset=BASE + [f"{k}_prez", f"{k}_dz"])
        f = cox(sub, BASE + [f"{k}_prez", f"{k}_dz"])
        ph[k] = float(proportional_hazard_test(f, sub[BASE + [f"{k}_prez", f"{k}_dz", "os_years", "death"]],
                                               time_transform="rank").summary.loc[f"{k}_dz", "p"])
    res["ph_p"] = ph
    res["splines"] = spline_figure(d, os.path.join(FIG, f"acrin_fig_splines{tag}.png"))

    # incremental value (complete cases)
    struct = [f"{k}_{s}z" for k in PRIMARY for s in ("pre", "d")]
    res["incremental_structural"] = incremental(d, BASE, BASE + struct, args.boot, rng)
    pet = d[d.pet_both]
    fdg = ["suvmean_full_prez", "suvmean_full_dz"]
    res["incremental_beyond_fdg"] = (dict(n=0, skipped="solo analisis principal") if args.primary_only else
                                     incremental(pet, BASE + fdg, BASE + fdg + struct, args.boot, rng))

    # replication of Miller et al. 2022: age + stage + change in cardiac SUVmean (absolute per SD and per 0.1; relative)
    rep = {}
    for kind in ("d", "rel") if not args.primary_only else ():
        sub = pet.dropna(subset=["age10", "stage3b", f"suvmean_full_{kind}"])
        if len(sub) < 30:
            rep[kind] = dict(n=len(sub), skipped="muestra insuficiente")
            continue
        f = cox(sub, ["age10", "stage3b", f"suvmean_full_{kind}"])
        b, se = f.params_[f"suvmean_full_{kind}"], f.standard_errors_[f"suvmean_full_{kind}"]
        step = 0.1 if kind == "d" else 10.0
        rep[kind] = dict(n=len(sub), events=int(sub.death.sum()), per=step, hr=float(np.exp(b * step)),
                         lo=float(np.exp((b - 1.96 * se) * step)), hi=float(np.exp((b + 1.96 * se) * step)),
                         p=float(f.summary.loc[f"suvmean_full_{kind}", "p"]))
    sub = pet.dropna(subset=["suvmean_full_d"])
    pos = sub.suvmean_full_d > 0
    if len(sub) >= 30 and not args.primary_only:
        rep["km_median_years"] = dict(
            aumento=float(KaplanMeierFitter().fit(sub.os_years[pos], sub.death[pos]).median_survival_time_),
            descenso=float(KaplanMeierFitter().fit(sub.os_years[~pos], sub.death[~pos]).median_survival_time_),
            n_aumento=int(pos.sum()), n=len(sub))
    res["miller_replication"] = rep

    # competing risks (cause-specific Cox)
    cr = []
    for k in PRIMARY:
        r = marker_model(d, k, "d", event="nsclc_death")
        cr.append(dict(marker=k, outcome="muerte por CPNM (ajuste base)", **r))
        for ev, lab in (("other_death", "muerte por otras causas"), ("other_unknown_death", "otras o desconocida")):
            sub = d.dropna(subset=[f"{k}_dz"])
            f = cox(sub, ["age10", f"{k}_dz"], event=ev)
            cr.append(dict(marker=k, outcome=f"{lab} (edad + cambio)", hr=float(np.exp(f.params_[f"{k}_dz"])),
                           lo=float(np.exp(f.confidence_intervals_.loc[f"{k}_dz"].iloc[0])),
                           hi=float(np.exp(f.confidence_intervals_.loc[f"{k}_dz"].iloc[1])),
                           p=float(f.summary.loc[f"{k}_dz", "p"]), n=len(sub), events=int(sub[ev].sum())))
    pd.DataFrame(cr).to_csv(os.path.join(TAB, f"acrin_tableS_competing{tag}.csv"), index=False)
    res["competing"] = cr

    # sensitivity (complete cases except the first row)
    sens = []
    subsets = {
        "principal (imputacion multiple)": (d, None, True),
        "casos completos": (d, None, False),
        "mismo modelo de equipo": (d[d.same_model], None, False),
        "sin marcas de control de calidad": (d[~d.qc_any], None, False),
        "sin marcas ni caja grande conservada": (d[~d.qc_any & ~d.large_any], None, False),
        "sin contraste IV (formulario o imagen)": (d[~d.iv_contrast_any], None, False),
        "sin muertes en 30 dias": (d[~((d.death == 1) & (d.os_years < 30 / 365.25))], None, False),
        "ajuste ampliado": (d, EXTENDED[len(BASE):], True),
    }
    d_all = load(args.permute, keep_visual_fail=True)
    subsets["incluye series con falla visual"] = (d_all, None, False)
    for k in PRIMARY:
        for name, (sub, extra, imp) in subsets.items():
            r = marker_model(sub, k, "d", extra=extra, impute=imp)
            sens.append(dict(marker=k, analysis=name, **r))
        r = marker_model(d, k, "rel", impute=False)
        sens.append(dict(marker=k, analysis="cambio relativo (%)", **r))
    pd.DataFrame(sens).to_csv(os.path.join(TAB, f"acrin_tableS_sensitivity{tag}.csv"), index=False)
    res["sensitivity"] = sens

    # validity of the measurements (no outcome)
    val = {}
    for a, b in (("eat_d", "weight_change_kg"), ("sac_d", "fluid_d"), ("eat_d", "sac_d"), ("eatd_d", "eat_d")):
        s = d[[a, b]].dropna()
        if len(s) < 30 or (args.primary_only and "fluid" in b):
            continue
        rho, p = stats.spearmanr(s[a], s[b])
        val[f"{a}~{b}"] = dict(rho=rho, p=p, n=len(s))
    for k in PRIMARY:
        val[f"sd_cambio_{k}_mismo_vs_distinto_equipo"] = [float(d[d.same_model][f"{k}_d"].std()),
                                                          float(d[~d.same_model][f"{k}_d"].std())]
    res["validity"] = val

    fr = [dict(label=f"{MARKERS[r['marker']][1]}", hr=r["hr"], lo=r["lo"], hi=r["hi"],
               group="primario" if r["marker"] in PRIMARY else "secundario")
          for r in res["cox"] if r["kind"] == "d" and r["marker"] in PRIMARY + SECONDARY]
    forest_figure(fr, os.path.join(FIG, f"acrin_fig_forest{tag}.png"))
    km_figure(d, os.path.join(FIG, f"acrin_fig_km{tag}.png"))
    paired_figure(d, os.path.join(FIG, f"acrin_fig_paired{tag}.png"),
                  PRIMARY + ["cac"] if args.primary_only else PRIMARY + ["fluid", "cac", "suvmean_full"])
    cif_figure(d, os.path.join(FIG, f"acrin_fig_cif{tag}.png"))
    with open(os.path.join(R, f"acrin_survival_results{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for r in res["cox"]:
        if r["kind"] == "d" and r["marker"] in PRIMARY + SECONDARY and not r.get("skipped"):
            print(f"  {r['label']:<40} HR/DE {r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.3f} | "
                  f"HR por {r['unit']} {r['hr_unit']:.2f} | n={r['n']} eventos={r['events']}")
    for name in ("incremental_structural", "incremental_beyond_fdg"):
        x = res[name]
        if "skipped" in x:
            print(f"  {name}: omitido ({x['skipped']}, n={x['n']})")
            continue
        print(f"  {name}: n={x['n']} LR p={x['lrt']['p']:.3f} | C Harrell A {x['harrell']['A_corrected']:.3f} -> B "
              f"{x['harrell']['B_corrected']:.3f} (dif corregida {x['harrell']['delta_corrected']:+.3f}) | Uno dif "
              f"{x['uno']['delta_corrected']:+.3f} | AUC2 dif {x['auc2']['delta_corrected']:+.3f}")


if __name__ == "__main__":
    main()
