"""Clinical cohort of ACRIN 6668 for the patients with baseline and restaging PET/CT (analysis plan, sections 2-6).

Joins the de-identified ACRIN forms (sets 6668 and 6668HB) by cn = number in the TCIA Patient ID and checks the link
with the images: sex and age in the DICOM against the forms, and the baseline-restaging interval of the DICOM dates
(shifted by TCIA, intervals kept) against the days on the TA form. Time zero = day of the restaging PET (landmark).
Overall survival from the landmark; cause of death from F1/F2 and DS. Does not look at the cardiac measurements.

Output: results/acrin_clinical_cohort.csv (one row per patient with images) and a summary on screen.
Usage: python scripts/03_acrin_clinical_cohort.py
"""
import glob
import os

import numpy as np
import pandas as pd
import pydicom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACRIN = os.environ.get("ACRIN_DATA", os.path.join(ROOT, "data"))      # TCIA download (see README)
FORMS = [os.path.join(ACRIN, "ACRIN-6668-NSCLC-FDG-PET-ACRIN-Data-TCIA-Anonymized-File-set-1",
                      "ACRIN 6668 NSCLC FDG PET - ACRIN Data TCIA Anonymized"),
         os.path.join(ACRIN, "ACRIN-6668HB-NSCLC-FDG-PET-ACRIN-Data-TCIA-Anonymized-File-set-2",
                      "ACRIN 6668HB NSCLC FDG PET - ACRIN Data TCIA Anonymized")]
IMG = os.path.join(ACRIN, "manifest-acrin6668-paired-petct", "ACRIN-NSCLC-FDG-PET")
OUT = os.path.join(ROOT, "results", "acrin_clinical_cohort.csv")
SITE_DAYS = ["e10d", "e15d", "e20d", "e25d", "e30d", "e35d", "e40d", "e45d", "e51d"]   # progression by site


def form(name):
    """Form from both sets (disjoint cn), with lower-case columns, numeric where possible."""
    t = pd.concat([pd.read_csv(os.path.join(d, f"{name}.csv"), dtype=str, encoding="latin-1") for d in FORMS],
                  ignore_index=True)
    t.columns = [c.lower() for c in t.columns]
    for c in t.columns:
        num = pd.to_numeric(t[c], errors="coerce")
        if num.notna().sum() >= t[c].notna().sum() * 0.9:
            t[c] = num
    return t


def imaging_studies():
    """One CT header per study: DICOM date (shifted), sex, age and weight."""
    rows = []
    for sdir in glob.glob(os.path.join(IMG, "*", "*", "*")):
        fs = glob.glob(os.path.join(sdir, "*.dcm"))
        if not fs:
            continue
        ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
        if ds.get("Modality") != "CT":
            continue
        rows.append(dict(cn=int(str(ds.PatientID)[-3:]), date=pd.to_datetime(ds.StudyDate, format="%Y%m%d"),
                         dcm_sex=str(ds.get("PatientSex", "")), dcm_age=str(ds.get("PatientAge", "")),
                         dcm_weight=pd.to_numeric(ds.get("PatientWeight", np.nan), errors="coerce")))
    s = pd.DataFrame(rows).sort_values(["cn", "date"])
    g = s.groupby("cn")
    out = g.agg(n_ct=("date", "size"), date_pre=("date", "first"), date_post=("date", "last"),
                dcm_sex=("dcm_sex", "first"), dcm_age=("dcm_age", "first"))
    out["dicom_interval"] = (out.date_post - out.date_pre).dt.days
    return out.reset_index()


def followup():
    """Last vital status, day of death or last contact, and cause (F1 and F2 have the same structure)."""
    fu = []
    for name in ("F1", "F2"):
        t = form(name)
        p = name.lower()
        t = t.rename(columns={c: c.replace(p, "", 1) for c in t.columns if c.startswith(p)})
        fu.append(t)
    fu = pd.concat(fu, ignore_index=True)
    ds = form("DS")
    out = []
    for cn, g in fu.groupby("cn"):
        dead = g[g.e2.isin([2, 9])]
        r = dict(cn=cn, n_fu_forms=len(g))
        prog = g[SITE_DAYS].min().min() if set(SITE_DAYS) <= set(g.columns) else np.nan
        r["first_progression_day"] = prog
        d = ds[ds.cn == cn]
        ds_death = len(d) and (d.dse3 == 2).any()
        if len(dead) or ds_death:
            r["death"] = 1
            day = dead.e3d.min() if len(dead) and dead.e3d.notna().any() else np.nan
            if np.isnan(day) and ds_death:
                day = d.dse2d.min()
            r["event_day"] = day
            causes = dead.e4.dropna()
            causes = causes[causes != 99] if (causes != 99).any() else causes
            c = causes.iloc[0] if len(causes) else np.nan
            if np.isnan(c) and len(d) and d.dse6.notna().any():
                c = 1 if (d.dse6 == 1).any() else 5
            r["cause_code"] = c
        else:
            r["death"] = 0
            last = np.nanmax([g.e3d.max(), g.e1d.max(), d.dse2d.max() if len(d) else np.nan])
            r["event_day"] = last
            r["cause_code"] = np.nan
        r["lost_to_fu"] = int((g.e2 == 3).any())
        out.append(r)
    o = pd.DataFrame(out)
    o["death_cause"] = np.select([o.death == 0, o.cause_code == 1, o.cause_code.isin([2, 3, 4, 5])],
                                 ["alive", "nsclc", "other"], "unknown")
    return o


def pet_timepoints():
    """TA form: PET day, weight, height, dose, glucose, fasting and IV contrast at baseline (1) and restaging (2)."""
    ta = form("TA")
    ta = ta[ta.tae1.isin([1, 2]) & (ta.tae3 == 2)].sort_values(["cn", "tae1", "tae7d"])
    ta = ta.groupby(["cn", "tae1"]).first().reset_index()
    keep = dict(tae7d="pet_day", tae15="weight_kg", tae16="height_cm", tae18="dose_mci", tae20="inj_time",
                tae14="glucose", tae13="fasting_h", tae33="iv_contrast")
    w = ta.pivot(index="cn", columns="tae1", values=list(keep))
    w.columns = [f"{keep[a]}_{'pre' if b == 1 else 'post'}" for a, b in w.columns]
    return w.reset_index()


def tumor_suvpeak():
    """SUVpeak of the primary tumor (lung) per time point: mean of the central readers (Q2; if missing, QZ) and, without
    a central read, the local read (IM). Baseline = evaluation 1 at time 1 (e18); restaging = evaluation 2 at time 2
    (e43). The SUVpeak fields of the SS form are 0 in TCIA."""
    out = {}
    for name, p in (("Q2", "q2e"), ("QZ", "qze"), ("IM", "ime")):
        t = form(name)
        for tp, var, lab in ((1, f"{p}18", "pre"), (2, f"{p}43", "post")):
            s = t[(t[f"{p}5"] == tp) & (t[var] > 0)].groupby("cn")[var].mean()
            for cn, v in s.items():
                out.setdefault((cn, lab), (v, name))
    rows = {}
    for (cn, lab), (v, src) in out.items():
        rows.setdefault(cn, {})[f"tumor_suvpeak_{lab}"] = v
        rows[cn][f"tumor_suvpeak_{lab}_source"] = src
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("cn").reset_index()


def main():
    img = imaging_studies()
    a0, i1, t1, tf, ss = form("A0"), form("I1"), form("T1"), form("TF"), form("SS")
    d = img.merge(a0[["cn", "entryage", "a0e11"]], on="cn", how="left")
    d = d.rename(columns={"entryage": "age", "a0e11": "sex_code"})
    d["male"] = (d.sex_code == 1).astype(float).where(d.sex_code.notna())
    i = i1.rename(columns=dict(i1e2="weight_i1", i1e3="weight_loss_pct", i1e5="zubrod", i1e6="stage_code",
                               i1e8="loc_rul", i1e9="loc_rll", i1e10="loc_rhilum_ml", i1e11="loc_lul", i1e12="loc_lll",
                               i1e88="diabetes_code", i1e45="hgb", i1e65="glucose_lab", i1e81="fev1_pct",
                               i1e83="dlco_pct"))
    d = d.merge(i[["cn", "weight_i1", "weight_loss_pct", "zubrod", "stage_code", "loc_rul", "loc_rll", "loc_rhilum_ml",
                   "loc_lul", "loc_lll", "diabetes_code", "hgb", "glucose_lab", "fev1_pct", "dlco_pct"]],
                on="cn", how="left")
    d["zubrod"] = d.zubrod.where(d.zubrod != 99)
    d["diabetes"] = d.diabetes_code.map({1: 0, 2: 1})
    for c in ("loc_rul", "loc_rll", "loc_rhilum_ml", "loc_lul", "loc_lll"):
        d[c] = d[c].map({1: 0, 2: 1})
    heart = []
    for _, r in t1.iterrows():
        doses = [r[f"t1e{dc}"] for tc, dc in ((34, 33), (37, 36), (40, 39))
                 if isinstance(r.get(f"t1e{tc}"), str) and "heart" in r[f"t1e{tc}"].lower()]
        heart.append(max(doses) if doses else np.nan)
    t1["heart_max_dose_gy"] = heart
    t = t1.rename(columns=dict(t1e1="rt_started", t1e2d="rt_start_day", t1e3d="rt_end_day", t1e12="gtv_fractions",
                               t1e13="gtv_dose_gy", t1e41="rt_stop_reason"))
    d = d.merge(t[["cn", "rt_started", "rt_start_day", "rt_end_day", "gtv_fractions", "gtv_dose_gy", "rt_stop_reason",
                   "heart_max_dose_gy"]], on="cn", how="left")
    agents = tf.set_index("cn")[["tfe2", "tfe10", "tfe18", "tfe26", "tfe34"]]
    chemo = agents.apply(lambda r: ";".join(str(int(v)) for v in r.dropna() if v != 9), axis=1).rename("chemo_agents")
    d = d.merge(chemo.reset_index(), on="cn", how="left").merge(
        tf[["cn", "tfe1"]].rename(columns={"tfe1": "chemo_completed_code"}), on="cn", how="left")
    s = ss.rename(columns=dict(sse1="cn_ss", sse3="ss_days_between_pet", sse4="tumor_side", sse6="n_stage_pre",
                               sse14="suvmax_all_pre", sse15="suvpeak_all_pre", sse16="mtv_all_pre",
                               sse19="suvmax_prim_pre", sse20="suvpeak_prim_pre", sse37="suvmax_all_post",
                               sse38="suvpeak_all_post", sse39="mtv_all_post", sse42="suvmax_prim_post",
                               sse43="suvpeak_prim_post", sse44="mtv_prim_post"))
    s = s[[c for c in s.columns if not c.startswith("sse")]].drop(columns=["rec"], errors="ignore")
    d = d.merge(s.groupby("cn").first().reset_index(), on="cn", how="left")
    d = d.merge(pet_timepoints(), on="cn", how="left").merge(followup(), on="cn", how="left")
    d = d.merge(tumor_suvpeak(), on="cn", how="left")

    # link and landmark
    d["clin_interval"] = d.pet_day_post - d.pet_day_pre
    d["interval_diff"] = d.dicom_interval - d.clin_interval
    # landmark = baseline PET day on the form + DICOM interval (the restaging PET in TCIA, read by the core lab). In 120
    # and 243 the TA form records a later restaging PET (the core-lab interval, SS, matches the DICOM); in
    # 229 the baseline PET day is impossible (706) and the restaging PET day on the form is used.
    pre_ok = d.pet_day_pre.between(d.rt_start_day - 90, d.rt_start_day + 30)
    d["landmark_day"] = np.where(pre_ok, d.pet_day_pre + d.dicom_interval, d.pet_day_post)
    d["landmark_source"] = np.where(pre_ok, "TA basal + intervalo DICOM", "TA post")
    d["os_days"] = d.event_day - d.landmark_day
    d["os_years"] = d.os_days / 365.25
    d["rt_end_to_landmark_days"] = d.landmark_day - d.rt_end_day
    d["dcm_age_years"] = pd.to_numeric(d.dcm_age.str.extract(r"(\d+)Y")[0], errors="coerce")
    d["sex_match"] = np.where(d.dcm_sex.isin(["M", "F"]) & d.male.notna(),
                              (d.dcm_sex == "M").astype(float) == d.male, np.nan)
    d["age_diff"] = d.dcm_age_years - d.age
    d["link_note"] = np.select([d.interval_diff.abs() > 3, d.sex_match == 0],
                               ["intervalo TA distinto del DICOM", "sexo DICOM distinto"], "")
    d["weight_change_kg"] = d.weight_kg_post - d.weight_kg_pre
    d["bmi_pre"] = d.weight_kg_pre / (d.height_cm_pre / 100) ** 2
    d["in_analysis"] = ((d.n_ct == 2) & (d.os_days > 0)).astype(int)
    d.to_csv(OUT, index=False)

    print(f"pacientes con imagen {len(d)} | con 2 TC {(d.n_ct == 2).sum()} | sin formulario A0 {d.age.isna().sum()}")
    print(f"enlace: sexo coincide {int(d.sex_match.sum())}/{d.sex_match.notna().sum()} | edad DICOM - formulario: "
          f"mediana {d.age_diff.median():.1f}, |dif|>2 en {(d.age_diff.abs() > 2).sum()} | intervalo DICOM - TA: "
          f"exacto {(d.interval_diff == 0).sum()}, |dif|<=3 d {(d.interval_diff.abs() <= 3).sum()} de "
          f"{d.interval_diff.notna().sum()}, sin dia TA post {d.pet_day_post.isna().sum()}")
    print(f"dia del PET basal: mediana {d.pet_day_pre.median():.0f} (rango {d.pet_day_pre.min():.0f}-"
          f"{d.pet_day_pre.max():.0f}); post: mediana {d.pet_day_post.median():.0f} (rango {d.pet_day_post.min():.0f}-"
          f"{d.pet_day_post.max():.0f}); fin de RT al landmark: mediana {d.rt_end_to_landmark_days.median():.0f} d")
    x = d[d.n_ct == 2]
    print(f"con 2 TC: muertes {int(x.death.sum())} | muerte o censura en el dia del PET post o antes "
          f"{(x.os_days <= 0).sum()} (muertes {int(((x.os_days <= 0) & (x.death == 1)).sum())})")
    a = d[d.in_analysis == 1]
    print(f"analisis: n={len(a)}, muertes {int(a.death.sum())} (CPNM {(a.death_cause == 'nsclc').sum()}, otras "
          f"{(a.death_cause == 'other').sum()}, desconocida {(a.death_cause == 'unknown').sum()}) | seguimiento "
          f"mediano de vivos {a[a.death == 0].os_years.median():.1f} a | perdidos {int(a.lost_to_fu.sum())}")
    print("SUVpeak tumoral post por fuente: " + a.tumor_suvpeak_post_source.fillna("falta").value_counts().to_string()
          .replace("\n", ", ") + f" | mediana {a.tumor_suvpeak_post.median():.1f}")
    miss = a[["age", "male", "zubrod", "stage_code", "tumor_suvpeak_post", "weight_kg_pre", "weight_kg_post",
              "gtv_dose_gy", "tumor_side", "diabetes", "iv_contrast_pre", "iv_contrast_post"]].isna().sum()
    print("faltantes en el analisis:\n" + miss[miss > 0].to_string())
    print(f"contraste IV declarado: pre {int((a.iv_contrast_pre == 2).sum())}, post {int((a.iv_contrast_post == 2).sum())}"
          f" | dosis maxima al corazon registrada en {a.heart_max_dose_gy.notna().sum()}")


if __name__ == "__main__":
    main()
