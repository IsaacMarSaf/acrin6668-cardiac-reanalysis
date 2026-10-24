"""Quality control of results/acrin_ct_eat_cac.csv and table of baseline/restaging pairs per patient.

Baseline and restaging follow the StudyDate order within each patient (dates shifted by TCIA; the interval is kept).
Flagged for manual review: errors, model flags, contrast on the image (sac core >= 60 HU), sac touching the z edge of
the crop, sac outside 400-1300 mL, crop < 250 mm or lungs < 1.5 L, and a scanner change between time points. Uses no
clinical data.

Usage: python scripts/02_acrin_qc_pairs.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "acrin_ct_eat_cac.csv")
OUT_QC = os.path.join(ROOT, "results", "acrin_ct_qc_flags.csv")
OUT_PAIRS = os.path.join(ROOT, "results", "acrin_ct_pairs.csv")
MEAS = ["sac_ml", "eat_ml", "eat_density_hu", "sac_core_median_hu", "cac_sac", "cac_pred", "heart_vol_ml_deepcac"]


def review_reasons(r):
    out = []
    if isinstance(r.error, str) and r.error:
        out.append("error")
    for c in ("eat_flags", "cac_flags"):
        if isinstance(r[c], str) and r[c]:
            out.append(c)
    if r.contrast_by_image == "yes":
        out.append("contraste")
    if str(r.sac_touches_z_edge) == "True":
        out.append("saco_borde_z")
    if not 400 <= r.sac_ml <= 1300:
        out.append("saco_fuera_rango")
    if r.crop_mm < 250 or r.lung_ml < 1500:
        out.append("recorte_o_pulmon_chico")
    return ";".join(out)


def main():
    d = pd.read_csv(IN, dtype={"StudyDate": str})
    d["pid"] = d.PatientID.str[-3:]
    d["date"] = pd.to_datetime(d.StudyDate, format="%Y%m%d")
    d = d.sort_values(["pid", "date"])
    d["n_tp"] = d.groupby("pid").pid.transform("size")
    d["tp"] = np.where(d.groupby("pid").cumcount() == 0, "pre", "post")
    d["review"] = d.apply(review_reasons, axis=1)
    d.to_csv(OUT_QC, index=False)

    print(f"{len(d)} series de {d.pid.nunique()} pacientes | errores {(d.error.fillna('') != '').sum()} | "
          f"pacientes sin par {(d.groupby('pid').size() != 2).sum()}")
    print(f"segundos por serie: mediana {d.secs.median():.0f}, max {d.secs.max():.0f}")
    for k in ("eat_flags", "cac_flags", "error"):
        v = d[k].fillna("")
        if (v != "").any():
            print(f"\n{k}:\n" + v[v != ""].value_counts().to_string())
    print("\nmotivos de revision:\n" + d.review.str.split(";").explode().replace("", np.nan).dropna()
          .value_counts().to_string())
    print("\npor equipo (medianas):")
    print(d.groupby("ManufacturerModelName")[["sac_ml", "eat_ml", "eat_density_hu", "sac_core_median_hu", "crop_mm",
                                              "secs"]].median().round(1).assign(n=d.ManufacturerModelName.value_counts()))
    print("\nrangos:\n" + d[MEAS + ["lung_ml", "crop_mm"]].describe(percentiles=[.05, .5, .95]).round(1).to_string())

    ok = d[(d.n_tp == 2) & d.groupby("pid").error.transform(lambda e: (e.fillna("") == "").all())]
    w = ok.pivot(index="pid", columns="tp", values=MEAS + ["date", "ManufacturerModelName", "review"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w["interval_days"] = (w.date_post - w.date_pre).dt.days
    w["same_model"] = w.ManufacturerModelName_pre == w.ManufacturerModelName_post
    for m in MEAS:
        w[f"d_{m}"] = w[f"{m}_post"].astype(float) - w[f"{m}_pre"].astype(float)
    w.reset_index().to_csv(OUT_PAIRS, index=False)
    print(f"\npares: {len(w)} | mismo equipo {w.same_model.sum()} | intervalo mediana {w.interval_days.median():.0f} d "
          f"(rango {w.interval_days.min()}-{w.interval_days.max()})")
    print("cambios post - pre (mediana [p25, p75]):")
    for m in ("sac_ml", "eat_ml", "eat_density_hu", "cac_sac"):
        x = w[f"d_{m}"].dropna()
        print(f"  {m}: {x.median():.1f} [{x.quantile(.25):.1f}, {x.quantile(.75):.1f}]  n={len(x)}")


if __name__ == "__main__":
    main()
