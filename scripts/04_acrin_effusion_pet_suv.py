"""Low-attenuation pericardial rim volume (CT) and cardiac SUV (PET) for each series measured by script 01.

Per CT series with masks (results/acrin_masks/<pid>_<date>.npz), without neural networks and on CPU only:
- Re-reads the CT, repeats the saved crop and the DeepCAC curation (dc.curate) and checks that the grid matches the
  saved one, so that the saved sac falls on the same voxels.
- Rim volume (analysis plan, section 4, and its change log): HU after a 1x3x3 median filter; volume between 0 and
  25 HU in the outer rim of the sac (<= 10 mm from the border). Sensitivity: 0-50 HU in the rim and in the whole sac,
  and 0-25 HU in a 4-mm rim (Modiri et al., Cancers 2025).
- PET of the same study (same table; matched through patient coordinates): SUV normalized to body weight. BQML with
  decay correction to START: dose decayed from injection to the start of the series. Philips CNTS: private SUV factor
  (7053,1000). GML: already in SUV. Weight and dose from the DICOM; if missing, from the TA form. Whole sac resampled
  to the PET (fraction >= 0.5): SUVmean and SUVmax, as the RTOG heart contour of Miller et al., Radiother Oncol 2022.
  Sac eroded by 5 mm (less spill-over from neighboring activity): SUVmax, SUVmean, SUVpeak (1 cm3) and median. EAT
  uptake is not measured: the fat layer is thinner than the PET resolution (~6 mm) and would be dominated by the
  neighboring myocardium.

Outputs: results/acrin_effusion_pet.csv (resumable by CT series) and qc/acrin_pet/<pid>_<date>.png.
Usage: python scripts/04_acrin_effusion_pet_suv.py [--only 001 --limit 2]
"""
import argparse
import csv
import datetime
import glob
import importlib.util
import os
import time
import traceback

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("run", os.path.join(ROOT, "scripts", "01_acrin_eat_cac_run.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)          # same reading, crop and curation as in the model run
dc, plt = run.dc, run.plt

OUT = os.path.join(ROOT, "results", "acrin_effusion_pet.csv")
QC_PET = os.path.join(ROOT, "qc", "acrin_pet")
LOG = os.path.join(ROOT, "logs", "acrin_effusion_pet.log")
COHORT = os.path.join(ROOT, "results", "acrin_clinical_cohort.csv")
SHELL_MM, ERODE_MM, F18_HALF_LIFE = 10.0, 5.0, 6586.2
FIELDS = ["PatientID", "StudyDate", "SeriesInstanceUID", "grid_ok", "shell_ml", "fluid_shell_ml", "fluid_sac_ml",
          "fluid25_shell_ml", "fluid25_shell4_ml", "pet_series_uid", "pet_manufacturer", "pet_units", "pet_decay",
          "weight_kg", "weight_source", "dose_mbq", "dose_source", "uptake_min", "suv_flags", "heart_full_roi_ml",
          "heart_suvmax_full", "heart_suvmean_full", "heart_roi_ml", "heart_suvmax", "heart_suvpeak", "heart_suvmean",
          "heart_suvmedian", "error", "secs", "run_at"]


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def secs(t):
    """'HHMMSS.ffffff' -> seconds since midnight."""
    t = str(t).strip()
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:] or 0)


def effusion(img, z, sac):
    """Rim volumes on the saved curated grid (sac: bool z,y,x)."""
    lo, hi = (int(v) for v in z["crop_z"])
    cur = dc.curate(img[:, :, lo:hi])
    ok = tuple(cur.GetSize()) == tuple(int(v) for v in z["cur_size"])
    if not ok:
        return dict(grid_ok=False)
    a = sitk.GetArrayFromImage(cur)
    sl = tuple(slice(max(0, s.start - 3), s.stop + 3) for s in ndimage.find_objects(sac.astype(np.uint8))[0])
    s, hu = sac[sl], ndimage.median_filter(a[sl], size=(1, 3, 3))
    sp = cur.GetSpacing()
    dist = ndimage.distance_transform_edt(s, sampling=(sp[2], sp[1], sp[0]))
    shell = s & (dist <= SHELL_MM)
    v = float(np.prod(sp)) / 1000
    fl, fl25 = (hu >= 0) & (hu <= 50), (hu >= 0) & (hu <= 25)
    return dict(grid_ok=True, shell_ml=round(shell.sum() * v, 1), fluid_shell_ml=round((shell & fl).sum() * v, 1),
                fluid_sac_ml=round((s & fl).sum() * v, 1), fluid25_shell_ml=round((shell & fl25).sum() * v, 1),
                fluid25_shell4_ml=round((s & (dist <= 4) & fl25).sum() * v, 1))


def read_pet(sdir):
    """PET series -> (SimpleITK image in DICOM units with per-slice slope, first header, earliest time)."""
    dss = [pydicom.dcmread(f) for f in glob.glob(os.path.join(sdir, "*.dcm"))]
    dss.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    arr = np.stack([d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1)) +
                    float(d.get("RescaleIntercept", 0)) for d in dss])
    d0 = dss[0]
    iop = [float(v) for v in d0.ImageOrientationPatient]
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-3):
        raise ValueError(f"orientacion PET no axial estandar {iop}")
    zs = [float(d.ImagePositionPatient[2]) for d in dss]
    img = sitk.GetImageFromArray(arr)
    ps = [float(v) for v in d0.PixelSpacing]
    img.SetSpacing((ps[1], ps[0], float(np.median(np.diff(zs)))))
    img.SetOrigin(tuple(float(v) for v in d0.ImagePositionPatient))
    acq = [secs(d.AcquisitionTime) for d in dss if d.get("AcquisitionTime")]
    return img, d0, (min(acq) if acq else None)


def suv_factor(d0, acq_min, ta):
    """Factor that converts the image to SUVbw and what was used. ta: row of the TA form (weight, dose, time)."""
    flags = []
    units, decay = str(d0.get("Units", "")).upper(), str(d0.get("DecayCorrection", "")).upper()
    w, wsrc = pd.to_numeric(d0.get("PatientWeight", np.nan), errors="coerce"), "dicom"
    if not (w and w > 20):
        w, wsrc = ta.get("weight_kg", np.nan), "TA"
    info = dict(pet_units=units, pet_decay=decay, weight_kg=w, weight_source=wsrc)
    if units == "GML":
        return 1.0, info, flags
    rp = d0.get("RadiopharmaceuticalInformationSequence", [None])[0]
    dose, dsrc = (float(rp.RadionuclideTotalDose), "dicom") if rp is not None and rp.get("RadionuclideTotalDose") else \
        (ta.get("dose_mci", np.nan) * 37e6, "TA")
    ta_bq = ta.get("dose_mci", np.nan) * 37e6
    if dsrc == "dicom" and dose < 3.7e7:                # < 37 MBq (1 mCi) in Bq: some GE scanners store the dose in MBq
        if 3.7e7 <= ta_bq <= 2e9:
            dose, dsrc = ta_bq, "TA (dosis DICOM en MBq)"
        elif 37 <= dose <= 2000:
            dose, dsrc = dose * 1e6, "dicom en MBq"
        flags.append("dosis_dicom_no_en_Bq")
    info.update(dose_mbq=round(dose / 1e6, 1), dose_source=dsrc)
    if units == "CNTS":
        f = d0.get((0x7053, 0x1000))
        if f is None:
            raise ValueError("Philips CNTS sin factor SUV (7053,1000)")
        return float(f.value), info, flags
    if units != "BQML" or decay != "START":
        raise ValueError(f"unidades/decaimiento no soportados: {units}/{decay}")
    inj = secs(rp.RadiopharmaceuticalStartTime) if rp is not None and rp.get("RadiopharmaceuticalStartTime") else None
    if inj is None:
        raise ValueError("sin hora de inyeccion")
    ref = secs(d0.SeriesTime)
    if acq_min is not None and acq_min < ref:
        ref = acq_min
        flags.append("ref=acq_min")
    dt = ref - inj + (86400 if ref < inj else 0)
    hl = float(rp.get("RadionuclideHalfLife", F18_HALF_LIFE) or F18_HALF_LIFE)
    info["uptake_min"] = round(dt / 60, 1)
    if not 30 <= dt / 60 <= 180:
        flags.append(f"captacion_{dt / 60:.0f}min")
    return w * 1000 / (dose * 2 ** (-dt / hl)), info, flags


def to_pet(mask, sl, z, pet):
    """Mask cropped to the box sl of the curated grid -> fraction on the PET grid (linear)."""
    m = sitk.GetImageFromArray(mask.astype(np.float32))
    sp, D = np.array(z["cur_spacing"]), np.array(z["cur_direction"]).reshape(3, 3)
    idx = np.array([sl[2].start, sl[1].start, sl[0].start])
    m.SetOrigin(tuple(np.array(z["cur_origin"]) + D @ (idx * sp)))
    m.SetSpacing(tuple(sp))
    m.SetDirection(tuple(z["cur_direction"]))
    return sitk.GetArrayFromImage(sitk.Resample(m, pet, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))


def render_pet(name, suv, roi, path):
    zc = int(np.median(np.where(roi)[0]))
    yc = int(np.median(np.where(roi)[1]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(suv[zc], cmap="gray_r", vmin=0, vmax=6)
    ax[0].contour(roi[zc], levels=[0.5], colors="red", linewidths=0.8)
    ax[1].imshow(suv[:, yc, :], cmap="gray_r", vmin=0, vmax=6, origin="lower")
    ax[1].contour(roi[:, yc, :], levels=[0.5], colors="red", linewidths=0.8, origin="lower")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name} SUV 0-6, saco erosionado {ERODE_MM:.0f} mm")
    plt.tight_layout()
    plt.savefig(path, dpi=55)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(QC_PET, exist_ok=True)
    meas = pd.read_csv(os.path.join(ROOT, "results", "acrin_ct_eat_cac.csv"), dtype={"StudyDate": str})
    meas = meas[meas.error.fillna("") == ""]
    if args.only:
        meas = meas[meas.PatientID.str[-3:].isin(args.only)]
    ta = pd.read_csv(COHORT)
    series = {s["SeriesInstanceUID"]: s for s in run.ct_series(run.DATA)}
    done = set(pd.read_csv(OUT, dtype=str).SeriesInstanceUID) if os.path.exists(OUT) and os.path.getsize(OUT) else set()
    todo = [r for _, r in meas.iterrows() if r.SeriesInstanceUID not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"derrame + SUV: {len(meas)} series medidas | hechas {len(done)} | en esta corrida {len(todo)}")
    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    with open(OUT, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        for i, r in enumerate(todo, 1):
            t = time.time()
            name = f"{r.PatientID[-3:]}_{r.StudyDate}"
            row = dict(PatientID=r.PatientID, StudyDate=r.StudyDate, SeriesInstanceUID=r.SeriesInstanceUID, error="",
                       run_at=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
            try:
                z = np.load(os.path.join(ROOT, "results", "acrin_masks", f"{name}.npz"))
                sac = z["sac"].astype(bool)
                sdir = series[r.SeriesInstanceUID]["sdir"]
                img = dc.read_dicom_series(sdir)
                row.update(effusion(img, z, sac))
                del img
                pets = [p for p in glob.glob(os.path.join(os.path.dirname(sdir), "*")) if p != sdir and
                        pydicom.dcmread(glob.glob(os.path.join(p, "*.dcm"))[0], stop_before_pixels=True).Modality == "PT"]
                if len(pets) != 1:
                    raise ValueError(f"{len(pets)} series PET en el estudio")
                pet, d0, acq_min = read_pet(pets[0])
                c = ta[ta.cn == int(r.PatientID[-3:])].iloc[0]
                tp = "pre" if r.StudyDate == min(meas[meas.PatientID == r.PatientID].StudyDate) else "post"
                f, info, flags = suv_factor(d0, acq_min, dict(weight_kg=c[f"weight_kg_{tp}"], dose_mci=c[f"dose_mci_{tp}"]))
                suv = sitk.GetArrayFromImage(pet) * f
                row.update(info, pet_series_uid=str(d0.SeriesInstanceUID), pet_manufacturer=str(d0.get("Manufacturer", "")),
                           suv_flags=";".join(flags))
                sp = z["cur_spacing"]
                sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in ndimage.find_objects(sac.astype(np.uint8))[0])
                core = sac[sl] & (ndimage.distance_transform_edt(sac[sl], sampling=(sp[2], sp[1], sp[0])) > ERODE_MM)
                roi = to_pet(core, sl, z, pet) >= 0.5
                full = to_pet(sac[sl], sl, z, pet) >= 0.5          # whole sac: definition of Miller 2022
                vpet = float(np.prod(pet.GetSpacing())) / 1000
                if full.any():
                    row.update(heart_full_roi_ml=round(full.sum() * vpet, 1),
                               heart_suvmax_full=round(float(suv[full].max()), 3),
                               heart_suvmean_full=round(float(suv[full].mean()), 3))
                    if not 0.3 <= row["heart_suvmean_full"] <= 10:
                        row["suv_flags"] = ";".join(filter(None, [row.get("suv_flags", ""), "suv_implausible"]))
                if roi.any():
                    psp = pet.GetSpacing()
                    rr = (3 / (4 * np.pi)) ** (1 / 3) * 10                       # radius of 1 cm3 in mm
                    k = np.array(np.meshgrid(*[np.arange(-int(rr // s) - 1, int(rr // s) + 2) * s
                                               for s in (psp[2], psp[1], psp[0])], indexing="ij"))
                    ker = (np.sqrt((k ** 2).sum(0)) <= rr).astype(float)
                    peak = ndimage.convolve(suv, ker / ker.sum(), mode="nearest")
                    row.update(heart_roi_ml=round(roi.sum() * vpet, 1), heart_suvmax=round(float(suv[roi].max()), 3),
                               heart_suvpeak=round(float(peak[roi].max()), 3),
                               heart_suvmean=round(float(suv[roi].mean()), 3),
                               heart_suvmedian=round(float(np.median(suv[roi])), 3))
                    render_pet(name, suv, roi, os.path.join(QC_PET, f"{name}.png"))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                log(traceback.format_exc())
            row["secs"] = round(time.time() - t, 1)
            w.writerow(row)
            fh.flush()
            log(f"[{i}/{len(todo)}] {name} grilla {row.get('grid_ok')} | liquido cascara {row.get('fluid_shell_ml')} "
                f"saco {row.get('fluid_sac_ml')} mL | {row.get('pet_units')} {row.get('pet_decay')} peso "
                f"{row.get('weight_kg')} ({row.get('weight_source')}) captacion {row.get('uptake_min')} min | corazon SUVmax "
                f"{row.get('heart_suvmax')} mean {row.get('heart_suvmean')} | {row.get('suv_flags', '')} | err "
                f"'{row['error']}' {row['secs']}s")
    log("FIN")


if __name__ == "__main__":
    main()
