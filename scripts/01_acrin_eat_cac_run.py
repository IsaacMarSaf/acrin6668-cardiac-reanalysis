"""ACRIN 6668 (ACRIN-NSCLC-FDG-PET, TCIA): EAT with DeepHeartSeg and CAC with DeepCAC on the CT of the baseline and
restaging PET/CT.

DeepCAC and DeepHeartSeg run with their published weights through a PyTorch implementation (deepcac_torch, not
distributed; see README), with the quality-control figures of lib/qc_render.py. These CTs cover the whole body, and
step 1 of DeepCAC looks for the heart in a 336-mm cube anchored at the first slice (with a 50-cm field of view it
covers 81 to 417 mm above the lower edge). The CT is therefore cropped in z to the thorax before the models run: from
the start of the cube, placed 60 mm below the lung base (141 mm below the base with a 50-cm field of view), to 350 mm
above the base. Lungs = inner parenchyma (-950 to -400 HU), large 3D components (thorax_crop). The crop is saved in
the CSV and in a figure. Second change, flagged: the upper limit of the step-1 heart mask is raised to 2.5 million
voxels (bb_center_acrin), because in large hearts the original fallback truncated the sac.

Per CT series: sac, EAT (-190 to -30 HU, volume and density), median HU of the sac core (contrast), Agatston score of
the original DeepCAC and restricted to the sac + 2 voxels, masks on the curated grid (for the PET and the rim volume
later) and quality-control figures. Only series that are complete according to metadata.csv. Resumable by
SeriesInstanceUID. Uses no clinical data.

Usage: python scripts/01_acrin_eat_cac_run.py [--only 001 --limit 1] [--min-ram-gb 2.0]
"""
import argparse
import csv
import datetime
import glob
import hashlib
import os
import re
import sys
import time
import traceback

import numpy as np
import pydicom
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACRIN = os.environ.get("ACRIN_DATA", os.path.join(ROOT, "data"))       # TCIA download (see README)
PORT = os.environ.get("DEEPCAC_PORT", os.path.join(ROOT, "models"))    # deepcac_torch.py, not distributed (README)
DATA = os.path.join(ACRIN, "manifest-acrin6668-paired-petct")
sys.path.insert(0, PORT)
import deepcac_torch as dc  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "lib"))
import qc_render as l1  # noqa: E402   quality-control figures and the 2-voxel sac margin
plt = l1.plt

QC_EAT = os.path.join(ROOT, "qc", "acrin_eat")
QC_CAC = os.path.join(ROOT, "qc", "acrin_cac")
QC_CROP = os.path.join(ROOT, "qc", "acrin_crop")
MASKS = os.path.join(ROOT, "results", "acrin_masks")
LOG = os.path.join(ROOT, "logs", "acrin_eat_cac_run.log")
OUT = os.path.join(ROOT, "results", "acrin_ct_eat_cac.csv")
CROP_BELOW_MM, CROP_ABOVE_MM = 60, 350      # below the start of the step-1 cube / above the lung base
BBOX_MAX = 2_500_000                        # see bb_center_acrin
TAGS = ["Manufacturer", "ManufacturerModelName", "ConvolutionKernel", "KVP", "SliceThickness", "Exposure",
        "CTDIvol", "ReconstructionDiameter", "PatientPosition", "ContrastBolusAgent", "SeriesDescription"]
FIELDS = (["PatientID", "StudyDate", "SeriesInstanceUID"] + TAGS +
          ["n_slices", "spacing", "z_extent_mm", "lung_z0", "lung_z1", "lung_n", "lung_ml", "crop_below_mm", "crop_z0",
           "crop_z1", "crop_mm",
           "sac_ml", "eat_ml", "eat_density_hu", "sac_core_median_hu", "sac_core_p90_hu", "contrast_by_image",
           "sac_z_slices", "sac_touches_z_edge", "eat_flags",
           "cac_pred", "class_pred", "n_lesions", "cac_vol_mm3", "cac_sac", "class_sac", "n_lesions_sac",
           "cac_vol_mm3_sac", "heart_vol_ml_deepcac", "cac_flags", "error", "secs", "run_at"])


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wait_for_ram(min_gb):
    """Do not start another series with little free RAM."""
    import gc
    import psutil
    gc.collect()
    while psutil.virtual_memory().available / 2**30 < min_gb:
        log(f"RAM libre {psutil.virtual_memory().available / 2**30:.1f} GB < {min_gb} GB: espero 30 s")
        time.sleep(30)
        gc.collect()


def complete_series(data):
    """SeriesInstanceUID -> number of images in the metadata.csv of the NBIA Data Retriever. The size has a decimal
    comma ('161,95 MB') that splits the line into one extra column, so the line is read with a regular expression."""
    pat = re.compile(r"^([\d.]+),.*,1\.2\.840\.10008\.[\d.]+,(\d+),.*,(\.\\[^,]+),[^,]*$")
    out = {}
    with open(os.path.join(data, "metadata.csv"), encoding="utf-8-sig") as fh:
        next(fh)
        for line in fh:
            mm = pat.match(line.strip())
            if mm:
                out[mm.group(1)] = (int(mm.group(2)), os.path.join(data, mm.group(3)[2:]))
    return out


def ct_series(data):
    """Downloaded complete CT series (one per study in the manifest)."""
    rows = []
    for uid, (n_img, sdir) in complete_series(data).items():
        files = glob.glob(os.path.join(sdir, "*.dcm"))
        if len(files) != n_img:
            continue
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        if ds.get("Modality") != "CT":
            continue
        rows.append(dict(sdir=sdir, PatientID=str(ds.PatientID), StudyDate=str(ds.StudyDate), SeriesInstanceUID=uid))
    return sorted(rows, key=lambda r: (r["PatientID"], r["StudyDate"]))


def header(sdir):
    ds = pydicom.dcmread(glob.glob(os.path.join(sdir, "*.dcm"))[0], stop_before_pixels=True)
    out = {}
    for t in TAGS:
        v = ds.get(t, "")
        out[t] = "\\".join(map(str, v)) if isinstance(v, pydicom.multival.MultiValue) else str(v)
    return out


def thorax_crop(img):
    """Crop in z to the thorax from the lung base, on a grid 4x coarser in plane.
    Lungs: air (< -400 HU) that does not touch the border of its slice (outside air removed), and of that air only the
    voxels with parenchymal density (> -950 HU) after a 3x3 opening per slice. This leaves out pure air (air pockets
    around the body, bowel gas, airway), the thin rims of those pockets and false bridges through the diaphragm. Foam
    mattresses or cushions under the patient (~ -910 HU, homogeneous) are left out too: a lung has vessels, so its
    90th percentile is > -800 HU, and its center in y lies in the anterior 75% of the image.
    The largest 3D component is taken, plus the second if it measures >= 30% of the first and its center in x lies on
    the other side (the other lung); base = lowest slice of both. A series without >= 200 mL of lung is an error (it
    does not cover the thorax).
    Lower margin: the step-1 cube starts 3*((n-112)//2) mm above the first slice (n = side in x / 3 mm), so that margin
    plus CROP_BELOW_MM is kept below the base; above, the crop reaches CROP_ABOVE_MM over the base."""
    from scipy import ndimage
    a = sitk.GetArrayFromImage(img)[:, ::4, ::4]
    air = a < -400
    inner = np.zeros_like(air)
    for z in range(len(air)):
        lab, _ = ndimage.label(air[z])
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        inner[z] = ndimage.binary_opening(air[z] & ~np.isin(lab, edge) & (a[z] > -950), structure=np.ones((3, 3)))
    lab, n = ndimage.label(inner)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    vox_ml = 16 * np.prod(img.GetSpacing()) / 1000
    objs = ndimage.find_objects(lab)
    cands = []                                              # (voxels, label, center x) of lung-like components
    for k in np.argsort(sizes)[::-1][:10]:
        sl = objs[k]
        m = lab[sl] == k + 1
        hu = a[sl][m]
        _, yy, xx = np.where(m)
        if np.percentile(hu, 90) > -800 and (yy.mean() + sl[1].start) / a.shape[1] < 0.75:
            cands.append((int(sizes[k]), int(k) + 1, xx.mean() + sl[2].start))
    if not cands or cands[0][0] * vox_ml < 200:
        raise ValueError("sin pulmon >= 200 mL en la imagen: la serie no cubre el torax")
    keep = [cands[0][1]]
    mid = a.shape[2] / 2
    if len(cands) > 1 and cands[1][0] >= 0.3 * cands[0][0] and (cands[0][2] - mid) * (cands[1][2] - mid) < 0:
        keep.append(cands[1][1])
    zs = np.where(np.isin(lab, keep).any((1, 2)))[0]
    dz = img.GetSpacing()[2]
    z0, z1 = int(zs.min()), int(zs.max())
    sx = int(img.GetSize()[0] * round(img.GetSpacing()[0], 2) / dc.CURATED_SPACING[0])
    n112 = int(sx * dc.CURATED_SPACING[0] / dc.HEARTLOC_SPACING)
    below = dc.HEARTLOC_SPACING * max(0, (n112 - dc.HEARTLOC_SIZE) // 2) + CROP_BELOW_MM
    lo = max(0, int(round(z0 - below / dz)))
    hi = min(img.GetSize()[2], int(round(z0 + CROP_ABOVE_MM / dz)) + 1)
    lung_ml = float(sum(sizes[k - 1] for k in keep)) * vox_ml
    return img[:, :, lo:hi], dict(lung_z0=z0, lung_z1=z1, lung_n=len(keep), lung_ml=round(lung_ml),
                                  crop_below_mm=below, crop_z0=lo, crop_z1=hi, crop_mm=round((hi - lo) * dz))


_bb_center_original = dc.bb_center


def bb_center_acrin(cube, flags):
    """dc.bb_center with the upper limit of the step-1 mask raised from 999 999 to BBOX_MAX voxels (1.16 L ->
    2.9 L on the curated grid). Large hearts, or hearts with effusion, exceed the original limit, and its fallback (the
    image center) left the sac truncated caudally by the 200-mm window of step 2. Flagged in flags; below 999 999 the
    original function runs unchanged. The weights and the rest of the pipeline are untouched."""
    c = cube.copy()
    c[c == 2] = 0
    c[c > 0] = 1
    s = int(c.sum())
    if 999999 < s <= BBOX_MAX:
        flags.append(f"step2_bbox_large_kept(sum={s})")
        w = np.where(c == 1)
        bb = [int(w[0].min()), int(w[0].max()), int(w[1].min()), int(w[1].max()), int(w[2].min()), int(w[2].max())]
        return [bb[0] + (bb[1] - bb[0]) // 2, bb[2] + (bb[3] - bb[2]) // 2, bb[4] + (bb[5] - bb[4]) // 2]
    return _bb_center_original(cube, flags)


dc.bb_center = bb_center_acrin          # dc.step2 looks up bb_center in the module when it is called


def render_crop(name, img, c, path):
    """Coronal image through the center of the full scan: crop lines (green) and lung lines (yellow)."""
    a = sitk.GetArrayFromImage(img)
    sp = img.GetSpacing()
    fig, ax = plt.subplots(1, 1, figsize=(5, 8))
    ax.imshow(np.clip(a[:, a.shape[1] // 2, :], -1000, 400), cmap="gray", origin="lower", aspect=sp[2] / sp[0])
    for z, col in ((c["crop_z0"], "lime"), (c["crop_z1"] - 1, "lime"), (c["lung_z0"], "yellow"), (c["lung_z1"], "yellow")):
        ax.axhline(z, color=col, lw=1)
    ax.set_title(f"{name} recorte {c['crop_mm']} mm", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def sha12(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="sufijos de PatientID, p. ej. 001 017")
    ap.add_argument("--limit", type=int, default=0, help="0 = todas")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-ram-gb", type=float, default=2.0)
    args = ap.parse_args()
    for d in (QC_EAT, QC_CAC, QC_CROP, MASKS, os.path.dirname(LOG), os.path.dirname(args.out)):
        os.makedirs(d, exist_ok=True)
    series = ct_series(DATA)
    if args.only:
        series = [s for s in series if s["PatientID"][-3:] in set(args.only)]
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out):
        with open(args.out, encoding="utf-8") as fh:
            done = {r["SeriesInstanceUID"] for r in csv.DictReader(fh)}
    todo = [s for s in series if s["SeriesInstanceUID"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    import psutil
    import torch
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    log(f"ACRIN: {len(series)} series CT completas | hechas {len(series) - len([s for s in series if s['SeriesInstanceUID'] not in done])} "
        f"| en esta corrida {len(todo)} | port {sha12(dc.__file__)} torch {torch.__version__} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    heart, cac = dc.DeepHeartModels(), dc.DeepCACModels()
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        for i, s in enumerate(todo, 1):
            wait_for_ram(args.min_ram_gb)
            t = time.time()
            name = f"{s['PatientID'][-3:]}_{s['StudyDate']}"
            row = dict(PatientID=s["PatientID"], StudyDate=s["StudyDate"], SeriesInstanceUID=s["SeriesInstanceUID"],
                       run_at=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", error="")
            try:
                row.update(header(s["sdir"]))
                img = dc.read_dicom_series(s["sdir"])
                row.update(n_slices=img.GetSize()[2], spacing="x".join(f"{v:.3f}" for v in img.GetSpacing()),
                           z_extent_mm=round(img.GetSize()[2] * img.GetSpacing()[2]))
                crop, c = thorax_crop(img)
                row.update(c)
                render_crop(name, img, c, os.path.join(QC_CROP, f"{name}.png"))
                geo = dict(orig_origin=np.array(img.GetOrigin()), orig_spacing=np.array(img.GetSpacing()),
                           orig_size=np.array(img.GetSize()), crop_z=np.array([c["crop_z0"], c["crop_z1"]]))
                del img
                res, m = dc.run_eat(heart, crop)
                row.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items() if k != "flags"})
                row["eat_flags"] = res["flags"]
                row["contrast_by_image"] = "yes" if res["sac_core_median_hu"] >= 60 else "no"
                l1.render(name, m, res, os.path.join(QC_EAT, f"{name}.png"))
                sac_dil = dc.ndimage.binary_dilation(m["sac"], structure=dc.STRUCT6, iterations=l1.SAC_MARGIN_ITER)
                cg = m["cur"]
                save = dict(sac=m["sac"].astype(np.uint8), cur_origin=np.array(cg.GetOrigin()),
                            cur_spacing=np.array(cg.GetSpacing()), cur_size=np.array(cg.GetSize()),
                            cur_direction=np.array(cg.GetDirection()), **geo)
                r, cur, s3 = dc.run_case(cac, crop, return_internals=True)
                row.update(cac_pred=r["cac_pred"], class_pred=r["class_pred"], n_lesions=r["n_lesions"],
                           cac_vol_mm3=r["cac_vol_mm3"], heart_vol_ml_deepcac=round(r["heart_vol_ml"], 1),
                           cac_flags=r["flags"])
                if s3 is not None:
                    assert cur.GetSize() == cg.GetSize()          # same curated grid in both models
                    ags, nls, vols = dc.agatston_within(s3, sac_dil, cur.GetSpacing())
                    row.update(cac_sac=ags, class_sac=dc.ag_class(ags), n_lesions_sac=nls, cac_vol_mm3_sac=vols)
                    save.update(cac_bin=(s3["prob"] >= dc.CAC_THR), bbox=np.array(s3["bbox"]))
                    l1.render_cac(name, m["arr"], s3, sac_dil, r["cac_pred"], ags, os.path.join(QC_CAC, f"{name}.png"))
                np.savez_compressed(os.path.join(MASKS, f"{name}.npz"), **save)
                del crop, m, res, s3, cur, sac_dil
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                log(traceback.format_exc())
            row["secs"] = round(time.time() - t, 1)
            w.writerow(row)
            fh.flush()
            log(f"[{i}/{len(todo)}] {name} {row.get('ManufacturerModelName', '')} recorte {row.get('crop_mm')} mm | "
                f"saco {row.get('sac_ml')} EAT {row.get('eat_ml')} mL {row.get('eat_density_hu')} HU core "
                f"{row.get('sac_core_median_hu')} | CAC {row.get('cac_pred')} saco {row.get('cac_sac')} | "
                f"flags '{row.get('eat_flags', '')}' '{row.get('cac_flags', '')}' | err '{row['error']}' {row['secs']}s")
    log("FIN")


if __name__ == "__main__":
    main()
