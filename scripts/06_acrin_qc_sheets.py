"""Contact sheets for the visual review of the segmentations (analysis plan, section 3): 8 quality-control figures of
the sac and EAT (qc/acrin_eat) per sheet, ordered by patient and date, with the sac volume and its change between
time points in the label. Output: qc/acrin_sheets/hoja_XX.png. The failures are recorded by hand in
results/acrin_visual_qc.csv (PatientID, StudyDate, falla 0/1, motivo).

Usage: python scripts/06_acrin_qc_sheets.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC = os.path.join(ROOT, "qc", "acrin_eat")
OUT = os.path.join(ROOT, "qc", "acrin_sheets")
PER_SHEET = 8


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(ROOT, "results", "acrin_ct_eat_cac.csv"), dtype={"StudyDate": str})
    d = d.sort_values(["PatientID", "StudyDate"])
    d["delta"] = d.groupby("PatientID").sac_ml.diff()
    d["delta"] = d.delta.fillna(-d.groupby("PatientID").sac_ml.diff(-1))
    rows = [r for _, r in d.iterrows() if os.path.exists(os.path.join(QC, f"{r.PatientID[-3:]}_{r.StudyDate}.png"))]
    for s in range(0, len(rows), PER_SHEET):
        chunk = rows[s:s + PER_SHEET]
        fig, ax = plt.subplots(len(chunk), 1, figsize=(16, 4.1 * len(chunk)))
        ax = [ax] if len(chunk) == 1 else ax
        for a, r in zip(ax, chunk):
            a.imshow(plt.imread(os.path.join(QC, f"{r.PatientID[-3:]}_{r.StudyDate}.png")))
            a.axis("off")
            a.set_title(f"{r.PatientID[-3:]}  {r.StudyDate}  {r.ManufacturerModelName}  saco {r.sac_ml:.0f} mL "
                        f"(cambio {r.delta:+.0f})  EAT {r.eat_ml:.0f} mL  flags {r.eat_flags if isinstance(r.eat_flags, str) else ''}",
                        fontsize=13, loc="left", color="darkred" if not 400 <= r.sac_ml <= 1300 else "black")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"hoja_{s // PER_SHEET + 1:02d}.png"), dpi=50)
        plt.close(fig)
    print(f"{len(rows)} figuras en {(len(rows) + PER_SHEET - 1) // PER_SHEET} hojas -> {OUT}")


if __name__ == "__main__":
    main()
