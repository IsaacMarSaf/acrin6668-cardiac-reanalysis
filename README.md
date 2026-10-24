# ACRIN 6668/RTOG 0235: cardiac changes after chemoradiation and survival

Analysis code for a secondary analysis of the public ACRIN 6668/RTOG 0235 PET/CT images. Open deep-learning models measure
the pericardial sac, epicardial adipose tissue (EAT) volume and density, coronary artery calcium and a low-attenuation
pericardial rim volume on the CT of the baseline and restaging PET/CT, and whole-sac cardiac FDG uptake on the PET.
Cox models relate the changes to overall survival from the restaging scan, and a published association between cardiac
FDG uptake change and survival in the same trial (Miller et al., *Radiother Oncol* 2022) is re-analyzed on every
possible unit of its hazard ratio.

The accompanying article is under review. The per-series measurements and the code for the figures and tables will be
added on publication.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/01`–`04` | Image measurements: thorax crop, sac, EAT and calcium on the CT (01), quality-control flags (02), clinical cohort from the trial forms (03), rim volume and cardiac SUV from the PET (04) |
| `scripts/05`–`11` | Survival analysis (05), visual-review sheets (06), equivalence and Bayesian bounds (07), positive controls (08), interactions (09), interval and technique (10), re-analysis of the published SUV finding (11) |
| `lib/qc_render.py` | Quality-control figures used by step 01 |
| `data/tcia_series.csv` | The CT and PET series used, by SeriesInstanceUID |

## Data

The images and the de-identified case report forms are public in the TCIA collection
[ACRIN-NSCLC-FDG-PET](https://doi.org/10.7937/tcia.2019.30ilqfcl) (CC BY 3.0). Download the series listed in
`data/tcia_series.csv` with the NBIA Data Retriever and the two clinical data archives of the collection, and arrange
them as follows (or set `ACRIN_DATA` to the folder that holds them):

```
data/
  manifest-acrin6668-paired-petct/            NBIA download, with its metadata.csv
  ACRIN-6668-NSCLC-FDG-PET-ACRIN-Data-TCIA-Anonymized-File-set-1/
  ACRIN-6668HB-NSCLC-FDG-PET-ACRIN-Data-TCIA-Anonymized-File-set-2/
```

## Models (not included)

Steps 01 and 04 run [DeepHeartSeg](https://doi.org/10.5281/zenodo.10724420) and
[DeepCAC](https://github.com/AIM-Harvard/DeepCAC) with their published weights, without retraining, through a PyTorch
implementation (`deepcac_torch.py`) that loads those weights unchanged. The models belong to their authors and are not
redistributed here; obtain them from the original repositories under their licenses. Set `DEEPCAC_PORT` to the folder
that holds `deepcac_torch.py` (default `models/`). For questions about this step, open an issue in this repository.

## Setup

Python 3.12:

```
pip install -r requirements.txt
```

## Running the analysis

```
python scripts/01_acrin_eat_cac_run.py            # CT: thorax crop, sac, EAT and calcium (images and models)
python scripts/02_acrin_qc_pairs.py               # quality-control flags and baseline/restaging pairs
python scripts/03_acrin_clinical_cohort.py        # clinical cohort and landmark from the trial forms
python scripts/04_acrin_effusion_pet_suv.py       # rim volume and cardiac SUV (images and models)
python scripts/06_acrin_qc_sheets.py              # contact sheets for the visual review of the contours
python scripts/05_acrin_survival.py               # primary and secondary analyses
python scripts/07_acrin_equivalence_bayes.py      # equivalence and Bayesian bounds (exploratory)
python scripts/08_acrin_controls_descriptive.py   # positive controls and description of the changes
python scripts/09_acrin_subgroups.py              # interactions (exploratory)
python scripts/10_acrin_interval_technique.py     # interval to restaging and acquisition technique (exploratory)
python scripts/11_acrin_metabolic_miller.py       # re-analysis of the published SUV finding (exploratory)
```

Each script describes its inputs and outputs in its header. Outputs are written to `results/`, `figures/` and `qc/`.
Step 05 reads the visual review of the sac contours from `results/acrin_visual_qc.csv` (one row per CT series:
PatientID, StudyDate, `falla` = failure 0/1, `motivo` = reason).

## Reproducibility

- Seed 2026, 2,000 bootstrap resamples and 20 imputations.
- `05_acrin_survival.py --permute` runs the whole analysis on shuffled outcomes, to test the code without looking at
  the results.
- Comments and documentation are in English. Some result keys and log messages are in Spanish, the working language of
  the authors.

## How to cite

- Article: citation pending (required for any use of this code; see `LICENSE`).
- Data: Kinahan P, Muzi M, Bialecki B, Herman B, Coombs L. Data from the ACRIN 6668 trial NSCLC-FDG-PET. Version 2.
  The Cancer Imaging Archive; 2019. https://doi.org/10.7937/tcia.2019.30ilqfcl
- Trial: Machtay M, et al. J Clin Oncol 2013;31:3823-30. https://doi.org/10.1200/JCO.2012.47.5947
- Models: Zeleznik R, et al. Nat Commun 2021;12:715 (DeepCAC); Foldyna B, et al. Commun Med 2024;4:44 (DeepHeartSeg).

## License

All rights reserved (see `LICENSE`). The code is shared so that reviewers and readers can inspect it and reproduce the
analyses of the article; any other use, copying, modification or redistribution requires the authors' written
permission (open an issue to ask), and any work that uses the code must cite the article. An open license will be
adopted when the article is published. DeepCAC, DeepHeartSeg and their weights are not part of this repository and
keep their own licenses.

## Acknowledgments

We thank the ACRIN 6668/RTOG 0235 investigators and patients, The Cancer Imaging Archive, and the developers of DeepCAC
and DeepHeartSeg. The repository was organized, documented and uploaded with the help of Claude Code (Anthropic).
