# DeepKriging Solar — Spatiotemporal GHI Generation for the IEEE 9500-Node Feeder

Generating synthetic, spatially-varying solar irradiance (GHI)
at 178 PV locations on the IEEE 9500-node test feeder.

Ground truth comes from 4 real GHI measurement stations (S1, S2, S3, P2).
DeepKriging (Chen et al., *Statistica Sinica* 2024) interpolates from
those 4 points to all 178 PV locations using a multi-resolution Wendland
RBF basis, trained with GOES-18 satellite features and NSRDB background
fields as covariates.

## Pipeline (two stages)

**Stage 1 — spatial generation.** DeepKriging (or IDW, as a baseline)
produces synthetic 5-minute GHI at all 178 PVs.

**Stage 2 — temporal downscaling.** NNRF (`nnrf/`) downscales Stage 1's
output, following Asiedu et al. 2025, using each PV's k=3 nearest
neighbors — confirmed non-circular with respect to Stage 1.

## Repo structure

```
src/
  data_prep/    station alignment, satellite feature extraction, basis
                functions, background fields, training matrix assembly
  train/        train.py (full model), train_nocov.py (basis-only ablation)
  predict/      predict.py, predict_nocov.py — PV-location inference
  correction/   spatial quantile-mapping bias correction (Bailey et al.
                2024 adaptation) + its validation/diagnostic scripts
  baselines/    IDW baseline, 5-method comparison against DeepKriging
  validation/   spatial diversity check, blend-threshold sweep
  viz/          spatial maps, basis/knot-grid diagrams, loss curves
  model.py      DeepKriging architecture (stays at src/ root — imported
                as src.model by train/predict scripts)

nnrf/           Stage 2 temporal downscaling
gee/            Google Earth Engine satellite data extraction (external
                step — run outside this pipeline, download results into
                data/raw/)
configs/config.py   all paths, station coordinates, model hyperparameters

outputs/
  models/, predictions/, validation/, figures/     main pipeline
  *_nocov                                          basis-only ablation
  idw/                                              IDW baseline
  nnrf_dk/, nnrf_idw/                              Stage 2 output
```

## Setup

```bash
conda env create -f environment.yml
conda activate deepkriging
```

GOES-18 extraction (`gee/`) additionally requires a Google Earth Engine
account — see `gee/extract_c13_pixels.py`'s docstring for the personal
OAuth flow.

## Running the pipeline

Run from the repo root so relative paths resolve correctly.

```bash
# 1. Data prep (in order)
python src/data_prep/prepare_station_data.py
python src/data_prep/clearsky_pvlib.py
python src/data_prep/pixel_mapping.py
python gee/extract_c13_pixels.py            # external — GEE export, then
                                              # download CSVs to
                                              # data/raw/goes_c13/extracted_pixels/
python src/data_prep/c13_features.py
python src/data_prep/background_field.py
python src/data_prep/basis_functions.py
python src/data_prep/residuals.py
python src/data_prep/training_matrix.py

# 2. Train + predict (main model)
python src/train/train.py
python src/predict/predict.py

# 3. Bias correction (spatial quantile mapping)
python src/correction/spatial_qm.py
python src/correction/validate_qm_accuracy.py   # honest accuracy check — see caveat below

# 4. Baselines / validation
python src/baselines/idw.py
python src/baselines/baseline_comparison.py
python src/validation/spatial_diversity_check.py

# 5. Ablation (basis functions only, no covariates)
python src/train/train_nocov.py
python src/predict/predict_nocov.py

# 6. Stage 2 — temporal downscaling
python nnrf/nnrf_downscale.py dk
```

## Important caveats

- **Quantile-mapping correction (`src/correction/spatial_qm.py`) improves
  spatial diversity but does not improve, and can worsen, point-in-time
  accuracy at the 4 real stations.** Confirmed via genuine
  leave-one-station-out testing in `validate_qm_accuracy.py` — 0/4
  stations improved, worst for S1 (the geographic and bias outlier).
  Decide whether `ghi_pvs_qm.parquet` feeds Stage 2 with this tradeoff
  in mind, not as a strict accuracy win over `ghi_pvs.parquet`.

- **Station coordinates in `configs/config.py` were corrected** (previously
  rounded, off by up to ~260m) — this affects every IDW weight computed
  from station positions throughout the pipeline. The practical effect is
  likely small relative to the feeder's ~10-20km domain, but a full
  re-run from `background_field.py` onward would be the rigorous fix.

- **The ablation pipeline (`train_nocov.py`) had a critical bug** (fixed):
  `N_COV` was 18, but the model only ever had 15 real covariates, silently
  truncating 3 real basis-function columns from every ablation run.
  Existing `outputs/*_nocov` results were computed before this fix and
  should be regenerated.