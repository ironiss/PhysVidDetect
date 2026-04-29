# `training_testing/` — train, evaluate, save the classifier

Reads three CSVs from `../feature_data/` (latent / physics / camera), aligns them by video filename, applies cleanup + safe-filtering, splits, trains XGBoost, evaluates under multiple protocols, and exports the artifacts consumed by [`../inference_pipeline/`](../inference_pipeline/).

All scripts use `random_state=42` → numbers are deterministic across runs.

## Folder map

| Subfolder | Script | What it does |
|---|---|---|
| `initial_setup/` | `train_baseline.py` | First-pass XGBoost on the 3 fake gens that were available initially. Reports CV + holdout. |
| `initial_setup/` | `run_stability_selection.py` | Bootstrap stability of feature ranking. |
| `extended_setup/` | `train_extended_baseline.py` | Branch-by-branch ablation (latent / physics / noise / camera) on the full 8-generator dataset. |
| `extended_setup/` | `run_feature_selection.py` | Stability + RFE feature selection. |
| `extended_setup/` | `run_ablation_full.py` | Drop-one-group ablations. |
| `extended_setup/model_tuning/` | `run_xgb_tuning_val.py` | XGB hyperparam grid search on validation. |
| `extended_setup/model_tuning/` | `run_model_comparison.py` | Compare XGB vs LR vs RF. |
| `extended_setup/threshold_tuning/` | `run_threshold_methods.py` | Sweep decision thresholds. |
| `comparison_protocols/` | `run_one_to_many.py` | Train on 1 fake gen, test on the rest. |
| `comparison_protocols/` | `run_two_to_many.py` | Train on 2, test on rest. |
| `comparison_protocols/` | `run_many_to_many.py` | All vs all. |
| `overfit_check/` | `run_overfit_final.py` | Sanity-check the production model on small subsets. |
| `save_model/` | `save_final_model.py` | **Production**: trains and writes `saved_models/final_*`. |
| `save_model/` | `save_models.py` | Trains the production model + 8 LOGO models, writes `saved_models/logo/*`. |
| `save_model/` | `run_final_model.py` | Evaluates LOGO + seen 90/10 (no save). |
| `plots/` | `plot_*.py`, `charts.ipynb` | Learning curves, error analysis, comparison charts. |

`utils.py` holds shared helpers: `load_h5`, `fill_nan`, `safe_auc`, `basic_cleanup`, `cleanup_features`, `stratified_split`, `holdout_split`, `train_and_eval`, plot helpers.

---

## Commands & expected outputs

All commands assume you're in the project root and have `feature_data/` populated (or use the pre-computed CSVs in [`../feature_data/`](../feature_data/)).

### 1. Save the production model — `save_model/save_final_model.py`

Trains XGBoost on the full dataset and saves artifacts to `saved_models/`. Test split is built so the test set is balanced 50 / 50 between real and fake:

- **Fake:** 200 clips per generator × 8 generators = **1600** (per-generator share is in the 7–14 % band, so on average ~10 %).
- **Real:** **1600** clips uniformly sampled from the real pool of ~16k → also ~10 %.
- Together this is a **conditional 90 / 10 train/test split**.

```bash
python training_testing/save_model/save_final_model.py
```

Expected tail:
```
features: 74D, samples: 33276
DynamicCrafter: 200 test from 1917
Latte: 200 test from 1870
...
train: 30076, test: 3200
results: AUC=0.9951, Acc=0.9706, F1=0.9707
per-generator:
DynamicCrafter: AUC=0.9990, Acc=0.9750
Latte: AUC=0.9924, Acc=0.9700
OpenSora: AUC=0.9940, Acc=0.9722
...
<TA-DAM DONE>, saved to <project>/saved_models
```

Outputs in `saved_models/`:
```
final_model.json        final_scaler.pkl       feature_names.json
final_metadata.json     safe_feature_mask.npy  final_train_indices.npy
final_test_indices.npy
```

### 2. Train + evaluate LOGO + production — `save_model/save_models.py`

Trains 1 production model (all data) + 8 LOGO models (one per held-out generator).

```bash
python training_testing/save_model/save_models.py
```

Expected tail:
```
PRODUCTION MODEL (all data):
train AUC (in-sample): 1.0000
LOGO MODELS:
DynamicCrafter: AUC=0.9984
Latte: AUC=0.9597
OpenSora: AUC=0.9892
Pika: AUC=0.9581
SEINE: AUC=0.9730
SVD: AUC=0.9862
VideoCrafter: AUC=0.9961
ZeroScope: AUC=0.9915
mean AUC: 0.9815
min AUC: 0.9581
```

Outputs `saved_models/production_model.json` + 8 `logo/model_holdout_*.json` + corresponding scalers.

### 3. Evaluate (no save) — `save_model/run_final_model.py`

Reports both LOGO and seen-split numbers in one go.

```bash
python training_testing/save_model/run_final_model.py
```

Expected tail:
```
--LOGO--
DynamicCrafter: AUC=0.998, Acc=0.977, F1=0.977
Latte: AUC=0.958, Acc=0.848, F1=0.864
...
Mean AUC: 0.981, Min AUC: 0.954
Mean Acc: 0.918, Mean F1: 0.924
--SEEN (90/10  split)--
AUC: 0.996, Acc: 0.973, F1: 0.972
DynamicCrafter: AUC=0.999, Acc=0.973
...
```

Add `--save-csv` to dump `results/final_model/{logo_results.csv, seen_per_generator.csv}`.

### 4. Initial baseline — `initial_setup/train_baseline.py`

Smaller dataset (only 3 fake gens that were available first), used for early experiments.

```bash
python training_testing/initial_setup/train_baseline.py
```

Expected tail:
```
cross-val results:
acc: 0.921+-0.005   f1: 0.918+-0.005   auc: 0.976+-0.002
best model: xgb (CV AUC=0.976)
test (xgb): acc=0.913, f1=0.909, auc=0.974
cv-test gap: +0.002 -- looks stable
```

### 5. Branch-by-branch ablation — `extended_setup/train_extended_baseline.py`

Trains XGBoost on each feature group separately + several combinations to show how complementary they are.

```bash
python training_testing/extended_setup/train_extended_baseline.py
```

Expected tail:
```
--SEEN--
physics              25D  AUC=0.820
latent               43D  AUC=0.950
noise                16D  AUC=0.800
camera               16D  AUC=0.947
phys+lat             68D  AUC=0.953
phys+lat+noise       84D  AUC=0.963
all                 100D  AUC=0.995
--LOGO--
physics  mean=0.772 min=0.636 ...
latent   mean=0.832 min=0.561 ...
camera   mean=0.922 min=0.869 ...
all      mean=0.978 min=0.938 ...
```

### 6. Feature selection / hyperparam tuning

```bash
# stability selection across bootstrap resamples (slow ~10 min)
python training_testing/extended_setup/run_feature_selection.py

# leave-one-out feature group ablation
python training_testing/extended_setup/run_ablation_full.py

# XGB grid search on validation split
python training_testing/extended_setup/model_tuning/run_xgb_tuning_val.py

# threshold sweep (precision/recall trade-off)
python training_testing/extended_setup/threshold_tuning/run_threshold_methods.py
```

### 7. Comparison protocols

Train on a subset of generators, test on the rest. Three flavours:

```bash
# train on 1 gen, test on remaining 7
python training_testing/comparison_protocols/run_one_to_many.py
# expected: per-gen AUC line, mean ~0.94

# train on 2, test on 6
python training_testing/comparison_protocols/run_two_to_many.py

# all combinations
python training_testing/comparison_protocols/run_many_to_many.py   # slow
```

Sample tail of `run_one_to_many.py`:
```
 train=      DC: AUC=0.973, Acc=0.623
 train=   Latte: AUC=0.890, Acc=0.470
 train=   OSora: AUC=0.977, Acc=0.735
 train=    Pika: AUC=0.881, Acc=0.440
 train=   SEINE: AUC=0.962, Acc=0.616
 train=     SVD: AUC=0.964, Acc=0.606
 train=  VCraft: AUC=0.961, Acc=0.574
 train=      ZS: AUC=0.905, Acc=0.329
mean AUC: 0.939, mean Acc: 0.549
```

### 8. Overfit check

Verifies the final model isn't memorising. Trains on shrinking subsets and watches the gap between train and test AUC.

```bash
python training_testing/overfit_check/run_overfit_final.py
```

### 9. Plots

```bash
python training_testing/plots/plot_learning_curves_epochs.py   # XGB tree-by-tree learning
python training_testing/plots/plot_error_analysis.py           # per-generator misclassifications
python training_testing/plots/plot_overfit_checks.py
```

PNGs land in `training_testing/plots/` next to the scripts.

---

## Notes

- **Reproducibility:** every script seeds `numpy` and `XGBClassifier` with `42`. Re-running on the same `feature_data/*.csv` should reproduce the headline numbers within ~0.005.
- **Class convention:** `1 = real`, `0 = fake`. Don't flip silently.
- **Filtering:** `basic_cleanup` (drop NaN-heavy + correlated) + `safe_filter` (drop generator-leaky, AUC > 0.65 in pairwise gen-vs-gen) → 100 → 74 features. The 74 surviving names are saved in `saved_models/feature_names.json`.
- **Where to look for canonical hyperparameters:** `save_final_model.py` (XGB params) and `final_metadata.json` (recorded after training).
