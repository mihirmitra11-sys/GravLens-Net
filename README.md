# GravLens-Net 🔭

Deep learning pipeline for detecting strong gravitational lenses in
wide-field galaxy survey imaging.

> **Status:** ✅ Phase 6 complete — redshift-matching negatives to SLACS's
> actual lens redshifts reveals the capstone finding of the real-data
> phases: AUC drops from ~0.99 to ~0.68 once confounds are properly
> controlled. Read **Results** below. Part of the SpaceAI portfolio
> alongside
> [LunarCrater-Net](https://github.com/mihirmitra11-sys/LunarCrater-Net)
> and [ExoTransit-Net](https://github.com/mihirmitra11-sys/ExoTransit-Net).

## Data Access Note

The original Bologna Challenge download links (`metcalf1.difa.unibo.it`) are
confirmed dead. Phase 4 instead sources **real observational data** from two
live public services:
- **VizieR** (`astroquery`) — real confirmed-lens coordinates, SLACS survey
  (Bolton et al. 2008, catalog `J/ApJ/682/964`)
- **DESI Legacy Imaging Surveys cutout service** — real observed sky cutouts
  at those coordinates, free, no registration

[`scripts/phase4_real_data_colab_auto.py`](scripts/phase4_real_data_colab_auto.py)
fetches this (run in Colab, not this repo's dev environment — see script
header). The resulting dataset (131 real lenses, 1310 real negatives) isn't
committed to the repo (57MB, and regeneratable from the script) — re-run it
to reproduce.

## Results

| Phase | Data | Precision | Recall | F1 | ROC-AUC |
|-------|------|-----------|--------|-----|---------|
| Phase 1 baseline | Synthetic, 64×64, bold arcs, 1:50 | 1.000 | 0.750 | 0.857 | — |
| Phase 3a — naive retrain | Synthetic v2, 101×101, PSF+masked, fainter arcs, 1:50 | 0.029 | 0.750 | 0.057 | — |
| Phase 3b — augmented + BN/GAP | Synthetic v2, 1:20, augmented | 0.084 | 0.867 | 0.154 | — |
| Phase 4 — real data (random negatives) | 131 real SLACS lenses + real random-sky negatives | 0.850–0.857 | 0.850–0.900 | 0.850–0.878 | 0.958–0.989 |
| Phase 5 — real data (matched galaxy type) | + matched LRG negatives, SLACS positions excluded | 0.762 | 0.842 | 0.800 | 0.956 |
| **Phase 6 — real data (redshift-matched)** | + negatives redshift-matched to SLACS's actual lens redshifts | 0.513 | 1.000 | 0.678 | **0.682** |

**⚠️ The real headline of Phases 4–6, read together:** every time a
confound gets identified and fixed, the model's apparent performance drops
— from ~0.99 → ~0.96 → **0.68**. That's not the model getting worse; it's
the evaluation becoming honest. AUC 0.682 is only modestly better than
chance (0.5), and with a 45-example test set (20 positives), that number
carries real uncertainty too.

- **Phase 4:** positives are always massive ellipticals, negatives were
  random sky/random galaxies → concentration alone gets AUC ~1.0 (pure
  confound).
- **Phase 5:** matched negatives to the same broad galaxy type (real LRGs)
  → concentration confound fixed (AUC 0.32), but total flux alone still
  gets AUC 0.79 (negatives systematically brighter/bigger — a likely
  redshift mismatch).
- **Phase 6:** matched negatives to SLACS's actual lens redshifts too →
  half-light-radius confound mostly fixed (0.319 → 0.442), but total flux
  confound persists and even strengthens slightly (0.81, flipped direction)
  — most likely because SLACS also selects on velocity dispersion (mass),
  which redshift-matching alone doesn't capture.

**This mirrors what the synthetic experiments found independently:** Phase
3's F1 collapsed from 0.857 to 0.057 once realistic confusers replaced easy
synthetic negatives. Two completely different tracks (synthetic simulation
vs. real observational data) converged on the same conclusion — strong-lens
detection is a genuinely hard, subtle-signal problem, and any result that
looks too easy is worth interrogating for what's actually being measured.

Full diagnostic chain across all three real-data phases:
[`Phase4_real_data.ipynb`](notebooks/Phase4_real_data.ipynb) →
[`Phase5_matched_negatives.ipynb`](notebooks/Phase5_matched_negatives.ipynb) →
[`Phase6_redshift_matched.ipynb`](notebooks/Phase6_redshift_matched.ipynb).

**Phase 7 (open):** the remaining flux confound suggests matching needs to
extend to velocity dispersion (mass), not just redshift — likely requiring
a spectroscopic catalog with sigma measurements rather than a photometric
LRG catalog. Sample size is also a real constraint by this point (131
positives, 167 negatives after strict matching) — worth weighing whether
further matching strictness is worth the shrinking dataset.

## Notebooks

| Notebook | Description |
|----------|--------------|
| [`notebooks/Phase1_baseline.ipynb`](notebooks/Phase1_baseline.ipynb) | SIS-lensing data simulation, sanity-check visualization, baseline CNN training + evaluation |
| [`notebooks/Phase2_data_pipeline.ipynb`](notebooks/Phase2_data_pipeline.ipynb) | Realistic-format simulator v2 (101×101, PSF, masking), preprocessing pipeline, real-data on-ramp |
| [`notebooks/Phase3_imbalance_architecture.ipynb`](notebooks/Phase3_imbalance_architecture.ipynb) | Naive-retrain cost measurement, positive-class augmentation, BatchNorm/GAP architecture |
| [`notebooks/Phase4_real_data.ipynb`](notebooks/Phase4_real_data.ipynb) | Real SLACS + Legacy Survey data, training, and the galaxy-type confound analysis |
| [`notebooks/Phase5_matched_negatives.ipynb`](notebooks/Phase5_matched_negatives.ipynb) | Matched LRG negative sample, confound re-check, and the residual size/flux confound |
| [`notebooks/Phase6_redshift_matched.ipynb`](notebooks/Phase6_redshift_matched.ipynb) | Redshift-matched negatives, full confound re-check, and the capstone AUC-degradation finding |

## Motivation

Strong gravitational lensing — where a massive foreground galaxy or
cluster bends and magnifies the light of a background source into
arcs, rings, or multiple images — is one of the few direct probes of
dark matter distribution and offers independent constraints on the
Hubble constant via time-delay cosmography. Wide-field surveys (KiDS,
DES, HSC, and soon LSST/Euclid) image billions of galaxies, but
confirmed strong lenses are extremely rare (roughly 1 in 10⁴–10⁶
galaxies), making this a severe class-imbalance detection problem —
conceptually similar to the rare-event challenge tackled in
ExoTransit-Net, but in 2D imaging rather than 1D time series.

## Problem Statement

Given a galaxy cutout image, classify whether it contains a strong
gravitational lensing signature (arcs, Einstein rings, multiple
images) versus a non-lensed galaxy. Key challenges:

- **Extreme class imbalance** — true lenses are a vanishing fraction
  of survey catalogs.
- **Visual subtlety** — lensing arcs can be faint, blended with the
  lens galaxy's light, or mimicked by spiral arms, mergers, and
  imaging artifacts (false positives).
- **Domain gap** — models trained on simulated lenses often struggle
  to generalize to real survey cutouts (the same synthetic-to-real
  gap documented in LunarCrater-Net Phase 4).

## Planned Phases

Following the same structured, iterate-and-document approach used in
LunarCrater-Net and ExoTransit-Net:

1. **Phase 1 — Foundations.** Literature review (Metcalf et al. 2019
   lens-finding challenge, Jacobs et al. 2017, Petrillo et al. 2017),
   establish a CNN baseline on a small labeled subset.
2. **Phase 2 — Data pipeline.** Build a loader for simulated lens
   datasets (Bologna Strong Gravitational Lens Finding Challenge) and
   real candidate cutouts (KiDS/DES/Space Warps citizen-science
   catalogs); implement train/val/test splits stratified for rarity.
3. **Phase 3 — Imbalance handling + architecture iteration.** Class
   weighting, oversampling/augmentation of positives, and comparison
   of CNN backbones (custom conv net vs. ResNet-style transfer
   learning) — analogous to the tiling fix that drove LunarCrater-Net's
   32× mAP improvement.
4. **Phase 4 — Real-data validation.** Evaluate on held-out real
   survey candidates, report false-positive/false-negative analysis,
   and quantify the sim-to-real gap.

## Dataset (planned)

- **Bologna Strong Gravitational Lens Finding Challenge** — simulated
  space- and ground-based mock lens/non-lens cutouts with ground truth.
- **Space Warps / Galaxy Zoo lens candidates** — citizen-science-
  vetted candidates for real-world validation.
- **KiDS / DES survey cutouts** — negative examples and additional
  real candidates for the domain-gap analysis in Phase 4.

## Installation

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn scipy astropy
```

## Repository Structure

```
GravLens-Net/
├── notebooks/     # Phase-by-phase Jupyter notebooks
├── src/           # Reusable modules (data_generation, simulate_v2, preprocessing)
├── scripts/       # Standalone scripts (e.g. real-data download helper)
├── data/          # Dataset download/prep scripts (raw data not tracked)
├── models/        # Saved model checkpoints
├── results/       # Metrics, plots, evaluation outputs
├── README.md
├── .gitignore
└── LICENSE
```

## Methodology (planned)

CNN-based binary classifier, starting from a lightweight custom
architecture (mirroring the 1D-CNN approach in ExoTransit-Net) with a
planned upgrade path to a ResNet-style backbone with transfer learning
if the baseline underperforms on real data. Evaluation will prioritize
recall and F1 over raw accuracy, given the class imbalance.

## Future Work

- Compare CNN backbones (ResNet-18/34, EfficientNet) against the
  custom baseline.
- Explore attention/saliency maps for interpretability of what the
  model flags as lens-like.
- Extend to arc-segmentation (not just binary classification) as a
  stretch goal.
- Cross-survey generalization test (train on KiDS, evaluate on DES).

## References

- Metcalf, R. B. et al. (2019). The strong gravitational lens finding
  challenge. *Astronomy & Astrophysics*, 625, A119.
- Jacobs, C. et al. (2017). Finding strong lenses in CFHTLS using
  convolutional neural networks. *MNRAS*, 471(1), 167–181.
- Petrillo, C. E. et al. (2017). Finding strong gravitational lenses
  in the Kilo Degree Survey with Convolutional Neural Networks.
  *MNRAS*, 472(1), 1129–1150.

## Author

Mihir Mitra — [mihirmitra11-sys](https://github.com/mihirmitra11-sys)
