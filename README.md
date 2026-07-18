# GravLens-Net 🔭

Deep learning pipeline for detecting strong gravitational lenses in
wide-field galaxy survey imaging.

> **Status:** 🚧 Phase 4 in progress — the original Bologna Challenge
> download links are dead (confirmed), so real data is being sourced instead
> from live public services (VizieR real lens catalogs + DESI Legacy Survey
> cutouts), run from Google Colab since this repo's dev sandbox can't reach
> either service. See **Data Access Note** below. Part of the SpaceAI
> portfolio alongside
> [LunarCrater-Net](https://github.com/mihirmitra11-sys/LunarCrater-Net)
> and [ExoTransit-Net](https://github.com/mihirmitra11-sys/ExoTransit-Net).

## Data Access Note

The original Bologna Strong Gravitational Lens Finding Challenge download
links (`metcalf1.difa.unibo.it`) are dead as of this writing — confirmed
directly, not just a sandbox access issue. Phase 4 instead uses:
- **VizieR** (`astroquery`) for real confirmed-lens coordinates (e.g. SLACS)
- **DESI Legacy Imaging Surveys cutout service** for real observed sky
  cutouts at those coordinates (free, no registration)

See [`scripts/phase4_real_data_colab.py`](scripts/phase4_real_data_colab.py)
— written for Google Colab, since neither service is reachable from this
repo's own dev environment either.

## Results

| Phase | Data | Precision | Recall | F1 |
|-------|------|-----------|--------|-----|
| Phase 1 baseline | Synthetic, 64×64, bold arcs, 1:50 | 1.000 | 0.750 | 0.857 |
| Phase 3a — naive retrain | Synthetic v2, 101×101, PSF+masked, fainter arcs, 1:50, same arch as Phase 1 | 0.029 | 0.750 | 0.057 |
| **Phase 3b — augmented + BN/GAP** | Same v2 data, 1:20, augmented positives, BatchNorm+GAP arch | 0.084 | 0.867 | 0.154 |

**The honest story:** Phase 1's strong numbers came from an easy synthetic
task (bold, high-contrast arcs), not a strong model. Phase 2's realism
upgrades (PSF blur, masking, fainter arcs matching the real challenge spec)
collapsed F1 to 0.057 with no other changes. Phase 3's fixes — positive-class
augmentation and a BatchNorm/GAP architecture better suited to rare, subtle
signals — recovered close to 3× the F1 (0.057 → 0.154) but land nowhere near
Phase 1, which is the expected and correct outcome for a genuinely hard
rare-event problem. Precision remains the weak point throughout (model
overpredicts positives); threshold-based F1 is also noisy at ~15 test
positives, so Phase 4 switches primary evaluation to ROC-AUC.

## Notebooks

| Notebook | Description |
|----------|--------------|
| [`notebooks/Phase1_baseline.ipynb`](notebooks/Phase1_baseline.ipynb) | SIS-lensing data simulation, sanity-check visualization, baseline CNN training + evaluation |
| [`notebooks/Phase2_data_pipeline.ipynb`](notebooks/Phase2_data_pipeline.ipynb) | Realistic-format simulator v2 (101×101, PSF, masking), preprocessing pipeline, real-data on-ramp |
| [`notebooks/Phase3_imbalance_architecture.ipynb`](notebooks/Phase3_imbalance_architecture.ipynb) | Naive-retrain cost measurement, positive-class augmentation, BatchNorm/GAP architecture |

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
