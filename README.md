# GravLens-Net 🔭

Deep learning pipeline for detecting strong gravitational lenses in
wide-field galaxy survey imaging.

> **Status:** 🚧 Planning phase — repo initialized, dataset acquisition
> and baseline model are next. Part of the SpaceAI portfolio alongside
> [LunarCrater-Net](https://github.com/mihirmitra11-sys/LunarCrater-Net)
> and [ExoTransit-Net](https://github.com/mihirmitra11-sys/ExoTransit-Net).

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
