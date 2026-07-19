# GravLens-Net 🔭

Deep learning pipeline for detecting strong gravitational lenses in
wide-field galaxy survey imaging.

> **Status:** 🚧 Phase 5 in progress — building a properly matched negative
> sample (massive ellipticals without confirmed lensing) to answer what
> Phase 4 actually couldn't. See
> [`scripts/phase5_matched_negatives_colab.py`](scripts/phase5_matched_negatives_colab.py).
> Part of the SpaceAI portfolio alongside
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
| **Phase 4a — real data (all)** | 131 real SLACS lenses + 1310 real random-sky negatives | 0.857 | 0.900 | 0.878 | **0.989** |
| **Phase 4b — real data (non-blank negatives)** | Same, 1310 → 398 negatives (blank sky removed) | 0.850 | 0.850 | 0.850 | **0.958** |

**⚠️ Important caveat on the Phase 4 numbers — read before citing these:**
these AUCs are real, but very likely do **not** mean "the model detects
gravitational lenses." SLACS positives are all massive elliptical galaxies
(that's how SLACS selects lens candidates), while the negatives — even after
removing blank sky — are an unmatched random sample of real galaxies of
every type. A concentration check (central flux / wide-annulus flux) found
positives at 9.88 vs. non-blank negatives at 1.10 — the model is most likely
separating "massive elliptical" from "everything else," not detecting arcs.
This is a known, well-documented failure mode in real lens-finding work,
which is why the actual Bologna Challenge and papers like Petrillo et al.
(2017) use negative samples matched in galaxy type to the positives. Ironically,
Phase 1-3's synthetic confuser negatives (all built from the same Sersic
profile family as the lens galaxy) were methodologically closer to correct
than this real-data setup. Full analysis in
[`notebooks/Phase4_real_data.ipynb`](notebooks/Phase4_real_data.ipynb).

**Phase 5 (open):** build a properly matched negative sample — massive
elliptical galaxies without confirmed lensing, ideally drawn from the same
SDSS spectroscopic selection SLACS used — to get a real answer to the
question Phase 4 didn't actually answer.

## Notebooks

| Notebook | Description |
|----------|--------------|
| [`notebooks/Phase1_baseline.ipynb`](notebooks/Phase1_baseline.ipynb) | SIS-lensing data simulation, sanity-check visualization, baseline CNN training + evaluation |
| [`notebooks/Phase2_data_pipeline.ipynb`](notebooks/Phase2_data_pipeline.ipynb) | Realistic-format simulator v2 (101×101, PSF, masking), preprocessing pipeline, real-data on-ramp |
| [`notebooks/Phase3_imbalance_architecture.ipynb`](notebooks/Phase3_imbalance_architecture.ipynb) | Naive-retrain cost measurement, positive-class augmentation, BatchNorm/GAP architecture |
| [`notebooks/Phase4_real_data.ipynb`](notebooks/Phase4_real_data.ipynb) | Real SLACS + Legacy Survey data, training, and the galaxy-type confound analysis |

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
