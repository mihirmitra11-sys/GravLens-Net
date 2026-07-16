"""
GravLens-Net Phase 2 — data pipeline utilities.

Designed to work on both:
  (a) the synthetic simulator in this repo (simulate_v2.py), and
  (b) real Bologna Ground-Based Challenge cutouts once downloaded locally
      (see scripts/download_bologna_challenge.py for the fetch step).

The real challenge format (per the official portal):
  - 101x101 pixel stamps, four bands: U, G, R, I
  - masked/artifact pixels are set to a fixed value of 100
  - a companion ASCII catalog/log gives lens (n_source_im > 0, mag_eff > 1.6,
    n_pix_source > 20) vs. non-lens labels per object ID

These utilities assume you've already turned whatever you downloaded (FITS
files + catalog) into two numpy arrays: `images` (N, H, W) or (N, H, W, bands)
and `labels` (N,). Point `load_real_dataset()` at your prepared .npy files,
or use `load_from_fits_dir()` if you still have raw FITS + catalog.
"""
import numpy as np
from pathlib import Path

MASK_VALUE = 100.0


def mask_to_nan_then_fill(images, fill="local_median"):
    """Real challenge data flags bad pixels at value 100. Replace them
    instead of letting the model learn 'grey square = lens' as a spurious
    correlation."""
    images = images.astype(np.float32).copy()
    mask = np.isclose(images, MASK_VALUE)
    if not mask.any():
        return images
    if fill == "local_median":
        for i in range(images.shape[0]):
            if mask[i].any():
                med = np.median(images[i][~mask[i]]) if (~mask[i]).any() else 0.0
                images[i][mask[i]] = med
    else:
        images[mask] = 0.0
    return images


def standardize(images):
    mean = images.mean(axis=(1, 2), keepdims=True)
    std = images.std(axis=(1, 2), keepdims=True) + 1e-6
    return (images - mean) / std


def stratified_split(images, labels, val_frac=0.15, test_frac=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed)
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_real_dataset(images_npy_path, labels_npy_path):
    """Load a real dataset already converted to .npy arrays."""
    images = np.load(images_npy_path)
    labels = np.load(labels_npy_path)
    assert len(images) == len(labels), "images/labels length mismatch"
    return images, labels


def load_from_fits_dir(fits_dir, catalog_csv, id_col="ID", label_col="is_lens", band="R"):
    """
    Build (images, labels) arrays from a directory of per-object FITS cutouts
    plus the challenge's catalog file, converted to CSV.

    Expects files named like f"{fits_dir}/{id}_{band}.fits" -- adjust the
    glob pattern below if your downloaded set names files differently.
    """
    from astropy.io import fits
    import pandas as pd

    catalog = pd.read_csv(catalog_csv)
    images, labels = [], []
    for _, row in catalog.iterrows():
        obj_id = row[id_col]
        fpath = Path(fits_dir) / f"{obj_id}_{band}.fits"
        if not fpath.exists():
            continue
        with fits.open(fpath) as hdul:
            images.append(hdul[0].data.astype(np.float32))
        labels.append(int(row[label_col]))
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    # quick self-test against the synthetic simulator
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from simulate_v2 import generate_dataset

    images, labels, _ = generate_dataset(n_neg=200, n_pos=8, seed=3)
    images = mask_to_nan_then_fill(images)
    images = standardize(images)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = stratified_split(images, labels)
    print("train/val/test:", len(y_tr), len(y_val), len(y_te))
    print("positives:", y_tr.sum(), y_val.sum(), y_te.sum())
