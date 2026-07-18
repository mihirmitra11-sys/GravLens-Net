# GravLens-Net -- Phase 4: Real Data via VizieR + Legacy Survey Cutouts
#
# FOR GOOGLE COLAB. Just paste this whole thing into one cell (or a few) and
# "Run all" -- no editing needed, it auto-detects catalog/column names.
# If something still errors, paste the error back and we'll adjust it.
#
# At the end it saves two files -- download them from Colab's file browser
# (left sidebar) and upload both back into our chat:
#   gravlens_real_images.npy
#   gravlens_real_labels.npy

!pip install astroquery -q

import numpy as np
import requests
from io import BytesIO
from astropy.io import fits
from astroquery.vizier import Vizier
import time

# ---------------------------------------------------------------------
# 1. Auto-find a real confirmed strong-lens catalog on VizieR
# ---------------------------------------------------------------------
print("Searching VizieR for a real lens catalog...")
catalog_list = Vizier.find_catalogs("SLACS gravitational lens")
if not catalog_list:
    catalog_list = Vizier.find_catalogs("strong gravitational lens catalog")

catalog_keys = list(catalog_list.keys())
print(f"Found {len(catalog_keys)} candidate catalogs, using the first: {catalog_keys[0]}")
for k in catalog_keys[:5]:
    print(" -", k, ":", catalog_list[k].description[:80])

CATALOG_KEY = catalog_keys[0]

Vizier.ROW_LIMIT = -1
result = Vizier.get_catalogs(CATALOG_KEY)
table = result[0]
print("Columns available:", table.colnames)

# ---------------------------------------------------------------------
# 2. Auto-detect RA/Dec columns from common naming conventions
# ---------------------------------------------------------------------
RA_CANDIDATES = ["RAJ2000", "RA_ICRS", "RAdeg", "_RAJ2000", "RA"]
DEC_CANDIDATES = ["DEJ2000", "DE_ICRS", "DEdeg", "_DEJ2000", "DEC", "DE"]

def find_col(candidates, colnames):
    for c in candidates:
        if c in colnames:
            return c
    # fall back: first column containing "RA" or "DE"
    for c in colnames:
        if any(cand.rstrip("J2000deg_ICRS").upper() in c.upper() for cand in candidates):
            return c
    return None

RA_COL = find_col(RA_CANDIDATES, table.colnames)
DEC_COL = find_col(DEC_CANDIDATES, table.colnames)
print(f"Using RA column: {RA_COL}, Dec column: {DEC_COL}")

if RA_COL is None or DEC_COL is None:
    raise ValueError(
        f"Could not auto-detect RA/Dec columns. Available columns: {table.colnames}. "
        "Paste this list back and we'll fix the detection."
    )

lens_ra = np.array(table[RA_COL], dtype=float)
lens_dec = np.array(table[DEC_COL], dtype=float)
print(f"Got {len(lens_ra)} real confirmed-lens coordinates")

# ---------------------------------------------------------------------
# 3. Cutout fetcher -- DESI Legacy Imaging Surveys (free, no auth)
# ---------------------------------------------------------------------
CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

def fetch_cutout(ra, dec, size=101, pixscale=0.262, layer="ls-dr10"):
    params = {"ra": ra, "dec": dec, "size": size, "pixscale": pixscale, "layer": layer}
    r = requests.get(CUTOUT_URL, params=params, timeout=30)
    r.raise_for_status()
    with fits.open(BytesIO(r.content)) as hdul:
        data = hdul[0].data
    if data is None:
        return None
    if data.ndim == 3:
        data = data.mean(axis=0)
    return data.astype(np.float32)

# ---------------------------------------------------------------------
# 4. Fetch positives (real lenses) -- capped so this doesn't run forever
# ---------------------------------------------------------------------
MAX_POSITIVES = 200  # plenty for a first real-data signal; raise later if it works

positives = []
n_try = min(len(lens_ra), MAX_POSITIVES)
for i in range(n_try):
    try:
        img = fetch_cutout(lens_ra[i], lens_dec[i])
        if img is not None:
            positives.append(img)
    except Exception as e:
        print(f"  [{i}] failed: {e}")
    time.sleep(0.3)
    if i % 20 == 0:
        print(f"positives: fetched {i}/{n_try}, kept {len(positives)}")

positives = np.array(positives)
print("Final positive cutouts:", positives.shape)

# ---------------------------------------------------------------------
# 5. Fetch negatives -- random sky positions (weak labels)
# ---------------------------------------------------------------------
rng = np.random.default_rng(42)
n_neg_target = len(positives) * 10

neg_ra = rng.uniform(0, 360, n_neg_target * 2)   # oversample RA/Dec pairs
neg_dec = rng.uniform(-20, 60, n_neg_target * 2)  # roughly Legacy Survey's footprint

negatives = []
for i in range(len(neg_ra)):
    if len(negatives) >= n_neg_target:
        break
    try:
        img = fetch_cutout(neg_ra[i], neg_dec[i])
        if img is not None and np.isfinite(img).all() and img.std() > 0:
            negatives.append(img)
    except Exception:
        pass
    time.sleep(0.3)
    if i % 50 == 0:
        print(f"negatives: tried {i}, kept {len(negatives)}/{n_neg_target}")

negatives = np.array(negatives)
print("Final negative cutouts:", negatives.shape)

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
images = np.concatenate([positives, negatives], axis=0)
labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])

np.save("gravlens_real_images.npy", images)
np.save("gravlens_real_labels.npy", labels)

print("\nDone.")
print(f"Total: {len(images)}  positives: {int(labels.sum())}  negatives: {int((1-labels).sum())}")
print("Download gravlens_real_images.npy and gravlens_real_labels.npy from the")
print("Colab file browser (left sidebar) and upload both back into our chat.")
