# GravLens-Net -- Phase 4: Real Data via VizieR + Legacy Survey Cutouts
#
# DESIGNED FOR GOOGLE COLAB. Not executed/verified in the repo's sandbox --
# both services below need real internet access this sandbox doesn't have.
# Run this cell-by-cell in Colab; if a step errors, paste the error back and
# we'll fix it together (catalog names / column names in particular are the
# most likely thing to need adjusting once we see real query results).
#
# What this does:
#   1. Finds a real confirmed-strong-lens catalog on VizieR (e.g. SLACS)
#      and pulls real RA/Dec coordinates -- no simulation involved.
#   2. Fetches real observed sky cutouts at those coordinates from the DESI
#      Legacy Imaging Surveys cutout service (free, no auth required).
#   3. Fetches cutouts at random sky positions as negatives (weak labels --
#      overwhelmingly non-lens, but not hand-verified; documented limitation).
#   4. Saves everything as .npy so it drops straight into
#      src/preprocessing.py from Phase 2/3.

# ---- Cell 1: setup ----
!pip install astroquery -q

import numpy as np
import requests
from io import BytesIO
from astropy.io import fits
from astroquery.vizier import Vizier
import time

# ---- Cell 2: find a real lens catalog on VizieR ----
# We search rather than hardcode a catalog ID, since guessing IDs from memory
# is a good way to query the wrong table. Inspect the printed list and pick
# the one that's actually the SLACS (or similar) confirmed-lens catalog.
catalog_list = Vizier.find_catalogs("SLACS gravitational lens")
for key, cat in catalog_list.items():
    print(key, "--", cat.description)

# Once you've identified the right key from the printout above, set it here:
CATALOG_KEY = None  # <-- e.g. "J/ApJ/682/964" -- fill in after inspecting Cell 2's output

# ---- Cell 3: pull the actual coordinates ----
Vizier.ROW_LIMIT = -1  # no row limit, get the full catalog
result = Vizier.get_catalogs(CATALOG_KEY)
table = result[0]
print(table.colnames)  # inspect to find the RA/Dec column names
print(table[:5])

# Adjust these two lines once you see the real column names from the printout:
RA_COL = "RAJ2000"
DEC_COL = "DEJ2000"

lens_ra = np.array(table[RA_COL])
lens_dec = np.array(table[DEC_COL])
print(f"Got {len(lens_ra)} real confirmed-lens coordinates")

# ---- Cell 4: cutout fetcher (Legacy Survey, free, no auth) ----
CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

def fetch_cutout(ra, dec, size=101, pixscale=0.262, layer="ls-dr10"):
    params = {"ra": ra, "dec": dec, "size": size, "pixscale": pixscale, "layer": layer}
    r = requests.get(CUTOUT_URL, params=params, timeout=30)
    r.raise_for_status()
    with fits.open(BytesIO(r.content)) as hdul:
        data = hdul[0].data
    # Legacy Survey multi-band cutouts come back as (bands, H, W) -- reduce
    # to a single reference band (or average) to match the Phase 1-3 pipeline
    if data.ndim == 3:
        data = data.mean(axis=0)
    return data.astype(np.float32)

# ---- Cell 5: fetch positives (real lenses) ----
positives = []
for i, (ra, dec) in enumerate(zip(lens_ra, lens_dec)):
    try:
        img = fetch_cutout(ra, dec)
        positives.append(img)
    except Exception as e:
        print(f"  [{i}] failed at ra={ra}, dec={dec}: {e}")
    time.sleep(0.3)  # be polite to the free service
    if i % 20 == 0:
        print(f"fetched {i}/{len(lens_ra)} positives")

positives = np.array(positives)
print("positive cutouts fetched:", positives.shape)

# ---- Cell 6: fetch negatives (random sky positions -- weak labels) ----
rng = np.random.default_rng(42)
n_neg = len(positives) * 20  # keep some imbalance, but tractable to fetch

neg_ra = rng.uniform(0, 360, n_neg)
neg_dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_neg))) * 0.5  # bias toward Legacy Survey's declination footprint

negatives = []
for i, (ra, dec) in enumerate(zip(neg_ra, neg_dec)):
    try:
        img = fetch_cutout(ra, dec)
        if img is not None and np.isfinite(img).any():
            negatives.append(img)
    except Exception as e:
        pass  # blank/off-footprint cutouts are expected and fine to skip
    time.sleep(0.3)
    if i % 50 == 0:
        print(f"fetched {i}/{n_neg} negative attempts, {len(negatives)} kept")

negatives = np.array(negatives)
print("negative cutouts fetched:", negatives.shape)

# ---- Cell 7: assemble + save ----
images = np.concatenate([positives, negatives], axis=0)
labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])

np.save("gravlens_real_images.npy", images)
np.save("gravlens_real_labels.npy", labels)
print("Saved gravlens_real_images.npy / gravlens_real_labels.npy")
print(f"Total: {len(images)}  positives: {int(labels.sum())}  negatives: {int((1-labels).sum())}")

# Next: download these two .npy files from Colab and either
#   (a) upload them back into our chat so I can push the retrain + results
#       to GitHub, or
#   (b) run src/preprocessing.py's standardize()/stratified_split() and the
#       Phase 3b model-building code here in Colab yourself, then share the
#       printed metrics so I can log them to the README.
