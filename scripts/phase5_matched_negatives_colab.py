# GravLens-Net -- Phase 5: Matched Negative Sample
#
# FOR GOOGLE COLAB. Paste the whole thing into one cell and run.
#
# Phase 4 found that random-sky / random-galaxy negatives don't isolate real
# lens detection, because SLACS positives are all massive elliptical
# galaxies -- a different population from "any galaxy." This script fetches
# real cutouts of massive elliptical/LRG-type galaxies specifically (the
# same broad population SLACS drew its candidates from), excluding anything
# near a confirmed SLACS lens position, so the negative class is finally
# matched in galaxy type to the positive class.
#
# At the end it saves ONE file:
#   gravlens_matched_negatives.npy
# Download it from Colab's file browser and upload it back into our chat --
# I already have the 131 real SLACS positives from Phase 4, so this is all
# that's needed to build the fairer Phase 5 dataset.

!pip install astroquery -q

import numpy as np
import requests
from io import BytesIO
from astropy.io import fits
from astroquery.vizier import Vizier
import time

# ---------------------------------------------------------------------
# 1. Real SLACS lens positions again (to exclude them from negatives)
# ---------------------------------------------------------------------
SLACS_KEY = "J/ApJ/682/964"
Vizier.ROW_LIMIT = 200000  # cap, not unlimited -- some LRG catalogs have millions of rows
slacs_result = Vizier.get_catalogs([SLACS_KEY])

RA_CANDIDATES = ["_RA", "RAJ2000", "RA_ICRS", "RAdeg", "_RAJ2000", "RA"]
DEC_CANDIDATES = ["_DE", "DEJ2000", "DE_ICRS", "DEdeg", "_DEJ2000", "DECJ2000", "DEC", "DE"]

def find_col(candidates, colnames):
    for c in candidates:
        if c in colnames:
            return c
    return None

slacs_table = None
SLACS_RA_COL = SLACS_DEC_COL = None
for t in slacs_result:
    ra_c = find_col(RA_CANDIDATES, t.colnames)
    de_c = find_col(DEC_CANDIDATES, t.colnames)
    if ra_c and de_c:
        slacs_table, SLACS_RA_COL, SLACS_DEC_COL = t, ra_c, de_c
        break

slacs_ra = np.array(slacs_table[SLACS_RA_COL], dtype=float)
slacs_dec = np.array(slacs_table[SLACS_DEC_COL], dtype=float)
print(f"Loaded {len(slacs_ra)} SLACS positions to exclude from negatives")

# ---------------------------------------------------------------------
# 2. Find a massive elliptical / LRG spectroscopic catalog on VizieR
# ---------------------------------------------------------------------
# We search rather than hardcode, same principle as Phase 4. Full SDSS
# "photometric catalog" hits (billions of rows) return nothing useful
# without extra constraints, so we specifically try curated LRG-selection
# catalogs first, and validate each candidate actually returns usable rows
# with RA/Dec before committing to it -- rather than blindly taking the
# first search hit.
print("\nSearching VizieR for a massive elliptical / LRG catalog...")
search_terms = [
    "Eisenstein 2001 luminous red galaxies",
    "SDSS LRG target selection",
    "SDSS luminous red galaxy spectroscopic",
    "BOSS LOWZ galaxies",
    "SDSS luminous red galaxies",
]

table = None
RA_COL = DEC_COL = None
CATALOG_KEY = None

for term in search_terms:
    found = Vizier.find_catalogs(term)
    if not found:
        continue
    print(f"\nSearch term '{term}' found {len(found)} candidate(s):")
    for k in list(found.keys())[:8]:
        desc = found[k].description or "(no description)"
        print(" -", k, ":", desc[:90])

    for k in found.keys():
        try:
            result = Vizier.get_catalogs([k])
        except Exception as e:
            print(f"   [{k}] failed to fetch: {e}")
            continue
        for t in result:
            if len(t) < 100:  # skip empty/trivial tables
                continue
            ra_c = find_col(RA_CANDIDATES, t.colnames)
            de_c = find_col(DEC_CANDIDATES, t.colnames)
            if ra_c and de_c:
                table, RA_COL, DEC_COL, CATALOG_KEY = t, ra_c, de_c, k
                break
        if table is not None:
            break
    if table is not None:
        break

if table is None:
    raise ValueError(
        "Could not find a usable LRG/elliptical catalog with RA/Dec across "
        "any search term. Paste everything printed above back and we'll "
        "pick a specific catalog key by hand instead of searching."
    )

print(f"\nUsing catalog: {CATALOG_KEY}, RA column: {RA_COL}, Dec column: {DEC_COL} ({len(table)} rows)")
cand_ra = np.array(table[RA_COL], dtype=float)
cand_dec = np.array(table[DEC_COL], dtype=float)

# ---------------------------------------------------------------------
# 3. Exclude anything within 10 arcsec of a confirmed SLACS lens
# ---------------------------------------------------------------------
def min_sep_deg(ra, dec, ref_ra, ref_dec):
    # small-angle approx, fine at these separations
    dra = (ra - ref_ra) * np.cos(np.radians(dec))
    ddec = dec - ref_dec
    return np.sqrt(dra**2 + ddec**2).min()

EXCLUDE_DEG = 10 / 3600  # 10 arcsec
keep_mask = np.array([
    min_sep_deg(ra, dec, slacs_ra, slacs_dec) > EXCLUDE_DEG
    for ra, dec in zip(cand_ra, cand_dec)
])
cand_ra, cand_dec = cand_ra[keep_mask], cand_dec[keep_mask]
print(f"After excluding SLACS positions: {len(cand_ra)} candidates remain")

# ---------------------------------------------------------------------
# 4. Fetch cutouts -- capped, same Legacy Survey service as Phase 4
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

MAX_NEGATIVES = 500  # matched negatives, capped so this finishes in reasonable time

rng = np.random.default_rng(42)
order = rng.permutation(len(cand_ra))[:MAX_NEGATIVES * 2]  # oversample, some will fail/be blank

matched_negatives = []
for count, i in enumerate(order):
    if len(matched_negatives) >= MAX_NEGATIVES:
        break
    try:
        img = fetch_cutout(cand_ra[i], cand_dec[i])
        if img is not None and np.isfinite(img).all() and img.std() >= 0.01:
            matched_negatives.append(img)
    except Exception:
        pass
    time.sleep(0.3)
    if count % 30 == 0:
        print(f"tried {count}, kept {len(matched_negatives)}/{MAX_NEGATIVES}")

matched_negatives = np.array(matched_negatives)
print(f"\nFinal matched negatives: {matched_negatives.shape}")

np.save("gravlens_matched_negatives.npy", matched_negatives)
print("Saved gravlens_matched_negatives.npy")
print("Download it from the Colab file browser and upload it back into our chat.")
