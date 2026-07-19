# GravLens-Net -- Phase 6: Redshift-Matched Negative Sample
#
# FOR GOOGLE COLAB. Paste the whole thing into one cell and run.
#
# Phase 5 fixed the galaxy-type confound (concentration) but found a
# residual size/flux confound (AUC 0.79 from total flux alone) -- most
# likely because the matched LRG catalog wasn't selected at the same
# REDSHIFT as the SLACS lenses. Closer galaxies look bigger and brighter
# regardless of galaxy type. This script fixes that specifically: it pulls
# each SLACS lens's actual redshift (column 'zFG' in the SLACS catalog) and
# only keeps LRG-catalog candidates whose redshift falls within the same
# range, in addition to excluding confirmed SLACS positions.
#
# At the end it saves ONE file:
#   gravlens_zmatched_negatives.npy
# Download it from Colab's file browser and upload it back into our chat.

!pip install astroquery -q

import numpy as np
import requests
from io import BytesIO
from astropy.io import fits
from astroquery.vizier import Vizier
import time

RA_CANDIDATES = ["_RA", "RAJ2000", "RA_ICRS", "RAdeg", "_RAJ2000", "RA"]
DEC_CANDIDATES = ["_DE", "DEJ2000", "DE_ICRS", "DEdeg", "_DEJ2000", "DECJ2000", "DEC", "DE"]
Z_CANDIDATES = ["zsp", "z_sp", "zSpec", "zphot", "z_phot", "Z", "z", "redshift", "Redshift", "zFG"]

def find_col(candidates, colnames):
    for c in candidates:
        if c in colnames:
            return c
    return None

# ---------------------------------------------------------------------
# 1. Real SLACS lens positions AND redshifts
# ---------------------------------------------------------------------
SLACS_KEY = "J/ApJ/682/964"
Vizier.ROW_LIMIT = 200000
slacs_result = Vizier.get_catalogs([SLACS_KEY])

slacs_table = None
SLACS_RA_COL = SLACS_DEC_COL = SLACS_Z_COL = None
for t in slacs_result:
    ra_c = find_col(RA_CANDIDATES, t.colnames)
    de_c = find_col(DEC_CANDIDATES, t.colnames)
    z_c = find_col(["zFG", "zLens", "z_l", "z"], t.colnames)
    if ra_c and de_c and z_c:
        slacs_table, SLACS_RA_COL, SLACS_DEC_COL, SLACS_Z_COL = t, ra_c, de_c, z_c
        break

if slacs_table is None:
    # fall back: at least get positions, print columns so we can add the
    # right z column name if this happens
    for t in slacs_result:
        ra_c = find_col(RA_CANDIDATES, t.colnames)
        de_c = find_col(DEC_CANDIDATES, t.colnames)
        if ra_c and de_c:
            slacs_table, SLACS_RA_COL, SLACS_DEC_COL = t, ra_c, de_c
            print(f"Could not auto-find a redshift column. Columns available: {t.colnames}")
            print("Paste this list back so we can add the right column name.")
            break

slacs_ra = np.array(slacs_table[SLACS_RA_COL], dtype=float)
slacs_dec = np.array(slacs_table[SLACS_DEC_COL], dtype=float)
if SLACS_Z_COL:
    slacs_z = np.array(slacs_table[SLACS_Z_COL], dtype=float)
    print(f"Loaded {len(slacs_ra)} SLACS positions + redshifts "
          f"(z range: {slacs_z.min():.3f} - {slacs_z.max():.3f})")
else:
    slacs_z = None
    print(f"Loaded {len(slacs_ra)} SLACS positions (no redshift column found -- "
          f"will exclude positions but can't redshift-match without it)")

# ---------------------------------------------------------------------
# 2. Find the same LRG-type catalog as Phase 5, check for a redshift column
# ---------------------------------------------------------------------
print("\nSearching VizieR for a massive elliptical / LRG catalog...")
search_terms = [
    "Eisenstein 2001 luminous red galaxies",
    "SDSS LRG target selection",
    "SDSS luminous red galaxy spectroscopic",
    "BOSS LOWZ galaxies",
    "SDSS luminous red galaxies",
]

table = None
RA_COL = DEC_COL = LRG_Z_COL = None
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
            if len(t) < 100:
                continue
            ra_c = find_col(RA_CANDIDATES, t.colnames)
            de_c = find_col(DEC_CANDIDATES, t.colnames)
            if ra_c and de_c:
                table, RA_COL, DEC_COL, CATALOG_KEY = t, ra_c, de_c, k
                LRG_Z_COL = find_col(Z_CANDIDATES, t.colnames)
                break
        if table is not None:
            break
    if table is not None:
        break

if table is None:
    raise ValueError(
        "Could not find a usable LRG/elliptical catalog. Paste everything "
        "printed above back and we'll pick one by hand."
    )

print(f"\nUsing catalog: {CATALOG_KEY}, RA: {RA_COL}, Dec: {DEC_COL}, "
      f"redshift column: {LRG_Z_COL} ({len(table)} rows)")
if LRG_Z_COL is None:
    print(f"NOTE: no redshift column auto-detected. Columns available: {table.colnames}")
    print("Paste this list back if you want exact redshift matching -- "
          "proceeding WITHOUT redshift filtering for now (same as Phase 5).")

cand_ra = np.array(table[RA_COL], dtype=float)
cand_dec = np.array(table[DEC_COL], dtype=float)
cand_z = np.array(table[LRG_Z_COL], dtype=float) if LRG_Z_COL else None

# ---------------------------------------------------------------------
# 3. Filter: exclude confirmed SLACS positions, keep only redshift-matched
# ---------------------------------------------------------------------
def min_sep_deg(ra, dec, ref_ra, ref_dec):
    dra = (ra - ref_ra) * np.cos(np.radians(dec))
    ddec = dec - ref_dec
    return np.sqrt(dra**2 + ddec**2).min()

EXCLUDE_DEG = 10 / 3600
keep_mask = np.array([
    min_sep_deg(ra, dec, slacs_ra, slacs_dec) > EXCLUDE_DEG
    for ra, dec in zip(cand_ra, cand_dec)
])

if cand_z is not None and slacs_z is not None:
    z_lo, z_hi = slacs_z.min() - 0.02, slacs_z.max() + 0.02
    z_mask = (cand_z >= z_lo) & (cand_z <= z_hi) & np.isfinite(cand_z)
    print(f"\nRedshift-matching to SLACS range [{z_lo:.3f}, {z_hi:.3f}]: "
          f"{z_mask.sum()}/{len(cand_z)} candidates pass")
    keep_mask = keep_mask & z_mask
else:
    print("\nProceeding without redshift filtering (column not found).")

cand_ra, cand_dec = cand_ra[keep_mask], cand_dec[keep_mask]
print(f"After all filtering: {len(cand_ra)} candidates remain")

if len(cand_ra) < 50:
    print("WARNING: very few candidates left. If this is too small, tell me "
          "and we'll widen the redshift window or pick a different catalog.")

# ---------------------------------------------------------------------
# 4. Fetch cutouts -- same Legacy Survey service as before
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

MAX_NEGATIVES = 500

rng = np.random.default_rng(42)
order = rng.permutation(len(cand_ra))[:MAX_NEGATIVES * 2]

zmatched_negatives = []
for count, i in enumerate(order):
    if len(zmatched_negatives) >= MAX_NEGATIVES:
        break
    try:
        img = fetch_cutout(cand_ra[i], cand_dec[i])
        if img is not None and np.isfinite(img).all() and img.std() >= 0.01:
            zmatched_negatives.append(img)
    except Exception:
        pass
    time.sleep(0.3)
    if count % 30 == 0:
        print(f"tried {count}, kept {len(zmatched_negatives)}/{MAX_NEGATIVES}")

zmatched_negatives = np.array(zmatched_negatives)
print(f"\nFinal redshift-matched negatives: {zmatched_negatives.shape}")

np.save("gravlens_zmatched_negatives.npy", zmatched_negatives)
print("Saved gravlens_zmatched_negatives.npy")
print("Download it from the Colab file browser and upload it back into our chat.")
