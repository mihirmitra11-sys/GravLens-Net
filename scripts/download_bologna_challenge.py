"""
Download helper for the real Bologna Strong Gravitational Lens Finding
Challenge ("Ground Based" training set).

*** RUN THIS LOCALLY, NOT IN A SANDBOXED/RESTRICTED ENVIRONMENT. ***
It needs unrestricted internet access to metcalf1.difa.unibo.it, and the
training pack is on the order of several GB.

Source / rules page (read this first -- links can change):
    http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html

As of the challenge page: the Ground-Based Challenge 1.0 training set is
20,000 objects x 4 bands (I, G, R, U), 101x101 px each, with an ASCII log
file giving the lens/non-lens classification and properties (Einstein
radius, source magnitude, etc.) per object ID.

The portal's download links are click-through (sometimes behind a short
registration form for the full challenge/test sets), so this script can't
hardcode a direct file URL reliably -- grab the exact link yourself:

  1. Open http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html
  2. Under "Challenge 1.0 (ended)" -> "Ground Based Training Set", copy the
     actual download link (right-click -> copy link address).
  3. Paste it into TRAINING_SET_URL below and run this script.
  4. Do the same for the truth-table / log file link if it's separate,
     into CATALOG_URL.

If the original challenge links are dead (the challenge ended in 2020),
check the "Data Set Release" section on the same page for the post-challenge
truth tables ("Ground set 1..5 and Truth table"), which are smaller,
already-labeled release files -- a better starting point than the full
20k training pack.
"""
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

TRAINING_SET_URL = ""   # <- paste the real download link here
CATALOG_URL = ""        # <- paste the log/truth-table link here
OUT_DIR = Path("data/bologna_ground_based")


def download(url, dest):
    if not url:
        print(f"[skip] no URL set for {dest.name} -- see the docstring for how to get it")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)
    if dest.suffix in (".gz", ".tgz"):
        with tarfile.open(dest) as tf:
            tf.extractall(dest.parent)
    elif dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(dest.parent)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download(TRAINING_SET_URL, OUT_DIR / "training_images.tar.gz")
    download(CATALOG_URL, OUT_DIR / "catalog.log")
    print("\nNext: point preprocessing.load_from_fits_dir() at",
          OUT_DIR, "and the downloaded catalog file.")
