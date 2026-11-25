"""Downloads the public data and the external code used in the paper and places them where the scripts expect them.

1. PulseDB version 1, VitalDB subsets (Kaggle dataset weinanwangrutgers/pulsedb-balanced-training-and-testing,
   CC BY-NC-SA 4.0, one public version, dated 8 December 2022): the five files
   VitalDB_{Train, CalBased_Test, CalFree_Test, AAMI_Cal, AAMI_Test}_Subset.mat (about 19 GB) go to AllDataSet/.
   A Kaggle API token (kaggle.json in ~/.kaggle/, or KAGGLE_USERNAME and KAGGLE_KEY) is used when present.
2. UCI Cuff-Less Blood Pressure Estimation (doi:10.24432/C5B602, CC BY 4.0): the archive (3.1 GB) is downloaded
   and Part_1.mat to Part_4.mat are extracted to data/uci/.
3. ModernTCN, official code of Luo and Wang (ICLR 2024), cloned to external/ModernTCN at commit 56a9a2c.

python data/download_data.py           download everything that is missing
python data/download_data.py --check   only check that every source answers (a few kilobytes, under 30 s)"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PULSEDB_DIR = ROOT / "AllDataSet"
UCI_DIR = ROOT / "data" / "uci"
MODERNTCN_DIR = ROOT / "external" / "ModernTCN"

KAGGLE_DATASET = "weinanwangrutgers/pulsedb-balanced-training-and-testing"
PULSEDB_FILES = ["VitalDB_Train_Subset.mat", "VitalDB_CalBased_Test_Subset.mat", "VitalDB_CalFree_Test_Subset.mat",
                 "VitalDB_AAMI_Cal_Subset.mat", "VitalDB_AAMI_Test_Subset.mat"]
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/" + KAGGLE_DATASET + "/{name}"
UCI_URL = "https://archive.ics.uci.edu/static/public/340/cuff+less+blood+pressure+estimation.zip"
UCI_PARTS = ["Part_1.mat", "Part_2.mat", "Part_3.mat", "Part_4.mat"]
MODERNTCN_URL = "https://github.com/luodhhh/ModernTCN.git"
MODERNTCN_COMMIT = "56a9a2c018385cd5acef015378cae7f084d1b11c"


def kaggle_auth():
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        token = Path.home() / ".kaggle" / "kaggle.json"
        if token.is_file():
            cred = json.loads(token.read_text())
            user, key = cred.get("username"), cred.get("key")
    if not (user and key):
        return None
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def fetch(url, dest, auth=None):
    print("downloading", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-downloader"})
    if auth:
        req.add_header("Authorization", auth)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    part.replace(dest)


def unpack_if_zip(path, members, target):
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        for m in members:
            hit = [n for n in names if n.split("/")[-1] == m and not n.startswith("__MACOSX")]
            if not hit:
                sys.exit(f"{m} not found in {path.name}")
            with zf.open(hit[0]) as src, open(target / m, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
    path.unlink()
    return True


def pulsedb():
    PULSEDB_DIR.mkdir(exist_ok=True)
    missing = [n for n in PULSEDB_FILES if not (PULSEDB_DIR / n).is_file()]
    if not missing:
        print("PulseDB files present")
        return
    auth = kaggle_auth()
    for name in missing:
        tmp = PULSEDB_DIR / (name + ".download")
        fetch(KAGGLE_URL.format(name=name), tmp, auth)
        if not unpack_if_zip(tmp, [name], PULSEDB_DIR):
            tmp.replace(PULSEDB_DIR / name)


def uci():
    UCI_DIR.mkdir(parents=True, exist_ok=True)
    if all((UCI_DIR / p).is_file() for p in UCI_PARTS):
        print("UCI files present")
        return
    tmp = UCI_DIR / "uci.zip"
    fetch(UCI_URL, tmp)
    unpack_if_zip(tmp, UCI_PARTS, UCI_DIR)


def moderntcn():
    if (MODERNTCN_DIR / "ModernTCN-classification" / "models" / "ModernTCN.py").is_file():
        print("ModernTCN present")
        return
    subprocess.run(["git", "clone", MODERNTCN_URL, str(MODERNTCN_DIR)], check=True)
    subprocess.run(["git", "-C", str(MODERNTCN_DIR), "checkout", MODERNTCN_COMMIT], check=True)


def probe(url, auth=None):
    """Asks for the first kilobyte only; returns the HTTP status (or the error) and the reported size."""
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-downloader", "Range": "bytes=0-1023"})
    if auth:
        req.add_header("Authorization", auth)
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            r.read(1024)
            return r.status, r.headers.get("Content-Range") or r.headers.get("Content-Length")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, None


def check():
    auth = kaggle_auth()
    ok = True
    for name in PULSEDB_FILES[-1:]:
        st, size = probe(KAGGLE_URL.format(name=name), auth)
        print(f"Kaggle {name}: {st} {size or ''}")
        ok &= st in (200, 206)
    st, size = probe(UCI_URL)
    print(f"UCI archive: {st} {size or ''}")
    ok &= st in (200, 206)
    r = subprocess.run(["git", "ls-remote", MODERNTCN_URL, "HEAD"], capture_output=True, text=True, timeout=20)
    print(f"ModernTCN repository: {'reachable' if r.returncode == 0 else 'NOT reachable'}")
    ok &= r.returncode == 0
    print("all sources answer" if ok else "some sources did not answer")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    pulsedb()
    uci()
    moderntcn()
