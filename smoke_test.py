"""Quick check of the package (about 30 s, no dataset needed): every script compiles, the shared model module
imports, a stored network loads and runs on a dummy batch, the reference results and splits are readable, the
protocol matches its published hash, and every data source answers (a few kilobytes).
python smoke_test.py"""
import hashlib
import json
import py_compile
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "f381b33934819a8f7bf1348f5209a5783609fe589701aac28e7a6192636fc0cc"
t0 = time.perf_counter()
fails = []


def step(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


scripts = sorted((ROOT / "scripts").glob("*.py"))
bad = []
for f in scripts:
    try:
        py_compile.compile(str(f), doraise=True)
    except py_compile.PyCompileError:
        bad.append(f.name)
step("scripts compile", not bad, f"({len(scripts)} files{'; failed: ' + ', '.join(bad) if bad else ''})")

sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from network_loader import load_q1_module  # noqa: E402

q1 = load_q1_module()
ck = torch.load(ROOT / "runs" / "s4_pat_control" / "WS_PAIRED_weights.pt", map_location="cpu", weights_only=False)
model = q1.RawResNet1D(dropout=0.5)
model.load_state_dict(ck["state_dict"])
model.eval()
x = torch.from_numpy(np.random.default_rng(0).standard_normal((4, 2, 1250)).astype(np.float32))
with torch.no_grad():
    y = model(x)
n_par = sum(p.numel() for p in model.parameters())
step("stored tracker loads and runs", tuple(y.shape) == (4, 3) and bool(torch.isfinite(y).all()),
     f"(output {tuple(y.shape)}, {n_par / 1e6:.1f} M parameters)")

ref = json.loads((ROOT / "reports_reference" / "s104_flare_branches.json").read_text())
sbp, dbp = ref["SBP"]["calfree"]["FLARE"][0], ref["DBP"]["calfree"]["FLARE"][0]
step("reference results readable", round(sbp, 2) == 11.28 and round(dbp, 2) == 7.51,
     f"(FLARE CalFree MAE {sbp:.2f} / {dbp:.2f} mmHg)")

n_calfree = len(json.loads((ROOT / "splits" / "calfree_test_subjects.json").read_text()))
step("splits readable", n_calfree == 144, f"({n_calfree} CalFree subjects)")

h = hashlib.sha256((ROOT / "protocol" / "protocol_aami_confirmatory.md").read_bytes()).hexdigest()
step("protocol hash", h == PROTOCOL_SHA256)

r = subprocess.run([sys.executable, str(ROOT / "data" / "download_data.py"), "--check"], capture_output=True,
                   text=True, timeout=60)
print("    " + r.stdout.strip().replace("\n", "\n    "))
step("data sources answer", r.returncode == 0)

print(f"{len(fails)} failed, {time.perf_counter() - t0:.1f} s")
sys.exit(1 if fails else 0)
