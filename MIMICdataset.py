#!/usr/bin/env python3
"""MIMIC‑Echo‑IV view‑classification – fast inference edition (bug‑fix)
────────────────────────────────────────────────────────────────────────
* One‑shot manifest + cached JSON to avoid repeated directory walks
* Vectorised skip of already processed paths (done_files set)
* Rank‑mismatch crash fixed (channels_last applied after slicing to 4‑D)
* Optional `--profile` flag prints per‑batch timing; end summary always printed.

Run examples
============
```bash
python mimic_echo_view_classification_fast.py                # fastest mode
python mimic_echo_view_classification_fast.py --profile      # verbose timings
```
"""

from __future__ import annotations
import json, os, time, argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch, torchvision
import numpy as np, pydicom
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm import tqdm

import utils, video_utils

# ─── CONFIG ────────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output")); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DONE_DIRS_FILE     = Path("done_dirs.txt")
PROCESSED_DCM_FILE = Path("processed_dcms.txt")
MANIFEST_FILE      = Path("manifest.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tune these to fill ≈ 80 % of 24 GB L4 mem
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 192))
NUM_WORKERS = min(28, (os.cpu_count() or 32) - 4)
PREFETCH    = 16
FLUSH_EVERY = 100

FRAMES_TAKE, FRAME_STRIDE, SIZE = 32, 2, 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3,1,1,1)
STD  = torch.tensor([47.989223, 46.456997, 47.200830]).reshape(3,1,1,1)

# ─── MODEL ────────────────────────────────────────────────────────────────
ckpt  = torch.load("model_data/weights/view_classifier.ckpt", map_location="cpu")
state = {k[6:]: v for k, v in ckpt["state_dict"].items()}
model = torchvision.models.convnext_base()
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,
                                       len(utils.COARSE_VIEWS))
model.load_state_dict(state, strict=False)
model = model.to(DEVICE).half().eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

if torch.cuda.device_count() > 1 and DEVICE.type == "cuda":
    model = torch.nn.DataParallel(model)

_autocast = torch.cuda.amp.autocast if DEVICE.type == "cuda" else torch.cpu.amp.autocast

@torch.inference_mode()
def classify_first_frames(batch_5d: torch.Tensor) -> List[str]:
    """Input `[B,C,T,H,W]` FP16 on device; returns list of coarse view names."""
    with _autocast(dtype=torch.float16):
        x = batch_5d[:, :, 0].contiguous(memory_format=torch.channels_last)  # [B,C,H,W]
        logits = model(x)
    return [utils.COARSE_VIEWS[i] for i in logits.argmax(1).tolist()]

# ─── RESUME STATE ─────────────────────────────────────────────────────────
print("🔄  Loading resume state …")
DONE_DIRS      = set(DONE_DIRS_FILE.read_text().splitlines()) if DONE_DIRS_FILE.exists() else set()
PROCESSED_DCMS = set(PROCESSED_DCM_FILE.read_text().splitlines()) if PROCESSED_DCM_FILE.exists() else set()
print(f"   ➜ {len(DONE_DIRS):,} dirs, {len(PROCESSED_DCMS):,} files already done.")

# ─── MANIFEST BUILD / LOAD ────────────────────────────────────────────────
print("🔍  Building / loading manifest …", end=" ")
if MANIFEST_FILE.exists():
    with MANIFEST_FILE.open() as f:
        manifest = json.load(f)
    ALL_DCMS      = [MOUNT_ROOT / p for p in manifest["all_dcms"]]
    TOTALS_BY_DIR = manifest["totals_by_dir"]
    print("cached ✅")
else:
    ALL_DCMS: List[Path] = []
    TOTALS_BY_DIR: Dict[str, int] = defaultdict(int)
    for dcm_path in tqdm(MOUNT_ROOT.rglob("*.dcm"), desc=" scanning", unit="file"):
        rel_dir = dcm_path.parent.relative_to(MOUNT_ROOT).as_posix()
        TOTALS_BY_DIR[rel_dir] += 1
        ALL_DCMS.append(dcm_path)
    MANIFEST_FILE.write_text(json.dumps({
        "all_dcms"     : [p.relative_to(MOUNT_ROOT).as_posix() for p in ALL_DCMS],
        "totals_by_dir": TOTALS_BY_DIR,
    }))
    print("new ✅")

TO_DO = [p for p in ALL_DCMS if p.relative_to(MOUNT_ROOT).as_posix() not in PROCESSED_DCMS]
print(f"   ➜ {len(TO_DO):,} remaining files to process.")

# ─── DATASET ──────────────────────────────────────────────────────────────
class EchoIterableDataset(IterableDataset):
    tv_resize = torchvision.transforms.Resize(SIZE, antialias=True)

    def __init__(self, paths: List[Path]):
        self.paths = paths

    @staticmethod
    def preprocess_tensor(px: np.ndarray) -> torch.Tensor:
        if px.ndim == 3:
            px = px[..., None]
        vid = torch.from_numpy(px).permute(0,3,1,2).float()
        vid = EchoIterableDataset.tv_resize(vid)
        vid = torchvision.transforms.functional.center_crop(vid, (SIZE,SIZE))
        t_pad = max(0, FRAMES_TAKE - vid.size(0))
        if t_pad:
            vid = torch.cat((vid, torch.zeros(t_pad, 3, SIZE, SIZE)), 0)
        vid = vid[:FRAMES_TAKE].permute(1,0,2,3)
        vid = (vid - MEAN).div_(STD)
        return vid.half()

    def __iter__(self):
        worker = get_worker_info()
        paths_slice = self.paths[worker.id::worker.num_workers] if worker else self.paths
        for dcm_path in paths_slice:
            rel_path = dcm_path.relative_to(MOUNT_ROOT).as_posix()
            rel_dir  = dcm_path.parent.relative_to(MOUNT_ROOT).as_posix()
            try:
                dcm = pydicom.dcmread(dcm_path, force=True)
                px  = dcm.pixel_array
                if px.ndim < 3 or (px.ndim == 3 and px.shape[-1] == 3):
                    raise ValueError
                px = video_utils.mask_outside_ultrasound(px)
                vid = self.preprocess_tensor(px)
                meta = {el.name: el.repval for el in dcm}
                yield vid, meta, rel_path, rel_dir
            except Exception:  # decode failure or shape error
                yield None, None, rel_path, rel_dir

# ─── UTILITIES ────────────────────────────────────────────────────────────

def _one_thread(_: int):
    torch.set_num_threads(1)


def _flush(res_per_dir: Dict[str, Dict[str, Any]], fail_per_dir: Dict[str, List[str]]):
    processed: List[str] = []
    touched = set(res_per_dir) | set(fail_per_dir)

    # successes
    for rdir, res in list(res_per_dir.items()):
        out_dir = OUTPUT_ROOT / rdir; out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.json"
        data = res
        if out_file.exists():
            with out_file.open() as f: data |= json.load(f)
        with out_file.open("w") as f: json.dump(data, f)
        processed.extend(res.keys()); res_per_dir.pop(rdir, None)

    # failures
    for rdir, fl in list(fail_per_dir.items()):
        out_dir = OUTPUT_ROOT / rdir; out_dir.mkdir(parents=True, exist_ok=True)
        fail_file = out_dir / "failed.txt"
        with fail_file.open("a") as f:
            f.writelines(k + "\n" for k in fl)
        processed.extend(fl); fail_per_dir.pop(rdir, None)

    # append to global processed list
    if processed:
        with PROCESSED_DCM_FILE.open("a") as pf:
            pf.writelines(p + "\n" for p in processed)

    # check directory completion via cached totals
    for rdir in touched:
        if rdir in DONE_DIRS:  # already marked done
            continue
        total = TOTALS_BY_DIR.get(rdir, 0)
        succ  = len(json.load(open(OUTPUT_ROOT/rdir/"results.json"))) if (OUTPUT_ROOT/rdir/"results.json").exists() else 0
        fail  = sum(1 for _ in open(OUTPUT_ROOT/rdir/"failed.txt")) if (OUTPUT_ROOT/rdir/"failed.txt").exists() else 0
        if succ + fail >= total > 0:
            with DONE_DIRS_FILE.open("a") as df:
                df.write(rdir + "\n")
            DONE_DIRS.add(rdir)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MIMIC‑Echo‑IV fast inference")
    parser.add_argument("--profile", action="store_true", help="Print per‑batch timings")
    args = parser.parse_args()

    ds = EchoIterableDataset(TO_DO)
    dl = DataLoader(ds,
                    batch_size=BATCH
