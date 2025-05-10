#!/usr/bin/env python3
"""MIMIC‑Echo‑IV view‑classification
-------------------------------------------------------------
Safe‑resume at directory **and** file level:
* `done_dirs.txt`   – completed folders
* `processed_dcms.txt` – every DICOM attempted (success or fail)

Optimised for a **single NVIDIA T4**:
* Batch 64, FP16, cuDNN benchmark on
* 16 dataloader workers, prefetch 8, pinned memory
* Vectorised, batched torchvision preprocessing
* Slice first frame **on GPU** to minimise host work
"""

from __future__ import annotations
import json, os, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import torch, torchvision
import numpy as np, pydicom
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import utils, video_utils

# ─── 1️⃣ CONFIG ───────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DONE_DIRS_FILE     = Path("done_dirs.txt")
PROCESSED_DCM_FILE = Path("processed_dcms.txt")

# GPU & batching tuned for 1× T4 (16 GB)
DEVICE       = torch.device("cuda")
BATCH_SIZE   = 64            # adjust if you hit OOM
NUM_WORKERS  = 16            # half of n1‑standard‑32 vCPUs
PREFETCH     = 8

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # pick fastest algo per shape

# Pre‑proc constants (FP32, we cast to FP16 later)
FRAMES_TAKE  = 32
FRAME_STRIDE = 2
SIZE         = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3,1,1,1)
STD  = torch.tensor([47.989223, 46.456997, 47.200830]).reshape(3,1,1,1)

# ─── 2️⃣ MODEL ────────────────────────────────────────────────────────────────
ckpt  = torch.load("model_data/weights/view_classifier.ckpt", map_location="cpu")
state = {k[6:]: v for k, v in ckpt["state_dict"].items()}
model = torchvision.models.convnext_base()
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,
                                       len(utils.COARSE_VIEWS))
model.load_state_dict(state, strict=False)
model = model.to(DEVICE).half().eval()
if torch.__version__ >= "2":
    model = torch.compile(model)
_autocast = torch.cuda.amp.autocast

@torch.inference_mode()
def classify_first_frames(batch: torch.Tensor) -> List[str]:
    """batch: [B, C, T, H, W] in FP16 on GPU"""
    with _autocast(enabled=True, dtype=torch.float16):
        logits = model(batch[:, :, 0])       # slice first frame on‑GPU
    idxs = logits.argmax(1).tolist()
    return [utils.COARSE_VIEWS[i] for i in idxs]

# ─── 3️⃣ RESUME STATE ─────────────────────────────────────────────────────────
print("🔄  Loading resume state …")
DONE_DIRS: set[str] = set(DONE_DIRS_FILE.read_text().splitlines()) if DONE_DIRS_FILE.exists() else set()
PROCESSED_DCMS: set[str] = set(PROCESSED_DCM_FILE.read_text().splitlines()) if PROCESSED_DCM_FILE.exists() else set()
print(f"   ➜ {len(DONE_DIRS):,} dirs, {len(PROCESSED_DCMS):,} files already done.")

# ─── 4️⃣ DATASET ──────────────────────────────────────────────────────────────
class EchoIterableDataset(IterableDataset):
    tv_resize = torchvision.transforms.Resize(SIZE, antialias=True)
    @staticmethod
    def preprocess_tensor(px: np.ndarray) -> torch.Tensor:  # [T,H,W] or [T,H,W,1]
        if px.ndim == 3:                                     # [T,H,W] → add channel
            px = px[..., None]
        t,c,h,w = px.shape[0],3,SIZE,SIZE
        vid = torch.from_numpy(px).permute(0,3,1,2).float()     # [T,3,H,W] (untouched H,W)
        vid = EchoIterableDataset.tv_resize(vid)                # vectorised C++ op
        vid = torchvision.transforms.functional.center_crop(vid, (SIZE,SIZE))
        if vid.size(0) < FRAMES_TAKE:
            pad = torch.zeros(FRAMES_TAKE-vid.size(0), 3, SIZE, SIZE)
            vid = torch.cat((vid, pad), 0)
        else:
            vid = vid[:FRAMES_TAKE]
        vid = vid.permute(1,0,2,3)  # [C,T,H,W]
        vid = (vid - MEAN).div_(STD)  # FP32 still
        return vid.half()            # cast once before GPU copy

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        patients = sorted([p for p in MOUNT_ROOT.iterdir() if p.is_dir()])
        if worker_info is not None:
            patients = patients[worker_info.id::worker_info.num_workers]
        for pat in patients:
            rel_pat = pat.relative_to(MOUNT_ROOT).as_posix()
            if rel_pat in DONE_DIRS:
                continue
            for dcm_path in pat.rglob("*.dcm"):
                rel_path = dcm_path.relative_to(MOUNT_ROOT).as_posix()
                if rel_path in PROCESSED_DCMS:
                    continue
                try:
                    dcm = pydicom.dcmread(dcm_path, force=True)
                    px  = dcm.pixel_array
                    if px.ndim < 3 or (px.ndim == 3 and px.shape[-1] == 3):
                        raise ValueError("invalid pixel array shape")
                    px = video_utils.mask_outside_ultrasound(px)
                    vid = self.preprocess_tensor(px)
                    meta = {el.name: el.repval for el in dcm}
                    rel_dir = dcm_path.parent.relative_to(MOUNT_ROOT).as_posix()
                    yield vid, meta, rel_path, rel_dir
                except Exception:
                    # could log here
                    continue

# ─── 5️⃣ MAIN LOOP ────────────────────────────────────────────────────────────

def _one_thread(_):
    torch.set_num_threads(1)

def main():
    ds = EchoIterableDataset()
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=PREFETCH,
                    collate_fn=lambda x: tuple(zip(*x)),
                    worker_init_fn=_one_thread)

    results_per_dir: Dict[str, Dict[str, Any]] = defaultdict(dict)
    fails_per_dir:   Dict[str, List[str]]      = defaultdict(list)

    for batch_idx, (vids, metas, keys, dirs) in enumerate(tqdm(dl, unit="batch")):
        vids = torch.stack(list(vids)).to(DEVICE, non_blocking=True)  # FP16 copy
        try:
            views = classify_first_frames(vids)
        except Exception:
            for k,d in zip(keys, dirs):
                fails_per_dir[d].append(k)
            _flush(results_per_dir, fails_per_dir)
            continue
        for k,m,v,d in zip(keys, metas, views, dirs):
            results_per_dir[d][k] = {"metadata": m, "predicted_view": v}
        _flush(results_per_dir, fails_per_dir)

    print("✅  All done!")

# ─── 6️⃣ FLUSH & BOOK‑KEEP ────────────────────────────────────────────────────

def _flush(res_per_dir: Dict[str, Dict[str, Any]],
           fail_per_dir: Dict[str, List[str]]):
    processed_files: List[str] = []
    dirs_touched = set(res_per_dir.keys()) | set(fail_per_dir.keys())

    # successes
    for rel_dir, res in list(res_per_dir.items()):
        out_dir = OUTPUT_ROOT / rel_dir; out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.json"
        data = res
        if out_file.exists():
            with out_file.open() as f: data |= json.load(f)
        with out_file.open("w") as f: json.dump(data, f, indent=2)
        processed_files.extend(res.keys())
        res_per_dir.pop(rel_dir, None)

    # failures
    for rel_dir, fl in list(fail_per_dir.items()):
        out_dir = OUTPUT_ROOT / rel_dir; out_dir.mkdir(parents=True, exist_ok=True)
        fail_file = out_dir / "failed.txt"
        with fail_file.open("a") as f:
            for k in fl: f.write(k + "\n")
        processed_files.extend(fl)
        fail_per_dir.pop(rel_dir, None)

    # record processed dcms
    if processed_files:
        with PROCESSED_DCM_FILE.open("a") as pf:
            for p in processed_files:
                if p not in PROCESSED_DCMS:
                    pf.write(p + "\n"); PROCESSED_DCMS.add(p)

    # check dir completion
    for rel_dir in dirs_touched:
        if rel_dir in DONE_DIRS: continue
        total = sum(1 for _ in (MOUNT_ROOT/rel_dir).glob("*.dcm"))
        succ  = len(json.load(open(OUTPUT_ROOT/rel_dir/"results.json"))) if (OUTPUT_ROOT/rel_dir/"results.json").exists() else 0
        fail  = sum(1 for _ in open(OUTPUT_ROOT/rel_dir/"failed.txt")) if (OUTPUT_ROOT/rel_dir/"failed.txt").exists() else 0
        print(f"ℹ️  {rel_dir}: success={succ} fail={fail} total={total}")
        if succ + fail >= total > 0:
            with DONE_DIRS_FILE.open("a") as df: df.write(rel_dir + "\n")
            DONE_DIRS.add(rel_dir)
            print(f"🏁  {rel_dir} marked complete.")

if __name__ == "__main__":
    main()