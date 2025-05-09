#!/usr/bin/env python3
"""MIMIC‐Echo‐IV view‐classification  ─ per‐folder outputs (immediate flush)
----------------------------------------------------------------------------
* Results and failed logs are written **immediately after every batch**
  to mirror the input directory hierarchy, eliminating any risk of data
  loss if the job is interrupted.
* IterableDataset streams lazily; clips are padded/trimmed to 32 frames.
* Works with or without CUDA.
"""

import json
import os
import traceback
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import pydicom
import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import utils
import video_utils

# ─── 1️⃣ CONFIG ─────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DONE_DIRS_FILE = Path("done_dirs.txt")

HAS_CUDA     = torch.cuda.is_available()
DEVICE       = torch.device("cuda")
BATCH_SIZE   = 1
NUM_WORKERS  = min(24, os.cpu_count() or 1)
PREFETCH     = 16

FRAMES_TAKE  = 32
FRAME_STRIDE = 2
SIZE         = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD  = torch.tensor([47.989223, 46.456997, 47.200830]).reshape(3, 1, 1, 1)

# ─── 2️⃣ MODEL ──────────────────────────────────────
ckpt = torch.load("model_data/weights/view_classifier.ckpt", map_location="cpu")
state = {k[6:]: v for k, v in ckpt["state_dict"].items()}
model = torchvision.models.convnext_base()
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(utils.COARSE_VIEWS))
model.load_state_dict(state, strict=False)

model = model.to(DEVICE).half().eval()
if torch.__version__ >= "2":
    model = torch.compile(model)
_autocast = torch.cuda.amp.autocast

@torch.inference_mode()
def classify_first_frames(videos: torch.Tensor) -> List[str]:
    with _autocast(enabled=True, dtype=torch.float16):
        logits = model(videos[:, :, 0])
    idxs = logits.argmax(1).cpu().tolist()
    return [utils.COARSE_VIEWS[i] for i in idxs]

# ─── 3️⃣ DISCOVER COMPLETED FOLDERS ──────────────────────────────
print("🔔 Loading completed folders from output...")
DONE_DIRS = set()
if DONE_DIRS_FILE.exists():
    with DONE_DIRS_FILE.open() as f:
        DONE_DIRS.update(line.strip() for line in f if line.strip())

if DONE_DIRS:
    for d in sorted(DONE_DIRS):
        print(f"✔️  {d} already done — skipping.")
    print(f"⚠️  Detected {len(DONE_DIRS):,} completed folders; they will be skipped.")

    # write the list of completed folders to a file or append to an existing one
    with DONE_DIRS_FILE.open("a") as done_file:
        for d in DONE_DIRS:
            done_file.write(d + "\n")


# ─── 4️⃣ ITERABLE DATASET ──────────────────────────────
class EchoIterableDataset(IterableDataset):
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        patients = sorted([p for p in MOUNT_ROOT.iterdir() if p.is_dir()])
        if worker_info is not None:
            patients = patients[worker_info.id::worker_info.num_workers]
        for pat in patients:
            rel_pat = pat.relative_to(MOUNT_ROOT).as_posix()
            if rel_pat in DONE_DIRS:
                continue
            for path in pat.rglob("*.dcm"):
                try:
                    yield self._proc(path)
                except Exception:
                    # traceback.print_exc()
                    continue

    @staticmethod
    def _proc(path: Path):
        dcm  = pydicom.dcmread(path, force=True)
        meta = {el.name: el.repval for el in dcm}
        px   = dcm.pixel_array
        if px.ndim < 3 or (px.ndim == 3 and px.shape[-1] == 3):
            raise ValueError(f"Invalid pixel array shape: {px.shape}")
        if px.ndim == 3:
            px = np.repeat(px[..., None], 3, axis=-1)
        px = video_utils.mask_outside_ultrasound(px)
        if hasattr(video_utils, "crop_and_scale_batch"):
            vid = video_utils.crop_and_scale_batch(px, out_h=SIZE, out_w=SIZE)
        else:
            vid = np.stack([video_utils.crop_and_scale(f) for f in px])
        vid = torch.from_numpy(vid).permute(3, 0, 1, 2).float()
        vid = vid[:, ::FRAME_STRIDE]
        if vid.shape[1] < FRAMES_TAKE:
            pad = torch.zeros(3, FRAMES_TAKE - vid.shape[1], SIZE, SIZE)
            vid = torch.cat((vid, pad), 1)
        else:
            vid = vid[:, :FRAMES_TAKE]
        vid.sub_(MEAN).div_(STD)
        rel_path = path.relative_to(MOUNT_ROOT).as_posix()
        rel_dir  = Path(rel_path).parent.as_posix()
        return vid, meta, rel_path, rel_dir

# ─── 5️⃣ MAIN ─────────────────────────────────────

def _one_thread(_):
    torch.set_num_threads(1)

def main():
    print("🔔 Setting up DataLoader...")
    ds = EchoIterableDataset()
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS if HAS_CUDA else 0,
                    pin_memory=HAS_CUDA,
                    persistent_workers=HAS_CUDA,
                    prefetch_factor=PREFETCH if HAS_CUDA else 2,
                    collate_fn=lambda x: tuple(zip(*x)),
                    worker_init_fn=_one_thread if HAS_CUDA else None)

    print(f"🔔 Starting inference loop on {DEVICE} with batch size {BATCH_SIZE}...")
    results_per_dir: Dict[str, Dict[str, Any]] = defaultdict(dict)
    failed_per_dir:  Dict[str, List[str]]      = defaultdict(list)
    total_done = 0

    for bidx, batch in enumerate(tqdm(dl, desc="🔍 inferring", unit="batch")):
        vids, metas, keys, dirs = batch
        vids = torch.stack(list(vids))
        try:
            vids = vids.to(DEVICE, dtype=torch.float16 if HAS_CUDA else torch.float32, non_blocking=HAS_CUDA)
            views = classify_first_frames(vids)
        except Exception:
            print(f"❌ Exception in batch {bidx+1} — flushing partial results.")
            for k, d in zip(keys, dirs):
                failed_per_dir[d].append(k)
            # traceback.print_exc()
            _flush(results_per_dir, failed_per_dir)
            continue

        for k, m, v, d in zip(keys, metas, views, dirs):
            results_per_dir[d][k] = {"metadata": m, "predicted_view": v}
            total_done += 1

        _flush(results_per_dir, failed_per_dir)
        print(f"✔️  Finished batch {bidx+1}: {len(keys)} files, total done {total_done}")

    print("✅  Complete run. Total successful samples:", total_done)

def _flush(results_per_dir: Dict[str, Dict[str, Any]],
           failed_per_dir: Dict[str, List[str]]):
    for rel_dir, res in list(results_per_dir.items()):
        out_dir = OUTPUT_ROOT / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.json"
        if out_file.exists():
            with out_file.open() as f:
                existing = json.load(f)
            existing.update(res)
            data = existing
        else:
            data = res
        with out_file.open("w") as f:
            json.dump(data, f, indent=2)
        with DONE_DIRS_FILE.open("a") as done_file:
            done_file.write(rel_dir + "\n")
        results_per_dir.pop(rel_dir, None)

    for rel_dir, fails in list(failed_per_dir.items()):
        out_dir = OUTPUT_ROOT / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        failed_file = out_dir / "failed.txt"
        with failed_file.open("a") as f:
            for fn in fails:
                f.write(fn + "\n")
        failed_per_dir.pop(rel_dir, None)

if __name__ == "__main__":
    assert HAS_CUDA, "CUDA is required for this script."
    torch.backends.cudnn.benchmark = True
    main()
