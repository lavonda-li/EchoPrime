#!/usr/bin/env python3
"""MIMIC‑Echo‑IV view‑classification  ─ per‑folder outputs (immediate flush)
----------------------------------------------------------------------------
* Resumes safely at both **folder** and **file** granularity.
  - `done_dirs.txt` records completed directories.
  - `processed_dcms.txt` records each *.dcm* file once attempted (success or fail).
* IterableDataset streams lazily; clips are padded/trimmed to 32 frames.
* Works with CUDA.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pydicom
import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import utils
import video_utils

# ─── 1️⃣ CONFIG ────────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DONE_DIRS_FILE      = Path("done_dirs.txt")
PROCESSED_DCM_FILE  = Path("processed_dcms.txt")

HAS_CUDA     = torch.cuda.is_available()
DEVICE       = torch.device("cuda")
BATCH_SIZE   = 64
NUM_WORKERS  = min(24, os.cpu_count() or 1)
PREFETCH     = 16

FRAMES_TAKE  = 32
FRAME_STRIDE = 2
SIZE         = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD  = torch.tensor([47.989223, 46.456997, 47.200830]).reshape(3, 1, 1, 1)

# ─── 2️⃣ LOAD MODEL ───────────────────────────────────────────────────────────
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

# ─── 3️⃣ LOAD RESUME STATE ─────────────────────────────────────────────────────
print("🔔 Loading resume state …")
DONE_DIRS: set[str] = set()
if DONE_DIRS_FILE.exists():
    with DONE_DIRS_FILE.open() as f:
        DONE_DIRS.update(line.strip() for line in f if line.strip())
PROCESSED_DCMS: set[str] = set()
if PROCESSED_DCM_FILE.exists():
    with PROCESSED_DCM_FILE.open() as f:
        PROCESSED_DCMS.update(line.strip() for line in f if line.strip())
print(f"🔔 {len(DONE_DIRS):,} completed dirs; {len(PROCESSED_DCMS):,} processed DICOMs loaded.")

# ─── 4️⃣ ITERABLE DATASET ─────────────────────────────────────────────────────
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
                rel_path = path.relative_to(MOUNT_ROOT).as_posix()
                if rel_path in PROCESSED_DCMS:
                    continue  # already seen
                try:
                    yield self._proc(path)
                except Exception:
                    continue

    @staticmethod
    def _proc(path: Path):
        dcm  = pydicom.dcmread(path, force=True)
        meta = {el.name: el.repval for el in dcm}
        px   = dcm.pixel_array
        if px.ndim < 3 or (px.ndim == 3 and px.shape[-1] == 3):
            raise ValueError("Invalid pixel array shape: %s" % (px.shape,))
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
        rel_dir  = path.parent.relative_to(MOUNT_ROOT).as_posix()
        return vid, meta, rel_path, rel_dir

# ─── 5️⃣ MAIN LOOP ────────────────────────────────────────────────────────────

def _one_thread(_):
    torch.set_num_threads(1)

def main():
    ds = EchoIterableDataset()
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS if HAS_CUDA else 0,
                    pin_memory=HAS_CUDA,
                    persistent_workers=HAS_CUDA,
                    prefetch_factor=PREFETCH if HAS_CUDA else 2,
                    collate_fn=lambda x: tuple(zip(*x)),
                    worker_init_fn=_one_thread if HAS_CUDA else None)

    results_per_dir: Dict[str, Dict[str, Any]] = defaultdict(dict)
    failed_per_dir:  Dict[str, List[str]]      = defaultdict(list)

    for bidx, batch in enumerate(tqdm(dl, desc="🔍 inferring", unit="batch")):
        vids, metas, keys, dirs = batch  # keys is rel_path per file
        vids = torch.stack(list(vids))
        try:
            vids = vids.to(DEVICE, dtype=torch.float16, non_blocking=HAS_CUDA)
            views = classify_first_frames(vids)
        except Exception:
            for k, d in zip(keys, dirs):
                failed_per_dir[d].append(k)
            _flush(results_per_dir, failed_per_dir)
            continue

        for k, m, v, d in zip(keys, metas, views, dirs):
            results_per_dir[d][k] = {"metadata": m, "predicted_view": v}

        _flush(results_per_dir, failed_per_dir)

    print("✅  Run complete.")

# ─── 6️⃣ FLUSH ────────────────────────────────────────────────────────────────

def _flush(results_per_dir: Dict[str, Dict[str, Any]],
           failed_per_dir: Dict[str, List[str]]):
    """Flush batch results, failures, and update progress trackers."""

    processed_files: List[str] = []
    # processed_dirs   = set(results_per_dir.keys()) | set(failed_per_dir.keys())

    # 1. successes
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
        processed_files.extend(res.keys())
        results_per_dir.pop(rel_dir, None)

    # 2. failures
    for rel_dir, fails in list(failed_per_dir.items()):
        out_dir = OUTPUT_ROOT / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        failed_file = out_dir / "failed.txt"
        with failed_file.open("a") as f:
            for fn in fails:
                f.write(fn + "\n")
        processed_files.extend(fails)
        failed_per_dir.pop(rel_dir, None)

    # 3. record processed files
    successes = sum(len(res) for res in results_per_dir.values())
    failures = sum(len(fails) for fails in failed_per_dir.values())
    print(f"🔔 {successes:,} successes, {failures:,} failures in this batch.")
    if processed_files:
        with PROCESSED_DCM_FILE.open("a") as pf:
            for pth in processed_files:
                if pth not in PROCESSED_DCMS:
                    pf.write(pth + "\n")
                    PROCESSED_DCMS.add(pth)

    # 4. directory completion check
    # for rel_dir in processed_dirs:
    #     if rel_dir in DONE_DIRS:
    #         continue
    #     out_dir = OUTPUT_ROOT / rel_dir
    #     total_dcm = sum(1 for _ in (MOUNT_ROOT / rel_dir).glob("*.dcm"))
    #     successes = 0
    #     res_file = out_dir / "results.json"
    #     if res_file.exists():
    #         with res_file.open() as f:
    #             successes = len(json.load(f))
    #     failures = 0
    #     fail_file = out_dir / "failed.txt"
    #     if fail_file.exists():
    #         with fail_file.open() as f:
    #             failures = sum(1 for _ in f)

    #     print(f"{rel_dir}: success={successes}, fail={failures}, total={total_dcm}")

    #     if successes + failures >= total_dcm > 0:
    #         with DONE_DIRS_FILE.open("a") as df:
    #             df.write(rel_dir + "\n")
    #         DONE_DIRS.add(rel_dir)
    #         print(f"🏁  {rel_dir} marked complete.")

if __name__ == "__main__":
    assert HAS_CUDA, "CUDA is required for this script."
    torch.backends.cudnn.benchmark = True
    main()
