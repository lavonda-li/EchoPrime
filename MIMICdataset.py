#!/usr/bin/env python3
"""MIMIC‑Echo‑IV view‑classification – optimized for a single NVIDIA L4
--------------------------------------------------------------------
* Conditional DataParallel – auto‑no‑op on 1 GPU
* Huge input‑pipeline: many DataLoader workers, large prefetch
* FP16, channels‑last, TensorCore friendly, optional torch.compile
* Batch increased – tune until GPU mem ≈ 80 %
* Flush results every 100 batches to reduce FS overhead
* Requires pylibjpeg (+ libjpeg‑turbo) for fast DICOM decode
"""

from __future__ import annotations
import json, os, time, math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch, torchvision
import numpy as np, pydicom
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import utils, video_utils

# ─── CONFIG ────────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output")); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DONE_DIRS_FILE     = Path("done_dirs.txt")
PROCESSED_DCM_FILE = Path("processed_dcms.txt")

DEVICE = torch.device("cuda")

# Tune these to fill ≈ 80 % of 24 GB L4 mem
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

# Automatic multi‑GPU support
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

_autocast = torch.cuda.amp.autocast

@torch.inference_mode()
def classify_first_frames(batch: torch.Tensor) -> List[str]:
    with _autocast(dtype=torch.float16):
        logits = model(batch[:, :, 0])              # first frame only
    return [utils.COARSE_VIEWS[i] for i in logits.argmax(1).tolist()]

# ─── RESUME STATE ─────────────────────────────────────────────────────────
print("🔄  Loading resume state …")
DONE_DIRS      = set(DONE_DIRS_FILE.read_text().splitlines()) if DONE_DIRS_FILE.exists() else set()
PROCESSED_DCMS = set(PROCESSED_DCM_FILE.read_text().splitlines()) if PROCESSED_DCM_FILE.exists() else set()
print(f"   ➜ {len(DONE_DIRS):,} dirs, {len(PROCESSED_DCMS):,} files already done.")

# ─── DATASET ──────────────────────────────────────────────────────────────
class EchoIterableDataset(IterableDataset):
    tv_resize = torchvision.transforms.Resize(SIZE, antialias=True)

    @staticmethod
    def preprocess_tensor(px: np.ndarray) -> torch.Tensor:  # [T,H,W] or [T,H,W,1]
        if px.ndim == 3:
            px = px[..., None]
        vid = torch.from_numpy(px).permute(0,3,1,2).float()   # [T,3,H,W]
        vid = EchoIterableDataset.tv_resize(vid)
        vid = torchvision.transforms.functional.center_crop(vid, (SIZE,SIZE))
        t_pad = max(0, FRAMES_TAKE - vid.size(0))
        if t_pad:
            vid = torch.cat((vid, torch.zeros(t_pad, 3, SIZE, SIZE)), 0)
        vid = vid[:FRAMES_TAKE].permute(1,0,2,3)             # [C,T,H,W]
        vid = (vid - MEAN).div_(STD)
        return vid.half()

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        patients = sorted([p for p in MOUNT_ROOT.iterdir() if p.is_dir()])
        if worker is not None:
            patients = patients[worker.id::worker.num_workers]
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
                    px  = dcm.pixel_array  # fast via pylibjpeg plugin
                    if px.ndim < 3 or (px.ndim == 3 and px.shape[-1] == 3):
                        raise ValueError("invalid pixel array shape")
                    px = video_utils.mask_outside_ultrasound(px)
                    vid = self.preprocess_tensor(px)
                    meta = {el.name: el.repval for el in dcm}
                    rel_dir = dcm_path.parent.relative_to(MOUNT_ROOT).as_posix()
                    yield vid, meta, rel_path, rel_dir
                except Exception:
                    continue

# ─── UTILITIES ────────────────────────────────────────────────────────────

def _one_thread(_: int):
    torch.set_num_threads(1)

# flush helpers

def _flush(res_per_dir: Dict[str, Dict[str, Any]],
           fail_per_dir: Dict[str, List[str]]):
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

    # processed book‑keep
    if processed:
        with PROCESSED_DCM_FILE.open("a") as pf:
            pf.writelines(p + "\n" for p in processed if p not in PROCESSED_DCMS)
            PROCESSED_DCMS.update(processed)

    # dir completion check
    for rdir in touched:
        if rdir in DONE_DIRS: continue
        total = sum(1 for _ in (MOUNT_ROOT/rdir).glob("*.dcm"))
        succ  = len(json.load(open(OUTPUT_ROOT/rdir/"results.json"))) if (OUTPUT_ROOT/rdir/"results.json").exists() else 0
        fail  = sum(1 for _ in open(OUTPUT_ROOT/rdir/"failed.txt")) if (OUTPUT_ROOT/rdir/"failed.txt").exists() else 0
        if succ + fail >= total > 0:
            with DONE_DIRS_FILE.open("a") as df:
                df.write(rdir + "\n"); DONE_DIRS.add(rdir)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────

def main() -> None:
    ds = EchoIterableDataset()
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=PREFETCH,
                    collate_fn=lambda x: tuple(zip(*x)),
                    worker_init_fn=_one_thread)

    res_per_dir: Dict[str, Dict[str, Any]] = defaultdict(dict)
    fail_per_dir: Dict[str, List[str]]      = defaultdict(list)

    for batch_idx, (vids, metas, keys, dirs) in enumerate(tqdm(dl, unit="batch")):
        vids_dev = torch.stack(list(vids)).to(
            DEVICE, non_blocking=True, memory_format=torch.channels_last)
        try:
            views = classify_first_frames(vids_dev)
        except Exception:
            for k, d in zip(keys, dirs): fail_per_dir[d].append(k)
        else:
            for k, m, v, d in zip(keys, metas, views, dirs):
                res_per_dir[d][k] = {"metadata": m, "predicted_view": v}

        if (batch_idx + 1) % FLUSH_EVERY == 0:
            _flush(res_per_dir, fail_per_dir)

    _flush(res_per_dir, fail_per_dir)
    print("✅  All done!")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()