#!/usr/bin/env python3
"""MIMIC‑Echo‑IV view‑classification – folder‑at‑a‑time batching
-----------------------------------------------------------------
Processes **all DICOMs in one folder as a single GPU batch**, then writes
/ merges exactly one `results.json` + `failed.txt` for that folder.

Advantages
==========
* No inter‑folder mixing ⇒ simpler bookkeeping.
* Disk I/O drops to one write per folder.
* Still leverages DataLoader workers for parallel decode.

Caveat: very large folders must still fit in GPU memory. If you hit OOM,
set `MAX_FOLDER_BATCH` (env or code) – the script will micro‑batch within
that folder.
"""

from __future__ import annotations
import json, os, math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch, torchvision
import numpy as np, pydicom
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import utils, video_utils

# ─── CONFIG ──────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output")); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DONE_DIRS_FILE     = Path("done_dirs.txt")
PROCESSED_DCM_FILE = Path("processed_dcms.txt")

DEVICE        = torch.device("cuda", 0)
MAX_FOLDER_BATCH = 128
NUM_WORKERS     = 28
PREFETCH        = 4   # few folders in flight

FRAMES_TAKE, SIZE = 32, 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD  = torch.tensor([47.989223, 46.456997, 47.200830]).reshape(3, 1, 1, 1)

# ─── MODEL ───────────────────────────────────────────────────────────────
ckpt  = torch.load("model_data/weights/view_classifier.ckpt", map_location="cpu")
state = {k[6:]: v for k, v in ckpt["state_dict"].items()}
model = torchvision.models.convnext_base()
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,
                                       len(utils.COARSE_VIEWS))
model.load_state_dict(state, strict=False)
model = model.to(DEVICE).half().eval()
if torch.__version__ >= "2":
    model = torch.compile(model)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

_autocast = torch.cuda.amp.autocast

@torch.inference_mode()
def classify_first_frames(tensor_bcthw: torch.Tensor) -> List[str]:
    with _autocast(dtype=torch.float16):
        logits = model(tensor_bcthw[:, :, 0])
    return [utils.COARSE_VIEWS[i] for i in logits.argmax(1).tolist()]

# ─── RESUME STATE ────────────────────────────────────────────────────────
DONE_DIRS      = set(DONE_DIRS_FILE.read_text().splitlines()) if DONE_DIRS_FILE.exists() else set()
PROCESSED_DCMS = set(PROCESSED_DCM_FILE.read_text().splitlines()) if PROCESSED_DCM_FILE.exists() else set()

# ─── DATASET ─────────────────────────────────────────────────────────────
class EchoFolderDataset(IterableDataset):
    """Yields one tuple (vids, metas, keys, rel_dir) **per folder**."""

    tv_resize = torchvision.transforms.Resize(SIZE, antialias=True)

    @staticmethod
    def preprocess_tensor(px: np.ndarray) -> torch.Tensor:
        if px.ndim == 3:
            px = px[..., None]
        vid = torch.from_numpy(px).permute(0, 3, 1, 2).float()
        vid = EchoFolderDataset.tv_resize(vid)
        vid = torchvision.transforms.functional.center_crop(vid, (SIZE, SIZE))
        if vid.size(0) < FRAMES_TAKE:
            vid = torch.cat((vid, torch.zeros(FRAMES_TAKE - vid.size(0), 3, SIZE, SIZE)), 0)
        vid = vid[:FRAMES_TAKE].permute(1, 0, 2, 3)
        return (vid - MEAN).div_(STD).half()

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        patients = sorted(p for p in MOUNT_ROOT.iterdir() if p.is_dir())
        if info is not None:
            patients = patients[info.id::info.num_workers]

        for pat in patients:
            rel_dir = pat.relative_to(MOUNT_ROOT).as_posix()
            if rel_dir in DONE_DIRS:
                continue

            vids, metas, keys = [], [], []
            for dcm_path in pat.rglob("*.dcm"):
                rel_path = dcm_path.relative_to(MOUNT_ROOT).as_posix()
                if rel_path in PROCESSED_DCMS:
                    continue
                try:
                    dcm = pydicom.dcmread(dcm_path, force=True)
                    px  = dcm.pixel_array
                    if px.ndim < 3 or (px.ndim == 3 and px.shape[-1] == 3):
                        raise ValueError("invalid pixel array shape")
                    px  = video_utils.mask_outside_ultrasound(px)
                    vids.append(self.preprocess_tensor(px))
                    metas.append({el.name: el.repval for el in dcm})
                    keys.append(rel_path)
                except Exception:
                    keys.append(rel_path)
                    vids.append(None)  # placeholder for fail
                    metas.append(None)
            if vids:
                yield vids, metas, keys, rel_dir

# ─── HELPERS ─────────────────────────────────────────────────────────────

def _write_folder(rel_dir: str, results: Dict[str, Any], fails: List[str]):
    out_dir = OUTPUT_ROOT / rel_dir; out_dir.mkdir(parents=True, exist_ok=True)

    # results.json (merge)
    r_path = out_dir / "results.json"
    merged = {}
    if r_path.exists():
        try:
            with r_path.open() as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged.update(results)
    if merged:
        with r_path.open("w") as f:
            json.dump(merged, f)

    # failed.txt (merge)
    f_path = out_dir / "failed.txt"
    merged_f = set(fails)
    if f_path.exists():
        try:
            with f_path.open() as f:
                merged_f |= {ln.strip() for ln in f if ln.strip()}
        except Exception:
            pass
    if merged_f:
        with f_path.open("w") as f:
            for k in sorted(merged_f):
                f.write(f"{k}\n")

    # book‑keeping files
    with PROCESSED_DCM_FILE.open("a") as pf:
        for p in list(results.keys()) + list(merged_f):
            if p not in PROCESSED_DCMS:
                pf.write(p + "\n"); PROCESSED_DCMS.add(p)
    with DONE_DIRS_FILE.open("a") as df:
        df.write(rel_dir + "\n"); DONE_DIRS.add(rel_dir)

    print(f"🏁  {rel_dir} done – wrote/merged in one go.")

# ─── MAIN ────────────────────────────────────────────────────────────────

def _init_worker(_: int):
    torch.set_num_threads(1)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ds = EchoFolderDataset()
    dl = DataLoader(ds,
                    batch_size=None,            # dataset already returns a folder
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=PREFETCH,
                    collate_fn=lambda x: x[0],  # unwrap the single item list
                    worker_init_fn=_init_worker)

    for vids, metas, keys, rel_dir in tqdm(dl, unit="folder"):
        # split fails vs good vids
        ok_indices = [i for i, v in enumerate(vids) if v is not None]
        fail_paths = [keys[i] for i, v in enumerate(vids) if v is None]

        results: Dict[str, Any] = {}
        if ok_indices:
            # optional micro‑batching if folder is huge
            step = MAX_FOLDER_BATCH or len(ok_indices)
            preds: List[str] = []
            for i in range(0, len(ok_indices), step):
                idx_chunk = ok_indices[i:i+step]
                tensor = torch.stack([vids[j] for j in idx_chunk]).to(
                    DEVICE, non_blocking=True, memory_format=torch.channels_last)
                preds.extend(classify_first_frames(tensor))

            for idx, view in zip(ok_indices, preds):
                results[keys[idx]] = {"metadata": metas[idx], "predicted_view": view}

        _write_folder(rel_dir, results, fail_paths)

    print("✅  All folders processed – job complete.")


if __name__ == "__main__":
    main()
