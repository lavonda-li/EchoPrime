#!/usr/bin/env python3
import json
import os
import traceback
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

# ─── 1️⃣ CONFIG ────────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64
NUM_WORKERS  = min(24, os.cpu_count() or 1)
FLUSH_EVERY  = 512
PREFETCH     = 16

FRAMES_TAKE  = 32
FRAME_STRIDE = 2
SIZE         = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD  = torch.tensor([47.989223, 46.456997, 47.200830]).reshape(3, 1, 1, 1)

# ─── 2️⃣ MODEL ────────────────────────────────────────────────────────────────
ckpt = torch.load("model_data/weights/view_classifier.ckpt", map_location="cpu")
state = {k[6:]: v for k, v in ckpt["state_dict"].items()}
model = torchvision.models.convnext_base()
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,
                                       len(utils.COARSE_VIEWS))
model.load_state_dict(state, strict=False)
model = model.to(DEVICE).half().eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

@torch.inference_mode()
@torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
def classify_first_frames(videos: torch.Tensor) -> List[str]:
    logits = model(videos[:, :, 0])
    idxs = logits.argmax(1).cpu().tolist()
    return [utils.COARSE_VIEWS[i] for i in idxs]

# ─── 3️⃣ DISCOVER COMPLETED FOLDERS ────────────────────────────────────────────
DONE_DIRS = {p.parent.relative_to(OUTPUT_ROOT).as_posix() for p in OUTPUT_ROOT.rglob("results.json")}
if DONE_DIRS:
    print(f"⚠️  Detected {len(DONE_DIRS):,} completed folders; they will be skipped.")
    for d in sorted(DONE_DIRS):
        print(f"✔️  {d} already done — skipping.")

# ─── 4️⃣ ITERABLE DATASET ──────────────────────────────────────────────────────
class EchoIterableDataset(IterableDataset):
    """Streams DICOM files lazily; each worker walks a shard of the tree."""

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, Dict[str, Any], str]]:
        worker_info = torch.utils.data.get_worker_info()
        # Determine which slice of folders this worker will process
        if worker_info is None:
            # single‑process – treat entire tree as one shard
            paths = list(MOUNT_ROOT.rglob("*.dcm"))
        else:
            # multi‑worker – split top‑level patient folders evenly
            patients = sorted([p for p in MOUNT_ROOT.iterdir() if p.is_dir()])
            per_worker = len(patients) // worker_info.num_workers
            remainder  = len(patients) % worker_info.num_workers
            start = worker_info.id * per_worker + min(worker_info.id, remainder)
            end   = start + per_worker + (1 if worker_info.id < remainder else 0)
            assigned = patients[start:end]
            paths = []
            for pat in assigned:
                paths.extend(pat.rglob("*.dcm"))

        for path in paths:
            rel_folder = path.parent.relative_to(MOUNT_ROOT).as_posix()
            if rel_folder in DONE_DIRS:
                continue  # whole folder already processed
            try:
                yield self._process_single(path)
            except Exception:
                # Skip corrupted file but continue streaming
                # traceback.print_exc()
                continue

    @staticmethod
    def _process_single(path: Path) -> Tuple[torch.Tensor, Dict[str, Any], str]:
        dcm  = pydicom.dcmread(path, force=True)
        meta = {el.name: el.repval for el in dcm}
        px   = dcm.pixel_array
        if px.ndim < 3 or px.shape[2] == 3:
            raise ValueError(f"Invalid pixel array shape: {px.shape}")
        if px.ndim == 3:
            px = np.repeat(px[..., None], 3, axis=-1)
        px = video_utils.mask_outside_ultrasound(px)
        if hasattr(video_utils, "crop_and_scale_batch"):
            vid = video_utils.crop_and_scale_batch(px, out_h=SIZE, out_w=SIZE)
        else:
            vid = np.stack([video_utils.crop_and_scale(f) for f in px])
        vid = torch.from_numpy(vid).permute(3, 0, 1, 2).float()
        if vid.shape[1] < FRAMES_TAKE:
            pad = torch.zeros(3, FRAMES_TAKE - vid.shape[1], SIZE, SIZE)
            vid = torch.cat([vid, pad], 1)
        else:
            vid = vid[:, ::FRAME_STRIDE, :, :][:, :FRAMES_TAKE]
        vid.sub_(MEAN).div_(STD)
        return vid, meta, str(path.relative_to(MOUNT_ROOT))

# ─── 5️⃣ MAIN ─────────────────────────────────────────────────────────────────

def _set_one_thread(_):
    torch.set_num_threads(1)  # avoid oversubscription in workers

def main():
    ds = EchoIterableDataset()
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=PREFETCH,
                    collate_fn=lambda x: tuple(zip(*x)),  # preserve tuple list
                    worker_init_fn=_set_one_thread)

    results: Dict[str, Any] = {}
    failed: List[str] = []

    for batch_idx, batch in enumerate(tqdm(dl, desc="🔍 inferring", unit="batch")):
        vids, metas, keys = batch
        vids = torch.stack(list(vids))
        try:
            vids = vids.half().to(DEVICE, non_blocking=True)
            views = classify_first_frames(vids)
        except Exception:
            failed.extend(keys)
            # traceback.print_exc()
            continue
        for k, m, v in zip(keys, metas, views):
            results[k] = {"metadata": m, "predicted_view": v}
        print(f"✔️  Finished batch {batch_idx + 1}: {len(keys)} files, total done {len(results)}")
        # if len(results) % FLUSH_EVERY == 0:
        _flush(results, failed)

    _flush(results, failed)
    print("✅  Complete run. successes=", len(results), "failures=", len(failed))


def _flush(results: Dict[str, Any], failed: List[str]):
    out_file = OUTPUT_ROOT / "results.json"
    failed_file = OUTPUT_ROOT / "failed.txt"
    if out_file.exists():
        with out_file.open() as f:
            existing = json.load(f)
        existing.update(results)
        results_to_write = existing
    else:
        results_to_write = results
    with out_file.open("w") as f:
        json.dump(results_to_write, f, indent=2)
    with failed_file.open("w") as f:
        f.write("\n".join(failed))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
