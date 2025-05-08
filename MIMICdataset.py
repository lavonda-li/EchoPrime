#!/usr/bin/env python3
import json
import os
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pydicom
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
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
PREFETCH     = 8

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

# ─── 3️⃣ DISCOVER ALREADY‑DONE FOLDERS ────────────────────────────────────────
DONE_DIRS = {p.parent.relative_to(OUTPUT_ROOT).as_posix() for p in OUTPUT_ROOT.rglob("results.json")}
if DONE_DIRS:
    print(f"⚠️  Detected {len(DONE_DIRS):,} completed folders; they will be skipped.")

# ─── 4️⃣ DATASET ──────────────────────────────────────────────────────────────
class EchoDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = [p for p in root.rglob("*.dcm")
                      if p.parent.relative_to(root).as_posix() not in DONE_DIRS]
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
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


def collate(batch):
    vids, metas, keys = zip(*batch)
    return torch.stack(vids), metas, keys

# ─── 5️⃣ MAIN ─────────────────────────────────────────────────────────────────

def main():
    ds = EchoDataset(MOUNT_ROOT)
    if len(ds) == 0:
        print("✅  Nothing to do – all folders already processed.")
        return

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    persistent_workers=True, prefetch_factor=PREFETCH,
                    collate_fn=collate)

    results: Dict[str, Any] = {}
    failed: List[str] = []

    print(f"🔍  Starting inference on {len(ds):,} files in {len(dl):,} batches of size {BATCH_SIZE:,}")
    for batch_idx, (vids, metas, keys) in enumerate(tqdm(dl, desc="🔍 inferring", unit="batch")):
        try:
            vids = vids.half().to(DEVICE, non_blocking=True)
            views = classify_first_frames(vids)
        except Exception:
            failed.extend(keys)
            traceback.print_exc()
            continue

        for k, m, v in zip(keys, metas, views):
            results[k] = {"metadata": m, "predicted_view": v}

        # 🖨️  Verbose feedback per iteration
        print(f"✔️  Finished batch {batch_idx + 1}: {len(keys)} files, total done {len(results)}")

        if len(results) % FLUSH_EVERY == 0:
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
