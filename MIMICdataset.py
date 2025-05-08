#!/usr/bin/env python3
import json
import os
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pydicom  # pylibjpeg rockets the decode speed ⚡️
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import utils  # project‑specific helpers (COARSE_VIEWS etc.)
import video_utils

# ─── 1️⃣ CONFIG ────────────────────────────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 32                      # safe on 24 GB w/ FP16; tune for your GPU
NUM_WORKERS  = min(32, os.cpu_count() or 1)
FLUSH_EVERY  = 256                    # serialise results every N samples

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
model = model.to(DEVICE).half().eval()   # FP16 weights
if torch.__version__ >= "2":
    model = torch.compile(model)          # fuse + optimise

@torch.inference_mode()
@torch.cuda.amp.autocast(dtype=torch.float16)
def classify_first_frames(videos: torch.Tensor) -> List[str]:
    """Predict coarse view of batch (uses only first frame)."""
    logits = model(videos[:, :, 0])  # [B,C,H,W]
    idxs = logits.argmax(1).cpu().tolist()
    return [utils.COARSE_VIEWS[i] for i in idxs]

# ─── 3️⃣ DATASET ───────────────────────────────────────────────────────────────
class EchoDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(root.rglob("*.dcm"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any], str]:
        path = self.paths[idx]
        dcm  = pydicom.dcmread(path, force=True)

        meta = {el.name: el.repval for el in dcm}
        px   = dcm.pixel_array  # pylibjpeg handles compressed syntaxes

        if px.ndim < 3 or px.shape[2] == 3:
            raise ValueError(f"Invalid pixel array shape: {px.shape}")

        # (H,W,T)  →  (T,H,W,1) → repeat channels
        if px.ndim == 3:
            px = np.repeat(px[..., None], 3, axis=-1)

        px = video_utils.mask_outside_ultrasound(px)

        # Vectorised crop/scale (batch‑wise)
        if hasattr(video_utils, "crop_and_scale_batch"):
            vid = video_utils.crop_and_scale_batch(px, out_h=SIZE, out_w=SIZE)
        else:  # fallback
            vid = np.stack([video_utils.crop_and_scale(f) for f in px])

        vid = torch.from_numpy(vid).permute(3, 0, 1, 2).float()  # [C,T,H,W]

        # Pad / stride
        if vid.shape[1] < FRAMES_TAKE:
            pad = torch.zeros(3, FRAMES_TAKE - vid.shape[1], SIZE, SIZE)
            vid = torch.cat([vid, pad], 1)
        else:
            vid = vid[:, ::FRAME_STRIDE, :, :][:, :FRAMES_TAKE]

        vid.sub_(MEAN).div_(STD)          # normalise in‑place
        return vid, meta, str(path.relative_to(MOUNT_ROOT))


def collate(batch):
    vids, metas, keys = zip(*batch)
    vids = torch.stack(vids, 0)  # [B,C,T,H,W]
    return vids, metas, keys

# ─── 4️⃣ MAIN ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("⚙️  Building dataset from", MOUNT_ROOT)
    ds = EchoDataset(MOUNT_ROOT)
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
                    collate_fn=collate)

    results: Dict[str, Any] = {}
    failed: List[str] = []

    for vids, metas, keys in tqdm(dl, desc="🔍 inferring", unit="batch"):
        try:
            vids = vids.half().to(DEVICE, non_blocking=True)
            views = classify_first_frames(vids)
        except Exception as e:
            failed.extend(keys)
            traceback.print_exc()
            continue

        for k, m, v in zip(keys, metas, views):
            results[k] = {"metadata": m, "predicted_view": v}

        # periodic flush to disk
        if len(results) % FLUSH_EVERY == 0:
            _flush(results, failed)

    _flush(results, failed)  # final write
    print("✅  Done. successes=", len(results), "failures=", len(failed))


def _flush(results: Dict[str, Any], failed: List[str]):
    out_file    = OUTPUT_ROOT / "results.json"
    failed_file = OUTPUT_ROOT / "failed.txt"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
    with failed_file.open("w") as f:
        f.write("\n".join(failed))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True      # autotune convolution kernels
    main()
