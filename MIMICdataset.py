#!/usr/bin/env python3
"""MIMIC‑Echo‑IV view‑classification – fast inference edition
────────────────────────────────────────────────────────────────────────
• One‑shot manifest (cached) eliminates repeated directory walks
• Vectorised skip of already‑processed paths
• Robust to `torch.compile` issues – auto‑fallback to eager if Dynamo fails
• Prints a log line **every time something is written to disk** (manifest,
  per‑dir results, failure logs, processed list, done‑dir flag) so you can
  monitor I/O behaviour in real‑time.

Set `NO_COMPILE=1` in the environment to skip `torch.compile` entirely.
"""
from __future__ import annotations
import json, os, time, argparse, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

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

BATCH_SIZE  = 64
NUM_WORKERS = 24
PREFETCH    = 16
FLUSH_EVERY = 1

FRAMES_TAKE, SIZE = 32, 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3,1,1,1)
STD  = torch.tensor([47.989223, 46.456997, 47.200830]).reshape(3,1,1,1)

# ─── HELPER: logging writes ───────────────────────────────────────────────

def _log_write(path: Path | str, note: str = "") -> None:
    """Print a single line to stdout whenever we touch the filesystem."""
    print(f"💾  {note} -> {path}", file=sys.stderr)

# ─── MODEL ────────────────────────────────────────────────────────────────
ckpt  = torch.load("model_data/weights/view_classifier.ckpt", map_location="cpu")
state = {k[6:]: v for k, v in ckpt["state_dict"].items()}
model = torchvision.models.convnext_base()
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,
                                       len(utils.COARSE_VIEWS))
model.load_state_dict(state, strict=False)
model = model.to(DEVICE).half().eval()

if DEVICE.type == "cuda" and torch.__version__ >= "2" and os.getenv("NO_COMPILE", "0") != "1":
    try:
        model = torch.compile(model, backend="inductor")
    except Exception as e:
        print(f"⚠️  torch.compile failed ({e.__class__.__name__}: {e}). Running eager.")
else:
    if os.getenv("NO_COMPILE", "0") == "1":
        print("ℹ️  Skipping torch.compile because NO_COMPILE=1")

if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

_autocast = torch.cuda.amp.autocast if DEVICE.type == "cuda" else torch.cpu.amp.autocast

@torch.inference_mode()
def classify_first_frames(batch_5d: torch.Tensor) -> List[str]:
    with _autocast(dtype=torch.float16):
        x = batch_5d[:, :, 0].contiguous(memory_format=torch.channels_last)
        logits = model(x)
    return [utils.COARSE_VIEWS[i] for i in logits.argmax(1).tolist()]

# ─── RESUME STATE ─────────────────────────────────────────────────────────
print("🔄  Loading resume state …")
DONE_DIRS      = set(DONE_DIRS_FILE.read_text().splitlines()) if DONE_DIRS_FILE.exists() else set()
PROCESSED_DCMS = set(PROCESSED_DCM_FILE.read_text().splitlines()) if PROCESSED_DCM_FILE.exists() else set()
print(f"   ➜ {len(DONE_DIRS):,} dirs, {len(PROCESSED_DCMS):,} files already done.")

# ─── MANIFEST ─────────────────────────────────────────────────────────────
print("🔍  Building / loading manifest …", end=" ")
if MANIFEST_FILE.exists():
    man = json.load(MANIFEST_FILE.open())
    ALL_DCMS      = [MOUNT_ROOT / p for p in man["all_dcms"]]
    TOTALS_BY_DIR: Dict[str,int] = man["totals_by_dir"]
    print("cached ✅")
else:
    ALL_DCMS: List[Path] = []
    TOTALS_BY_DIR = defaultdict(int)
    for p in tqdm(MOUNT_ROOT.rglob("*.dcm"), desc=" scan", unit="file"):
        TOTALS_BY_DIR[p.parent.relative_to(MOUNT_ROOT).as_posix()] += 1
        ALL_DCMS.append(p)
    MANIFEST_FILE.write_text(json.dumps({
        "all_dcms"     : [p.relative_to(MOUNT_ROOT).as_posix() for p in ALL_DCMS],
        "totals_by_dir": TOTALS_BY_DIR,
    }))
    _log_write(MANIFEST_FILE, "manifest saved")
    print("new ✅")

TO_DO = [p for p in ALL_DCMS if p.relative_to(MOUNT_ROOT).as_posix() not in PROCESSED_DCMS]
print(f"   ➜ {len(TO_DO):,} remaining files to process.")

# ─── DATASET ──────────────────────────────────────────────────────────────
class EchoIterableDataset(IterableDataset):
    resize = torchvision.transforms.Resize(SIZE, antialias=True)

    def __init__(self, paths: List[Path]):
        self.paths = paths

    @staticmethod
    def preprocess(px: np.ndarray) -> torch.Tensor:
        if px.ndim == 3:
            px = px[..., None]
        vid = torch.from_numpy(px).permute(0,3,1,2).float()
        vid = EchoIterableDataset.resize(vid)
        vid = torchvision.transforms.functional.center_crop(vid, (SIZE,SIZE))
        if vid.size(0) < FRAMES_TAKE:
            vid = torch.cat((vid, torch.zeros(FRAMES_TAKE - vid.size(0), 3, SIZE, SIZE)), 0)
        vid = vid[:FRAMES_TAKE].permute(1,0,2,3)
        vid = (vid - MEAN).div_(STD)
        return vid.half()

    def __iter__(self):
        worker = get_worker_info()
        paths = self.paths[worker.id::worker.num_workers] if worker else self.paths
        for dcm_path in paths:
            rel_path = dcm_path.relative_to(MOUNT_ROOT).as_posix()
            rel_dir  = dcm_path.parent.relative_to(MOUNT_ROOT).as_posix()
            try:
                dcm = pydicom.dcmread(dcm_path, force=True)
                px  = dcm.pixel_array
                if px.ndim < 3 or (px.ndim == 3 and px.shape[-1] == 3):
                    raise ValueError
                px = video_utils.mask_outside_ultrasound(px)
                yield EchoIterableDataset.preprocess(px), {el.name: el.repval for el in dcm}, rel_path, rel_dir
            except Exception:
                yield None, None, rel_path, rel_dir

# ─── HELPERS ──────────────────────────────────────────────────────────────

def _one_thread(_: int):
    torch.set_num_threads(1)

def _flush(res: Dict[str, Dict[str, Any]], fail: Dict[str, List[str]]):
    processed = []
    touched = set(res) | set(fail)

    for rdir, items in list(res.items()):
        out_dir = OUTPUT_ROOT / rdir; out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.json"
        data = items | (json.load(out_file.open()) if out_file.exists() else {})
        json.dump(data, out_file.open("w"))
        _log_write(out_file, f"results ({len(items)})")
        processed.extend(items.keys()); res.pop(rdir)

    for rdir, lst in list(fail.items()):
        out_dir = OUTPUT_ROOT / rdir; out_dir.mkdir(parents=True, exist_ok=True)
        fail_fp = out_dir / "failed.txt"
        fail_fp.open("a").writelines(p+"\n" for p in lst)
        _log_write(fail_fp, f"failures (+{len(lst)})")
        processed.extend(lst); fail.pop(rdir)

    if processed:
        PROCESSED_DCM_FILE.open("a").writelines(p+"\n" for p in processed if p not in PROCESSED_DCMS)
        _log_write(PROCESSED_DCM_FILE, f"update (+{len(processed)})")
        PROCESSED_DCMS.update(processed)

    for rdir in touched - DONE_DIRS:
        total = TOTALS_BY_DIR.get(rdir, 0)
        succ  = len(json.load((OUTPUT_ROOT/rdir/"results.json").open())) if (OUTPUT_ROOT/rdir/"results.json").exists() else 0
        failc = sum(1 for _ in (OUTPUT_ROOT/rdir/"failed.txt").open()) if (OUTPUT_ROOT/rdir/"failed.txt").exists() else 0
        print(f"   ➜ {rdir}: {succ:,} successes, {failc:,} failures, {total:,}")
        if succ + failc >= total > 0:
            DONE_DIRS_FILE.open("a").write(rdir+"\n"); DONE_DIRS.add(rdir)

# ─── MAIN ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fast MIMIC-Echo-IV inference")
    parser.add_argument("--profile", action="store_true", help="verbose per-batch timings")
    args = parser.parse_args()

    dl = DataLoader(EchoIterableDataset(TO_DO),
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=PREFETCH,
                    collate_fn=lambda x: tuple(zip(*x)),
                    worker_init_fn=_one_thread)

    res, fail = defaultdict(dict), defaultdict(list)
    n, t0 = 0, time.perf_counter()
    t_dl=t_inf=t_post=t_flush=0.0; t_prev=t0

    for bidx, (vids, metas, keys, dirs) in enumerate(tqdm(dl, unit="batch")):
        now=time.perf_counter(); t_dl+=now-t_prev; n+=1
        ok=[v is not None for v in vids]
        if any(ok):
            batch=torch.stack([v for v in vids if v is not None]).to(DEVICE, non_blocking=True)
            if DEVICE.type=="cuda": torch.cuda.synchronize(); tic=time.perf_counter()
            views=classify_first_frames(batch)
            if DEVICE.type=="cuda": torch.cuda.synchronize(); t_inf+=time.perf_counter()-tic
        else:
            views=[]
        tic=time.perf_counter()
        viter=iter(views)
        for k,m,good,d in zip(keys,metas,ok,dirs):
            if not good:
                fail[d].append(k); continue
            res[d][k] = {"metadata": m, "predicted_view": next(viter)}
        t_post+=time.perf_counter()-tic
        if (bidx+1)%FLUSH_EVERY==0:
            tic=time.perf_counter(); _flush(res, fail); t_flush+=time.perf_counter()-tic
        t_prev=time.perf_counter()
        if args.profile:
            print(f"Batch {bidx:>5d}: DL={t_dl/n:.4f}s INF={t_inf/n:.4f}s POST={t_post/n:.4f}s")

    tic=time.perf_counter(); _flush(res, fail); t_flush+=time.perf_counter()-tic
    t_tot=time.perf_counter()-t0
    print("\n────────── SUMMARY ──────────")
    print(f"Batches: {n:,}  |  Total: {t_tot:.2f}s  |  Throughput: {n/t_tot:.2f} batch/s")
    if n:
        print(f"Avg DL  : {t_dl/n:.4f}s  |  Avg INF: {t_inf/n:.4f}s  |  Avg POST: {t_post/n:.4f}s")
    print(f"Flush time total: {t_flush:.2f}s (every {FLUSH_EVERY} batches)")
    print("✅  Done.")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
