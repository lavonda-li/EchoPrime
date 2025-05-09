import os
from pathlib import Path


# ─── 1️⃣ CONFIG ─────────────────────────────────────────
MOUNT_ROOT  = Path(os.path.expanduser("~/mount-folder/MIMIC-Echo-IV"))
OUTPUT_ROOT = Path(os.path.expanduser("~/inference_output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DONE_DIRS_FILE = "done_dirs.txt"


BATCH_SIZE   = 64
NUM_WORKERS  = min(24, os.cpu_count() or 1)
PREFETCH     = 16

FRAMES_TAKE  = 32
FRAME_STRIDE = 2
SIZE         = 224

print("🔔 Loading completed folders from output...")
DONE_DIRS = set()
if DONE_DIRS_FILE.exists():
    with DONE_DIRS_FILE.open() as f:
        DONE_DIRS.update(line.strip() for line in f if line.strip())

json_dirs = {p.parent.relative_to(OUTPUT_ROOT).as_posix() for p in OUTPUT_ROOT.rglob("results.json")}
DONE_DIRS.update(json_dirs)

if DONE_DIRS:
    for d in sorted(DONE_DIRS):
        print(f"✔️  {d} already done — skipping.")
    print(f"⚠️  Detected {len(DONE_DIRS):,} completed folders; they will be skipped.")

    # write the list of completed folders to a file or append to an existing one
    with DONE_DIRS_FILE.open("a") as done_file:
        for d in DONE_DIRS:
            done_file.write(d + "\n")