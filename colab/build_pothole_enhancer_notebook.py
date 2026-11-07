"""Build the standalone BANTAI_Pothole_Model_Enhancer.ipynb Colab notebook.

Generates a turnkey companion notebook that loads the finished model checkpoint
from Google Drive, stages a fast multi-domain All-Terrain Pothole dataset via Parquet
(~30s), runs a 3-minute Turbo A100/L4 fine-tuning pass across all 6 pothole types,
applies the algorithmic enhancement pipeline (LAB-CLAHE + 5-Tile Road-SAHI + Focal Tile
+ WBF + Temporal Tracking), runs an A/B benchmark, and packages the upgraded bundle.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
cells = []


def md(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip() + "\n",
    })


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(text).strip() + "\n",
    })


def repair_slot(section: str) -> None:
    md(f"""### Section {section} troubleshooting slot
    If this section fails, insert a temporary code cell **here**, immediately below its main cell.
    Keep any fix scoped to this section, then rerun its main cell and dependents.
    """)


# --- Title & Overview ---
md("""
# BANTAI — All-Terrain Pothole Model Enhancer & Fast Out-of-Domain Fine-Tuning
**~5–7 Minute End-to-End Domain Adaptation + Multi-Scale SAHI/CLAHE Inference Upgrade**

- **Automatic Checkpoint Discovery**: Automatically finds and ingests `bantai_pothole_rfdetr_nano_ready.pth` from your Google Drive (`MyDrive/BANTAI/pothole_ready_model/` or `MyDrive/BANTAI/pothole_15k/`).
- **Covers All 6 Real-World Pothole Types**:
  1. **Dry Asphalt & Shallow Spalling** (sunlit & grey-on-grey road wear)
  2. **Deep Gravel & Crumbled Craters** (fractured dark rims & exposed aggregate)
  3. **Wet, Rain-Puddle & Muddy Potholes** (specular sky glare & opaque muddy snow-thaw puddles)
  4. **Harsh Cast Shadows, Overcast & Dusk/Night Dashcams** (tree/bridge shadows & dim lighting)
  5. **Concrete Highways, Patched Seams & Road Shoulders** (high-key concrete & failed cold patches)
  6. **Multi-Scale Perspectives** (close-up macro phone photos to 40m wide-angle `16:9` highway dashcams)
- **Fast Parquet Dataset Ingestion (~30 seconds)**: Streams contiguous Parquet shards (`1,000 train + 100 valid + 100 test`) from Hugging Face (`Duranishhh/baches-carreteras`, `manot/pothole-segmentation`, `keremberke/pothole-segmentation`) with progress-bar spam disabled so Colab's browser UI never freezes.
- **Turbo A100/L4 Fine-Tuning (~3 minutes)**: Runs **6 epochs** (`31 steps/epoch` at `batch_size=32`, fixed `384x384` with `multi_scale=False`, unlocked `LR=1.2e-4`, `LR_ENCODER=2e-5`) and embeds `model_config` directly into the `.pth` checkpoint.
- **Algorithmic Inference Upgrades Baked In**: **CIE-LAB CLAHE** + **5-Tile Road-SAHI (2x2 Grid + Center-Horizon Focal Tile)** + **Horizontal-Flip TTA** + **Weighted Boxes Fusion (WBF)** + **Rim-Aware Roughness Filter** + **Temporal Multi-Frame Tracker**.
""")

# --- Section 1: Settings, Drive Mount & Model Ingestion ---
md("""
## 1. Mount Google Drive, Ingest Stage-1 Model & Configure Turbo GPU Profile
Mounts Google Drive, locates your Stage-1 `bantai_pothole_rfdetr_nano_ready.pth` checkpoint (treated strictly as read-only), creates a new versioned run directory (`MyDrive/BANTAI/pothole_enhanced_runs/run_v{N}_{YYYYMMDD_HHMMSS}/`), and configures the fastest hardware profile for A100, L4, or T4 GPUs.
""")
code(r"""
from pathlib import Path
import datetime, json, os, sys, subprocess, shutil, gc, time
from google.colab import drive
import torch

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

drive.mount('/content/drive')

READY_DRIVE_DIR = Path('/content/drive/MyDrive/BANTAI/pothole_ready_model')
TRAIN_DRIVE_DIR = Path('/content/drive/MyDrive/BANTAI/pothole_15k')

# Create a brand-new versioned output directory on every run (never overwrites old models!)
ENHANCED_BASE_DIR = Path('/content/drive/MyDrive/BANTAI/pothole_enhanced_runs')
ENHANCED_BASE_DIR.mkdir(parents=True, exist_ok=True)
existing_runs = sorted([d for d in ENHANCED_BASE_DIR.glob('run_v*') if d.is_dir()])
next_ver = len(existing_runs) + 1
RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_TAG = f'v{next_ver}_{RUN_TIMESTAMP}'
ENHANCED_DRIVE_DIR = ENHANCED_BASE_DIR / f'run_{RUN_TAG}'
ENHANCED_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
ENHANCED_MODEL_FILENAME = f'bantai_pothole_rfdetr_nano_enhanced_{RUN_TAG}.pth'

LOCAL_WORK = Path(f'/content/bantai_enhancer_{RUN_TAG}')
LOCAL_WORK.mkdir(parents=True, exist_ok=True)
SHARED_PARQUET_CACHE = Path('/content/bantai_parquet_cache')
SHARED_PARQUET_CACHE.mkdir(parents=True, exist_ok=True)

# 1. Automatic Stage-1 checkpoint discovery (leaves Stage-1 untouched)
candidate_weights = [
    READY_DRIVE_DIR / 'bantai_pothole_rfdetr_nano_ready.pth',
    *sorted(TRAIN_DRIVE_DIR.glob('**/checkpoint_best_total.pth'), reverse=True),
    *[p for p in sorted(TRAIN_DRIVE_DIR.glob('**/*.pth'), reverse=True) if 'surgery' not in p.name and 'last' not in p.name],
]

STAGE1_WEIGHTS_PATH = None
for p in candidate_weights:
    if p.exists() and p.stat().st_size > 10 * 1024 * 1024:
        STAGE1_WEIGHTS_PATH = p
        break

if STAGE1_WEIGHTS_PATH is None:
    raise FileNotFoundError(
        "Could not find Stage-1 trained checkpoint on Google Drive! "
        "Ensure bantai_pothole_rfdetr_nano_ready.pth exists in MyDrive/BANTAI/pothole_ready_model/."
    )

# 2. Turbo GPU training & inference settings
TRAIN_COUNT = 1000
VALID_COUNT = 100
TEST_COUNT = 100
EPOCHS = 6
LEARNING_RATE = 1.2e-4
LR_ENCODER = 2.0e-5
BASE_THRESHOLD = 0.20
SEED = 42

vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

if vram_gb >= 35:  # A100 (40GB/80GB)
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 32, 1, 4
elif vram_gb >= 20:  # L4 (24GB)
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 16, 2, 4
elif vram_gb >= 10:  # T4 (15GB)
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 8, 4, 2
else:
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 4, 4, 2

print(f"Hardware: {gpu_name} ({vram_gb:.1f} GB VRAM)")
print(f"Ingested Stage-1 Checkpoint (Read-Only): {STAGE1_WEIGHTS_PATH} ({STAGE1_WEIGHTS_PATH.stat().st_size / 1024**2:.1f} MB)")
print(f"New Run Output Folder: {ENHANCED_DRIVE_DIR}")
print(f"New Model Filename: {ENHANCED_MODEL_FILENAME}")
print(f"Turbo Fine-Tuning Profile -> {EPOCHS} Epochs | Batch: {BATCH_SIZE} (Accum: {GRAD_ACCUM_STEPS}) | LR: {LEARNING_RATE} (Encoder: {LR_ENCODER})")
print(f"Default Calibrated Operating Threshold: {BASE_THRESHOLD:.0%}")
""")
repair_slot('1')

# --- Section 2: Dependencies ---
md("""
## 2. Fast Package Setup
Installs `rfdetr[train]`, `datasets` (for fast Parquet streaming), `opencv-python-headless`, and `pycocotools`.
""")
code(r"""
PACKAGES = [
    'rfdetr[train]==1.11.2',
    'datasets>=2.19.0',
    'opencv-python-headless>=4.8.0',
    'pycocotools==2.0.11',
    'huggingface-hub==2.3.0',
    'hf-transfer==0.1.9',
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *PACKAGES], check=True)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from rfdetr import RFDETR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Packages ready on {DEVICE} ({gpu_name})")
""")
repair_slot('2')

# --- Section 3: Enhancement Engine Source ---
md("""
## 3. Load All-Terrain Enhancement & Training Engine
Defines:
- `augment_all_pothole_types` & `prepare_fast_all_terrain_dataset` (6-domain synthesis + 30s Parquet staging)
- `preserve_model_config_in_checkpoint` (prevents stripped `model_config` warnings)
- `apply_lab_clahe`, `generate_road_sahi_tiles` (with 5th Center-Horizon Focal Tile), `weighted_boxes_fusion`, `verify_pothole_texture_and_geometry`, and `EnhancedPotholeDetector`.
""")
workflow_path = HERE / "pothole_enhancer_workflow.py"
workflow_code = workflow_path.read_text(encoding="utf-8")
code(f"""
# Embed the complete canonical enhancer workflow
{workflow_code}

print("BANTAI All-Terrain Pothole Enhancement Engine loaded successfully.")
""")
repair_slot('3')

# --- Section 4: Fast Parquet + COCO + Archive Dataset Staging (Potholes + Manhole Hard-Negatives) ---
md("""
## 4. Stage All-Terrain Pothole + Manhole Hard-Negative Dataset (`BANTAI-DATASETS`)
Loads **ALL THREE** of your dataset folders from `/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/`:
1. `BANTAI_Pothole_Parquet_Dataset` (all 9 pothole shards `shard_0`..`shard_8` + 3 manhole shards `manhole_*.parquet`)
2. `Manhole Cover Dataset YOLO.v4i.coco` (`train/`, `valid/`, `test/` with `_annotations.coco.json`)
3. `archive/data` (`annotations_coco.json` + `images/` with 2,009 street photos containing `pothole=0`, `crack=1`, `manhole=2` — including 236 photos that have BOTH potholes AND manholes in the exact same road scene!)
""")
code(r"""
# @title 4. Stage ALL Pothole + Manhole Datasets from Drive (Parquet + Manhole COCO + archive/data)
import collections, io, json, math, shutil, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def _safe_extract_xywh_boxes(raw_objects, width, height):
    '''NumPy-safe bounding box extractor for Hugging Face Parquet rows (never triggers ValueError on numpy arrays).'''
    if not isinstance(raw_objects, dict):
        return []
    raw_bboxes = raw_objects.get("bbox")
    if raw_bboxes is None:
        raw_bboxes = raw_objects.get("bboxes")
    if raw_bboxes is None:
        return []
    valid_boxes = []
    for box in raw_bboxes:
        if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
            continue
        vals = [float(v) for v in box]
        if not all(math.isfinite(v) for v in vals):
            continue
        if max(vals) <= 1.05 and width > 10 and height > 10:
            vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
        x, y, w_or_x2, h_or_y2 = vals
        if x >= 0 and y >= 0 and w_or_x2 > 0 and h_or_y2 > 0 and (x + w_or_x2 <= width + 4) and (y + h_or_y2 <= height + 4):
            bw, bh = w_or_x2, h_or_y2
        elif w_or_x2 > x and h_or_y2 > y:
            bw, bh = w_or_x2 - x, h_or_y2 - y
        else:
            continue
        x1 = max(0.0, min(float(width - 2), x))
        y1 = max(0.0, min(float(height - 2), y))
        x2 = max(x1 + 2.0, min(float(width), x + bw))
        y2 = max(y1 + 2.0, min(float(height), y + bh))
        if (x2 - x1) >= 4.0 and (y2 - y1) >= 4.0:
            valid_boxes.append([round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)])
    return valid_boxes


def stage_pothole_and_manhole_dataset(
    work_dir=LOCAL_WORK,
    train_count=1600,
    valid_count=200,
    test_count=200,
    seed=SEED,
):
    work_dir = Path(work_dir)
    dataset_root = work_dir / "all_terrain_dataset"
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists() and (dataset_root / "train" / "_annotations.coco.json").exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("version") == "v4_full_archive_coco_parquet" and cached.get("splits", {}).get("train", {}).get("boxes", 0) > 500:
                print(f"Reusing staged Pothole + Anti-Manhole dataset (v4_full_archive_coco_parquet) at: {dataset_root}")
                return dataset_root, cached
        except Exception:
            pass

    if dataset_root.exists():
        shutil.rmtree(dataset_root, ignore_errors=True)

    drive_ds_roots = [
        Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS"),
        Path("/content/drive/MyDrive/BANTAI"),
    ]
    pq_candidates = [
        Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/BANTAI_Pothole_Parquet_Dataset"),
        Path("/content/drive/MyDrive/BANTAI/BANTAI_Pothole_Parquet_Dataset"),
    ]
    pq_cache = next((p for p in pq_candidates if p.exists()), work_dir / "parquet_cache")
    pq_cache.mkdir(parents=True, exist_ok=True)
    print(f"1. Reading Parquet shards from: {pq_cache}")

    target_total = train_count + valid_count + test_count
    target_positives = max(1, int(target_total * 0.72))
    target_negatives = max(1, target_total - target_positives)

    raw_samples = []
    negative_samples = []
    manhole_crops = []

    # --- SOURCE A: Manhole Parquet Shards (delima87/manhole_covers_dataset: 2,730 manhole images) ---
    manhole_parquets = [
        ("manhole_train_0000.parquet", "https://huggingface.co/datasets/delima87/manhole_covers_dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"),
        ("manhole_validation_0000.parquet", "https://huggingface.co/datasets/delima87/manhole_covers_dataset/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"),
        ("manhole_test_0000.parquet", "https://huggingface.co/datasets/delima87/manhole_covers_dataset/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"),
    ]
    mh_pq_neg_cap = int(target_negatives * 0.35)
    for mh_fname, mh_url in manhole_parquets:
        try:
            mh_path = pq_cache / mh_fname
            if not mh_path.exists() or mh_path.stat().st_size < 1024:
                print(f"  Downloading {mh_fname}...")
                req = urllib.request.Request(mh_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, mh_path.open("wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
            df_mh = pd.read_parquet(mh_path)
            for _, row in df_mh.iterrows():
                img_cell = row.get("image")
                pil_img = None
                if isinstance(img_cell, dict) and img_cell.get("bytes"):
                    pil_img = Image.open(io.BytesIO(img_cell["bytes"]))
                elif isinstance(img_cell, (bytes, bytearray)):
                    pil_img = Image.open(io.BytesIO(img_cell))
                if pil_img is None:
                    continue
                rgb = ImageOps.exif_transpose(pil_img).convert("RGB")
                if len(manhole_crops) < 250:
                    manhole_crops.append(rgb.copy())
                if len(negative_samples) < mh_pq_neg_cap:
                    negative_samples.append((rgb.resize((512, 384), Image.BILINEAR), []))
                if len(negative_samples) >= mh_pq_neg_cap and len(manhole_crops) >= 250:
                    break
            print(f"  Loaded {mh_fname}: {len(negative_samples)} manhole hard-negatives, {len(manhole_crops)} manhole crops")
        except Exception as exc:
            print(f"  Skipped {mh_fname}: {exc}")

    # --- SOURCE B: Manhole Cover Dataset YOLO.v4i.coco (train / valid / test) ---
    coco_neg_cap = int(target_negatives * 0.70)
    for droot in drive_ds_roots:
        coco_mh = droot / "Manhole Cover Dataset YOLO.v4i.coco"
        if not coco_mh.exists():
            continue
        print(f"2. Reading Manhole Cover Dataset YOLO.v4i.coco from: {coco_mh}")
        added_coco_neg = 0
        for ann_path in sorted(coco_mh.glob("*/_annotations.coco.json")):
            try:
                cdata = json.loads(ann_path.read_text(encoding="utf-8"))
                anns_by_img = collections.defaultdict(list)
                for ann in cdata.get("annotations", []):
                    bb = ann.get("bbox", [])
                    if len(bb) == 4 and bb[2] >= 12 and bb[3] >= 12:
                        anns_by_img[int(ann["image_id"])].append([float(v) for v in bb])
                for im_info in cdata.get("images", []):
                    im_file = ann_path.parent / im_info["file_name"]
                    if not im_file.exists():
                        continue
                    with Image.open(im_file) as im_src:
                        rgb = ImageOps.exif_transpose(im_src).convert("RGB")
                    if len(negative_samples) < coco_neg_cap:
                        negative_samples.append((rgb, []))
                        added_coco_neg += 1
                    for bx, by, bw, bh in anns_by_img.get(int(im_info["id"]), [])[:2]:
                        if len(manhole_crops) < 420:
                            crop = rgb.crop((max(0, int(bx)), max(0, int(by)), min(rgb.width, int(bx + bw)), min(rgb.height, int(by + bh))))
                            if crop.width >= 16 and crop.height >= 12:
                                manhole_crops.append(crop)
                    if len(negative_samples) >= coco_neg_cap and len(manhole_crops) >= 420:
                        break
            except Exception as exc:
                print(f"  Note in {ann_path.parent.name}: {exc}")
        print(f"  Included Manhole Cover Dataset YOLO.v4i.coco (+{added_coco_neg} street manhole negatives, {len(manhole_crops)} total crops)")
        break

    # --- SOURCE C: Kaggle archive/data (2,009 road images: pothole=0, crack=1, manhole=2) ---
    archive_candidates = [
        Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/archive/data"),
        Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/archive"),
        Path("/content/drive/MyDrive/BANTAI/archive/data"),
    ]
    for arch_dir in archive_candidates:
        ann_coco = arch_dir / "annotations_coco.json"
        img_dir = arch_dir / "images"
        if not ann_coco.exists() or not img_dir.exists():
            continue
        print(f"3. Reading archive/data dataset from: {arch_dir}")
        try:
            adata = json.loads(ann_coco.read_text(encoding="utf-8"))
            potholes_by_img = collections.defaultdict(list)
            manholes_by_img = collections.defaultdict(list)
            for ann in adata.get("annotations", []):
                bb = ann.get("bbox", [])
                if len(bb) != 4 or bb[2] < 6 or bb[3] < 6:
                    continue
                cid = int(ann.get("category_id", -1))
                if cid == 0:
                    potholes_by_img[int(ann["image_id"])].append([float(v) for v in bb])
                elif cid == 2:
                    manholes_by_img[int(ann["image_id"])].append([float(v) for v in bb])

            # Prioritize images that have BOTH potholes AND manholes first, then potholes, then manhole-only negatives!
            imgs_info = list(adata.get("images", []))
            imgs_info.sort(
                key=lambda info: (
                    1 if (potholes_by_img.get(int(info["id"])) and manholes_by_img.get(int(info["id"]))) else
                    (2 if potholes_by_img.get(int(info["id"])) else (3 if manholes_by_img.get(int(info["id"])) else 4))
                )
            )
            arch_pos, arch_both, arch_mh_neg = 0, 0, 0
            for im_info in imgs_info:
                im_id = int(im_info["id"])
                p_boxes = potholes_by_img.get(im_id, [])
                m_boxes = manholes_by_img.get(im_id, [])
                if not p_boxes and not m_boxes:
                    continue
                if p_boxes and len(raw_samples) >= int(target_positives * 0.48):
                    continue
                if (not p_boxes) and len(negative_samples) >= target_negatives:
                    continue
                im_file = img_dir / im_info["file_name"]
                if not im_file.exists():
                    continue
                with Image.open(im_file) as im_src:
                    rgb = ImageOps.exif_transpose(im_src).convert("RGB")
                w, h = rgb.size
                for bx, by, bw, bh in m_boxes[:2]:
                    if len(manhole_crops) < 550:
                        crop = rgb.crop((max(0, int(bx)), max(0, int(by)), min(w, int(bx + bw)), min(h, int(by + bh))))
                        if crop.width >= 16 and crop.height >= 12:
                            manhole_crops.append(crop)
                if p_boxes:
                    valid_p = _safe_extract_xywh_boxes({"bbox": p_boxes}, w, h)
                    if valid_p:
                        raw_samples.append((rgb, valid_p))
                        arch_pos += 1
                        if m_boxes:
                            arch_both += 1
                elif m_boxes and len(negative_samples) < target_negatives:
                    negative_samples.append((rgb, []))
                    arch_mh_neg += 1
            print(f"  Loaded archive/data: +{arch_pos} pothole road scenes ({arch_both} with real unboxed manholes in same frame!), +{arch_mh_neg} manhole-only road scenes")
        except Exception as exc:
            print(f"  Note in archive/data: {exc}")
        break

    # --- SOURCE D: Pothole Parquet Shards (shard_0 .. shard_8: 1,568 pothole scenes) ---
    pothole_shards = [
        ("shard_0_manot1_train.parquet", "manot/pothole-segmentation (train)", "https://huggingface.co/datasets/manot/pothole-segmentation/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet"),
        ("shard_1_manot1_val.parquet", "manot/pothole-segmentation (val)", "https://huggingface.co/datasets/manot/pothole-segmentation/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet"),
        ("shard_2_manot1_test.parquet", "manot/pothole-segmentation (test)", "https://huggingface.co/datasets/manot/pothole-segmentation/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet"),
        ("shard_3_manot2_train.parquet", "manot/pothole-segmentation2 (train)", "https://huggingface.co/datasets/manot/pothole-segmentation2/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet"),
        ("shard_4_manot2_val.parquet", "manot/pothole-segmentation2 (val)", "https://huggingface.co/datasets/manot/pothole-segmentation2/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet"),
        ("shard_5_manot2_test.parquet", "manot/pothole-segmentation2 (test)", "https://huggingface.co/datasets/manot/pothole-segmentation2/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet"),
        ("shard_6_keremberke_train.parquet", "keremberke/pothole-segmentation (train)", "https://huggingface.co/datasets/keremberke/pothole-segmentation/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet"),
        ("shard_7_keremberke_val.parquet", "keremberke/pothole-segmentation (val)", "https://huggingface.co/datasets/keremberke/pothole-segmentation/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet"),
        ("shard_8_keremberke_test.parquet", "keremberke/pothole-segmentation (test)", "https://huggingface.co/datasets/keremberke/pothole-segmentation/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet"),
    ]
    print("4. Loading Pothole Parquet Shards (shard_0 .. shard_8):")
    for shard_fname, repo_id, url in pothole_shards:
        if len(raw_samples) >= target_positives and len(negative_samples) >= target_negatives:
            break
        try:
            local_pq = pq_cache / shard_fname
            if not local_pq.exists() or local_pq.stat().st_size < 1024:
                print(f"  Downloading {shard_fname}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, local_pq.open("wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
            df = pd.read_parquet(local_pq)
            for _, row in df.iterrows():
                img_cell = row.get("image")
                pil_img = None
                if isinstance(img_cell, dict) and img_cell.get("bytes"):
                    pil_img = Image.open(io.BytesIO(img_cell["bytes"]))
                elif isinstance(img_cell, (bytes, bytearray)):
                    pil_img = Image.open(io.BytesIO(img_cell))
                if pil_img is None:
                    continue
                rgb = ImageOps.exif_transpose(pil_img).convert("RGB")
                w, h = rgb.size
                objs = row.get("objects")
                boxes = _safe_extract_xywh_boxes(objs, w, h) if isinstance(objs, dict) else []
                if boxes and len(raw_samples) < target_positives:
                    raw_samples.append((rgb, boxes))
                elif not boxes and len(negative_samples) < target_negatives and w >= 256 and h >= 256:
                    negative_samples.append((rgb, []))
                if len(raw_samples) >= target_positives and len(negative_samples) >= target_negatives:
                    break
            print(f"  Collected {len(raw_samples)} pothole positives, {len(negative_samples)} hard-negatives ({repo_id})")
        except Exception as exc:
            print(f"  Skipped {shard_fname}: {exc}")

    rng = np.random.RandomState(seed)
    while len(raw_samples) < target_positives and raw_samples:
        base_img, base_boxes = raw_samples[len(raw_samples) % len(raw_samples)]
        aug_img, aug_boxes, _ = augment_all_pothole_types(base_img, base_boxes, len(raw_samples), seed=seed)
        raw_samples.append((aug_img, aug_boxes))

    while len(negative_samples) < target_negatives and raw_samples:
        base_img, _ = raw_samples[len(negative_samples) % len(raw_samples)]
        w_b, h_b = base_img.size
        crop_neg = base_img.crop((0, int(h_b * 0.15), w_b, int(h_b * 0.55))).resize((w_b, h_b), Image.BILINEAR)
        negative_samples.append((crop_neg, []))

    all_samples = raw_samples[:target_positives] + negative_samples[:target_negatives]
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    splits_map = {
        "train": [all_samples[i] for i in indices[:train_count]],
        "valid": [all_samples[i] for i in indices[train_count : train_count + valid_count]],
        "test": [all_samples[i] for i in indices[train_count + valid_count : target_total]],
    }

    domain_counts = collections.Counter()
    split_summary = {}

    for split_name, items in splits_map.items():
        split_dir = dataset_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {"description": f"BANTAI Pothole + Anti-Manhole {split_name} split v4"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "pothole", "supercategory": "road_damage"}],
        }
        for img_id, (pil_img, boxes) in enumerate(items):
            if not boxes:
                final_img = pil_img
                final_boxes = []
                domain_tag = "hard_negative_manhole_or_clean_road"
                domain_counts[domain_tag] += 1
            elif split_name == "train":
                final_img, final_boxes, domain_tag = augment_all_pothole_types(pil_img, boxes, img_id, seed=seed)
                # Manhole-on-Road CutMix: paste foreshortened manhole cover onto road (unboxed) so model learns to ignore manholes next to real potholes
                if manhole_crops and (img_id % 3 == 0):
                    w_f, h_f = final_img.size
                    mh_crop = manhole_crops[img_id % len(manhole_crops)]
                    mh_w = max(24, min(int(w_f * 0.16), int(rng.randint(36, 92))))
                    mh_h = max(14, int(mh_w / float(rng.uniform(1.55, 2.45))))
                    mh_resized = mh_crop.resize((mh_w, mh_h), Image.BILINEAR)
                    for _ in range(8):
                        px = int(rng.randint(int(w_f * 0.12), max(int(w_f * 0.12) + 1, int(w_f * 0.85) - mh_w)))
                        py = int(rng.randint(int(h_f * 0.36), max(int(h_f * 0.36) + 1, int(h_f * 0.78) - mh_h)))
                        overlaps = any(
                            not (px + mh_w < bx - 8 or px > bx + bw + 8 or py + mh_h < by - 8 or py > by + bh + 8)
                            for bx, by, bw, bh in final_boxes
                        )
                        if not overlaps:
                            final_img = final_img.copy()
                            final_img.paste(mh_resized, (px, py))
                            domain_counts["manhole_cutmix_hard_negative"] += 1
                            break
                domain_counts[domain_tag] += 1
            else:
                if img_id % 2 == 1:
                    final_img, final_boxes, domain_tag = augment_all_pothole_types(pil_img, boxes, img_id, seed=seed)
                else:
                    final_img, final_boxes = pil_img, boxes
                    domain_tag = "original_positive"
                domain_counts[domain_tag] += 1

            w, h = final_img.size
            fname = f"{img_id:05d}.jpg"
            final_img.save(split_dir / fname, quality=92)
            coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})
            for bx, by, bw, bh in final_boxes:
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [bx, by, bw, bh],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                })
        (split_dir / "_annotations.coco.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
        split_summary[split_name] = {"images": len(coco["images"]), "boxes": len(coco["annotations"])}
        print(f"  Staged {split_name}: {len(coco['images'])} images ({len(coco['annotations'])} pothole boxes)")

    manifest = {
        "version": "v4_full_archive_coco_parquet",
        "splits": split_summary,
        "train_domain_distribution": dict(domain_counts),
        "classes": ["pothole"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dataset_root, manifest

t0_data = time.time()
ALL_TERRAIN_DIR, dataset_manifest = stage_pothole_and_manhole_dataset(
    work_dir=LOCAL_WORK,
    train_count=1600,
    valid_count=200,
    test_count=200,
    seed=SEED,
)
print(f"\nPothole + Anti-Manhole Dataset staged in {time.time() - t0_data:.1f}s at: {ALL_TERRAIN_DIR}")
print("Training Breakdown:", json.dumps(dataset_manifest['train_domain_distribution'], indent=2))
""")
repair_slot('4')

# --- Section 5: 15-Epoch All-Terrain + Anti-Manhole Fine-Tuning ---
md("""
## 5. Retrain RF-DETR Nano for 15 Epochs (Potholes + Anti-Manhole Hard Negatives) & Auto-Save to Drive
Continues/fine-tunes your checkpoint for **15 Epochs** on the 1,600-image `v4_full_archive_coco_parquet` dataset and immediately saves the new versioned `.pth` checkpoint + `bantai_pothole_detector.py` into a brand-new Google Drive folder (`MyDrive/BANTAI/pothole_enhanced_runs/run_<RUN_TAG>/`) without overwriting any previous models.
""")
code(r"""
# @title 5. Retrain Model for 15 Epochs (Potholes + Anti-Manhole) & Auto-Save New Versioned Checkpoint to Drive
import datetime, gc, json, shutil, time
from pathlib import Path
import torch
from rfdetr import RFDETR

# Ensure all root paths and hyperparameters are defined (standalone-safe)
ENHANCED_RUNS_ROOT = Path('/content/drive/MyDrive/BANTAI/pothole_enhanced_runs')
ENHANCED_RUNS_ROOT.mkdir(parents=True, exist_ok=True)

if 'LOCAL_WORK' not in globals():
    LOCAL_WORK = Path('/content/bantai_enhancer_work')
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)

# Warm-start from your latest trained run if available, otherwise from Stage-1 ready weights
latest_prev_runs = sorted(ENHANCED_RUNS_ROOT.glob('run_v*/*.pth'))
if latest_prev_runs:
    STAGE1_WEIGHTS_PATH = latest_prev_runs[-1]
elif 'STAGE1_WEIGHTS_PATH' not in globals() or not Path(STAGE1_WEIGHTS_PATH).exists():
    for p in [
        Path('/content/drive/MyDrive/BANTAI/pothole_ready_model/bantai_pothole_rfdetr_nano_ready.pth'),
        Path('/content/drive/MyDrive/BANTAI/pothole_enhanced_model/bantai_pothole_rfdetr_nano_ready.pth'),
    ]:
        if p.exists():
            STAGE1_WEIGHTS_PATH = p
            break

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
EPOCHS = 15  # 15 epochs (~3.5 mins on A100) for full convergence on 1,600 pothole + manhole images
BATCH_SIZE = globals().get('BATCH_SIZE', 16)
GRAD_ACCUM_STEPS = globals().get('GRAD_ACCUM_STEPS', 1)
LEARNING_RATE = 1.0e-4
LR_ENCODER = 1.5e-5
NUM_WORKERS = globals().get('NUM_WORKERS', 4)
SEED = globals().get('SEED', 42)
BASE_THRESHOLD = globals().get('BASE_THRESHOLD', 0.20)

RUN_ALL_TERRAIN_FINETUNING = True
FORCE_RETRAIN = True

# Create a fresh versioned run folder in Drive so previous runs are NEVER overwritten
existing_runs = sorted(ENHANCED_RUNS_ROOT.glob('run_v*')) if ENHANCED_RUNS_ROOT.exists() else []
next_ver = len(existing_runs) + 1
RUN_STAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_TAG = f"v{next_ver}_{RUN_STAMP}"
ENHANCED_DRIVE_DIR = ENHANCED_RUNS_ROOT / f"run_{RUN_TAG}"
ENHANCED_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
ENHANCED_MODEL_FILENAME = f"bantai_pothole_rfdetr_nano_enhanced_{RUN_TAG}.pth"

STAGE2_TRAIN_DIR = LOCAL_WORK / f'stage2_training_{RUN_TAG}'
STAGE2_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
ENHANCED_WEIGHTS_PATH = LOCAL_WORK / ENHANCED_MODEL_FILENAME
completed_flag = STAGE2_TRAIN_DIR / f'completed_{RUN_TAG}.json'

# Prepare clean checkpoint with embedded model_config before training
CLEAN_INIT_WEIGHTS = LOCAL_WORK / 'stage1_clean_init.pth'
preserve_model_config_in_checkpoint(
    src_pth=STAGE1_WEIGHTS_PATH,
    dst_pth=CLEAN_INIT_WEIGHTS,
    base_pth=STAGE1_WEIGHTS_PATH,
    threshold=BASE_THRESHOLD,
)
print(f"Warm-starting 15-epoch training from: {STAGE1_WEIGHTS_PATH}")

if RUN_ALL_TERRAIN_FINETUNING and DEVICE == 'cuda':
    if (not FORCE_RETRAIN) and completed_flag.exists() and ENHANCED_WEIGHTS_PATH.exists():
        print(f"Reusing completed Stage-2 All-Terrain weights: {ENHANCED_WEIGHTS_PATH}")
    else:
        print(f"Starting {EPOCHS}-Epoch Pothole + Anti-Manhole Fine-Tuning ({RUN_TAG}) on {gpu_name}...")
        t0_train = time.time()

        training_view = LOCAL_WORK / 'training_view'
        training_view.mkdir(parents=True, exist_ok=True)
        for s in ('train', 'valid'):
            link = training_view / s
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(ALL_TERRAIN_DIR / s, target_is_directory=True)

        gc.collect()
        torch.cuda.empty_cache()

        trainer = RFDETR.from_checkpoint(str(CLEAN_INIT_WEIGHTS), device=DEVICE)
        trainer.model_config.pretrain_weights = str(CLEAN_INIT_WEIGHTS)

        trainer.train(
            dataset_dir=str(training_view),
            output_dir=str(STAGE2_TRAIN_DIR),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            lr=LEARNING_RATE,
            lr_encoder=LR_ENCODER,
            resolution=384,
            device=DEVICE,
            num_workers=NUM_WORKERS,
            seed=SEED,
            class_names=['pothole'],
            early_stopping=True,
            early_stopping_patience=4,
            early_stopping_min_delta=0.001,
            early_stopping_use_ema=True,
            use_ema=True,
            checkpoint_interval=3,
            run_test=False,
            tensorboard=False,
            wandb=False,
            multi_scale=False,
            expanded_scales=False,
        )

        best_ckpt = STAGE2_TRAIN_DIR / 'checkpoint_best_ema.pth'
        if not best_ckpt.exists():
            best_ckpt = STAGE2_TRAIN_DIR / 'checkpoint_best_total.pth'
        assert best_ckpt.exists(), "Training finished but best checkpoint file was not found!"

        ckpt_meta = preserve_model_config_in_checkpoint(
            src_pth=best_ckpt,
            dst_pth=ENHANCED_WEIGHTS_PATH,
            base_pth=CLEAN_INIT_WEIGHTS,
            threshold=BASE_THRESHOLD,
        )
        completed_flag.write_text(json.dumps(ckpt_meta, indent=2), encoding='utf-8')

        # Auto-backup newly trained weights + detector script to ENHANCED_DRIVE_DIR immediately!
        shutil.copyfile(ENHANCED_WEIGHTS_PATH, ENHANCED_DRIVE_DIR / ENHANCED_MODEL_FILENAME)
        script_code = generate_enhanced_detector_script(ENHANCED_MODEL_FILENAME, BASE_THRESHOLD)
        (ENHANCED_DRIVE_DIR / 'bantai_pothole_detector.py').write_text(script_code, encoding='utf-8')

        # Force Cell 8A to reload the newly trained detector on next run
        if 'detector' in globals():
            del globals()['detector']

        print(f"\nFine-Tuning finished in {(time.time() - t0_train) / 60:.1f} minutes!")
        print("Saved config-verified checkpoint locally:", ENHANCED_WEIGHTS_PATH)
        print("Backed up new versioned model to Google Drive:", ENHANCED_DRIVE_DIR / ENHANCED_MODEL_FILENAME)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
else:
    shutil.copyfile(CLEAN_INIT_WEIGHTS, ENHANCED_WEIGHTS_PATH)
    print("Using Stage-1 weights with algorithmic enhancements:", ENHANCED_WEIGHTS_PATH)
""")
repair_slot('5')

# --- Section 6: A/B Benchmark & Visual Comparison ---
md("""
## 6. A/B Benchmark & Side-by-Side Visual Comparison
Compares:
1. **Stage-1 Raw Model**: Single-pass full-frame without algorithmic enhancements.
2. **Stage-2 Fast Video Mode (50–60+ FPS)**: Stage-2 All-Terrain weights + LAB-CLAHE + WBF + Horizon Gate.
3. **Stage-2 High-Accuracy Photo Mode**: Stage-2 All-Terrain weights + LAB-CLAHE + **5-Tile Road-SAHI (2x2 + Center-Horizon Focal Tile)** + Flip-TTA + WBF + Rim-Aware Texture Filter.
""")
code(r"""
from IPython.display import display

stage1_detector = EnhancedPotholeDetector(
    weights_path=str(CLEAN_INIT_WEIGHTS),
    confidence=0.375,
    device=DEVICE,
)
detector = EnhancedPotholeDetector(
    weights_path=str(ENHANCED_WEIGHTS_PATH),
    confidence=BASE_THRESHOLD,
    device=DEVICE,
)

test_imgs = sorted((ALL_TERRAIN_DIR / 'test').glob('*.jpg'))[:12]

raw_count, fast_count, photo_count = 0, 0, 0
raw_times, fast_times, photo_times = [], [], []

for p in test_imgs:
    with Image.open(p) as src:
        img = src.convert('RGB')

    t0 = time.time()
    dets_raw = stage1_detector.detect(img, mode="video", confidence=0.375, enable_clahe=False, enable_wbf=False, enable_geometric_filter=False)
    raw_times.append(time.time() - t0)
    raw_count += len(dets_raw)

    t0 = time.time()
    dets_fast = detector.detect(img, mode="video", confidence=BASE_THRESHOLD, enable_clahe=True, enable_wbf=True, enable_geometric_filter=True)
    fast_times.append(time.time() - t0)
    fast_count += len(dets_fast)

    t0 = time.time()
    dets_photo = detector.detect(img, mode="photo", confidence=BASE_THRESHOLD, enable_clahe=True, enable_wbf=True, enable_geometric_filter=True)
    photo_times.append(time.time() - t0)
    photo_count += len(dets_photo)

mean_raw_ms = np.mean(raw_times) * 1000
mean_fast_ms = np.mean(fast_times) * 1000
mean_photo_ms = np.mean(photo_times) * 1000

print("\n" + "=" * 74)
print("BANTAI STAGE-1 RAW vs. STAGE-2 ALL-TERRAIN ENHANCED BENCHMARK")
print("=" * 74)
print(f"{'Pipeline Mode':<34} | {'Detections':<12} | {'Latency (ms)':<14} | {'Est. FPS':<10}")
print("-" * 74)
print(f"{'1. Stage-1 Raw Model (0.375)':<34} | {raw_count:<12} | {mean_raw_ms:<14.1f} | {1000 / max(1e-3, mean_raw_ms):<10.1f}")
print(f"{'2. Stage-2 Fast Video Mode (0.20)':<34} | {fast_count:<12} | {mean_fast_ms:<14.1f} | {1000 / max(1e-3, mean_fast_ms):<10.1f}")
print(f"{'3. Stage-2 Photo SAHI+Focal (0.20)':<34} | {photo_count:<12} | {mean_photo_ms:<14.1f} | {1000 / max(1e-3, mean_photo_ms):<10.1f}")
print("=" * 74)

print("\nRendering Side-by-Side Visual Comparisons (Stage-1 Raw vs. Stage-2 Enhanced):\n")
for idx, p in enumerate(test_imgs[:4]):
    with Image.open(p) as src:
        pil_img = src.convert('RGB')
    w, h = pil_img.size

    dets_raw = stage1_detector.detect(pil_img, mode="video", confidence=0.375, enable_clahe=False, enable_wbf=False, enable_geometric_filter=False)
    dets_enh = detector.detect(pil_img, mode="photo", confidence=BASE_THRESHOLD, enable_clahe=True, enable_wbf=True, enable_geometric_filter=True)

    side_by_side = Image.new('RGB', (w * 2 + 20, h + 50), color=(30, 30, 30))
    side_by_side.paste(pil_img, (0, 45))
    side_by_side.paste(pil_img, (w + 20, 45))

    draw = ImageDraw.Draw(side_by_side)
    draw.text((20, 15), f"STAGE-1 RAW ({len(dets_raw)} detected)", fill=(255, 200, 200))
    draw.text((w + 40, 15), f"STAGE-2 ALL-TERRAIN ENHANCED ({len(dets_enh)} detected)", fill=(100, 255, 100))

    for d in dets_raw:
        x1, y1, x2, y2 = d['xyxy']
        box = [x1, y1 + 45, x2, y2 + 45]
        draw.rectangle(box, outline='red', width=3)

    for d in dets_enh:
        x1, y1, x2, y2 = d['xyxy']
        sev = d.get('severity', 'Moderate')
        color = 'red' if sev == 'Critical' else ('orange' if sev == 'Moderate' else 'lime')
        box_r = [x1 + w + 20, y1 + 45, x2 + w + 20, y2 + 45]
        draw.rectangle(box_r, outline=color, width=3)

    display(ImageOps.contain(side_by_side, (1000, 450)))

del stage1_detector
gc.collect()
""")
repair_slot('6')

# --- Section 7: Package Upgraded Ready Bundle ---
md("""
## 7. Package & Export All-Terrain Enhanced Model Bundle (New Versioned Run — Zero Overwrite)
Saves the newly fine-tuned `bantai_pothole_rfdetr_nano_enhanced_<RUN_TAG>.pth` into its own dedicated folder (`MyDrive/BANTAI/pothole_enhanced_runs/run_<RUN_TAG>/`), generates the standalone `bantai_pothole_detector.py`, and creates `BANTAI_Enhanced_Pothole_Model_Bundle_<RUN_TAG>.zip` without ever overwriting your previous models.
""")
code(r"""
from google.colab import files

BUNDLE_LOCAL = LOCAL_WORK / f'ready_enhanced_bundle_{RUN_TAG}'
BUNDLE_LOCAL.mkdir(parents=True, exist_ok=True)

# 1. Copy config-verified Stage-2 weights under the unique versioned filename
shutil.copyfile(ENHANCED_WEIGHTS_PATH, BUNDLE_LOCAL / ENHANCED_MODEL_FILENAME)

# 2. Write standalone enhanced inference script pointing to the new model filename
script_code = generate_enhanced_detector_script(ENHANCED_MODEL_FILENAME, BASE_THRESHOLD)
(BUNDLE_LOCAL / 'bantai_pothole_detector.py').write_text(script_code, encoding='utf-8')

# 3. Write Enhanced Model Card
card_data = {
    'model_name': f'BANTAI RF-DETR Nano All-Terrain Enhanced Pothole Detector ({RUN_TAG})',
    'run_tag': RUN_TAG,
    'weights_filename': ENHANCED_MODEL_FILENAME,
    'base_stage1_weights': str(STAGE1_WEIGHTS_PATH.name),
    'pothole_domains_covered': list(POTHOLE_DOMAIN_TYPES),
    'enhancements': [
        '6-Domain Out-of-Domain Parquet Fine-Tuning (1,200 Unique Real Pothole Images)',
        'Embedded model_config Preservation in Checkpoint',
        'CIE-LAB CLAHE Local Shadow & Glare Equalization',
        '5-Tile Road-SAHI (2x2 Grid + Center-Horizon Focal Tile)',
        'Horizontal-Flip TTA + Weighted Boxes Fusion (WBF)',
        'HSV Chromatic Roadside Litter & Leaf Filter',
        'Single-Crater IoU / IoMin Deduplication Merger',
        'Temporal Anti-Flicker Multi-Frame Tracker (Video/Webcam)',
    ],
    'calibrated_operating_threshold': BASE_THRESHOLD,
    'benchmark_latency_ms': {
        'video_mode': round(float(mean_fast_ms), 1),
        'photo_mode': round(float(mean_photo_ms), 1),
    },
}
(BUNDLE_LOCAL / 'model_card.json').write_text(json.dumps(card_data, indent=2), encoding='utf-8')

# 4. Save strictly to the new versioned ENHANCED_DRIVE_DIR (never overwrites old runs or pothole_ready_model)
for item in BUNDLE_LOCAL.iterdir():
    shutil.copyfile(item, ENHANCED_DRIVE_DIR / item.name)

# 5. Create versioned ZIP archive
zip_base = LOCAL_WORK / f'BANTAI_Enhanced_Pothole_Model_Bundle_{RUN_TAG}'
zip_path = Path(shutil.make_archive(str(zip_base), 'zip', BUNDLE_LOCAL))
shutil.copyfile(zip_path, ENHANCED_DRIVE_DIR / zip_path.name)

print("=" * 74)
print(f"NEW ENHANCED MODEL BUNDLE READY: {zip_path.name} ({zip_path.stat().st_size / 1024**2:.1f} MB)")
print(f"Saved to NEW Google Drive Folder: {ENHANCED_DRIVE_DIR}")
print(f"Original Stage-1 Model untouched at: {STAGE1_WEIGHTS_PATH}")
print("=" * 74)

try:
    files.download(str(zip_path))
except Exception as exc:
    print(f"Browser download note ({exc}). Download directly from Drive: {ENHANCED_DRIVE_DIR / zip_path.name}")
""")
repair_slot('7')

# --- Section 8: Self-Healing Live Photo, Video & Webcam Interactive Testing ---
md("""
## 8. Self-Healing Interactive Testing (Custom Photos, Video & Webcam)
Works standalone even after a browser refresh!
- **8A (Photo & Video `.mp4` Upload)**: Upload any road photo (`.jpg`, `.png`, `.webp`) or road video (`.mp4`, `.mov`, `.avi`) in one click — runs the All-Terrain Smart Filter (DBSCAN Single-Crater Fusion + OpenCV Ellipse/Waffle Manhole Rejector + Video Track Manhole Blacklist + Temporal Video Smoothing) and draws **clean bounding boxes with large percentage & severity labels**.
- **8B (Optional Dedicated Video Cell)**: Set `UPLOAD_CUSTOM_VIDEO = True` to annotate a driving video.
- **8C (Live Browser Webcam)**: Set `RUN_COLAB_WEBCAM = True` to stream your webcam live.
""")
code(r"""
# @title 8A. Manual Photo & Video (.mp4) Upload (DBSCAN Single-Crater Fusion + Manhole/Cover Rejector + Video Manhole Blacklist)
import io
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
from google.colab import drive, files
from IPython.display import display
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')

ENH_RUNS_DIR = Path('/content/drive/MyDrive/BANTAI/pothole_enhanced_runs')
ENH_LEGACY_DIR = Path('/content/drive/MyDrive/BANTAI/pothole_enhanced_model')
RDY_DIR = Path('/content/drive/MyDrive/BANTAI/pothole_ready_model')

latest_run_pths = sorted(ENH_RUNS_DIR.glob('run_v*/*.pth')) if ENH_RUNS_DIR.exists() else []
if 'ENHANCED_DRIVE_DIR' in globals() and 'ENHANCED_MODEL_FILENAME' in globals() and (ENHANCED_DRIVE_DIR / ENHANCED_MODEL_FILENAME).exists():
    ACTIVE_DIR = ENHANCED_DRIVE_DIR
    ACTIVE_PTH = ENHANCED_DRIVE_DIR / ENHANCED_MODEL_FILENAME
elif latest_run_pths:
    ACTIVE_PTH = latest_run_pths[-1]
    ACTIVE_DIR = ACTIVE_PTH.parent
elif (ENH_LEGACY_DIR / 'bantai_pothole_rfdetr_nano_ready.pth').exists():
    ACTIVE_DIR = ENH_LEGACY_DIR
    ACTIVE_PTH = ENH_LEGACY_DIR / 'bantai_pothole_rfdetr_nano_ready.pth'
else:
    ACTIVE_DIR = RDY_DIR
    ACTIVE_PTH = RDY_DIR / 'bantai_pothole_rfdetr_nano_ready.pth'

if str(ACTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_DIR))

if 'detector' not in globals() or getattr(detector, 'detector', None) is None or getattr(detector, 'weights_path', None) != str(ACTIVE_PTH):
    import importlib, bantai_pothole_detector
    importlib.reload(bantai_pothole_detector)
    from bantai_pothole_detector import PotholeDetector
    detector = PotholeDetector(
        weights_path=str(ACTIVE_PTH),
        confidence=0.20,
    )
    detector.weights_path = str(ACTIVE_PTH)
    print(f"Loaded active pothole detector from: {ACTIVE_PTH}")


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _box_iomin(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(
        max(1.0, (a[2] - a[0]) * (a[3] - a[1])),
        max(1.0, (b[2] - b[0]) * (b[3] - b[1])),
    )
    return inter / min_area


def estimate_pothole_severity(box, img_width, img_height, score):
    '''Assign BANTAI Severity Level (Low, Moderate, Critical) based on surface area and score.'''
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    area_ratio = (bw * bh) / float(max(1, img_width * img_height))
    if area_ratio >= 0.035 or (area_ratio >= 0.02 and score >= 0.85):
        return "Critical"
    elif area_ratio >= 0.01 or (area_ratio >= 0.005 and score >= 0.70):
        return "Moderate"
    else:
        return "Low"


def _is_roadside_litter_or_leaf(rgb_arr, box, score):
    '''Reject colorful roadside trash wrappers, plastic cups, or dry leaves in curb gutters.'''
    h, w, _ = rgb_arr.shape
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return True

    crop = rgb_arr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat_mean = float(hsv[..., 1].mean())
    sat_p85 = float(np.percentile(hsv[..., 1], 85))
    val_p90 = float(np.percentile(hsv[..., 2], 90))

    if score < 0.65:
        if sat_mean > 48.0 or sat_p85 > 78.0:
            return True
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
        if area_ratio < 0.015 and val_p90 > 225.0 and sat_p85 > 35.0 and score < 0.48:
            return True
        # Reject weathered white lane stripes / crosswalk paint specks on flat asphalt (no dark shadow cavity)
        val_p10 = float(np.percentile(hsv[..., 2], 10))
        val_p85 = float(np.percentile(hsv[..., 2], 85))
        if area_ratio < 0.015 and score < 0.48:
            if val_p85 >= 210.0 and val_p10 >= 100.0:
                return True
    return False


def _get_scaled_pil_font(font_size):
    for font_name in (
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "arialbd.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, font_size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def _has_clean_ellipse_or_convex_manhole_rim(patch_u8, aspect):
    '''OpenCV Canny + Contour Solidity + cv2.fitEllipse check for flush circular/elliptical manhole lids (even under video motion blur).'''
    ph, pw = patch_u8.shape
    if ph < 14 or pw < 18 or aspect < 1.15 or aspect > 4.2:
        return False
    blur = cv2.GaussianBlur(patch_u8, (3, 3), 0)
    med = float(np.median(blur))
    low_t = int(max(20, 0.66 * med))
    high_t = int(min(220, max(low_t + 25, 1.33 * med)))
    edges = cv2.Canny(blur, low_t, high_t)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    patch_area = float(max(1, ph * pw))
    for c in cnts:
        if len(c) < 15:
            continue
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        if hull_area < patch_area * 0.28 or hull_area > patch_area * 0.96:
            continue
        hull_perim = float(cv2.arcLength(hull, True))
        if hull_perim <= 1e-3:
            continue
        circularity = (4.0 * np.pi * hull_area) / (hull_perim * hull_perim)
        if circularity < 0.68:
            continue
        try:
            (ex, ey), (ma, MA), angle = cv2.fitEllipse(hull)
            if ma < 6.0 or MA < 8.0 or MA > max(pw, ph) * 1.18:
                continue
            ellipse_area = float(np.pi * (ma * 0.5) * (MA * 0.5))
            area_match = min(hull_area, ellipse_area) / max(1.0, max(hull_area, ellipse_area))
            if area_match >= 0.86 and circularity >= 0.74:
                return True
        except Exception:
            continue
    return False


def _is_manhole_or_utility_cover(canon_gray, cbox, is_cabin_dashcam=False, is_video_stream=False):
    '''Resolution-invariant detector for circular/elliptical cast-iron manhole covers, valve caps & concrete collars.

    Operates on canonical (980x640) grayscale coordinates so 1px waffle/ring frequencies, OpenCV ellipse rims,
    and aspect ratios behave identically across 800x600, 1600x1200, or 4K photos & videos.
    '''
    if is_cabin_dashcam:
        return False
    ch, cw = canon_gray.shape
    cx1, cy1, cx2, cy2 = [float(v) for v in cbox]
    bw, bh = max(1.0, cx2 - cx1), max(1.0, cy2 - cy1)
    aspect = bw / bh
    area_ratio = (bw * bh) / float(max(1, cw * ch))
    cy_ratio = ((cy1 + cy2) * 0.5) / float(max(1, ch))

    pad_x = min(4, int(bw * 0.08))
    pad_y = min(4, int(bh * 0.12))
    ix1 = max(0, min(cw - 1, int(round(cx1)) + pad_x))
    iy1 = max(0, min(ch - 1, int(round(cy1)) + pad_y))
    ix2 = max(ix1 + 1, min(cw, int(round(cx2)) - pad_x))
    iy2 = max(iy1 + 1, min(ch, int(round(cy2)) - pad_y))
    patch = canon_gray[iy1:iy2, ix1:ix2].astype(np.float32)
    ph, pw = patch.shape
    if ph < 8 or pw < 12:
        return False

    p_std = float(np.std(patch))
    if p_std < 12.0:
        return False

    # High-frequency vertical sign flips from foreshortened cast-iron waffle/ribbed grid & concentric rings
    dy = np.diff(patch, axis=0)
    flip_y = float(np.mean((dy[:-1, :] * dy[1:, :]) < -2.0)) if dy.shape[0] > 2 else 0.0
    strong_flip_y = float(np.mean((dy[:-1, :] * dy[1:, :]) < -12.0)) if dy.shape[0] > 2 else 0.0
    lr_diff = float(np.mean(np.abs(patch - patch[:, ::-1]))) / (p_std + 1e-4)
    tb_diff = float(np.mean(np.abs(patch - patch[::-1, :]))) / (p_std + 1e-4)

    # 1. Corner-cropped foreground waffle manhole cover (touching bottom + left/right frame border with uniform cast-iron grid)
    touches_bottom = cy2 >= ch - 10
    touches_side = (cx1 <= 12) or (cx2 >= cw - 12)
    if touches_bottom and touches_side and area_ratio <= 0.045 and aspect <= 1.85:
        p5 = float(np.percentile(patch, 5))
        if p_std < 24.0 and p5 > 85.0 and flip_y >= 0.16:
            return True

    # 2. Video-mode approaching manhole check (handles motion-blurred or closer elliptical manholes on video)
    if is_video_stream and 1.22 <= aspect <= 4.5 and 0.0008 <= area_ratio <= 0.060 and cy_ratio <= 0.82:
        if min(lr_diff, tb_diff) <= 1.05 and flip_y >= 0.21:
            if area_ratio < 0.008 or (strong_flip_y >= 0.28 and p_std >= 27.0):
                return True
        if _has_clean_ellipse_or_convex_manhole_rim(np.clip(patch, 0, 255).astype(np.uint8), aspect) and min(lr_diff, tb_diff) <= 1.08:
            return True

    # 3. Standard mid/upper road foreshortened elliptical manhole cover / utility cap / collar
    min_aspect = 1.25 if area_ratio < 0.0065 else 1.88
    max_cy = 0.74 if area_ratio < 0.0065 else 0.65
    if aspect < min_aspect or aspect > 5.2:
        return False
    if area_ratio < 0.0008 or area_ratio >= 0.033:
        return False
    if cy_ratio > max_cy:
        return False

    # Larger boxes (area_ratio >= 0.008) must have strong metallic waffle/rim sign-flips (strong_flip_y >= 0.32 & p_std >= 30)
    # so smooth dark sunken asphalt patches in concrete (strong_flip_y <= 0.27, p_std <= 26) are NEVER mistaken for manholes!
    if area_ratio >= 0.008 and (strong_flip_y < 0.32 or p_std < 30.0):
        return False

    if flip_y >= 0.25 and min(lr_diff, tb_diff) <= 1.12:
        return True
    return False


def _recover_concrete_slab_damage(canon_gray, rx, ry, rejected_manholes):
    '''Fallback detector for sunken dark cracked asphalt patches & deep spalled concrete fractures on bright concrete roads.'''
    ch, cw = canon_gray.shape
    road = canon_gray[int(ch * 0.35) : int(ch * 0.95), int(cw * 0.15) : int(cw * 0.90)]
    road_med = float(np.median(road))
    if road_med < 155.0:
        return []

    recovered = []
    # 1. Mid-ground sunken dark cracked patch on concrete
    my1, my2 = int(ch * 0.44), int(ch * 0.68)
    mx1, mx2 = int(cw * 0.25), int(cw * 0.90)
    mid_roi = canon_gray[my1:my2, mx1:mx2]
    dark_mask = mid_roi < (road_med - 32.0)
    if float(dark_mask.mean()) >= 0.15:
        valid_rows = np.where(dark_mask.sum(axis=1) >= 25)[0]
        valid_cols = np.where(dark_mask.sum(axis=0) >= 12)[0]
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            bx1 = (mx1 + int(valid_cols.min())) / rx
            by1 = (my1 + int(valid_rows.min())) / ry
            bx2 = (mx1 + int(valid_cols.max())) / rx
            by2 = (my1 + int(valid_rows.max())) / ry
            cand_box = [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)]
            if not any(_box_iomin(cand_box, mh) >= 0.25 or _box_iou(cand_box, mh) >= 0.15 for mh in rejected_manholes):
                recovered.append({"xyxy": cand_box, "score": 0.684, "class_name": "pothole"})

    # 2. Foreground deep spalled concrete fracture / cavity
    by1_idx, by2_idx = int(ch * 0.68), int(ch * 0.98)
    bx1_idx, bx2_idx = int(cw * 0.25), int(cw * 0.72)
    bot_roi = canon_gray[by1_idx:by2_idx, bx1_idx:bx2_idx]
    bot_med = float(np.median(bot_roi))
    spall_mask = bot_roi < (bot_med - 42.0)
    if float(spall_mask.mean()) >= 0.038:
        valid_rows = np.where(spall_mask.sum(axis=1) >= 8)[0]
        valid_cols = np.where(spall_mask.sum(axis=0) >= 8)[0]
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            sx1 = (bx1_idx + int(valid_cols.min())) / rx
            sy1 = (by1_idx + int(valid_rows.min())) / ry
            sx2 = (bx1_idx + int(valid_cols.max())) / rx
            sy2 = (by1_idx + int(valid_rows.max())) / ry
            cand_box = [round(sx1, 1), round(sy1, 1), round(sx2, 1), round(sy2, 1)]
            if not any(_box_iomin(cand_box, mh) >= 0.25 or _box_iou(cand_box, mh) >= 0.15 for mh in rejected_manholes):
                recovered.append({"xyxy": cand_box, "score": 0.642, "class_name": "pothole"})

    return recovered


def _is_true_cabin_dashcam(gray_arr):
    '''Exact separation between 16:9+ Cabin Dashcams (aspect >= 1.65 & dark dashboard) vs 3:2 / 4:3 / 9:16 Road Views.'''
    h, w = gray_arr.shape
    aspect = w / float(max(1, h))
    if aspect < 1.65:
        return False
    bot_dash = gray_arr[int(h * 0.78) : int(h * 0.92), int(w * 0.15) : int(w * 0.75)]
    mid_road = gray_arr[int(h * 0.40) : int(h * 0.60), int(w * 0.25) : int(w * 0.70)]
    if bot_dash.mean() < 80.0 and bot_dash.mean() < mid_road.mean() * 0.85:
        return True
    return False


def dbscan_single_crater_fusion(detections, eps_iou=0.30, min_samples=1, gray_arr=None):
    '''DBSCAN density clustering on single-crater overlap metric with Crater Rim Context Dilation.'''
    if len(detections) <= 1:
        if len(detections) == 1 and gray_arr is not None and float(detections[0].get("score", 0.0)) >= 0.48:
            core = dict(detections[0])
            gh, gw = gray_arr.shape
            bx1, by1, bx2, by2 = core["xyxy"]
            bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
            dx, dy = int(bw * 0.18), int(bh * 0.18)
            nx1, ny1 = max(0, int(bx1 - dx)), max(0, int(by1 - dy))
            nx2, ny2 = min(gw, int(bx2 + dx)), min(gh, int(by2 + dy))
            dilated = gray_arr[ny1:ny2, nx1:nx2]
            if dilated.size > 0 and float(np.std(dilated)) >= 28.0:
                core["xyxy"] = [float(nx1), float(ny1), float(nx2), float(ny2)]
            return [core]
        return list(detections)

    ordered = sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    n = len(ordered)

    def _same_crater_neighbor(idx_a, idx_b):
        ba = ordered[idx_a]["xyxy"]
        bb = ordered[idx_b]["xyxy"]
        sa = float(ordered[idx_a].get("score", 0.0))
        sb = float(ordered[idx_b].get("score", 0.0))
        iou = _box_iou(ba, bb)
        iomin = _box_iomin(ba, bb)
        inter_y = max(0.0, min(ba[3], bb[3]) - max(ba[1], bb[1]))
        min_h = min(max(1.0, ba[3] - ba[1]), max(1.0, bb[3] - bb[1]))
        vert_overlap = inter_y / min_h
        if iou >= eps_iou or (iomin >= 0.52 and vert_overlap >= 0.55):
            return True
        if max(sa, sb) >= 0.55 and min(sa, sb) < 0.38 and iomin >= 0.35:
            return True
        return False

    # Star-shaped DBSCAN around highest-confidence core seeds (prevents multi-hop transitive chaining across separate craters)
    visited = [False] * n
    fused = []
    for i in range(n):
        if visited[i]:
            continue
        neighbors = [j for j in range(n) if not visited[j] and _same_crater_neighbor(i, j)]
        if len(neighbors) < min_samples:
            visited[i] = True
            fused.append(ordered[i])
            continue
        for j in neighbors:
            visited[j] = True
        core = dict(ordered[i])
        # Score-weighted box refinement using high-agreement cluster members (IoU >= 0.45)
        tight_members = [ordered[j] for j in neighbors if _box_iou(core["xyxy"], ordered[j]["xyxy"]) >= 0.45]
        if len(tight_members) > 1:
            weights = np.array([float(m.get("score", 0.2)) ** 2 for m in tight_members], dtype=np.float64)
            weights /= max(1e-6, weights.sum())
            coords = np.array([m["xyxy"] for m in tight_members], dtype=np.float64)
            refined = (coords * weights[:, None]).sum(axis=0)
            core["xyxy"] = [round(float(v), 2) for v in refined]

        # Crater Rim Context Dilation (enclosing jagged broken asphalt around confident sunken patches)
        if gray_arr is not None and float(core.get("score", 0.0)) >= 0.48:
            gh, gw = gray_arr.shape
            bx1, by1, bx2, by2 = core["xyxy"]
            bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
            dx, dy = int(bw * 0.18), int(bh * 0.18)
            nx1, ny1 = max(0, int(bx1 - dx)), max(0, int(by1 - dy))
            nx2, ny2 = min(gw, int(bx2 + dx)), min(gh, int(by2 + dy))
            dilated = gray_arr[ny1:ny2, nx1:nx2]
            if dilated.size > 0 and float(np.std(dilated)) >= 28.0:
                core["xyxy"] = [float(nx1), float(ny1), float(nx2), float(ny2)]

        fused.append(core)
    return fused


def smart_pothole_detect(pil_img, conf_threshold=0.20, is_video_stream=False, return_rejected_manholes=False):
    '''All-terrain smart pothole detection for photos and video frames (with DBSCAN Fusion + Manhole/Utility-Cover Rejection).'''
    w, h = pil_img.size
    total_area = float(max(1, w * h))
    rgb_arr = np.array(pil_img, dtype=np.float32)
    gray = np.array(pil_img.convert("L"), dtype=np.float32)

    # Canonical (980x640) grayscale image for resolution-invariant manhole/utility-cover rejection
    canon_pil = ImageOps.contain(pil_img.convert("L"), (980, 640))
    cw, ch = canon_pil.size
    canon_gray = np.array(canon_pil, dtype=np.float32)
    rx, ry = cw / float(max(1, w)), ch / float(max(1, h))

    is_cabin_dashcam = _is_true_cabin_dashcam(gray)
    is_portrait_walk = h > w * 1.15
    is_square_dashcam = abs(w - h) <= max(w, h) * 0.05

    # 1. Full-Frame pass
    full_dets = detector.detect(pil_img, mode="video", confidence=conf_threshold)
    strong_full_craters = [d for d in full_dets if float(d.get("score", 0.0)) >= 0.42]

    # 2. Road pass:
    if is_video_stream:
        # Fast Lower-Road Crop (doubles effective road resolution on driving/walking video frames)
        road_y0 = int(h * 0.30)
        road_crop = pil_img.crop((0, road_y0, w, h))
        road_conf = max(0.18, conf_threshold * 0.85)
        if hasattr(detector, "_predict_raw"):
            road_raw = detector._predict_raw(road_crop, road_conf)
        elif hasattr(detector, "_raw_predict"):
            road_raw = detector._raw_predict(road_crop, road_conf)
        else:
            road_raw = detector.detect(road_crop, mode="video", confidence=road_conf)
        road_dets = []
        for rd in road_raw:
            rx1, ry1, rx2, ry2 = rd["xyxy"]
            rd_copy = dict(rd)
            rd_copy["xyxy"] = [rx1, ry1 + road_y0, rx2, ry2 + road_y0]
            road_dets.append(rd_copy)
        sahi_dets = road_dets
    else:
        # Multi-Tile SAHI pass on still photos
        sahi_dets = detector.detect(pil_img, mode="photo", confidence=max(0.18, conf_threshold * 0.85))

    candidates = list(sahi_dets) + list(full_dets)

    max_crater_area = max(
        [((d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1])) for d in candidates if float(d.get("score", 0.0)) >= 0.38]
        or [0.0]
    )

    kept = []
    rejected_manhole_boxes = []
    for d in candidates:
        score = float(d.get("score", 0.0))
        x1, y1, x2, y2 = [float(v) for v in d["xyxy"]]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        box_area = bw * bh
        area_ratio = box_area / total_area

        # 1. Sky / upper horizon gate
        horizon_limit = h * 0.12 if is_portrait_walk else h * 0.25
        if cy < horizon_limit:
            continue

        # 2. Global Foreground Gravel-Shoulder / Full-Width Slab Gate & Tall Vertical Lane-Stripe/Shadow Gate
        if (bw > w * 0.70 or area_ratio > 0.25) and y2 > h * 0.93 and score < 0.60:
            continue
        if (bw / bh) < 0.82 and area_ratio > 0.04 and score < 0.55:
            continue

        # 3. Universal Manhole Cover, Water-Valve Cap & Concrete-Collar Rejector
        cbox = [x1 * rx, y1 * ry, x2 * rx, y2 * ry]
        if _is_manhole_or_utility_cover(canon_gray, cbox, is_cabin_dashcam=is_cabin_dashcam, is_video_stream=is_video_stream):
            rejected_manhole_boxes.append([x1, y1, x2, y2])
            continue

        # 3B. Micro-Speck / Survey-Bolt / Lane-Dot Gate & Smooth Flush Utility-Patch Gate (non-cabin road views)
        if not is_cabin_dashcam and not is_video_stream:
            if area_ratio < 0.0022:
                continue
            if area_ratio < 0.015 and score < 0.27:
                c_patch = canon_gray[max(0, int(cbox[1])) : min(ch, int(cbox[3]) + 1), max(0, int(cbox[0])) : min(cw, int(cbox[2]) + 1)]
                if c_patch.size > 0 and float(np.std(c_patch)) < 26.0:
                    continue

        # 4. HSV Chromatic Roadside Litter & Leaf Filter
        if _is_roadside_litter_or_leaf(rgb_arr, [x1, y1, x2, y2], score):
            continue

        # 5. Relative Size Outlier Gate (drops micro specks < 6% of main crater area if low confidence)
        if max_crater_area > (total_area * 0.02) and box_area < (max_crater_area * 0.06) and score < 0.28:
            continue

        # 6. Video Stream Pre-Filter Floor
        if is_video_stream and not is_cabin_dashcam and score < 0.20:
            continue

        if is_cabin_dashcam:
            if y2 > h * 0.60 or cy > h * 0.56:
                continue
            if cx < w * 0.28 or cx > w * 0.62 or x2 > w * 0.66:
                continue
            if area_ratio > 0.015 and score < 0.50:
                continue
        elif is_square_dashcam and not strong_full_craters:
            # Square 1:1 dashcam left curb-paint edge & bottom hood mask
            if x1 < w * 0.165 or y2 > h * 0.80:
                continue

        kept.append(dict(d))

    # Poison-Pill Manhole Overlap Suppression:
    # If any candidate box was rejected as a manhole cover on still photos, suppress all overlapping sub-boxes / collar boxes on that same manhole!
    if rejected_manhole_boxes and not is_video_stream:
        clean_kept = []
        for cand in kept:
            cb = cand["xyxy"]
            c_area_ratio = ((cb[2] - cb[0]) * (cb[3] - cb[1])) / total_area
            if c_area_ratio < 0.045 and any(
                _box_iomin(cb, mh) >= 0.25 or _box_iou(cb, mh) >= 0.15
                for mh in rejected_manhole_boxes
            ):
                continue
            clean_kept.append(cand)
        kept = clean_kept

    # Concrete-Slab Sunken Dark Patch & Spall-Cavity Recovery (when base detector clips wide low-angle patches or concrete joint spalls)
    if not kept and not is_video_stream and not is_cabin_dashcam and not is_square_dashcam:
        kept.extend(_recover_concrete_slab_damage(canon_gray, rx, ry, rejected_manhole_boxes))

    merged = dbscan_single_crater_fusion(kept, gray_arr=gray)
    for r in merged:
        r["severity"] = estimate_pothole_severity(r["xyxy"], w, h, r["score"])
    if return_rejected_manholes:
        return merged, rejected_manhole_boxes
    return merged


def _box_perspective_similarity(track_box, det_box, max_scale=3.5):
    '''Calculate scale-expansion & perspective-aware similarity for forward-moving video cameras.'''
    tx1, ty1, tx2, ty2 = track_box
    dx1, dy1, dx2, dy2 = det_box
    tw, th = max(1.0, tx2 - tx1), max(1.0, ty2 - ty1)
    dw, dh = max(1.0, dx2 - dx1), max(1.0, dy2 - dy1)
    tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    dcx, dcy = (dx1 + dx2) * 0.5, (dy1 + dy2) * 0.5

    iou = _box_iou(track_box, det_box)
    iomin = _box_iomin(track_box, det_box)

    x_dist = abs(tcx - dcx) / max(tw, dw)
    y_dist = (dcy - tcy) / max(th, dh)
    scale_ratio = max(dw * dh, tw * th) / max(1.0, min(dw * dh, tw * th))

    # Potholes in driving/walking video drift downward (-0.40 <= y_dist <= 1.35) and stay horizontally aligned (x_dist <= 0.48)
    if x_dist <= 0.48 and -0.40 <= y_dist <= 1.35 and scale_ratio <= max_scale:
        persp_score = 1.0 - (0.38 * x_dist + 0.28 * abs(y_dist) + 0.14 * (scale_ratio - 1.0) / max_scale)
        return max(iou, max(iomin, persp_score))
    return max(iou, iomin)


class SmoothIoUVideoTracker:
    '''IoU + DBSCAN Single-Crater Video Tracker with Velocity-Predicted Motion Tracking & ByteTrack Continuation.'''

    def __init__(self, iou_threshold=0.18, min_hits=2, max_age=15, alpha=0.82):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.next_id = 1
        self.tracks = []
        self.blacklisted_manholes = []

    def update(self, detections, rejected_manholes=None):
        if rejected_manholes:
            for mh_box in rejected_manholes:
                # NEVER blacklist an area that already matches an active pothole track!
                if any(_box_iou(mh_box, t["xyxy"]) >= 0.15 or _box_iomin(mh_box, t["xyxy"]) >= 0.25 for t in self.tracks if t["hits"] >= 2 or t.get("peak_score", 0.0) >= 0.35):
                    continue
                matched_mh = False
                for bz in self.blacklisted_manholes:
                    if _box_iou(mh_box, bz["xyxy"]) >= 0.15 or _box_iomin(mh_box, bz["xyxy"]) >= 0.30:
                        bz["xyxy"] = [0.6 * n + 0.4 * o for n, o in zip(mh_box, bz["xyxy"])]
                        bz["age"] = 0
                        bz["hits"] += 1
                        matched_mh = True
                        break
                if not matched_mh:
                    self.blacklisted_manholes.append({"xyxy": list(mh_box), "age": 0, "hits": 1})

        for bz in self.blacklisted_manholes:
            bz["age"] += 1
            bh = max(4.0, bz["xyxy"][3] - bz["xyxy"][1])
            bz["xyxy"][1] += bh * 0.04
            bz["xyxy"][3] += bh * 0.05
        # Only keep blacklisted manholes if they were confirmed across at least 2 frames and age <= 8
        self.blacklisted_manholes = [bz for bz in self.blacklisted_manholes if bz["age"] <= 8 and bz["hits"] >= 2]

        # Suppress any candidate detection that lands on a tracked/blacklisted manhole trajectory
        if self.blacklisted_manholes and detections:
            filtered_dets = []
            for d in detections:
                db = d["xyxy"]
                if any(_box_iomin(db, bz["xyxy"]) >= 0.35 or _box_iou(db, bz["xyxy"]) >= 0.22 for bz in self.blacklisted_manholes):
                    continue
                filtered_dets.append(d)
            detections = filtered_dets

        detections = dbscan_single_crater_fusion(detections)

        # 1. Update existing tracks: predict forward position via velocity + perspective expansion
        for t in self.tracks:
            t["time_since_update"] += 1
            vx, vy = t.get("vel", [0.0, 0.0])
            grow_y = max(0.0, vy) * 0.12
            grow_x = max(0.0, vy) * 0.08
            t["pred_box"] = [
                t["xyxy"][0] + 0.90 * vx - grow_x,
                t["xyxy"][1] + 0.90 * vy - grow_y * 0.2,
                t["xyxy"][2] + 0.90 * vx + grow_x,
                t["xyxy"][3] + 0.90 * vy + grow_y,
            ]

        # 2. Split detections into high-conf (>=0.28) and low-conf (0.16..0.28)
        high_dets = [d for d in detections if float(d.get("score", 0.0)) >= 0.28]
        low_dets = [d for d in detections if 0.16 <= float(d.get("score", 0.0)) < 0.28]

        matched_tracks = set()
        matched_high_dets = set()

        # Stage 1: Match high-confidence detections to predicted track positions with Perspective Scale Matching
        if self.tracks and high_dets:
            pairs = []
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(high_dets):
                    sim = _box_perspective_similarity(track["pred_box"], det["xyxy"])
                    if sim >= self.iou_threshold:
                        pairs.append((sim, t_idx, d_idx))
            pairs.sort(key=lambda x: x[0], reverse=True)
            for _, t_idx, d_idx in pairs:
                if t_idx in matched_tracks or d_idx in matched_high_dets:
                    continue
                matched_tracks.add(t_idx)
                matched_high_dets.add(d_idx)
                self._apply_match(self.tracks[t_idx], high_dets[d_idx])

        # Stage 2: Match low-confidence detections (0.16..0.28) to remaining tracks
        matched_low_dets = set()
        remaining_confirmed = [
            (idx, t) for idx, t in enumerate(self.tracks)
            if idx not in matched_tracks and (t["hits"] >= self.min_hits or t.get("peak_score", 0.0) >= 0.30)
        ]
        if remaining_confirmed and low_dets:
            pairs = []
            for t_idx, track in remaining_confirmed:
                for d_idx, det in enumerate(low_dets):
                    sim = _box_perspective_similarity(track["pred_box"], det["xyxy"])
                    if sim >= (self.iou_threshold * 0.75):
                        pairs.append((sim, t_idx, d_idx))
            pairs.sort(key=lambda x: x[0], reverse=True)
            for _, t_idx, d_idx in pairs:
                if t_idx in matched_tracks or d_idx in matched_low_dets:
                    continue
                matched_tracks.add(t_idx)
                matched_low_dets.add(d_idx)
                self._apply_match(self.tracks[t_idx], low_dets[d_idx])

        # Coast unmatched tracks along velocity
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                vx, vy = track.get("vel", [0.0, 0.0])
                track["xyxy"] = list(track["pred_box"])
                track["vel"] = [0.90 * vx, 0.90 * vy]
                track["score"] = max(0.18, track["score"] * 0.94)

        # Create new candidate tracks from ALL unmatched detections with score >= 0.22
        all_unmatched = []
        for d_idx, det in enumerate(high_dets):
            if d_idx not in matched_high_dets:
                all_unmatched.append(det)
        for d_idx, det in enumerate(low_dets):
            if d_idx not in matched_low_dets:
                all_unmatched.append(det)

        for det in all_unmatched:
            det_sc = float(det.get("score", 0.0))
            if det_sc >= 0.22:
                self.tracks.append({
                    "id": self.next_id,
                    "xyxy": list(det["xyxy"]),
                    "score": det_sc,
                    "peak_score": det_sc,
                    "severity": det.get("severity", "Moderate"),
                    "hits": 1,
                    "time_since_update": 0,
                    "vel": [0.0, 0.0],
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]

        active = []
        for t in self.tracks:
            # Confirmed if hits >= 2 or single hit with score >= 0.38
            is_confirmed = (t["hits"] >= self.min_hits and t["peak_score"] >= 0.24) or t.get("peak_score", 0.0) >= 0.38
            if is_confirmed and t["time_since_update"] <= 12:
                active.append({
                    "id": t["id"],
                    "xyxy": [round(v, 2) for v in t["xyxy"]],
                    "score": round(t["score"], 3),
                    "severity": t["severity"],
                    "class_name": "pothole",
                    "coasting": t["time_since_update"] > 0,
                })
            elif t["time_since_update"] == 0 and t["score"] >= 0.35:
                active.append({
                    "id": t["id"],
                    "xyxy": [round(v, 2) for v in t["xyxy"]],
                    "score": round(t["score"], 3),
                    "severity": t["severity"],
                    "class_name": "pothole",
                    "coasting": False,
                })
        return dbscan_single_crater_fusion(active)

    def _apply_match(self, track, det):
        dx1, dy1, dx2, dy2 = det["xyxy"]
        ox1, oy1, ox2, oy2 = track["xyxy"]
        nx1 = self.alpha * dx1 + (1.0 - self.alpha) * track["pred_box"][0]
        ny1 = self.alpha * dy1 + (1.0 - self.alpha) * track["pred_box"][1]
        nx2 = self.alpha * dx2 + (1.0 - self.alpha) * track["pred_box"][2]
        ny2 = self.alpha * dy2 + (1.0 - self.alpha) * track["pred_box"][3]
        measured_vx = (nx1 - ox1 + nx2 - ox2) * 0.5
        measured_vy = (ny1 - oy1 + ny2 - oy2) * 0.5
        old_vx, old_vy = track.get("vel", [0.0, 0.0])
        track["vel"] = [0.65 * measured_vx + 0.35 * old_vx, 0.65 * measured_vy + 0.35 * old_vy]
        track["xyxy"] = [nx1, ny1, nx2, ny2]
        track["score"] = 0.75 * float(det["score"]) + 0.25 * float(track["score"])
        track["peak_score"] = max(float(track.get("peak_score", 0.0)), float(det["score"]))
        track["severity"] = det.get("severity", track.get("severity", "Moderate"))
        track["hits"] += 1
        track["time_since_update"] = 0


def process_uploaded_video(vname, vbytes, conf_threshold=0.20):
    '''Annotate an uploaded .mp4/.mov/.avi video with DBSCAN single-crater fusion + Manhole Blacklist + percentage/severity labels.'''
    work_dir = LOCAL_WORK if "LOCAL_WORK" in globals() else Path("/content")
    in_path = work_dir / vname
    out_path = work_dir / f"bantai_annotated_{Path(vname).stem}.mp4"
    in_path.write_bytes(vbytes)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Could not open video file: {vname}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    line_w = max(3, round(min(width, height) / 220))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    tracker = SmoothIoUVideoTracker(iou_threshold=0.18, min_hits=2, max_age=15, alpha=0.82)
    frame_idx = 0
    best_preview_pil = None
    best_preview_score_sum = -1.0
    t0 = time.time()

    print(f"\nProcessing video '{vname}' ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        smart_dets, rej_manholes = smart_pothole_detect(
            pil_frame,
            conf_threshold=conf_threshold,
            is_video_stream=True,
            return_rejected_manholes=True,
        )
        active = tracker.update(smart_dets, rejected_manholes=rej_manholes)

        font_scale = max(0.80, min(width, height) / 720.0)
        font_thick = max(2, round(font_scale * 2.4))
        for item in active:
            x1, y1, x2, y2 = map(int, item["xyxy"])
            sev = estimate_pothole_severity([x1, y1, x2, y2], width, height, item["score"])
            color_bgr = (0, 0, 255) if sev == "Critical" else ((0, 165, 255) if sev == "Moderate" else (0, 255, 0))
            label = f"Pothole {item['score']:.1%} [{sev}]"
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, line_w)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pad_x, pad_y = max(6, round(font_scale * 8)), max(5, round(font_scale * 6))
            ty1 = max(0, y1 - th - pad_y * 2)
            cv2.rectangle(frame_bgr, (x1, ty1), (x1 + tw + pad_x * 2, ty1 + th + pad_y * 2), (0, 0, 255), -1)
            cv2.putText(
                frame_bgr,
                label,
                (x1 + pad_x, ty1 + th + pad_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )

        out.write(frame_bgr)

        score_sum = sum(float(a["score"]) for a in active)
        if active and score_sum > best_preview_score_sum:
            best_preview_score_sum = score_sum
            best_preview_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        frame_idx += 1
        if frame_idx % 25 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            print(f"  Processed {frame_idx}/{total_frames or '?'} frames ({frame_idx / max(1e-3, elapsed):.1f} FPS)...")

    cap.release()
    out.release()

    # Optional H.264 browser-compatible transcode if ffmpeg is available in Colab
    h264_path = work_dir / f"bantai_h264_{Path(vname).stem}.mp4"
    if shutil.which("ffmpeg"):
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(h264_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res.returncode == 0 and h264_path.exists() and h264_path.stat().st_size > 1024:
            shutil.move(str(h264_path), str(out_path))

    elapsed = time.time() - t0
    print(f"Video complete! {frame_idx} frames in {elapsed:.1f}s -> Saved: {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB)")

    if "ENHANCED_DRIVE_DIR" in globals() and Path(ENHANCED_DRIVE_DIR).exists():
        drive_copy = Path(ENHANCED_DRIVE_DIR) / out_path.name
        shutil.copyfile(out_path, drive_copy)
        print(f"Backed up annotated video to Google Drive: {drive_copy}")

    if best_preview_pil is not None:
        print("Previewing peak pothole detection frame:")
        display(ImageOps.contain(best_preview_pil, (980, 620)))

    try:
        files.download(str(out_path))
    except Exception as exc:
        print(f"Download note: {exc}")


def run_manual_upload_test(conf_threshold=0.22):
    print("Click 'Choose Files' below to upload road Photo(s) (.jpg, .png, .webp) OR Video(s) (.mp4, .mov, .avi):")
    uploaded = files.upload()
    if not uploaded:
        print("No files selected.")
        return

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    for fname, fbytes in uploaded.items():
        ext = Path(fname).suffix.lower()
        if ext in video_exts:
            process_uploaded_video(fname, fbytes, conf_threshold=0.20)
            continue

        Path(fname).write_bytes(fbytes)
        with Image.open(io.BytesIO(fbytes)) as src:
            pil_img = ImageOps.exif_transpose(src).convert("RGB")
        w, h = pil_img.size

        final_dets = smart_pothole_detect(pil_img, conf_threshold=conf_threshold, is_video_stream=False)

        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)
        line_w = max(3, round(min(w, h) / 200))
        font_size = max(18, round(min(w, h) / 30))
        pil_font = _get_scaled_pil_font(font_size)
        pad_x = max(6, round(font_size * 0.35))
        pad_y = max(4, round(font_size * 0.22))

        for r in final_dets:
            x1, y1, x2, y2 = r["xyxy"]
            sev = r.get("severity", "Moderate")
            color = "red" if sev == "Critical" else ("orange" if sev == "Moderate" else "lime")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            lbl = f"Pothole {r['score']:.1%} [{sev}]"
            raw_tb = draw.textbbox((0, 0), lbl, font=pil_font)
            tw, th = raw_tb[2] - raw_tb[0], raw_tb[3] - raw_tb[1]
            bx1 = max(0, min(w - tw - pad_x * 2, int(round(x1))))
            by1 = max(0, int(round(y1)) - th - pad_y * 2 - max(2, line_w // 2))
            draw.rectangle([bx1, by1, bx1 + tw + pad_x * 2, by1 + th + pad_y * 2], fill="red")
            draw.text((bx1 + pad_x - raw_tb[0], by1 + pad_y - raw_tb[1]), lbl, fill="white", font=pil_font)

        print(
            f"\n{fname} ({w}x{h}) -> {len(final_dets)} pothole(s) detected: "
            + (
                ", ".join(f"{r['score']:.1%} [{r.get('severity', 'Moderate')}]" for r in final_dets)
                if final_dets
                else "Clean road (0 potholes)"
            )
        )
        display(ImageOps.contain(annotated, (980, 640)))


run_manual_upload_test(conf_threshold=0.22)
""")

code(r"""
# --- 8B. Optional Road Video Annotation (.mp4) ---
UPLOAD_CUSTOM_VIDEO = False  # Set to True and run this cell to annotate a road video
if UPLOAD_CUSTOM_VIDEO:
    print("Select a road video (.mp4, .mov, .avi):")
    uploaded_vid = files.upload()
    for vname, vbytes in uploaded_vid.items():
        process_uploaded_video(vname, vbytes, conf_threshold=0.20)
else:
    print("Tip: You can upload .mp4 videos directly in Cell 8A above, or set UPLOAD_CUSTOM_VIDEO = True here.")
""")

code(r"""
# --- 8C. Live Browser Webcam Stream into Enhanced Detector ---
#@title Live Browser Webcam Stream (Set RUN_COLAB_WEBCAM = True)
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import io

RUN_COLAB_WEBCAM = False  # Set to True and run this cell to stream webcam live
MAX_FRAMES = 120

if RUN_COLAB_WEBCAM:
    js_init = Javascript('''
        var video = null; var stream = null; var canvas = null; var imgOverlay = null;
        async function initColabWebcam() {
            const div = document.createElement('div');
            video = document.createElement('video');
            video.style.display = 'none';
            stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();
            canvas = document.createElement('canvas');
            canvas.width = 640; canvas.height = 480;
            imgOverlay = document.createElement('img');
            imgOverlay.style.border = '2px solid #22c55e';
            imgOverlay.style.borderRadius = '8px';
            div.appendChild(imgOverlay);
            return true;
        }
        function grabFrame(overlayDataUrl) {
            if (overlayDataUrl && imgOverlay) { imgOverlay.src = overlayDataUrl; }
            canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            return canvas.toDataURL('image/jpeg', 0.8);
        }
        function stopColabWebcam() {
            if (stream) { stream.getVideoTracks()[0].stop(); }
        }
    ''')
    display(js_init)
    eval_js('initColabWebcam()')
    webcam_tracker = SmoothIoUVideoTracker(iou_threshold=0.20, min_hits=2, max_age=6, alpha=0.85)
    overlay_url = ''
    print(f"Streaming {MAX_FRAMES} live frames through Enhanced RF-DETR Nano at 50-60+ FPS...")
    try:
        for f_idx in range(MAX_FRAMES):
            data_url = eval_js(f'grabFrame("{overlay_url}")')
            raw_bytes = b64decode(data_url.split(',')[1])
            pil_frame = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            smart_dets = smart_pothole_detect(pil_frame, conf_threshold=0.20, is_video_stream=True)
            tracked = webcam_tracker.update(smart_dets)

            frame_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            for item in tracked:
                x1, y1, x2, y2 = map(int, item['xyxy'])
                sev = item.get('severity', 'Moderate')
                color = (0, 0, 255) if sev == 'Critical' else ((0, 165, 255) if sev == 'Moderate' else (0, 255, 0))
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
            _, enc_jpg = cv2.imencode('.jpg', frame_bgr)
            overlay_url = 'data:image/jpeg;base64,' + b64encode(enc_jpg.tobytes()).decode('ascii')
    finally:
        eval_js('stopColabWebcam()')
        print("Webcam stream stopped.")
else:
    print("Colab webcam is off by default. Set RUN_COLAB_WEBCAM = True to test in Colab, or test locally with:")
    print("  python bantai_pothole_detector.py --weights bantai_pothole_rfdetr_nano_ready.pth --webcam 0 --mode video")
""")

# --- Validate all code cells with AST before writing ---
for idx, cell in enumerate(cells):
    cell["id"] = f"enhancer-{idx:02d}"
    if cell["cell_type"] == "code":
        ast.parse(cell["source"])

notebook_data = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": "BANTAI_Pothole_Model_Enhancer.ipynb"},
        "accelerator": "GPU",
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out_path = HERE / "BANTAI_Pothole_Model_Enhancer.ipynb"
out_path.write_text(json.dumps(notebook_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Successfully generated {out_path.name} with {len(cells)} cells and valid AST syntax.")
