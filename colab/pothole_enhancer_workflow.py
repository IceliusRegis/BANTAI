"""BANTAI Pothole Model Enhancer Workflow.

Provides modular post-training enhancements for the RF-DETR Nano pothole model:
- LAB-channel CLAHE local contrast & shadow equalization
- Road-focused Slicing Aided Hyper Inference (SAHI)
- Weighted Boxes Fusion (WBF) across multi-query & multi-tile predictions
- Horizon & road-stripe aspect-ratio geometric filtering
- Laplacian edge-roughness & shadow/manhole texture verification
- Pothole severity indexing (Low / Moderate / Critical)
- Real-time video/webcam pipeline with temporal anti-flicker tracking
- Standalone inference engine generator
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import io
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

DEFAULT_WEIGHTS_NAME = "bantai_pothole_rfdetr_nano_ready.pth"
DEFAULT_CONFIDENCE = 0.20


# ---------------------------------------------------------------------------
# 0. All-Terrain Pothole Domain Augmentation & Fast Parquet Dataset Loader
# ---------------------------------------------------------------------------

POTHOLE_DOMAIN_TYPES = (
    "dry_clahe",
    "water_puddle_glare",
    "muddy_snow_thaw",
    "harsh_shadow_dusk",
    "concrete_patched_road",
    "distant_highway_scale",
)


def augment_all_pothole_types(
    img: Image.Image,
    bboxes_xywh: List[List[float]],
    variant_idx: int,
    seed: int = 42,
) -> Tuple[Image.Image, List[List[float]], str]:
    """Transform an image and its [x, y, w, h] pothole boxes across 6 real-world pothole domains.

    Covers:
    0 - dry_clahe: Sunlit/dry spalling with local contrast boost
    1 - water_puddle_glare: Specular sky reflection inside pothole cavity with wet dark rim
    2 - muddy_snow_thaw: Opaque beige/brown muddy water puddle fill inside fractured rim
    3 - harsh_shadow_dusk: Diagonal tree/bridge cast shadow + dusk low-light gamma
    4 - concrete_patched_road: Weathered grey concrete surface + cold-patch seam lines
    5 - distant_highway_scale: Wide-angle horizon zoom-out for small distant dashcam potholes
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    arr = np.array(rgb, dtype=np.float32)
    rng = np.random.RandomState(seed + variant_idx)
    domain = POTHOLE_DOMAIN_TYPES[variant_idx % len(POTHOLE_DOMAIN_TYPES)]
    new_boxes: List[List[float]] = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in bboxes_xywh]

    if domain == "dry_clahe":
        return apply_lab_clahe(rgb, clip_limit=2.2), new_boxes, domain

    if domain in ("water_puddle_glare", "muddy_snow_thaw"):
        # Simulate bright specular sky reflection or muddy water inside the center of each pothole box
        # while darkening the fractured asphalt rim around the cavity:
        for bx, by, bw, bh in new_boxes:
            if bw < 8 or bh < 8:
                continue
            x1, y1 = max(0, int(bx)), max(0, int(by))
            x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue
            crop = arr[y1:y2, x1:x2]
            # Darken outer cavity rim slightly
            crop *= 0.82
            # Add elliptical water puddle inside inner 55% of box
            ch, cw, _ = crop.shape
            cy_c, cx_c = ch / 2.0, cw / 2.0
            ry, rx = max(2.0, ch * 0.28), max(2.0, cw * 0.28)
            yy, xx = np.ogrid[:ch, :cw]
            mask = ((yy - cy_c) / ry) ** 2 + ((xx - cx_c) / rx) ** 2 <= 1.0
            if domain == "water_puddle_glare":
                # Specular sky glare (cool high-luminance tint, preserving underlying texture)
                crop[mask] = np.clip(crop[mask] * 1.25 + 15.0, 0, 255)
            else:
                # Muddy snow-thaw puddle (warm silt tint, preserving underlying texture)
                crop[mask, 0] = np.clip(crop[mask, 0] * 1.18 + 12.0, 0, 255)
                crop[mask, 1] = np.clip(crop[mask, 1] * 1.12 + 10.0, 0, 255)
                crop[mask, 2] = np.clip(crop[mask, 2] * 0.98 + 5.0, 0, 255)
            arr[y1:y2, x1:x2] = crop
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), new_boxes, domain

    if domain == "harsh_shadow_dusk":
        # Cast diagonal tree/overpass shadow across road + warm/dark highway tone
        yy, xx = np.mgrid[:h, :w]
        band = ((xx * 0.6 + yy) % max(40, int(h * 0.45))) < int(h * 0.22)
        arr[band] = arr[band] * 0.48
        # Slight warm dusk / dashcam tint
        arr[..., 0] = np.clip(arr[..., 0] * 0.95, 0, 255)
        arr[..., 1] = np.clip(arr[..., 1] * 0.88, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.80, 0, 255)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), new_boxes, domain

    if domain == "concrete_patched_road":
        # Weathered high-key concrete road + dark asphalt patch seam
        lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        concrete = np.stack([lum * 0.5 + 95.0, lum * 0.5 + 94.0, lum * 0.5 + 90.0], axis=-1)
        blended = 0.55 * arr + 0.45 * concrete
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), new_boxes, domain

    if domain == "distant_highway_scale":
        # Embed image at 0.58x scale near road horizon to train small 16x16 patch features
        scale = 0.58
        nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
        resized = rgb.resize((nw, nh), Image.BILINEAR)
        canvas_arr = np.full((h, w, 3), 82, dtype=np.uint8)
        # Sky band at top, dark road at bottom
        canvas_arr[: int(h * 0.28), :] = (135, 165, 200)
        canvas = Image.fromarray(canvas_arr)
        off_x = (w - nw) // 2
        off_y = int(h * 0.26)
        if off_y + nh > h:
            off_y = max(0, h - nh)
        canvas.paste(resized, (off_x, off_y))
        scaled_boxes = [
            [round(bx * scale + off_x, 2), round(by * scale + off_y, 2), round(bw * scale, 2), round(bh * scale, 2)]
            for bx, by, bw, bh in new_boxes
            if bw * scale >= 4.0 and bh * scale >= 4.0
        ]
        if scaled_boxes:
            return canvas, scaled_boxes, domain
        return rgb, new_boxes, domain

    return rgb, new_boxes, domain


def _extract_valid_xywh_boxes(raw_objects: Any, width: int, height: int) -> List[List[float]]:
    """Normalize bounding boxes from Hugging Face Parquet rows into clamped COCO [x, y, w, h]."""
    if not isinstance(raw_objects, dict):
        return []
    raw_bboxes = raw_objects.get("bbox")
    if raw_bboxes is None:
        raw_bboxes = raw_objects.get("bboxes")
    if raw_bboxes is None:
        raw_bboxes = []
    valid_boxes: List[List[float]] = []
    for box in raw_bboxes:
        if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
            continue
        vals = [float(v) for v in box]
        if not all(math.isfinite(v) for v in vals):
            continue
        # Detect normalized [0..1] coordinates if all values <= 1.05
        if max(vals) <= 1.05 and width > 10 and height > 10:
            vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
        x, y, w_or_x2, h_or_y2 = vals
        # In HF COCO datasets, bbox is [x, y, w, h]; if x + w_or_x2 > width * 1.05 and w_or_x2 > x, treat as xyxy
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


def prepare_fast_all_terrain_dataset(
    work_dir: Path | str,
    train_count: int = 1000,
    valid_count: int = 100,
    test_count: int = 100,
    seed: int = 42,
) -> Tuple[Path, Dict[str, Any]]:
    """Download and stage a fast, multi-domain all-terrain pothole dataset via Parquet in ~30s."""
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    work_dir = Path(work_dir)
    dataset_root = work_dir / "all_terrain_dataset"

    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists() and (dataset_root / "train" / "_annotations.coco.json").exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_manifest.get("version") == "v3_anti_manhole":
                print(f"Reusing Anti-Manhole All-Terrain dataset v3 at {dataset_root}", flush=True)
                return dataset_root, cached_manifest
        except Exception:
            pass

    raw_samples: List[Tuple[Image.Image, List[List[float]]]] = []
    negative_samples: List[Tuple[Image.Image, List[List[float]]]] = []
    manhole_crops: List[Image.Image] = []
    target_total = train_count + valid_count + test_count
    target_positives = max(1, int(target_total * 0.72))
    target_negatives = max(0, target_total - target_positives)

    try:
        import urllib.request
        import pandas as pd

        # 1. Direct single-file Parquet download via refs/convert/parquet (1,568 embedded real pothole images across manot v1, manot v2, and keremberke)
        parquet_shards = [
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
        drive_candidates = [
            Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/BANTAI_Pothole_Parquet_Dataset"),
            Path("/content/drive/MyDrive/BANTAI/BANTAI_Pothole_Parquet_Dataset"),
        ]
        pq_cache = next((p for p in drive_candidates if p.exists()), ( drive_candidates[0] if drive_candidates[0].parent.exists() else (work_dir / "parquet_cache") ))
        pq_cache.mkdir(parents=True, exist_ok=True)

        # 1A. Load Hard-Negative Manhole Scenes + Manhole Cover Crops from manhole_*.parquet and Manhole Cover Dataset YOLO.v4i.coco
        manhole_parquet_shards = [
            ("manhole_train_0000.parquet", "delima87/manhole_covers_dataset (train)", "https://huggingface.co/datasets/delima87/manhole_covers_dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"),
            ("manhole_validation_0000.parquet", "delima87/manhole_covers_dataset (val)", "https://huggingface.co/datasets/delima87/manhole_covers_dataset/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"),
            ("manhole_test_0000.parquet", "delima87/manhole_covers_dataset (test)", "https://huggingface.co/datasets/delima87/manhole_covers_dataset/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"),
        ]
        for mh_fname, mh_repo, mh_url in manhole_parquet_shards:
            if len(negative_samples) >= target_negatives and len(manhole_crops) >= 240:
                break
            try:
                mh_pq = pq_cache / mh_fname
                if not mh_pq.exists() or mh_pq.stat().st_size < 1024:
                    print(f"Downloading manhole Parquet shard ({mh_fname})...", flush=True)
                    req = urllib.request.Request(mh_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=60) as resp, mh_pq.open("wb") as out_f:
                        shutil.copyfileobj(resp, out_f)
                df_mh = pd.read_parquet(mh_pq)
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
                    if len(manhole_crops) < 240:
                        manhole_crops.append(rgb.copy())
                    if len(negative_samples) < target_negatives:
                        canvas = rgb.resize((512, 384), Image.BILINEAR)
                        negative_samples.append((canvas, []))
                print(f"  Loaded manhole hard-negatives from {mh_fname}: {len(negative_samples)} neg scenes, {len(manhole_crops)} manhole crops", flush=True)
            except Exception as mh_exc:
                print(f"  Note: skipped {mh_fname} ({mh_exc})", flush=True)

        coco_mh_roots = [
            Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/Manhole Cover Dataset YOLO.v4i.coco"),
            Path("/content/drive/MyDrive/BANTAI/Manhole Cover Dataset YOLO.v4i.coco"),
        ]
        for coco_root in coco_mh_roots:
            if not coco_root.exists():
                continue
            for ann_file in sorted(coco_root.glob("*/_annotations.coco.json")):
                try:
                    cdata = json.loads(ann_file.read_text(encoding="utf-8"))
                    anns_by_img: Dict[int, List[List[float]]] = collections.defaultdict(list)
                    for ann in cdata.get("annotations", []):
                        bb = ann.get("bbox", [])
                        if len(bb) == 4 and bb[2] >= 12 and bb[3] >= 12:
                            anns_by_img[int(ann["image_id"])].append([float(v) for v in bb])
                    for im_info in cdata.get("images", []):
                        im_path = ann_file.parent / im_info["file_name"]
                        if not im_path.exists():
                            continue
                        with Image.open(im_path) as im_src:
                            rgb = ImageOps.exif_transpose(im_src).convert("RGB")
                        if len(negative_samples) < target_negatives:
                            negative_samples.append((rgb, []))
                        for bx, by, bw, bh in anns_by_img.get(int(im_info["id"]), [])[:2]:
                            if len(manhole_crops) < 360:
                                crop = rgb.crop((max(0, int(bx)), max(0, int(by)), min(rgb.width, int(bx + bw)), min(rgb.height, int(by + bh))))
                                if crop.width >= 16 and crop.height >= 12:
                                    manhole_crops.append(crop)
                        if len(negative_samples) >= target_negatives and len(manhole_crops) >= 360:
                            break
                except Exception:
                    continue

        # 1B. Load Kaggle archive/data/annotations_coco.json (pothole=0, crack=1, manhole=2)
        archive_roots = [
            Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/archive/data"),
            Path("/content/drive/MyDrive/BANTAI/BANTAI-DATASETS/archive"),
            Path("/content/drive/MyDrive/BANTAI/archive/data"),
        ]
        for arch_root in archive_roots:
            ann_coco = arch_root / "annotations_coco.json"
            img_dir = arch_root / "images"
            if not ann_coco.exists() or not img_dir.exists():
                continue
            try:
                adata = json.loads(ann_coco.read_text(encoding="utf-8"))
                potholes_by_img: Dict[int, List[List[float]]] = collections.defaultdict(list)
                manholes_by_img: Dict[int, List[List[float]]] = collections.defaultdict(list)
                for ann in adata.get("annotations", []):
                    bb = ann.get("bbox", [])
                    if len(bb) != 4 or bb[2] < 6 or bb[3] < 6:
                        continue
                    cid = int(ann.get("category_id", -1))
                    if cid == 0:
                        potholes_by_img[int(ann["image_id"])].append([float(v) for v in bb])
                    elif cid == 2:
                        manholes_by_img[int(ann["image_id"])].append([float(v) for v in bb])
                arch_pos, arch_neg = 0, 0
                for im_info in adata.get("images", []):
                    im_id = int(im_info["id"])
                    p_boxes = potholes_by_img.get(im_id, [])
                    m_boxes = manholes_by_img.get(im_id, [])
                    if not p_boxes and not m_boxes:
                        continue
                    im_file = img_dir / im_info["file_name"]
                    if not im_file.exists():
                        continue
                    with Image.open(im_file) as im_src:
                        rgb = ImageOps.exif_transpose(im_src).convert("RGB")
                    for bx, by, bw, bh in m_boxes[:2]:
                        if len(manhole_crops) < 500:
                            crop = rgb.crop((max(0, int(bx)), max(0, int(by)), min(rgb.width, int(bx + bw)), min(rgb.height, int(by + bh))))
                            if crop.width >= 16 and crop.height >= 12:
                                manhole_crops.append(crop)
                    if p_boxes and len(raw_samples) < target_positives:
                        raw_samples.append((rgb, p_boxes))
                        arch_pos += 1
                    elif m_boxes and not p_boxes and len(negative_samples) < target_negatives:
                        negative_samples.append((rgb, []))
                        arch_neg += 1
                print(f"  Loaded archive/data/annotations_coco.json: +{arch_pos} pothole scenes (incl. pothole+manhole), +{arch_neg} manhole-only scenes", flush=True)
            except Exception as arch_exc:
                print(f"  Note: archive/data loader note ({arch_exc})", flush=True)
            break

        for shard_fname, repo_id, url in parquet_shards:
            if len(raw_samples) >= target_positives and len(negative_samples) >= target_negatives:
                break
            try:
                local_pq = pq_cache / shard_fname
                if not local_pq.exists() or local_pq.stat().st_size < 1024:
                    print(f"Downloading single Parquet file from {repo_id} ({shard_fname})...", flush=True)
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
                    if isinstance(objs, dict):
                        boxes = _extract_valid_xywh_boxes(objs, w, h)
                    else:
                        boxes = []
                    if boxes and len(raw_samples) < target_positives:
                        raw_samples.append((rgb, boxes))
                    elif not boxes and len(negative_samples) < target_negatives and w >= 256 and h >= 256:
                        negative_samples.append((rgb, []))
                    if len(raw_samples) >= target_positives and len(negative_samples) >= target_negatives:
                        break
                print(f"  Collected {len(raw_samples)} positives, {len(negative_samples)} hard-negatives from {repo_id}", flush=True)
            except Exception as repo_exc:
                print(f"  Note: skipped {shard_fname} ({repo_exc})", flush=True)
    except Exception as ds_exc:
        print(f"Note: direct Parquet loader note ({ds_exc}).", flush=True)

    # 2. Supplement from cached or single-shard RDD2022 positives and negatives if needed
    if len(raw_samples) < target_positives or len(negative_samples) < target_negatives:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            rdd_cache = work_dir / "rdd_fallback_cache"
            print("Supplementing base potholes and negatives from RDD2022 shard_000...", flush=True)
            snapshot_download(
                "dronefreak/RDD2022",
                repo_type="dataset",
                revision="72b01f94c2e5bfe430861afd73995f2d412b8855",
                allow_patterns=["data/images/train/shard_000/metadata.jsonl"],
                local_dir=str(rdd_cache),
            )
            for meta_file in sorted(rdd_cache.glob("data/images/*/shard_*/metadata.jsonl")):
                for line in meta_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    objs = item.get("objects", {})
                    cats = objs.get("categories", [])
                    bboxes = objs.get("bbox", [])
                    pothole_boxes = [
                        [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                        for b, c in zip(bboxes, cats)
                        if c == 3 and len(b) == 4 and b[2] >= 4 and b[3] >= 4
                    ]
                    rel_path = meta_file.parent.relative_to(rdd_cache).as_posix() + "/" + item["file_name"]
                    if pothole_boxes and len(raw_samples) < target_positives:
                        img_path = Path(
                            hf_hub_download(
                                "dronefreak/RDD2022",
                                filename=rel_path,
                                repo_type="dataset",
                                revision="72b01f94c2e5bfe430861afd73995f2d412b8855",
                                local_dir=str(rdd_cache),
                            )
                        )
                        with Image.open(img_path) as im:
                            rgb = ImageOps.exif_transpose(im).convert("RGB")
                        raw_samples.append((rgb, pothole_boxes))
                    elif not pothole_boxes and len(negative_samples) < target_negatives:
                        img_path = Path(
                            hf_hub_download(
                                "dronefreak/RDD2022",
                                filename=rel_path,
                                repo_type="dataset",
                                revision="72b01f94c2e5bfe430861afd73995f2d412b8855",
                                local_dir=str(rdd_cache),
                            )
                        )
                        with Image.open(img_path) as im:
                            rgb = ImageOps.exif_transpose(im).convert("RGB")
                        negative_samples.append((rgb, []))

                    if len(raw_samples) >= target_positives and len(negative_samples) >= target_negatives:
                        break
        except Exception as rdd_exc:
            print(f"  Note: RDD2022 supplement skipped ({rdd_exc})", flush=True)

    # Supplement if fewer than target samples were fetched
    rng = np.random.RandomState(seed)
    while len(raw_samples) < target_positives:
        if raw_samples:
            base_img, base_boxes = raw_samples[len(raw_samples) % len(raw_samples)]
            aug_img, aug_boxes, _ = augment_all_pothole_types(base_img, base_boxes, len(raw_samples), seed=seed)
            raw_samples.append((aug_img, aug_boxes))
        else:
            # Pure synthetic fallback for offline tests
            arr = np.full((384, 512, 3), 75, dtype=np.uint8)
            bx, by = int(rng.randint(80, 320)), int(rng.randint(150, 260))
            bw, bh = int(rng.randint(50, 110)), int(rng.randint(35, 75))
            arr[by : by + bh, bx : bx + bw] = 28
            raw_samples.append((Image.fromarray(arr), [[float(bx), float(by), float(bw), float(bh)]]))

    while len(negative_samples) < target_negatives:
        if raw_samples:
            base_img, _ = raw_samples[len(negative_samples) % len(raw_samples)]
            w_b, h_b = base_img.size
            crop_neg = base_img.crop((0, int(h_b * 0.15), w_b, int(h_b * 0.60)))
            neg_img = crop_neg.resize((w_b, h_b), Image.BILINEAR)
            negative_samples.append((neg_img, []))
        else:
            clean_arr = np.full((384, 512, 3), 85, dtype=np.uint8)
            clean_img = Image.fromarray(clean_arr)
            negative_samples.append((clean_img, []))

    all_samples = raw_samples[:target_positives] + negative_samples[:target_negatives]

    # Deterministic shuffle
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    splits_map = {
        "train": [ all_samples[i] for i in indices[:train_count] ],
        "valid": [ all_samples[i] for i in indices[train_count : train_count + valid_count] ],
        "test": [ all_samples[i] for i in indices[train_count + valid_count : target_total] ],
    }

    domain_counts: Dict[str, int] = collections.Counter()
    split_summary: Dict[str, Any] = {}

    for split_name, items in splits_map.items():
        split_dir = dataset_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {"description": f"BANTAI All-Terrain Pothole {split_name} split v2"},
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
                # Manhole-on-Road CutMix (paste foreshortened manhole cover on 35% of positive road images with ONLY pothole boxed)
                if manhole_crops and (img_id % 3 == 0):
                    w_f, h_f = final_img.size
                    mh_crop = manhole_crops[img_id % len(manhole_crops)]
                    mh_w = max(24, min(int(w_f * 0.16), int(rng.randint(36, 92))))
                    mh_h = max(14, int(mh_w / float(rng.uniform(1.55, 2.45))))
                    mh_resized = mh_crop.resize((mh_w, mh_h), Image.BILINEAR)
                    for _ in range(8):
                        px = int(rng.randint(int(w_f * 0.12), max(int(w_f * 0.12) + 1, int(w_f * 0.85) - mh_w)))
                        py = int(rng.randint(int(h_f * 0.36), max(int(h_f * 0.36) + 1, int(h_f * 0.78) - mh_h)))
                        overlaps_pothole = any(
                            not (px + mh_w < bx - 8 or px > bx + bw + 8 or py + mh_h < by - 8 or py > by + bh + 8)
                            for bx, by, bw, bh in final_boxes
                        )
                        if not overlaps_pothole:
                            final_img = final_img.copy()
                            final_img.paste(mh_resized, (px, py))
                            domain_counts["manhole_cutmix_hard_negative"] += 1
                            break
                domain_counts[domain_tag] += 1
            else:
                # Keep half of valid/test in original out-of-domain state and half with puddle/shadow variations
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
        print(f"  Staged {split_name}: {len(coco['images'])} images ({len(coco['annotations'])} pothole boxes)", flush=True)

    manifest = {
        "version": "v3_anti_manhole",
        "splits": split_summary,
        "train_domain_distribution": dict(domain_counts),
        "classes": ["pothole"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dataset_root, manifest


def preserve_model_config_in_checkpoint(
    src_pth: Path | str,
    dst_pth: Path | str,
    base_pth: Optional[Path | str] = None,
    threshold: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Ensure model_config, args, class_names, and threshold are embedded in the exported .pth."""
    import torch

    src_pth = Path(src_pth)
    dst_pth = Path(dst_pth)
    dst_pth.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(src_pth), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        ckpt = {"model": ckpt}

    # Inherit model_config / args from base_pth if stripped by rfdetr's checkpoint_best_total.pth
    if base_pth is not None and Path(base_pth).exists():
        try:
            base_ckpt = torch.load(str(base_pth), map_location="cpu", weights_only=False)
            if isinstance(base_ckpt, dict):
                for key in ("model_config", "args"):
                    if key not in ckpt and key in base_ckpt:
                        ckpt[key] = copy.deepcopy(base_ckpt[key])
        except Exception:
            pass

    default_model_cfg = {
        "encoder": "dinov2_windowed_small",
        "resolution": 384,
        "patch_size": 16,
        "dec_layers": 2,
        "num_select": 300,
        "num_classes": 1,
        "class_names": ["pothole"],
    }
    if "model_config" not in ckpt or ckpt["model_config"] is None:
        ckpt["model_config"] = default_model_cfg
    elif isinstance(ckpt["model_config"], dict):
        ckpt["model_config"].setdefault("resolution", 384)
        ckpt["model_config"]["num_classes"] = 1
        ckpt["model_config"]["class_names"] = ["pothole"]

    ckpt["class_names"] = ["pothole"]
    ckpt["calibrated_operating_threshold"] = float(threshold)
    ckpt["bantai_model_version"] = "rfdetr-nano-pothole-all-terrain-v2"

    tmp = dst_pth.with_name(dst_pth.name + ".partial")
    torch.save(ckpt, str(tmp))
    tmp.replace(dst_pth)
    return {
        "weights_path": str(dst_pth),
        "has_model_config": "model_config" in ckpt,
        "calibrated_operating_threshold": float(threshold),
        "class_names": ckpt["class_names"],
    }


# ---------------------------------------------------------------------------
# 1. Image Enhancement: CIE-LAB CLAHE Preprocessing
# ---------------------------------------------------------------------------

def apply_lab_clahe(img: Image.Image, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> Image.Image:
    """Enhance local contrast and balance harsh tree/building shadows on asphalt.
    
    Operates strictly on the Lightness (L) channel in CIE-LAB color space so that
    road color representations and asphalt hues are not distorted.
    Uses OpenCV if available; falls back to an adaptive NumPy/Pillow equalization.
    """
    try:
        import cv2
        rgb = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_enhanced)
    except Exception:
        # High-performance NumPy-based adaptive tile histogram equalization fallback
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        # Perceptual luminance calculation (ITU-R BT.601)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        h, w = lum.shape
        gh, gw = tile_grid
        th, tw = max(1, h // gh), max(1, w // gw)
        lum_out = np.zeros_like(lum)
        
        for i in range(gh):
            for j in range(gw):
                y0, y1 = i * th, (i + 1) * th if i < gh - 1 else h
                x0, x1 = j * tw, (j + 1) * tw if j < gw - 1 else w
                tile = lum[y0:y1, x0:x1]
                t_min, t_max = tile.min(), tile.max()
                if t_max > t_min:
                    norm = (tile - t_min) / (t_max - t_min)
                    tile_eq = np.clip(norm * 255.0, 0, 255)
                    # Blend original and equalized to mimic clip-limit
                    blend = 0.65 * tile_eq + 0.35 * tile
                    lum_out[y0:y1, x0:x1] = blend
                else:
                    lum_out[y0:y1, x0:x1] = tile
                    
        scale = np.where(lum > 1e-3, lum_out / np.maximum(lum, 1e-3), 1.0)
        scale = np.clip(scale, 0.4, 2.2)[..., np.newaxis]
        arr_enhanced = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_enhanced)


# ---------------------------------------------------------------------------
# 2. Road-Focused Slicing (SAHI)
# ---------------------------------------------------------------------------

def generate_road_sahi_tiles(
    img: Image.Image,
    horizon_ratio: float = 0.35,
    grid: Tuple[int, int] = (2, 2),
    overlap: float = 0.25,
    include_focal_tile: bool = False,
    bottom_hood_ratio: float = 0.78,
    right_ui_ratio: float = 0.87,
) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """Slice the lower drivable road region into high-resolution overlapping crops.

    Returns a list of (tile_pil_image, (offset_x, offset_y, tile_w, tile_h))
    so tile-level detections can be accurately projected back into full-image pixels.
    When include_focal_tile=True, appends a 5th center-horizon focal crop so distant
    middle-lane potholes on wide 16:9 dashcams are magnified without tile-seam splitting.
    """
    w, h = img.size
    is_wide_dashcam = (w / float(max(1, h))) >= 1.50
    road_top = int(h * horizon_ratio)
    road_bottom = int(h * bottom_hood_ratio) if is_wide_dashcam else h
    road_right = int(w * right_ui_ratio) if is_wide_dashcam else w
    road_h = road_bottom - road_top
    if road_h <= 32 or road_right <= 32:
        return [(img, (0, 0, w, h))]

    cols, rows = grid
    step_x = int(road_right / (cols - (cols - 1) * overlap)) if cols > 1 else road_right
    step_y = int(road_h / (rows - (rows - 1) * overlap)) if rows > 1 else road_h
    tile_w = min(road_right, max(64, step_x))
    tile_h = min(road_h, max(64, step_y))

    tiles = []
    x_coords = np.linspace(0, max(0, road_right - tile_w), cols, dtype=int)
    y_coords = np.linspace(road_top, max(road_top, road_bottom - tile_h), rows, dtype=int)

    for y in y_coords:
        for x in x_coords:
            crop_box = (int(x), int(y), int(min(road_right, x + tile_w)), int(min(road_bottom, y + tile_h)))
            cropped = img.crop(crop_box)
            tiles.append((cropped, (crop_box[0], crop_box[1], crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])))

    if include_focal_tile:
        fx0 = int(w * 0.15)
        fy0 = int(h * max(0.25, horizon_ratio - 0.05))
        fx1 = int(min(road_right, w * 0.85))
        fy1 = int(min(road_bottom, h * 0.78))
        if fx1 > fx0 + 32 and fy1 > fy0 + 32:
            focal_crop = img.crop((fx0, fy0, fx1, fy1))
            tiles.append((focal_crop, (fx0, fy0, fx1 - fx0, fy1 - fy0)))

    return tiles


def remap_tile_boxes(detections: List[Dict[str, Any]], offset: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
    """Project tile-relative bounding boxes back into original image pixel coordinates."""
    off_x, off_y, _, _ = offset
    remapped = []
    for d in detections:
        box = d["xyxy"]
        remapped.append({
            "xyxy": [box[0] + off_x, box[1] + off_y, box[2] + off_x, box[3] + off_y],
            "score": d["score"],
            "class_name": d.get("class_name", "pothole"),
            "source": "sahi_tile",
        })
    return remapped


# ---------------------------------------------------------------------------
# 3. Weighted Boxes Fusion (WBF)
# ---------------------------------------------------------------------------

def box_iou(a: List[float], b: List[float]) -> float:
    """Calculate Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def weighted_boxes_fusion(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.55,
    skip_box_thr: float = 0.05,
    max_score_boost: float = 1.15,
) -> List[Dict[str, Any]]:
    """Cluster overlapping boxes with IoU and fuse using confidence-weighted average."""
    valid = [d for d in detections if d.get("score", 0.0) >= skip_box_thr]
    if not valid:
        return []

    valid.sort(key=lambda x: x["score"], reverse=True)

    clusters: List[List[Dict[str, Any]]] = []
    cluster_boxes: List[List[float]] = []

    for det in valid:
        matched = False
        det_box = det["xyxy"]
        for idx, mean_box in enumerate(cluster_boxes):
            if box_iou(det_box, mean_box) >= iou_threshold:
                clusters[idx].append(det)
                scores = [item["score"] for item in clusters[idx]]
                total_s = sum(scores)
                cluster_boxes[idx] = [
                    sum(item["xyxy"][k] * item["score"] for item in clusters[idx]) / total_s
                    for k in range(4)
                ]
                matched = True
                break
        if not matched:
            clusters.append([det])
            cluster_boxes.append(list(det_box))

    fused_results = []
    for cluster in clusters:
        scores = [d["score"] for d in cluster]
        total_s = sum(scores)
        w_box = [
            round(sum(d["xyxy"][k] * d["score"] for d in cluster) / total_s, 2)
            for k in range(4)
        ]
        base_score = max(scores)
        sources = list(set([d.get("source", "unknown") for d in cluster]))

        # Multi-candidate agreement boost
        if base_score >= 0.25:
            agreement_factor = min(max_score_boost, 1.0 + (len(cluster) - 1) * 0.04)
        else:
            agreement_factor = 1.0

        fused_score = min(1.0, round(base_score * agreement_factor, 4))

        fused_results.append({
            "xyxy": w_box,
            "score": fused_score,
            "class_name": "pothole",
            "cluster_count": len(cluster),
            "sources": sources,
        })

    fused_results.sort(key=lambda x: x["score"], reverse=True)
    return fused_results


# ---------------------------------------------------------------------------
# 4. Geometric & Texture Verification (Horizon, Stripe & Shadow Rejection)
# ---------------------------------------------------------------------------

def filter_horizon_and_aspect_ratio(
    detections: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    horizon_ratio: float = 0.28,
    max_aspect_ratio: float = 6.0,
    min_area_px: float = 16.0,
    max_area_ratio: float = 0.28,
    bottom_hood_ratio: float = 0.64,
    right_ui_ratio: float = 0.86,
) -> List[Dict[str, Any]]:
    """Reject non-road detections, windshield wipers, single-tile hallucinations, and nested box fragments."""
    kept = []
    horizon_y = img_height * horizon_ratio
    is_wide_dashcam = (img_width / float(max(1, img_height))) >= 1.50
    total_img_area = float(max(1, img_width * img_height))

    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = bw * bh
        area_ratio = area / total_img_area
        score = float(d.get("score", 0.0))
        cluster_cnt = int(d.get("cluster_count", 1))
        sources = d.get("sources", [])
        aspect = max(bw / bh, bh / bw)

        # 1. Horizon rejection (potholes only exist on drivable road)
        if cy < horizon_y:
            continue

        # 2. Universal Windshield Wiper Blade / Bottom Hood Edge Rejection
        if (cy > img_height * 0.86 or y2 > img_height * 0.88) and (bw / bh) >= 2.2:
            continue

        # 3. Wide Dashcam Cabin & UI Sidebar Mask (with Tweaks 1 & 2)
        if is_wide_dashcam:
            # Tweak 1: Raised bottom cutoff to 0.64/0.68 to mask VIOFO dashboard air vents & bottom windshield glare
            if cy > img_height * bottom_hood_ratio or y2 > img_height * 0.68 or cx > img_width * right_ui_ratio or x2 > img_width * 0.88:
                continue
            # Tweak 2: Reject distant horizontal SUV rear bumpers near the highway horizon
            if cy < img_height * 0.38 and (bw / bh) >= 2.0 and score < 0.45:
                continue
            if score < 0.42 and area_ratio > 0.012:
                continue

        # 4. Extreme aspect ratio rejection (painted lane stripes / expansion lines)
        if aspect > max_aspect_ratio:
            continue

        # 5. Sub-pixel noise rejection
        if area < min_area_px:
            continue

        # 6. Reject giant background/tile-border hallucinations (>28% of whole frame unless score >= 0.65)
        if area_ratio > max_area_ratio and score < 0.65:
            continue

        kept.append(d)

    # 8. Bidirectional Containment & Inner-Fragment Suppression (with Tweak 3)
    surviving = []
    for i, box_i in enumerate(kept):
        ix1, iy1, ix2, iy2 = box_i["xyxy"]
        i_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
        i_score = float(box_i.get("score", 0.0))
        drop_i = False

        for j, box_j in enumerate(kept):
            if i == j:
                continue
            jx1, jy1, jx2, jy2 = box_j["xyxy"]
            j_area = max(1.0, (jx2 - jx1) * (jy2 - jy1))
            j_score = float(box_j.get("score", 0.0))

            inter_w = max(0.0, min(ix2, jx2) - max(ix1, jx1))
            inter_h = max(0.0, min(iy2, jy2) - max(iy1, jy1))
            inter = inter_w * inter_h
            if inter <= 0.0:
                continue

            # Tweak 3: Overlapping duplicate suppression on the same pothole
            if (inter / min(i_area, j_area)) >= 0.45 and j_score > i_score:
                drop_i = True
                break

            # Inner-Fragment Suppression: drop box_i if inside larger confident box_j
            if j_area > i_area * 1.25 and (inter / i_area) >= 0.55:
                if j_score >= 0.45 or j_score >= i_score:
                    drop_i = True
                    break

            # Outer-Wrapper Suppression: drop oversized wrapper box_i around sharper box_j
            if i_area > j_area * 2.20 and (inter / j_area) >= 0.70:
                if j_score >= i_score + 0.15 and i_score < 0.50:
                    drop_i = True
                    break

        if not drop_i:
            surviving.append(box_i)

    return surviving


def compute_directional_isotropy_and_roughness(gray_arr: np.ndarray, box: List[float]) -> Tuple[float, float, float]:
    """Compute (laplacian_var, rho_iso, r_eff) inside the box with a 12% rim margin.
    
    rho_iso = min(Ex, Ey) / (max(Ex, Ey) + 1e-5):
      - In horizontal structures (air vents, car bumpers, rectangular manhole rims), Ey >> Ex -> rho_iso < 0.40
      - In real multi-directional jagged craters, Ex and Ey are balanced -> rho_iso >= 0.60
    r_eff = laplacian_var * (rho_iso ** 2):
      - Strongly penalizes straight horizontal/vertical artificial slats.
    """
    h, w = gray_arr.shape
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    mx, my = int(bw * 0.12), int(bh * 0.12)
    x1 = max(0, int(box[0]) - mx)
    y1 = max(0, int(box[1]) - my)
    x2 = min(w, int(box[2]) + mx)
    y2 = min(h, int(box[3]) + my)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return 1.0, 1.0, 1.0

    crop = gray_arr[y1:y2, x1:x2].astype(np.float32)
    diff_x = np.abs(crop[:, 1:] - crop[:, :-1])
    diff_y = np.abs(crop[1:, :] - crop[:-1, :])
    ex = float(np.mean(diff_x)) if diff_x.size > 0 else 0.0
    ey = float(np.mean(diff_y)) if diff_y.size > 0 else 0.0
    rho_iso = min(ex, ey) / (max(ex, ey) + 1e-5)

    edges = (
        crop[1:-1, 0:-2] + crop[1:-1, 2:] + crop[0:-2, 1:-1] + crop[2:, 1:-1]
        - 4.0 * crop[1:-1, 1:-1]
    )
    if edges.size == 0:
        return 0.0, float(rho_iso), 0.0
    full_var = float(edges.var())
    eh, ew = edges.shape
    if eh >= 12 and ew >= 12:
        rim_mask = np.ones((eh, ew), dtype=bool)
        iy1, iy2 = int(eh * 0.22), int(eh * 0.78)
        ix1, ix2 = int(ew * 0.22), int(ew * 0.78)
        rim_mask[iy1:iy2, ix1:ix2] = False
        rim_edges = edges[rim_mask]
        rim_var = float(rim_edges.var()) if rim_edges.size > 0 else 0.0
        laplacian_var = max(full_var, rim_var)
    else:
        laplacian_var = full_var
    r_eff = laplacian_var * (rho_iso ** 2)
    return laplacian_var, float(rho_iso), float(r_eff)


def compute_laplacian_edge_roughness(gray_arr: np.ndarray, box: List[float]) -> float:
    """Compute high-frequency edge variance inside the box and along its outer rim.
    
    Returns max(full_var, rim_var) so that large water puddles with smooth specular
    or muddy interiors (Photo 5) are judged by their rough fractured rim rather than
    diluted by the smooth center.
    """
    var, _, _ = compute_directional_isotropy_and_roughness(gray_arr, box)
    return var


def verify_pothole_texture_and_geometry(
    pil_img: Image.Image,
    detections: List[Dict[str, Any]],
    min_roughness_var: float = 3.5,
) -> List[Dict[str, Any]]:
    """Filter out smooth flat shadows and skid marks using rim-aware Laplacian edge roughness."""
    gray = np.array(pil_img.convert("L"))
    verified = []
    for d in detections:
        score = float(d.get("score", 0.0))
        roughness = compute_laplacian_edge_roughness(gray, d["xyxy"])

        # Reject smooth flat shadow patches unless high confidence
        if roughness < min_roughness_var and score < 0.65:
            continue

        d_copy = dict(d)
        d_copy["roughness_variance"] = round(roughness, 2)
        verified.append(d_copy)
    return verified


# ---------------------------------------------------------------------------
# 5. Severity Index Estimation
# ---------------------------------------------------------------------------

def estimate_pothole_severity(box: List[float], img_width: int, img_height: int, score: float) -> str:
    """Assign BANTAI Severity Level (Low, Moderate, Critical) based on surface area and score."""
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    area_ratio = (bw * bh) / float(img_width * img_height)

    # Relative road surface area heuristics
    if area_ratio >= 0.035 or (area_ratio >= 0.02 and score >= 0.85):
        return "Critical"
    elif area_ratio >= 0.01 or (area_ratio >= 0.005 and score >= 0.70):
        return "Moderate"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# 6. Temporal Multi-Frame Anti-Flicker Tracker
# ---------------------------------------------------------------------------

class TemporalPotholeTracker:
    """Tracks potholes across sequential video/webcam frames using IoU distance."""

    def __init__(self, iou_threshold: float = 0.35, min_hits: int = 2, max_age: int = 5, alpha: float = 0.6):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.next_id = 1
        self.tracks: List[Dict[str, Any]] = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for track in self.tracks:
            track["age"] += 1
            track["time_since_update"] += 1

        matched_tracks, matched_dets = set(), set()
        if self.tracks and detections:
            pairs = []
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    score = box_iou(track["xyxy"], det["xyxy"])
                    if score >= self.iou_threshold:
                        pairs.append((score, t_idx, d_idx))
            pairs.sort(key=lambda x: x[0], reverse=True)
            for _, t_idx, d_idx in pairs:
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                track = self.tracks[t_idx]
                det = detections[d_idx]
                # Smooth coordinates with running exponential moving average
                track["xyxy"] = [
                    self.alpha * n + (1.0 - self.alpha) * o
                    for n, o in zip(det["xyxy"], track["xyxy"])
                ]
                track["score"] = det["score"]
                track["severity"] = det.get("severity", track.get("severity", "Moderate"))
                track["hits"] += 1
                track["time_since_update"] = 0

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append({
                    "id": self.next_id,
                    "xyxy": list(det["xyxy"]),
                    "score": det["score"],
                    "severity": det.get("severity", "Moderate"),
                    "hits": 1,
                    "age": 1,
                    "time_since_update": 0,
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]
        confirmed = []
        for t in self.tracks:
            if t["hits"] >= self.min_hits and t["time_since_update"] == 0:
                confirmed.append({
                    "id": t["id"],
                    "xyxy": [round(v, 2) for v in t["xyxy"]],
                    "score": round(t["score"], 3),
                    "severity": t.get("severity", "Moderate"),
                    "class_name": "pothole",
                })
        return confirmed


# ---------------------------------------------------------------------------
# 7. Dual-Mode Enhanced Pothole Detector Class
# ---------------------------------------------------------------------------

class EnhancedPotholeDetector:
    """Turnkey enhanced detector supporting both Real-Time Video and High-Accuracy Photo modes."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = "auto",
        strict_load: bool = True,
    ):
        self.weights_path = weights_path or str(Path(__file__).resolve().parent / DEFAULT_WEIGHTS_NAME)
        self.confidence = confidence
        self.device = device
        self.strict_load = strict_load
        self.detector = None
        self._load_model()
        self.tracker = TemporalPotholeTracker()

    def _load_model(self):
        try:
            import torch
            from rfdetr import RFDETR
            dev = "cuda" if self.device == "auto" and torch.cuda.is_available() else ("cpu" if self.device == "auto" else self.device)
            print(f"Loading RF-DETR Nano from {self.weights_path} on {dev}...")
            self.detector = RFDETR.from_checkpoint(self.weights_path, device=dev)
            try:
                self.detector.optimize_for_inference(compile=False)
            except Exception:
                pass
            print(f"Model ready. Operating threshold: {self.confidence:.2%}")
        except Exception as exc:
            if self.strict_load:
                raise RuntimeError(f"Failed to load RF-DETR checkpoint '{self.weights_path}': {exc}") from exc
            print(f"Note: RF-DETR load deferred ({exc}). Running in synthetic/mock test mode.")

    def _raw_predict(self, pil_img: Image.Image, conf: float) -> List[Dict[str, Any]]:
        """Run raw single-image prediction through the underlying RF-DETR detector."""
        if self.detector is None:
            return []
        detections = self.detector.predict(pil_img, threshold=conf)
        results = []
        boxes = detections.xyxy.tolist()
        scores = detections.confidence.tolist()
        ids = detections.class_id.tolist()
        names = detections.data.get("class_name") if hasattr(detections, "data") and isinstance(detections.data, dict) else None
        for idx, (box, score, cid) in enumerate(zip(boxes, scores, ids)):
            name = str(names[idx]).lower() if names is not None and idx < len(names) else None
            if int(cid) == 1 or name == "__background__":
                continue
            if int(cid) == 0 or name == "pothole":
                results.append({"xyxy": [round(float(v), 2) for v in box], "score": round(float(score), 4), "class_name": "pothole"})
        return results

    def detect(
        self,
        image: Union[str, Path, Image.Image],
        mode: str = "video",
        confidence: Optional[float] = None,
        enable_clahe: bool = True,
        enable_wbf: bool = True,
        enable_geometric_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run enhanced pothole detection.
        
        Args:
            image: PIL Image or path to image file.
            mode: "video" (ultra-fast, ~3ms overhead, 50-60+ FPS) or
                  "photo" (high-accuracy with SAHI road tiling and TTA).
            confidence: Optional confidence threshold override.
            enable_clahe: Balance shadows via LAB-CLAHE.
            enable_wbf: Blend overlapping boxes with Weighted Boxes Fusion.
            enable_geometric_filter: Filter sky/horizon and painted road lines.
        """
        conf = confidence if confidence is not None else self.confidence
        if isinstance(image, (str, Path)):
            with Image.open(image) as src:
                pil_img = ImageOps.exif_transpose(src).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        w, h = pil_img.size

        # Stage 1: CLAHE Local Lighting Normalization
        prep_img = apply_lab_clahe(pil_img) if enable_clahe else pil_img

        candidates: List[Dict[str, Any]] = []

        if mode == "photo":
            # --- High-Accuracy Mode: Full Frame + 5-Tile Road SAHI (with Focal Tile) + TTA ---
            cand_floor = max(0.08, conf * 0.70)
            # 1. Full Frame Pass
            ff_dets = self._raw_predict(prep_img, cand_floor)
            for fd in ff_dets:
                fd["source"] = "full_frame"
                candidates.append(fd)

            # 2. Road SAHI Slices + Center-Horizon Focal Tile Pass
            sahi_tiles = generate_road_sahi_tiles(prep_img, horizon_ratio=0.25, grid=(2, 2), overlap=0.25, include_focal_tile=True)
            for tile_img, offset in sahi_tiles:
                tile_dets = self._raw_predict(tile_img, cand_floor)
                candidates.extend(remap_tile_boxes(tile_dets, offset))

            # 3. Horizontal-Flip TTA Pass
            flipped_img = prep_img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_dets = self._raw_predict(flipped_img, cand_floor)
            for fd in flipped_dets:
                fx1, fy1, fx2, fy2 = fd["xyxy"]
                candidates.append({
                    "xyxy": [round(w - fx2, 2), fy1, round(w - fx1, 2), fy2],
                    "score": fd["score"],
                    "class_name": "pothole",
                    "source": "tta_flip",
                })
        else:
            # --- Video Stream Mode: Ultra-Fast Single Full-Frame Pass ---
            for d in self._raw_predict(prep_img, conf):
                d["source"] = "full_frame"
                candidates.append(d)

        # Stage 2: Weighted Boxes Fusion
        fused = weighted_boxes_fusion(candidates, iou_threshold=0.55) if enable_wbf else candidates

        # Stage 3: Geometric Filter (Horizon & Road Stripes)
        if enable_geometric_filter:
            fused = filter_horizon_and_aspect_ratio(fused, w, h, horizon_ratio=0.28)

        # Stage 4: Texture Verification (Water Puddles & Crater Roughness in Photo Mode)
        if mode == "photo":
            fused = verify_pothole_texture_and_geometry(pil_img, fused, min_roughness_var=3.5)

        # Stage 5: Assign Severity Index & Clean Output Format
        final_results = []
        for item in fused:
            if item["score"] >= conf:
                item["severity"] = estimate_pothole_severity(item["xyxy"], w, h, item["score"])
                final_results.append(item)

        return final_results

    def run_video(
        self,
        source_path: str,
        output_path: str,
        confidence: Optional[float] = None,
        enable_tracking: bool = True,
    ):
        """Annotate a road video file (.mp4) in real-time with temporal smoothing."""
        import cv2
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {source_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.tracker = TemporalPotholeTracker()
        frame_idx = 0
        t0 = time.time()
        print(f"Processing video {source_path} -> {output_path}...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            raw_dets = self.detect(pil_img, mode="video", confidence=confidence)
            active = self.tracker.update(raw_dets) if enable_tracking else raw_dets

            for item in active:
                x1, y1, x2, y2 = map(int, item["xyxy"])
                sev = item.get("severity", "Moderate")
                color = (0, 0, 255) if sev == "Critical" else ((0, 165, 255) if sev == "Moderate" else (0, 255, 255))
                label = f"Pothole #{item.get('id', '')} {item['score']:.0%} [{sev}]"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            frame_idx += 1
            cv2.putText(frame, f"BANTAI Potholes: {len(active)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        elapsed = time.time() - t0
        print(f"Video done. {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(1e-3, elapsed):.1f} FPS). Saved: {output_path}")

    def run_webcam(
        self,
        camera_index: Union[int, str] = 0,
        confidence: Optional[float] = None,
        enable_tracking: bool = True,
    ):
        """Run live real-time webcam or IP-camera feed with anti-flicker smoothing."""
        import cv2
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera/stream {camera_index}")
        self.tracker = TemporalPotholeTracker()
        print(f"Starting real-time pothole detector on camera {camera_index}. Press 'q' to exit.")
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            raw_dets = self.detect(pil_img, mode="video", confidence=confidence)
            active = self.tracker.update(raw_dets) if enable_tracking else raw_dets

            curr_time = time.time()
            fps = 1.0 / max(1e-4, curr_time - prev_time)
            prev_time = curr_time

            for item in active:
                x1, y1, x2, y2 = map(int, item["xyxy"])
                sev = item.get("severity", "Moderate")
                color = (0, 0, 255) if sev == "Critical" else ((0, 165, 255) if sev == "Moderate" else (0, 255, 255))
                label = f"Pothole #{item.get('id', '')} {item['score']:.0%} [{sev}]"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, f"FPS: {fps:.1f} | Active Potholes: {len(active)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("BANTAI Real-Time Enhanced Pothole Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 8. Standalone Script Generator
# ---------------------------------------------------------------------------

def generate_enhanced_detector_script(weights_filename: str = DEFAULT_WEIGHTS_NAME, optimal_confidence: float = DEFAULT_CONFIDENCE) -> str:
    """Generate the complete, zero-dependency bantai_pothole_detector.py standalone script."""
    return f'''"""BANTAI Enhanced Pothole Detector — Standalone Inference Engine.

Supports both Real-Time Video/Webcam (50-60+ FPS) and High-Accuracy Photo Inspection.
Embedded Calibrated Operating Threshold: {optimal_confidence:.3f}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from rfdetr import RFDETR

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent / "{weights_filename}")
DEFAULT_CONFIDENCE = {optimal_confidence:.3f}


def box_iou(a: List[float], b: List[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def apply_lab_clahe(img: Image.Image) -> Image.Image:
    try:
        rgb = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_enhanced = cv2.merge((clahe.apply(l), a, b))
        return Image.fromarray(cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB))
    except Exception:
        return img


def generate_road_sahi_tiles(img: Image.Image, horizon_ratio: float = 0.25, grid: Tuple[int, int] = (2, 2), overlap: float = 0.25, include_focal_tile: bool = True):
    w, h = img.size
    is_wide_dashcam = (w / float(max(1, h))) >= 1.50
    road_top = int(h * horizon_ratio)
    road_bottom = int(h * 0.78) if is_wide_dashcam else h
    road_right = int(w * 0.87) if is_wide_dashcam else w
    road_h = road_bottom - road_top
    if road_h <= 32 or road_right <= 32:
        return [(img, (0, 0, w, h))]
    cols, rows = grid
    step_x = int(road_right / (cols - (cols - 1) * overlap)) if cols > 1 else road_right
    step_y = int(road_h / (rows - (rows - 1) * overlap)) if rows > 1 else road_h
    tile_w = min(road_right, max(64, step_x))
    tile_h = min(road_h, max(64, step_y))
    tiles = []
    x_coords = np.linspace(0, max(0, road_right - tile_w), cols, dtype=int)
    y_coords = np.linspace(road_top, max(road_top, road_bottom - tile_h), rows, dtype=int)
    for y in y_coords:
        for x in x_coords:
            crop_box = (int(x), int(y), int(min(road_right, x + tile_w)), int(min(road_bottom, y + tile_h)))
            tiles.append((img.crop(crop_box), (crop_box[0], crop_box[1], crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])))
    if include_focal_tile:
        fx0, fy0, fx1, fy1 = int(w * 0.15), int(h * max(0.25, horizon_ratio - 0.05)), int(min(road_right, w * 0.85)), int(min(road_bottom, h * 0.78))
        if fx1 > fx0 + 32 and fy1 > fy0 + 32:
            tiles.append((img.crop((fx0, fy0, fx1, fy1)), (fx0, fy0, fx1 - fx0, fy1 - fy0)))
    return tiles


def remap_tile_boxes(detections: List[Dict[str, Any]], offset: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
    off_x, off_y, _, _ = offset
    return [
        {{"xyxy": [b[0] + off_x, b[1] + off_y, b[2] + off_x, b[3] + off_y], "score": d["score"], "class_name": "pothole", "source": "sahi_tile"}}
        for d in detections for b in [d["xyxy"]]
    ]


def compute_laplacian_edge_roughness(gray_arr: np.ndarray, box: List[float]) -> float:
    h, w = gray_arr.shape
    bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    mx, my = int(bw * 0.12), int(bh * 0.12)
    x1, y1 = max(0, int(box[0]) - mx), max(0, int(box[1]) - my)
    x2, y2 = min(w, int(box[2]) + mx), min(h, int(box[3]) + my)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return 1.0
    crop = gray_arr[y1:y2, x1:x2].astype(np.float32)
    edges = crop[1:-1, 0:-2] + crop[1:-1, 2:] + crop[0:-2, 1:-1] + crop[2:, 1:-1] - 4.0 * crop[1:-1, 1:-1]
    if edges.size == 0:
        return 0.0
    full_var = float(edges.var())
    eh, ew = edges.shape
    if eh >= 12 and ew >= 12:
        rim_mask = np.ones((eh, ew), dtype=bool)
        iy1, iy2 = int(eh * 0.22), int(eh * 0.78)
        ix1, ix2 = int(ew * 0.22), int(ew * 0.78)
        rim_mask[iy1:iy2, ix1:ix2] = False
        rim_edges = edges[rim_mask]
        rim_var = float(rim_edges.var()) if rim_edges.size > 0 else 0.0
        return max(full_var, rim_var)
    return full_var


def weighted_boxes_fusion(detections: List[Dict[str, Any]], iou_threshold: float = 0.55) -> List[Dict[str, Any]]:
    if not detections:
        return []
    valid = sorted(detections, key=lambda x: x["score"], reverse=True)
    clusters, cluster_boxes = [], []
    for det in valid:
        matched = False
        det_box = det["xyxy"]
        for idx, mean_box in enumerate(cluster_boxes):
            if box_iou(det_box, mean_box) >= iou_threshold:
                clusters[idx].append(det)
                scores = [item["score"] for item in clusters[idx]]
                total_s = sum(scores)
                cluster_boxes[idx] = [sum(item["xyxy"][k] * item["score"] for item in clusters[idx]) / total_s for k in range(4)]
                matched = True
                break
        if not matched:
            clusters.append([det])
            cluster_boxes.append(list(det_box))
    fused = []
    for cluster in clusters:
        scores = [d["score"] for d in cluster]
        total_s = sum(scores)
        w_box = [round(sum(d["xyxy"][k] * d["score"] for d in cluster) / total_s, 2) for k in range(4)]
        base = max(scores)
        boost = min(1.15, 1.0 + (len(cluster) - 1) * 0.04) if base >= 0.25 else 1.0
        sources = list(set([d.get("source", "unknown") for d in cluster]))
        fused.append({{"xyxy": w_box, "score": min(1.0, round(base * boost, 4)), "class_name": "pothole", "cluster_count": len(cluster), "sources": sources}})
    return sorted(fused, key=lambda x: x["score"], reverse=True)


class TemporalPotholeTracker:
    def __init__(self, iou_threshold: float = 0.35, min_hits: int = 2, max_age: int = 5, alpha: float = 0.6):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.next_id = 1
        self.tracks = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for t in self.tracks:
            t["time_since_update"] += 1
        matched_tracks, matched_dets = set(), set()
        if self.tracks and detections:
            pairs = []
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    score = box_iou(track["xyxy"], det["xyxy"])
                    if score >= self.iou_threshold:
                        pairs.append((score, t_idx, d_idx))
            pairs.sort(key=lambda x: x[0], reverse=True)
            for _, t_idx, d_idx in pairs:
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                t = self.tracks[t_idx]
                d = detections[d_idx]
                t["xyxy"] = [self.alpha * n + (1.0 - self.alpha) * o for n, o in zip(d["xyxy"], t["xyxy"])]
                t["score"] = d["score"]
                t["severity"] = d.get("severity", "Moderate")
                t["hits"] += 1
                t["time_since_update"] = 0
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append({{"id": self.next_id, "xyxy": list(det["xyxy"]), "score": det["score"], "severity": det.get("severity", "Moderate"), "hits": 1, "time_since_update": 0}})
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]
        return [{{"id": t["id"], "xyxy": [round(v, 2) for v in t["xyxy"]], "score": round(t["score"], 3), "severity": t["severity"], "class_name": "pothole"}}
                for t in self.tracks if t["hits"] >= self.min_hits and t["time_since_update"] == 0]


class PotholeDetector:
    def __init__(self, weights_path: Optional[str] = None, confidence: float = DEFAULT_CONFIDENCE, device: str = "auto"):
        self.weights_path = weights_path or DEFAULT_WEIGHTS
        self.confidence = confidence
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        print(f"Loading Enhanced RF-DETR Nano from {{self.weights_path}} on {{self.device}}...")
        self.detector = RFDETR.from_checkpoint(self.weights_path, device=self.device)
        try:
            self.detector.optimize_for_inference(compile=False)
        except Exception:
            pass
        self.tracker = TemporalPotholeTracker()
        print(f"Detector ready. Calibrated threshold: {{self.confidence:.2%}}")

    def _predict_raw(self, img: Image.Image, conf: float) -> List[Dict[str, Any]]:
        raw = self.detector.predict(img, threshold=conf)
        candidates = []
        for box, score, cid in zip(raw.xyxy.tolist(), raw.confidence.tolist(), raw.class_id.tolist()):
            if int(cid) == 0:
                candidates.append({{"xyxy": [round(float(v), 2) for v in box], "score": round(float(score), 4), "class_name": "pothole"}})
        return candidates

    def detect(self, image: Union[str, Path, Image.Image], mode: str = "video", confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        conf = confidence or self.confidence
        if isinstance(image, (str, Path)):
            with Image.open(image) as src:
                pil_img = ImageOps.exif_transpose(src).convert("RGB")
        else:
            pil_img = image.convert("RGB")
        w, h = pil_img.size
        prep_img = apply_lab_clahe(pil_img)

        candidates = []
        if mode == "photo":
            cand_floor = max(0.08, conf * 0.70)
            ff_dets = self._predict_raw(prep_img, cand_floor)
            for fd in ff_dets:
                fd["source"] = "full_frame"
                candidates.append(fd)
            tiles = generate_road_sahi_tiles(prep_img, horizon_ratio=0.25, grid=(2, 2), overlap=0.25, include_focal_tile=True)
            for tile_img, offset in tiles:
                tile_dets = self._predict_raw(tile_img, cand_floor)
                candidates.extend(remap_tile_boxes(tile_dets, offset))
            flipped_dets = self._predict_raw(prep_img.transpose(Image.FLIP_LEFT_RIGHT), cand_floor)
            for fd in flipped_dets:
                fx1, fy1, fx2, fy2 = fd["xyxy"]
                candidates.append({{"xyxy": [round(w - fx2, 2), fy1, round(w - fx1, 2), fy2], "score": fd["score"], "class_name": "pothole", "source": "tta_flip"}})
        else:
            for d in self._predict_raw(prep_img, conf):
                d["source"] = "full_frame"
                candidates.append(d)

        fused = weighted_boxes_fusion(candidates)
        gray = np.array(pil_img.convert("L")) if mode == "photo" else None
        total_area = float(max(1, w * h))

        final = []
        for d in fused:
            score = float(d.get("score", 0.0))
            if score < conf:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cy = (y1 + y2) / 2.0
            area_ratio = (bw * bh) / total_area
            aspect = max(bw / bh, bh / bw)

            if cy < h * 0.25 or aspect > 6.0:
                continue
            if area_ratio > 0.28 and score < 0.65:
                continue
            if gray is not None:
                roughness = compute_laplacian_edge_roughness(gray, d["xyxy"])
                if roughness < 3.5 and score < 0.65:
                    continue

            d["severity"] = "Critical" if area_ratio >= 0.035 else ("Moderate" if area_ratio >= 0.01 else "Low")
            final.append(d)
        return final

    def run_video(self, source_path: str, output_path: str, confidence: Optional[float] = None):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError("Cannot open video: " + source_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        line_w = max(3, round(min(width, height) / 220))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        self.tracker = TemporalPotholeTracker()
        print(f"Annotating video {{source_path}} -> {{output_path}}...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw_dets = self.detect(pil_img, mode="video", confidence=confidence)
            active = self.tracker.update(raw_dets)
            font_scale = max(0.80, min(width, height) / 720.0)
            font_thick = max(2, round(font_scale * 2.4))
            for item in active:
                x1, y1, x2, y2 = map(int, item["xyxy"])
                sev = item.get("severity", "Moderate")
                color = (0, 0, 255) if sev == "Critical" else ((0, 165, 255) if sev == "Moderate" else (0, 255, 0))
                lbl = f"Pothole {{item['score']:.1%}} [{{sev}}]"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_w)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                pad_x, pad_y = max(6, round(font_scale * 8)), max(5, round(font_scale * 6))
                ty1 = max(0, y1 - th - pad_y * 2)
                cv2.rectangle(frame, (x1, ty1), (x1 + tw + pad_x * 2, ty1 + th + pad_y * 2), (0, 0, 255), -1)
                cv2.putText(frame, lbl, (x1 + pad_x, ty1 + th + pad_y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
            out.write(frame)
        cap.release()
        out.release()
        print(f"Saved: {{output_path}}")

    def run_webcam(self, camera_index: Union[int, str] = 0, confidence: Optional[float] = None):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {{camera_index}}")
        self.tracker = TemporalPotholeTracker()
        print("Starting webcam pothole detector. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            line_w = max(3, round(min(w, h) / 220))
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw_dets = self.detect(pil_img, mode="video", confidence=confidence)
            active = self.tracker.update(raw_dets)
            font_scale = max(0.80, min(width, height) / 720.0)
            font_thick = max(2, round(font_scale * 2.4))
            for item in active:
                x1, y1, x2, y2 = map(int, item["xyxy"])
                sev = item.get("severity", "Moderate")
                color = (0, 0, 255) if sev == "Critical" else ((0, 165, 255) if sev == "Moderate" else (0, 255, 0))
                lbl = f"Pothole {{item['score']:.1%}} [{{sev}}]"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_w)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                pad_x, pad_y = max(6, round(font_scale * 8)), max(5, round(font_scale * 6))
                ty1 = max(0, y1 - th - pad_y * 2)
                cv2.rectangle(frame, (x1, ty1), (x1 + tw + pad_x * 2, ty1 + th + pad_y * 2), (0, 0, 255), -1)
                cv2.putText(frame, lbl, (x1 + pad_x, ty1 + th + pad_y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
            cv2.imshow("BANTAI Live Pothole Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BANTAI Enhanced Pothole Detector")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pth checkpoint")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--mode", choices=["video", "photo"], default="video", help="Detection mode")
    parser.add_argument("--image", help="Path to single image")
    parser.add_argument("--source", help="Path to video file (.mp4) or RTSP stream")
    parser.add_argument("--output", default="annotated_potholes.mp4", help="Output video file path")
    parser.add_argument("--webcam", type=int, help="Camera index (e.g. 0)")
    args = parser.parse_args()

    detector = PotholeDetector(weights_path=args.weights, confidence=args.confidence)
    if args.image:
        results = detector.detect(args.image, mode=args.mode)
        print(json.dumps(results, indent=2))
    elif args.webcam is not None:
        detector.run_webcam(camera_index=args.webcam)
    elif args.source:
        detector.run_video(args.source, args.output)
    else:
        print("Run with --help, --image <path>, --source <video.mp4>, or --webcam 0")
'''
