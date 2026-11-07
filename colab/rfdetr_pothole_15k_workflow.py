"""BANTAI RF-DETR Nano 15K Pothole Workflow.

Includes:
- Single-class pothole head surgery preserving pretrained weights and class-count offsets
- 15,000-image multi-country dataset preparation with pothole enrichment and hard negatives
- Resilient parallel download with automatic retry/backoff and zero-copy hardlinking
- Safe duplicate and EXIF rotation quarantining with contiguous COCO image IDs
- Temporal IoU tracker for video/webcam anti-flicker
- Ready-to-use model bundle packaging (weights, threshold, inference engine)
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

SOURCE_CLASSES = ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole']
TARGET_CLASSES = ['pothole']
POTHOLE_SOURCE_INDEX = 3
BACKGROUND_SOURCE_INDEX = 4

MODEL_REPO = 'dronefreak/rdd2022-rfdetr-nano'
MODEL_REVISION = '66429f39b916e56c6a33243d776718088f5f43ad'
DATA_REPO = 'dronefreak/RDD2022'
DATA_REVISION = '72b01f94c2e5bfe430861afd73995f2d412b8855'
PACKAGES = [
    'rfdetr[train]==1.11.2',
    'huggingface-hub==2.3.0',
    'pycocotools==2.0.11',
    'hf-transfer==0.1.9',
    'opencv-python-headless>=4.8.0',
]


def write_json(path: Path | str, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(p)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path | str, target: Path | str) -> None:
    t = Path(target)
    t.parent.mkdir(parents=True, exist_ok=True)
    temp = t.with_name(t.name + '.partial')
    shutil.copyfile(source, temp)
    temp.replace(t)


def backup_results(source: Path | str, target: Path | str) -> None:
    for path in sorted(Path(source).rglob('*')):
        if path.is_file() and not path.name.endswith('.partial'):
            atomic_copy(path, Path(target) / path.relative_to(source))


def guard_json(path: Path | str, value: Any) -> None:
    p = Path(path)
    if p.exists() and read_json(p) != json.loads(json.dumps(value)):
        raise ValueError('Saved settings/data differ. Choose a new RUN_NAME; do not overwrite this run.')
    write_json(p, value)


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe dataset path: ' + repr(value))
    return path


# ---------------------------------------------------------------------------
# 1. Checkpoint 1-Class Pothole Head Surgery
# ---------------------------------------------------------------------------

def slice_classification_tensor(
    tensor: Any,
    source_idx: int = POTHOLE_SOURCE_INDEX,
    bg_idx: int = BACKGROUND_SOURCE_INDEX,
) -> Any:
    """Slice a class projection weight or bias tensor to keep pothole (0) and background (1)."""
    if hasattr(tensor, 'shape') and len(tensor.shape) > 0:
        first_dim = tensor.shape[0]
        if first_dim >= 5:
            sliced = tensor[[source_idx, bg_idx]]
            return sliced.contiguous().clone() if hasattr(sliced, 'contiguous') else sliced
        if first_dim == 4:
            sliced = tensor[[source_idx]]
            return sliced.contiguous().clone() if hasattr(sliced, 'contiguous') else sliced
    return tensor


def perform_head_surgery(
    state_dict: Dict[str, Any],
    source_idx: int = POTHOLE_SOURCE_INDEX,
    bg_idx: int = BACKGROUND_SOURCE_INDEX,
) -> Tuple[Dict[str, Any], int, Optional[int], Optional[int]]:
    """Surgically slice class_embed layers from 4-class (or 5 with bg) to 1-class (or 2 with bg)."""
    modified = 0
    orig_rows: Optional[int] = None
    new_rows: Optional[int] = None
    new_state: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if any(term in key for term in ('class_embed', 'enc_out_class_embed', 'score_head', 'cls_score')):
            if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] in (4, 5):
                orig_rows = int(value.shape[0])
                sliced = slice_classification_tensor(value, source_idx, bg_idx)
                new_rows = int(sliced.shape[0])
                new_state[key] = sliced
                modified += 1
                continue
        new_state[key] = value
    return new_state, modified, orig_rows, new_rows


def _compute_target_num_classes(orig_nc: Optional[int], orig_rows: Optional[int], new_rows: Optional[int]) -> int:
    """Preserve the checkpoint's exact num_classes convention (with or without background slot)."""
    if orig_nc is not None and orig_rows is not None and new_rows is not None:
        offset = orig_rows - int(orig_nc)
        if offset in (0, 1):
            return max(1, new_rows - offset)
    return 1


def convert_checkpoint_to_single_class_pothole(src_ckpt_path: Path | str, dst_ckpt_path: Path | str) -> Dict[str, Any]:
    """Convert a 4-class RF-DETR Nano checkpoint to a native 1-class Pothole checkpoint."""
    import torch
    src = Path(src_ckpt_path)
    dst = Path(dst_ckpt_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(str(src), map_location='cpu', weights_only=False)
    surgery_info: Dict[str, Any] = {'source_file': src.name, 'source_sha256': sha256(src)}
    orig_rows: Optional[int] = None
    new_rows: Optional[int] = None

    if isinstance(data, dict):
        if 'model' in data and isinstance(data['model'], dict):
            new_model, count_m, orig_rows, new_rows = perform_head_surgery(data['model'])
            data['model'] = new_model
            surgery_info['model_tensors_sliced'] = count_m
        elif any('class_embed' in k for k in data.keys()):
            new_model, count_m, orig_rows, new_rows = perform_head_surgery(data)
            data = new_model
            surgery_info['model_tensors_sliced'] = count_m

        if isinstance(data, dict) and 'ema_model' in data and isinstance(data['ema_model'], dict):
            new_ema, count_e, _, _ = perform_head_surgery(data['ema_model'])
            data['ema_model'] = new_ema
            surgery_info['ema_tensors_sliced'] = count_e

        if isinstance(data, dict):
            for config_key in ('args', 'model_config'):
                if config_key in data:
                    cfg = data[config_key]
                    if isinstance(cfg, dict):
                        orig_nc = cfg.get('num_classes')
                        cfg['num_classes'] = _compute_target_num_classes(orig_nc, orig_rows, new_rows)
                        cfg['class_names'] = list(TARGET_CLASSES)
                    elif hasattr(cfg, 'num_classes'):
                        orig_nc = getattr(cfg, 'num_classes', None)
                        cfg.num_classes = _compute_target_num_classes(orig_nc, orig_rows, new_rows)
                        if hasattr(cfg, 'class_names'):
                            cfg.class_names = list(TARGET_CLASSES)

            if 'class_names' in data:
                data['class_names'] = list(TARGET_CLASSES)

    temp = dst.with_name(dst.name + '.partial')
    torch.save(data, str(temp))
    temp.replace(dst)

    surgery_info['orig_head_rows'] = orig_rows
    surgery_info['new_head_rows'] = new_rows
    surgery_info['target_file'] = dst.name
    surgery_info['target_sha256'] = sha256(dst)
    surgery_info['target_classes'] = list(TARGET_CLASSES)
    return surgery_info


def foreground_pothole_predictions(detections: Any, image_meta: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Extract foreground pothole predictions (class 0) and filter out RF-DETR's background slot."""
    boxes = detections.xyxy.tolist()
    ids = detections.class_id.tolist()
    scores = detections.confidence.tolist()
    names = detections.data.get('class_name') if hasattr(detections, 'data') and isinstance(detections.data, dict) else None

    if not (len(boxes) == len(ids) == len(scores)):
        raise ValueError('Detection array lengths do not match')

    predictions: List[Dict[str, Any]] = []
    background_filtered = 0
    for i, (box, cls_id, score) in enumerate(zip(boxes, ids, scores)):
        name = str(names[i]).lower() if names is not None and i < len(names) else None
        cid = int(cls_id)
        if cid == len(TARGET_CLASSES) or name == '__background__':
            background_filtered += 1
            continue
        if cid == 0 or name == 'pothole':
            predictions.append({
                'xyxy': [round(float(v), 2) for v in box],
                'category_id': 0,
                'score': round(float(score), 4),
                'class_name': 'pothole',
            })
        else:
            raise ValueError(f'Unexpected class ID {cls_id!r} (name={name!r}) in single-class pothole model; image={image_meta}')
    return predictions, background_filtered


# ---------------------------------------------------------------------------
# 2. Dataset Selection & 1-Class COCO Preparation (10k - 20k Scale)
# ---------------------------------------------------------------------------

def validate_objects_single_class(
    objects: Dict[str, Any],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter to pothole annotations (source category 3) and remap to category 0."""
    boxes, labels = objects.get('bbox', []), objects.get('categories', [])
    if len(boxes) != len(labels):
        raise ValueError('Box and label counts differ')
    annotations = []
    for box, label in zip(boxes, labels):
        if type(label) is not int or not (0 <= label < len(SOURCE_CLASSES)):
            raise ValueError('Unknown source class ID: ' + repr(label))
        if label != POTHOLE_SOURCE_INDEX:
            # Cracks are ignored in annotations, turning crack-only images into hard negatives
            continue
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Invalid bounding box: ' + repr(box))
        x, y, w, h = map(float, box)
        if min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError('Invalid box dimensions: ' + repr(box))
        if width is not None and (x + w > width or y + h > height):
            raise ValueError('Box extends outside image boundary')
        annotations.append({
            'category_id': 0,  # Single-class Pothole
            'bbox': [x, y, w, h],
            'area': w * h,
            'iscrowd': 0,
        })
    return annotations


def select_pothole_enriched_records(records: List[Dict[str, Any]], target_count: int, seed: int) -> List[Dict[str, Any]]:
    """Select a deterministic sample with maximum potholes (up to 70%) and balanced hard negatives."""
    positives = sorted(
        (r for r in records if POTHOLE_SOURCE_INDEX in r['objects']['categories']),
        key=lambda r: r['path'],
    )
    hard_negatives = sorted(
        (r for r in records if POTHOLE_SOURCE_INDEX not in r['objects']['categories'] and len(r['objects']['categories']) > 0),
        key=lambda r: r['path'],
    )
    clean_backgrounds = sorted(
        (r for r in records if len(r['objects']['categories']) == 0),
        key=lambda r: r['path'],
    )

    total_available = len(positives) + len(hard_negatives) + len(clean_backgrounds)
    if total_available < target_count:
        raise ValueError(
            f'Requested {target_count} images, but only {total_available} available '
            f'(positives={len(positives)}, cracks={len(hard_negatives)}, clean={len(clean_backgrounds)}).'
        )

    # Take up to 70% potholes, 20% crack hard-negatives, and 10% clean road backgrounds
    n_pos = min(len(positives), int(target_count * 0.70))
    n_crack = min(len(hard_negatives), int(target_count * 0.20))
    n_bg = target_count - (n_pos + n_crack)

    if len(clean_backgrounds) < n_bg:
        shortfall = n_bg - len(clean_backgrounds)
        n_bg = len(clean_backgrounds)
        extra_cracks = min(shortfall, len(hard_negatives) - n_crack)
        n_crack += extra_cracks
        shortfall -= extra_cracks
        n_pos = min(len(positives), n_pos + shortfall)

    if n_pos + n_crack + n_bg < target_count:
        rem = target_count - (n_pos + n_crack + n_bg)
        extra_bg = min(rem, len(clean_backgrounds) - n_bg)
        n_bg += extra_bg
        rem -= extra_bg
        extra_cracks = min(rem, len(hard_negatives) - n_crack)
        n_crack += extra_cracks

    rng = random.Random(seed)
    chosen_pos = rng.sample(positives, n_pos)
    chosen_crack = rng.sample(hard_negatives, n_crack)
    chosen_bg = rng.sample(clean_backgrounds, n_bg)

    selected = sorted(chosen_pos + chosen_crack + chosen_bg, key=lambda r: r['path'])
    return selected


def metadata_records(cache: Path | str, countries: Optional[Tuple[str, ...]] = None) -> Dict[str, List[Dict[str, Any]]]:
    records: Dict[str, List[Dict[str, Any]]] = {'train': [], 'valid': [], 'test': []}
    for path in sorted(Path(cache).glob('data/images/*/shard_*/metadata.jsonl')):
        split = path.parent.parent.name
        if split not in records:
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            name = item['file_name']
            safe_relative(name)
            if PurePosixPath(name).name != name:
                raise ValueError('Unexpected metadata filename layout: ' + repr(name))
            if countries:
                if isinstance(countries, (list, tuple, set)):
                    if not any(name.startswith(c + '_') for c in countries):
                        continue
                elif not name.startswith(str(countries) + '_'):
                    continue
            remote = path.parent.relative_to(cache).as_posix() + '/' + name
            records[split].append({'path': remote, 'file_name': name, 'objects': item['objects']})
    for split, rows in records.items():
        if not rows:
            raise ValueError('No metadata records found for split: ' + split)
        if len({r['path'] for r in rows}) != len(rows):
            raise ValueError('Duplicate metadata entries in ' + split)
    return records


def prepare_15k_dataset(
    root: Path | str,
    backup: Path | str,
    counts: Dict[str, int],
    seed: int,
    countries: Optional[Tuple[str, ...]] = None,
    max_workers: int = 16,
) -> Tuple[Path, Dict[str, Any]]:
    """Download and stage a multi-country 10k-20k pothole dataset with resilient downloads and hardlinks."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import snapshot_download, hf_hub_download
    from PIL import Image

    root = Path(root)
    backup = Path(backup)
    cache = root / 'download_cache'

    print('Fetching dataset metadata shards...', flush=True)
    snapshot_download(
        DATA_REPO,
        repo_type='dataset',
        revision=DATA_REVISION,
        allow_patterns=['data/images/*/shard_*/metadata.jsonl'],
        local_dir=str(cache),
    )

    candidates = metadata_records(cache, countries)
    chosen = {
        s: select_pothole_enriched_records(candidates[s], counts[s], seed + i)
        for i, s in enumerate(('train', 'valid', 'test'))
    }

    selection = {
        'dataset': DATA_REPO,
        'revision': DATA_REVISION,
        'seed': seed,
        'countries': list(countries) if countries else 'all',
        'counts': counts,
        'splits': chosen,
    }
    guard_json(backup / 'selection.json', selection)

    all_paths = [r['path'] for rows in chosen.values() for r in rows]
    print(f'Staging {len(all_paths)} images with {max_workers} parallel workers...', flush=True)

    def fetch(p: str) -> str:
        local_target = cache.joinpath(*safe_relative(p).parts)
        if local_target.exists() and local_target.stat().st_size > 0:
            return str(local_target)
        for attempt in range(6):
            try:
                return hf_hub_download(
                    DATA_REPO,
                    filename=p,
                    repo_type='dataset',
                    revision=DATA_REVISION,
                    local_dir=str(cache),
                )
            except Exception as err:
                if attempt == 5:
                    raise
                wait = min(60.0, (2.0 ** attempt) + random.uniform(0.1, 1.0))
                if '429' in str(err):
                    wait = max(wait, 15.0 * (attempt + 1))
                time.sleep(wait)
        raise RuntimeError('Download retries exhausted for ' + p)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for _ in pool.map(fetch, all_paths):
            completed_count += 1
            if completed_count % 500 == 0 or completed_count == len(all_paths):
                print(f'  {completed_count} / {len(all_paths)} images downloaded ({completed_count / len(all_paths):.1%})', flush=True)

    prepared = root / 'dataset'
    hashes: Dict[str, str] = {}
    exclusions: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        'dataset': DATA_REPO,
        'revision': DATA_REVISION,
        'classes': list(TARGET_CLASSES),
        'splits': {},
    }

    for split in ('train', 'valid', 'test'):
        rows = chosen[split]
        folder = prepared / split
        folder.mkdir(parents=True, exist_ok=True)
        coco: Dict[str, Any] = {
            'info': {'description': 'BANTAI 15K Single-Class Pothole Dataset'},
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': [{'id': 0, 'name': 'pothole', 'supercategory': 'road_damage'}],
        }
        manifest['splits'][split] = []
        pothole_count = 0
        positive_images = 0

        for row in rows:
            src = cache.joinpath(*safe_relative(row['path']).parts)
            digest = sha256(src)
            if digest in hashes:
                exclusions.append({
                    'split': split,
                    'path': row['path'],
                    'sha256': digest,
                    'retained': hashes[digest],
                    'reason': 'Exact duplicate image skipped to prevent cross-split leakage',
                })
                continue

            try:
                with Image.open(src) as img:
                    img.load()
                    orientation = img.getexif().get(274, 1)
                    if orientation != 1:
                        exclusions.append({
                            'split': split,
                            'path': row['path'],
                            'sha256': digest,
                            'orientation': orientation,
                            'reason': 'EXIF rotation/mirroring excluded without distortion',
                        })
                        continue
                    width, height = img.size
                anns = validate_objects_single_class(row['objects'], width, height)
            except (OSError, ValueError) as exc:
                exclusions.append({
                    'split': split,
                    'path': row['path'],
                    'sha256': digest,
                    'reason': str(exc),
                })
                continue

            hashes[digest] = row['path']
            img_id = len(coco['images'])
            filename = f'{img_id:05d}.jpg'
            target_image = folder / filename

            if target_image.exists() and sha256(target_image) != digest:
                target_image.unlink()
            if not target_image.exists():
                try:
                    os.link(src, target_image)
                except OSError:
                    atomic_copy(src, target_image)

            coco['images'].append({'id': img_id, 'file_name': filename, 'width': width, 'height': height})
            if anns:
                positive_images += 1
            for ann in anns:
                pothole_count += 1
                coco['annotations'].append({**ann, 'id': len(coco['annotations']) + 1, 'image_id': img_id})

            manifest['splits'][split].append({
                'id': img_id,
                'path': row['path'],
                'local_file': filename,
                'width': width,
                'height': height,
                'sha256': digest,
                'pothole_boxes': len(anns),
            })

        if not coco['images'] or pothole_count == 0:
            raise ValueError(f'No usable pothole images found in split {split}')

        write_json(folder / '_annotations.coco.json', coco)
        print(
            f'{split}: {len(coco["images"])} usable images '
            f'({positive_images} pothole-positive, {len(coco["images"]) - positive_images} hard-negative/background), '
            f'{pothole_count} pothole boxes',
            flush=True,
        )

    write_json(root / 'results/data_exclusions.json', exclusions)
    write_json(backup / 'data_exclusions.json', exclusions)
    manifest['id'] = identity(manifest)
    guard_json(backup / 'manifest.json', manifest)
    write_json(root / 'results/manifest.json', manifest)
    return prepared, manifest


# ---------------------------------------------------------------------------
# 3. GPU Profile Auto-Configuration (A100 / L4 / T4)
# ---------------------------------------------------------------------------

def auto_gpu_training_profile() -> Dict[str, Any]:
    """Inspect torch CUDA device and return optimized batch size, gradient accumulation, and workers."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {'device': 'cpu', 'batch_size': 2, 'grad_accum_steps': 8, 'num_workers': 0, 'gpu_name': 'CPU', 'vram_gb': 0.0}
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 35:  # A100 (40GB or 80GB)
            return {'device': 'cuda', 'batch_size': 32, 'grad_accum_steps': 1, 'num_workers': 4, 'gpu_name': name, 'vram_gb': round(vram_gb, 1)}
        elif vram_gb >= 20:  # L4 (24GB) or RTX 3090/4090
            return {'device': 'cuda', 'batch_size': 16, 'grad_accum_steps': 2, 'num_workers': 4, 'gpu_name': name, 'vram_gb': round(vram_gb, 1)}
        elif vram_gb >= 10:  # T4 (15GB)
            return {'device': 'cuda', 'batch_size': 8, 'grad_accum_steps': 4, 'num_workers': 2, 'gpu_name': name, 'vram_gb': round(vram_gb, 1)}
        else:
            return {'device': 'cuda', 'batch_size': 4, 'grad_accum_steps': 4, 'num_workers': 2, 'gpu_name': name, 'vram_gb': round(vram_gb, 1)}
    except Exception:
        return {'device': 'cpu', 'batch_size': 2, 'grad_accum_steps': 8, 'num_workers': 0, 'gpu_name': 'Unknown', 'vram_gb': 0.0}


# ---------------------------------------------------------------------------
# 4. Box Matching, Evaluation & Threshold Calibration
# ---------------------------------------------------------------------------

def box_iou(a: List[float], b: List[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0.0 else 0.0


def match_boxes(truth: List[List[float]], predictions: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, int]:
    unmatched = set(range(len(truth)))
    tp = 0
    for p in sorted(predictions, key=lambda x: x['score'], reverse=True):
        choices = [(box_iou(p['xyxy'], truth[i]), i) for i in unmatched]
        overlap, idx = max(choices, default=(0.0, -1))
        if idx >= 0 and overlap >= threshold:
            unmatched.remove(idx)
            tp += 1
    return {'tp': tp, 'fp': len(predictions) - tp, 'fn': len(truth) - tp}


def calibrate_operating_threshold(
    coco: Dict[str, Any],
    rows: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """Sweep confidence thresholds to find the optimal F1 operating point."""
    truth_by_image = collections.defaultdict(list)
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        truth_by_image[ann['image_id']].append([x, y, x + w, y + h])

    candidates = []
    for step in range(2, 37):
        conf = round(step * 0.025, 4)
        total_tp = total_fp = total_fn = 0
        bg_images = 0
        bg_alarms = 0

        for r in rows:
            img_truth = truth_by_image[r['image_id']]
            preds = [p for p in r['predictions'] if p['score'] >= conf]
            matched = match_boxes(img_truth, preds, iou_thresh)
            total_tp += matched['tp']
            total_fp += matched['fp']
            total_fn += matched['fn']
            if not img_truth:
                bg_images += 1
                if preds:
                    bg_alarms += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        alarm_rate = (bg_alarms / bg_images) if bg_images > 0 else 0.0

        candidates.append({
            'confidence': conf,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'false_alarm_rate': round(alarm_rate, 4),
        })

    best = max(candidates, key=lambda c: (c['f1'], c['recall'], -c['false_alarm_rate'], -c['confidence']))
    return {
        'optimal_confidence': best['confidence'],
        'best_operating_point': best,
        'sweep': candidates,
    }


# ---------------------------------------------------------------------------
# 5. Temporal Video & Webcam Anti-Flicker Tracker
# ---------------------------------------------------------------------------

class TemporalPotholeTracker:
    """Frame-to-frame IoU tracker with Exponential Moving Average (EMA) box smoothing."""

    def __init__(self, iou_threshold: float = 0.35, min_hits: int = 2, max_age: int = 5, alpha: float = 0.6):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.next_id = 1
        self.tracks: List[Dict[str, Any]] = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for track in self.tracks:
            track['age'] += 1
            track['time_since_update'] += 1

        matched_tracks = set()
        matched_dets = set()

        if self.tracks and detections:
            pairs = []
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    score = box_iou(track['xyxy'], det['xyxy'])
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

                track['xyxy'] = [
                    self.alpha * n + (1 - self.alpha) * o
                    for n, o in zip(det['xyxy'], track['xyxy'])
                ]
                track['score'] = det['score']
                track['hits'] += 1
                track['time_since_update'] = 0

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append({
                    'id': self.next_id,
                    'xyxy': list(det['xyxy']),
                    'score': det['score'],
                    'hits': 1,
                    'age': 1,
                    'time_since_update': 0,
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]

        confirmed = []
        for t in self.tracks:
            if t['hits'] >= self.min_hits and t['time_since_update'] == 0:
                confirmed.append({
                    'id': t['id'],
                    'xyxy': [round(v, 2) for v in t['xyxy']],
                    'score': round(t['score'], 3),
                    'class_name': 'pothole',
                })
        return confirmed


# ---------------------------------------------------------------------------
# 6. Turnkey Ready-to-Use Inference Module Generator
# ---------------------------------------------------------------------------

def generate_ready_pothole_detector_script(weights_filename: str, optimal_confidence: float) -> str:
    """Generate self-contained bantai_pothole_detector.py for images, videos, and webcams."""
    return f'''"""BANTAI Pothole Detector — Ready-to-Use Inference Engine.

Pretrained on 15,000+ multi-country road images (RF-DETR Nano).
Embedded Calibrated Operating Threshold: {optimal_confidence:.3f}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import cv2
from PIL import Image, ImageOps
import torch
from rfdetr import RFDETR

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent / "{weights_filename}")
DEFAULT_CONFIDENCE = {optimal_confidence:.3f}
CLASSES = ["pothole"]


def box_iou(a: List[float], b: List[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0.0 else 0.0


class TemporalPotholeTracker:
    def __init__(self, iou_threshold: float = 0.35, min_hits: int = 2, max_age: int = 5, alpha: float = 0.6):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.next_id = 1
        self.tracks = []

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
                track["xyxy"] = [self.alpha * n + (1 - self.alpha) * o for n, o in zip(det["xyxy"], track["xyxy"])]
                track["score"] = det["score"]
                track["hits"] += 1
                track["time_since_update"] = 0
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append({{"id": self.next_id, "xyxy": list(det["xyxy"]), "score": det["score"], "hits": 1, "age": 1, "time_since_update": 0}})
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]
        return [{{"id": t["id"], "xyxy": [round(v, 2) for v in t["xyxy"]], "score": round(t["score"], 3), "class_name": "pothole"}}
                for t in self.tracks if t["hits"] >= self.min_hits and t["time_since_update"] == 0]


class PotholeDetector:
    def __init__(self, weights_path: Optional[str] = None, confidence: float = DEFAULT_CONFIDENCE, device: str = "auto"):
        self.weights_path = weights_path or DEFAULT_WEIGHTS
        self.confidence = confidence
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        print(f"Loading RF-DETR Nano Pothole Model on {{self.device}}...")
        self.detector = RFDETR.from_checkpoint(self.weights_path, device=self.device)
        try:
            self.detector.optimize_for_inference(compile=False)
        except Exception as exc:
            print(f"Note: using standard eager inference ({{exc}})")
        self.tracker = TemporalPotholeTracker()
        print(f"Model ready. Calibrated confidence threshold: {{self.confidence:.2%}}")

    def detect(self, image: Union[str, Path, Image.Image], confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        conf = confidence if confidence is not None else self.confidence
        if isinstance(image, (str, Path)):
            with Image.open(image) as src:
                img = ImageOps.exif_transpose(src).convert("RGB")
        else:
            img = image.convert("RGB")
        detections = self.detector.predict(img, threshold=conf)
        results = []
        boxes = detections.xyxy.tolist()
        scores = detections.confidence.tolist()
        ids = detections.class_id.tolist()
        names = detections.data.get("class_name") if hasattr(detections, "data") and isinstance(detections.data, dict) else None
        for idx, (box, score, cid) in enumerate(zip(boxes, scores, ids)):
            name = str(names[idx]).lower() if names is not None and idx < len(names) else None
            if int(cid) == len(CLASSES) or name == "__background__":
                continue
            if int(cid) == 0 or name == "pothole":
                results.append({{"xyxy": [round(float(v), 2) for v in box], "score": round(float(score), 4), "class_name": "pothole"}})
        return results

    def run_video(self, source_path: str, output_path: str, confidence: Optional[float] = None, enable_tracking: bool = True):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError("Cannot open video: " + source_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.tracker = TemporalPotholeTracker()
        frame_idx = 0
        t0 = time.time()
        print(f"Processing video {{source_path}} -> {{output_path}}...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            raw_dets = self.detect(pil_img, confidence=confidence)
            active = self.tracker.update(raw_dets) if enable_tracking else raw_dets

            for item in active:
                x1, y1, x2, y2 = map(int, item["xyxy"])
                label = f"Pothole #{{item.get('id', '')}} {{item['score']:.0%}}".strip()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_idx += 1
            cv2.putText(frame, f"BANTAI Potholes: {{len(active)}}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        elapsed = time.time() - t0
        print(f"Video done. {{frame_idx}} frames in {{elapsed:.1f}}s ({{frame_idx / max(1e-3, elapsed):.1f}} FPS). Saved: {{output_path}}")

    def run_webcam(self, camera_index: Union[int, str] = 0, confidence: Optional[float] = None, enable_tracking: bool = True):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera/stream {{camera_index}}")
        self.tracker = TemporalPotholeTracker()
        print("Starting real-time pothole detection stream. Press 'q' to exit.")
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            raw_dets = self.detect(pil_img, confidence=confidence)
            active = self.tracker.update(raw_dets) if enable_tracking else raw_dets

            curr_time = time.time()
            fps = 1.0 / max(1e-4, curr_time - prev_time)
            prev_time = curr_time

            for item in active:
                x1, y1, x2, y2 = map(int, item["xyxy"])
                label = f"Pothole #{{item.get('id', '')}} {{item['score']:.0%}}".strip()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame, f"FPS: {{fps:.1f}} | Potholes: {{len(active)}}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("BANTAI Real-Time Pothole Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BANTAI Ready Pothole Detector")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to checkpoint")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--image", help="Detect single image")
    parser.add_argument("--source", help="Input video file path or RTSP URL")
    parser.add_argument("--output", default="annotated_potholes.mp4", help="Output video file path")
    parser.add_argument("--webcam", type=int, help="Camera index for real-time webcam (e.g. 0)")
    args = parser.parse_args()

    detector = PotholeDetector(weights_path=args.weights, confidence=args.confidence)
    if args.image:
        results = detector.detect(args.image)
        print(json.dumps(results, indent=2))
    elif args.webcam is not None:
        detector.run_webcam(camera_index=args.webcam)
    elif args.source:
        detector.run_video(args.source, args.output)
    else:
        print("No action specified. Run with --help, --image <path>, --source <video.mp4>, or --webcam 0")
'''
