"""Standalone helpers embedded in the BANTAI RF-DETR Colab notebook.

Importing this module performs no downloads, package installs, or training.
"""
from pathlib import Path, PurePosixPath
import collections
import hashlib
import json
import math
import random
import shutil
import time

CLASSES = ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole']
MODEL_REPO = 'dronefreak/rdd2022-rfdetr-nano'
MODEL_REVISION = '66429f39b916e56c6a33243d776718088f5f43ad'
DATA_REPO = 'dronefreak/RDD2022'
DATA_REVISION = '72b01f94c2e5bfe430861afd73995f2d412b8855'
PACKAGES = ['rfdetr[train]==1.11.2', 'huggingface-hub==2.3.0', 'pycocotools==2.0.11']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.partial')
    shutil.copyfile(source, temp)
    temp.replace(target)


def backup_results(source, target):
    for path in sorted(Path(source).rglob('*')):
        if path.is_file() and not path.name.endswith('.partial'):
            atomic_copy(path, Path(target) / path.relative_to(source))


def guard_json(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != json.loads(json.dumps(value)):
        raise ValueError('Saved settings/data differ. Choose a new RUN_NAME; do not overwrite this run.')
    write_json(path, value)


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe dataset path')
    return path


def validate_objects(objects, width=None, height=None):
    boxes, labels = objects['bbox'], objects['categories']
    if len(boxes) != len(labels):
        raise ValueError('Box/label counts differ')
    annotations = []
    for box, label in zip(boxes, labels):
        if type(label) is not int or not 0 <= label < len(CLASSES):
            raise ValueError('Unknown class ID')
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Invalid bounding box')
        x, y, w, h = map(float, box)
        if min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError('Invalid box extent')
        if width is not None and (x + w > width or y + h > height):
            raise ValueError('Box extends outside image; inspect source labels')
        annotations.append({'category_id': label, 'bbox': [x, y, w, h],
                            'area': w * h, 'iscrowd': 0})
    return annotations


def select_records(records, count, seed):
    """A fixed half-pothole diagnostic sample, preserving upstream assignments."""
    positive = sorted((r for r in records if 3 in r['objects']['categories']), key=lambda r: r['path'])
    negative = sorted((r for r in records if 3 not in r['objects']['categories']), key=lambda r: r['path'])
    npos, nneg = count // 2, count - count // 2
    if len(positive) < npos or len(negative) < nneg:
        raise ValueError(f'Insufficient images: need {npos} pothole/{nneg} no-pothole; '
                         f'available {len(positive)}/{len(negative)}. Reduce sample sizes in section 1.')
    rng = random.Random(seed)
    return sorted(rng.sample(positive, npos) + rng.sample(negative, nneg), key=lambda r: r['path'])


def metadata_records(cache, country=None):
    records = {split: [] for split in ('train', 'valid', 'test')}
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
                raise ValueError('Unexpected metadata filename layout')
            if country:
                if isinstance(country, (list, tuple, set)):
                    if not any(name.startswith(c + '_') for c in country):
                        continue
                elif not name.startswith(country + '_'):
                    continue
            validate_objects(item['objects'])
            remote = path.parent.relative_to(cache).as_posix() + '/' + name
            records[split].append({'path': remote, 'file_name': name, 'objects': item['objects']})
    for split, rows in records.items():
        if not rows:
            raise ValueError('No metadata records for ' + split)
        if len({r['path'] for r in rows}) != len(rows):
            raise ValueError('Duplicate metadata entries in ' + split)
    return records


def prepare_dataset(root, backup, counts, seed, country):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import snapshot_download, hf_hub_download
    from PIL import Image
    root, backup = Path(root), Path(backup)
    cache = root / 'download_cache'
    snapshot_download(DATA_REPO, repo_type='dataset', revision=DATA_REVISION,
                      allow_patterns=['data/images/*/shard_*/metadata.jsonl'], local_dir=cache)
    candidates = metadata_records(cache, country)
    chosen = {s: select_records(candidates[s], n, seed + i)
              for i, (s, n) in enumerate(counts.items())}
    selection = {'dataset': DATA_REPO, 'revision': DATA_REVISION, 'seed': seed,
                 'country': country, 'counts': counts, 'splits': chosen}
    guard_json(backup / 'selection.json', selection)
    prepared = root / 'dataset'
    all_paths = [r['path'] for rows in chosen.values() for r in rows]
    def fetch(path):
        return hf_hub_download(DATA_REPO, filename=path, repo_type='dataset',
                               revision=DATA_REVISION, local_dir=cache)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(fetch, all_paths)):
            if (i + 1) % 50 == 0:
                print(i + 1, '/', len(all_paths), 'images ready', flush=True)
    hashes = {}
    exclusions = []
    manifest = {'dataset': DATA_REPO, 'revision': DATA_REVISION, 'splits': {}, 'classes': CLASSES}
    for split, rows in chosen.items():
        folder = prepared / split
        folder.mkdir(parents=True, exist_ok=True)
        coco = {'info': {'description': 'BANTAI RDD2022 diagnostic subset'}, 'licenses': [],
                'images': [], 'annotations': [],
                'categories': [{'id': i, 'name': n, 'supercategory': 'road_damage'}
                               for i, n in enumerate(CLASSES)]}
        manifest['splits'][split] = []
        coverage = collections.Counter()
        for i, row in enumerate(rows):
            src = cache.joinpath(*safe_relative(row['path']).parts)
            digest = sha256(src)
            if digest in hashes:
                raise ValueError('Exact duplicate image in selected sample: ' + row['path']
                                 + ' and ' + hashes[digest] + '. Revise selection before proceeding.')
            with Image.open(src) as image:
                image.load()
                orientation = image.getexif().get(274, 1)
                if orientation != 1:
                    exclusions.append({'split': split, 'path': row['path'], 'sha256': digest,
                                       'orientation': orientation,
                                       'reason': 'EXIF rotation/mirroring requires annotation review; excluded without transforming pixels or boxes.'})
                    print('Quarantined:', row['path'], '| EXIF orientation:', orientation, flush=True)
                    write_json(root / 'results/data_exclusions.json', exclusions)
                    write_json(backup / 'data_exclusions.json', exclusions)
                    continue
                width, height = image.size
            hashes[digest] = row['path']
            anns = validate_objects(row['objects'], width, height)
            filename = f'{i:05d}.jpg'
            atomic_copy(src, folder / filename)
            coco['images'].append({'id': i, 'file_name': filename, 'width': width, 'height': height})
            for ann in anns:
                coverage[ann['category_id']] += 1
                coco['annotations'].append({**ann, 'id': len(coco['annotations']) + 1, 'image_id': i})
            manifest['splits'][split].append({**row, 'id': i, 'local_file': filename,
                                              'width': width, 'height': height, 'sha256': digest})
        if not coco['images'] or not coverage[3]:
            raise ValueError('No usable pothole evaluation/training data in ' + split)
        if split == 'train' and set(coverage) != set(range(4)):
            raise ValueError('Training sample must include all four classes; increase TRAIN_IMAGES.')
        write_json(folder / '_annotations.coco.json', coco)
        print(split, len(coco['images']), 'usable images of', len(rows), 'requested; boxes:',
              {CLASSES[c]: n for c, n in sorted(coverage.items())})
    write_json(root / 'results/data_exclusions.json', exclusions)
    write_json(backup / 'data_exclusions.json', exclusions)
    manifest['id'] = identity(manifest)
    guard_json(backup / 'manifest.json', manifest)
    write_json(root / 'results/manifest.json', manifest)
    return prepared, manifest


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def match_boxes(truth, predictions, threshold=.5):
    unmatched = set(range(len(truth)))
    tp = 0
    for prediction in sorted(predictions, key=lambda p: p['score'], reverse=True):
        choices = [(box_iou(prediction['xyxy'], truth[i]), i) for i in unmatched]
        overlap, index = max(choices, default=(0., -1))
        if index >= 0 and overlap >= threshold:
            unmatched.remove(index)
            tp += 1
    return {'tp': tp, 'fp': len(predictions)-tp, 'fn': len(truth)-tp}


def fixed_metrics(coco, rows, confidence):
    truth = collections.defaultdict(list)
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        truth[(ann['image_id'], ann['category_id'])].append([x, y, x+w, y+h])
    output = {}
    for category, name in enumerate(CLASSES):
        totals = collections.Counter()
        bg, alarms = 0, 0
        for row in rows:
            labels = truth[(row['image_id'], category)]
            predictions = [p for p in row['predictions'] if p['category_id'] == category and p['score'] >= confidence]
            totals.update(match_boxes(labels, predictions))
            if not labels:
                bg += 1
                alarms += bool(predictions)
        tp, fp, fn = (totals[k] for k in ('tp', 'fp', 'fn'))
        precision = tp / (tp+fp) if tp+fp else 0.
        recall = tp / (tp+fn) if tp+fn else 0.
        output[name] = {**dict(totals), 'precision': precision, 'recall': recall,
                        'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
                        'images_without_class': bg, 'false_alarm_images': alarms,
                        'image_false_alarm_rate': alarms/bg if bg else None}
    return output


def validate_model_class_id(value, image, detected_ids):
    """Reject unknown classes without silently dropping or remapping predictions."""
    try:
        class_id = int(value)
        valid = (
            not isinstance(value, bool)
            and class_id == value
            and 0 <= class_id < len(CLASSES)
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if valid:
        return class_id
    expected = dict(enumerate(CLASSES))
    raise ValueError(
        f"Unexpected model class ID: {value!r}; "
        f"image_id={image.get('id')!r}; file={image.get('file_name')!r}; "
        f"detected IDs in this image={detected_ids!r}; "
        f"expected mapping={expected!r}. "
        "Evaluation stopped to avoid incorrect metrics. Inspect the checkpoint's "
        "class-name mapping before changing IDs. Do not subtract 1 or discard "
        "detections without confirming that mapping."
    )


def foreground_predictions(detections, image):
    """Exclude only RF-DETR's explicitly named final background slot."""
    boxes = detections.xyxy.tolist()
    ids = detections.class_id.tolist()
    scores = detections.confidence.tolist()
    names = detections.data.get('class_name')
    if not (len(boxes) == len(ids) == len(scores)):
        raise ValueError('Detection array lengths do not match')
    if names is not None and len(names) != len(ids):
        raise ValueError('Detection class-name array length does not match')
    predictions = []
    background_filtered = 0
    for i, (box, cls, score) in enumerate(zip(boxes, ids, scores)):
        name = names[i] if names is not None else None
        # Confirmed from the supplied installed predict() implementation:
        # the final slot is returned with the explicit __background__ label.
        if cls == len(CLASSES) and name == '__background__':
            background_filtered += 1
            continue
        class_id = validate_model_class_id(cls, image, ids)
        predictions.append({'xyxy': box, 'category_id': class_id, 'score': float(score)})
    return predictions, background_filtered


def evaluate(detector, weights, prepared, split, out, backup, manifest_id, confidence, label):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from PIL import Image
    import torch
    out, backup, prepared = Path(out) / label, Path(backup) / label, Path(prepared)
    out.mkdir(parents=True, exist_ok=True)
    annotation_file = prepared / split / '_annotations.coco.json'
    coco = read_json(annotation_file)
    signature = identity({'model': sha256(weights), 'manifest': manifest_id, 'split': split,
                          'annotations': sha256(annotation_file), 'confidence': confidence,
                          'floor': .001, 'version': 'rfdetr-eval-v1'})
    guard_json(backup / 'identity.json', {'id': signature})
    rows = []
    # Warm up once; reported timings exclude this, IO, download and metric calculation.
    first = prepared / split / coco['images'][0]['file_name']
    detector.predict(str(first), threshold=.001)
    for i, image in enumerate(coco['images']):
        target = out / 'predictions' / f"{image['id']:05d}.json"
        saved = backup / 'predictions' / target.name
        if not target.exists() and saved.exists():
            atomic_copy(saved, target)
        if target.exists():
            row = read_json(target)
            if row['signature'] != signature or row['image_id'] != image['id']:
                raise ValueError('Prediction cache mismatch; use a new RUN_NAME')
        else:
            with Image.open(prepared / split / image['file_name']) as source:
                rgb = source.convert('RGB')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            detections = detector.predict(rgb, threshold=.001)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            predictions, background_filtered = foreground_predictions(detections, image)
            row = {'signature': signature, 'image_id': image['id'],
                   'predictions': predictions, 'predict_seconds': elapsed,
                   'background_filtered': background_filtered}
            write_json(target, row)
            atomic_copy(target, saved)
        rows.append(row)
        if (i+1) % 25 == 0:
            print(label, i+1, '/', len(coco['images']), 'saved', flush=True)
    ground_truth = COCO(str(annotation_file))
    detections = []
    for row in rows:
        for p in row['predictions']:
            x1, y1, x2, y2 = p['xyxy']
            detections.append({'image_id': row['image_id'], 'category_id': p['category_id'],
                               'score': p['score'], 'bbox': [x1, y1, x2-x1, y2-y1]})
    if detections:
        predicted = ground_truth.loadRes(detections)
    else:
        predicted = COCO()
        predicted.dataset = {**coco, 'annotations': []}
        predicted.createIndex()
    evaluator = COCOeval(ground_truth, predicted, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    per_class = {}
    values = evaluator.eval['precision']
    for c, name in enumerate(CLASSES):
        all_iou = values[:, :, c, 0, -1]
        iou50 = values[0, :, c, 0, -1]
        per_class[name] = {'map50_95': float(all_iou[all_iou > -1].mean()) if (all_iou > -1).any() else None,
                           'map50': float(iou50[iou50 > -1].mean()) if (iou50 > -1).any() else None}
    report = {'signature': signature, 'label': label, 'split': split, 'images': len(rows),
              'background_filtered': sum(r.get('background_filtered', 0) for r in rows),
              'model_sha256': sha256(weights), 'map50_95': float(evaluator.stats[0]),
              'map50': float(evaluator.stats[1]), 'per_class_ap': per_class,
              'operating_confidence': confidence, 'matching_iou': .5,
              'fixed_confidence': fixed_metrics(coco, rows, confidence),
              'mean_predict_ms': sum(r['predict_seconds'] for r in rows)/len(rows)*1000,
              'speed_note': 'Warm-start predict call, excludes file IO; cached records may span sessions.',
              'map_note': 'COCO maxDets=100, low prediction floor=0.001.'}
    write_json(out / 'summary.json', report)
    write_json(out / 'coco_predictions.json', detections)
    save_examples(prepared, split, coco, rows, out / 'examples', confidence)
    backup_results(out, backup)
    return report


def save_examples(prepared, split, coco, rows, destination, confidence):
    from PIL import Image, ImageDraw, ImageOps
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    index = []
    for image, row in zip(coco['images'], rows):
        anns = [a for a in coco['annotations'] if a['image_id'] == image['id'] and a['category_id'] == 3]
        truth = [[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in anns]
        predictions = [p for p in row['predictions'] if p['category_id'] == 3 and p['score'] >= confidence]
        match = match_boxes(truth, predictions)
        kind = 'miss' if match['fn'] else 'false_alarm' if match['fp'] else 'correct' if match['tp'] else 'background'
        if counts[kind] >= 3:
            continue
        counts[kind] += 1
        with Image.open(Path(prepared) / split / image['file_name']) as source:
            picture = source.convert('RGB')
        draw = ImageDraw.Draw(picture)
        for box in truth:
            draw.rectangle(box, outline='lime', width=3)
        for p in predictions:
            draw.rectangle(p['xyxy'], outline='red', width=2)
            draw.text(tuple(p['xyxy'][:2]), f"{p['score']:.2f}", fill='red')
        name = f"{kind}_{image['id']:05d}.jpg"
        ImageOps.contain(picture, (1280, 960)).save(destination / name)
        index.append({'file': name, 'image_id': image['id'], **match})
    write_json(destination / 'index.json', index)
