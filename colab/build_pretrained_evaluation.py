"""Build the standalone evaluation notebook; no model training or downloads here."""
import ast
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
cells = []
def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'})
def code(text):
    cells.append({'cell_type': 'code', 'execution_count': None, 'outputs': [], 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'})

md('''
# BANTAI — Pretrained pothole evaluation (v1)
**Evaluation only. No fine-tuning, training, or changes to your app.**

Upload this notebook to Colab. Select a T4 GPU if available and run sections in order.
We use PeterHdd's pothole YOLOv8s and a fixed sample of **200 pothole images + 100
images with no annotated potholes** from RDD2022 India. This deliberately balanced
sample is a diagnostic test, not an estimate of how often potholes occur in traffic.

Dataset/model work stays on Colab's temporary SSD (`/content`). Results are backed up
to Drive. A new runtime requires downloading/rebuilding the data again; completed
per-image predictions are restored from Drive. An interrupted mAP pass restarts that
short pass. Download the results ZIP in section 7 and send it back in our chat.

**Maintenance:** edit the existing section that fails, then rerun it and its dependent
sections. Keep one configuration cell. Use a new RUN_NAME when settings change.

Sources: [model](https://huggingface.co/peterhdd/pothole-detection-yolov8),
[RDD2022 official release](https://github.com/sekilab/RoadDamageDetector#dataset),
[RDD2022 paper](https://doi.org/10.1002/gdj3.260).
Training-data overlap has not been independently ruled out. Local Philippine camera
performance and physical pothole depth/severity are not measured by this notebook.
''')
md('''
## 1. Settings and persistent results
Run this first. Allow the Drive mount so completed predictions survive runtime loss.
The confidence threshold is fixed before evaluation; do not tune it on these results
and describe the rerun as an untouched test. Use a separate validation set for tuning.
''')
code('''
from pathlib import Path
from google.colab import drive
import hashlib, json, random, shutil, sys, subprocess, os

RUN_NAME = "peterhdd_rdd_india_eval_001"
SEED = 42
POSITIVE_IMAGES = 200
BACKGROUND_IMAGES = 100
IMAGE_SIZE = 640
CONFIDENCE = 0.25
MATCH_IOU = 0.50
MAP_SCORE_FLOOR = 0.001
ULTRALYTICS_VERSION = "8.3.221"
MODEL_REVISION = "da7747e"  # Fixed Hugging Face commit, never a moving 'main'.
MODEL_REPO = "peterhdd/pothole-detection-yolov8"

assert RUN_NAME.replace('_', '').isalnum(), "Use letters, numbers and underscores."
assert POSITIVE_IMAGES > 0 and BACKGROUND_IMAGES > 0
assert 0 < CONFIDENCE < 1 and 0 < MATCH_IOU <= 1
ROOT = Path('/content/bantai_pretrained') / RUN_NAME
OUT = ROOT / 'results'
drive.mount('/content/drive')
BACKUP = Path('/content/drive/MyDrive/BANTAI/pretrained_evaluation') / RUN_NAME
for folder in (ROOT, OUT, BACKUP):
    folder.mkdir(parents=True, exist_ok=True)
config = dict(version='pretrained-eval-v1', seed=SEED, positive=POSITIVE_IMAGES,
              background=BACKGROUND_IMAGES, imgsz=IMAGE_SIZE, confidence=CONFIDENCE,
              match_iou=MATCH_IOU, map_score_floor=MAP_SCORE_FLOOR,
              ultralytics=ULTRALYTICS_VERSION, repo=MODEL_REPO, revision=MODEL_REVISION)
config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
for previous in (OUT / 'config.json', BACKUP / 'config.json'):
    if previous.exists():
        assert json.loads(previous.read_text()) == config, 'Settings changed: choose a new RUN_NAME.'
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
shutil.copy2(OUT / 'config.json', BACKUP / 'config.json')
print('Run:', RUN_NAME, '| results backup:', BACKUP)
''')
md('''
## 2. Install the evaluation package
Running this cell installs the pinned Ultralytics package into this Colab runtime.
It reuses Colab's PyTorch/GPU setup. Transitive versions depend on Colab; the complete
installed environment is exported with the results. No packages are installed on your PC.
If Colab requests a runtime restart, restart and rerun from section 1.
''')
code('''
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                'ultralytics==' + ULTRALYTICS_VERSION], check=True)
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'ultralytics_settings')
import torch
import ultralytics
from ultralytics import YOLO, settings
from PIL import Image, ImageDraw, ImageOps
settings.update({'sync': False})
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print('Ultralytics:', ultralytics.__version__, '| PyTorch:', torch.__version__)
print('Device:', torch.cuda.get_device_name(0) if DEVICE == 0 else 'CPU (slower, still usable)')
(OUT / 'environment.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze'], text=True))
''')
md('''
## 3. Download the pretrained model and India dataset
The dataset is approximately 502 MB. Only the labeled `train` folders are extracted;
RDD's official challenge test folders have no labels and cannot produce local scores.
We are evaluating an existing model, so using RDD's labeled `train` folder does not
train anything. No Kaggle/Roboflow account is required.

If the original S3 link is unavailable, use the official Figshare mirror and retrieve
only the nested India ZIP. A server that refuses partial downloads stops safely.
Incomplete downloads restart when this section is rerun.
''')
# Reuse the established, audited downloader/parser, embedding them into the notebook.
source = (HERE / 'pothole_workflow.py').read_text(encoding='utf-8')
tree = ast.parse(source)
names = {'write_json', 'read_json', 'sha256', 'identity', 'atomic_copy', 'backup_tree',
         'extract_training', 'RangeDownload', 'mirror_country', 'download_country',
         'download_sources', 'difference_hash', 'parse_annotation', 'audit_sources'}
helpers = '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
code('''
import collections, io, math, re, stat, uuid, zipfile
import urllib.request, urllib.error
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
VERSION = 'pretrained-eval-v1'
COUNTRIES = ('India',)
BASE_URL = 'https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/'
URLS = {'India': BASE_URL + 'RDD2022_India.zip'}
FIGSHARE_ARTICLE_API = 'https://api.figshare.com/v2/articles/21431547'
FIGSHARE_FILENAME = 'RDD2022_released_through_CRDDC2022.zip'
''' + '\n' + helpers)
code('''
# Use the canonical weight file from a fixed model revision, and record its hash.
MODEL_PATH = ROOT / 'peterhdd_pothole.pt'
model_url = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/best.pt'
if not MODEL_PATH.exists():
    temporary = MODEL_PATH.with_suffix('.partial')
    with urllib.request.urlopen(model_url, timeout=120) as response, temporary.open('wb') as target:
        expected = response.headers.get('Content-Length')
        total = 0
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            target.write(chunk)
            total += len(chunk)
    if expected and total != int(expected):
        raise IOError('Incomplete model download. Rerun section 3.')
    if total < 1024 * 1024:
        raise ValueError('Unexpected model download size. Check the model URL.')
    temporary.replace(MODEL_PATH)
model_digest = sha256(MODEL_PATH)
model_info = dict(repo=MODEL_REPO, revision=MODEL_REVISION, url=model_url,
                  sha256=model_digest, bytes=MODEL_PATH.stat().st_size)
for previous in (OUT / 'model.json', BACKUP / 'model.json'):
    if previous.exists():
        assert read_json(previous)['sha256'] == model_digest, 'Model changed; choose a new RUN_NAME.'
write_json(OUT / 'model.json', model_info)
# Loading this PyTorch checkpoint trusts the selected publisher's model file.
model = YOLO(str(MODEL_PATH))
assert len(model.names) == 1 and 'pothole' in str(model.names[0]).lower(), model.names
print('Loaded:', model.names)
provenance = download_sources(ROOT)
write_json(OUT / 'source_provenance.json', provenance)
backup_tree(OUT, BACKUP)
''')
md('''
## 4. Prepare the fixed evaluation sample
Check image/XML dimensions, quarantine invalid labels, and remove exact duplicates.
Keep D40 potholes as class 0; other damage types are not potholes. Valid XML with no
D40 is treated as a background candidate. Small/difficult boxes remain included and
are flagged. Missing XML is never treated as an empty road.

Sampling is deterministic. The manifest records selected image and annotation hashes.
We do not claim route independence or absence of near duplicates. Background labels
may omit defects; inspect examples after evaluation. No sample images are shown
before the fixed evaluation.
''')
code('''
audit = audit_sources(ROOT)
records = audit['records']
rng = random.Random(SEED)
positives = [r for r in records if r['boxes']]
backgrounds = [r for r in records if not r['boxes']]
assert len(positives) >= POSITIVE_IMAGES, f'Only {len(positives)} usable positive images; review counts.'
assert len(backgrounds) >= BACKGROUND_IMAGES, f'Only {len(backgrounds)} usable backgrounds; review counts.'
selected = sorted(rng.sample(positives, POSITIVE_IMAGES) +
                  rng.sample(backgrounds, BACKGROUND_IMAGES), key=lambda r: r['id'])
manifest = dict(config_id=config_id, records=selected)
manifest['sample_id'] = identity(manifest)
for previous in (OUT / 'manifest.json', BACKUP / 'manifest.json'):
    if previous.exists():
        assert read_json(previous) == manifest, 'Sample changed; use a new RUN_NAME.'
write_json(OUT / 'manifest.json', manifest)
write_json(OUT / 'audit_summary.json', dict(summary=audit['summary'], exclusions=audit['exclusions']))
prepared = ROOT / ('sample_' + manifest['sample_id'][:12])
for folder in (prepared / 'images/val', prepared / 'labels/val'):
    folder.mkdir(parents=True, exist_ok=True)
for index, r in enumerate(selected):
    stem = f'{index:04d}'
    target = prepared / 'images/val' / (stem + Path(r['image']).suffix.lower())
    atomic_copy(ROOT / r['image'], target)
    r_width, r_height = r['width'], r['height']
    labels = []
    for x1, y1, x2, y2 in r['boxes']:
        labels.append(f'0 {(x1+x2)/2/r_width:.10f} {(y1+y2)/2/r_height:.10f} '
                      f'{(x2-x1)/r_width:.10f} {(y2-y1)/r_height:.10f}')
    (prepared / 'labels/val' / (stem + '.txt')).write_text('\n'.join(labels))
# YOLO requires a train key in its schema; no training method is ever called.
yaml_path = prepared / 'data.yaml'
yaml_path.write_text(f'path: {prepared.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n  0: pothole\n')
run_id = identity(dict(config_id=config_id, sample_id=manifest['sample_id'], model_sha256=model_digest))
write_json(OUT / 'run_identity.json', dict(run_id=run_id))
backup_tree(OUT, BACKUP)
print('Selected:', len(selected), '| pothole images:', POSITIVE_IMAGES,
      '| backgrounds:', BACKGROUND_IMAGES,
      '| total pothole boxes:', sum(len(r['boxes']) for r in selected))
''')
md('''
## 5. Standard evaluation: mAP
mAP checks detection quality across confidence thresholds. It uses a low prediction
score floor (0.001), not the operating confidence of 0.25. Section 6 reports the
actual false alarms/misses at that operating confidence. These are distinct measures.

This pass is cached when complete. If interrupted, rerun it. Start with batch 4;
if GPU memory runs out, change only `batch=4` to `batch=1` here and rerun.
''')
code('''
map_file = OUT / 'map_metrics.json'
if not map_file.exists() and (BACKUP / map_file.name).exists():
    atomic_copy(BACKUP / map_file.name, map_file)
if map_file.exists():
    map_report = read_json(map_file)
    assert map_report['run_id'] == run_id, 'Cached evaluation belongs to another run.'
else:
    metrics = model.val(data=str(yaml_path), split='val', imgsz=IMAGE_SIZE,
                        batch=4, device=DEVICE, workers=0, conf=MAP_SCORE_FLOOR,
                        iou=0.7, max_det=300, half=False, plots=True,
                        project=str(ROOT / 'map_runs'), name='evaluation', exist_ok=True,
                        save_json=True, verbose=False)
    map_report = dict(run_id=run_id, map50=float(metrics.box.map50),
                      map50_95=float(metrics.box.map), speed_ms_per_image=metrics.speed,
                      device=str(DEVICE), note='Speed excludes download; not app latency.')
    # Preserve standard charts/JSON, without copying raw dataset or weights.
    for p in Path(metrics.save_dir).rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.json', '.csv', '.yaml'}:
            atomic_copy(p, OUT / 'standard_evaluation' / p.relative_to(metrics.save_dir))
    write_json(map_file, map_report)
backup_tree(OUT, BACKUP)
print(json.dumps(map_report, indent=2))
''')
md('''
## 6. Fixed-confidence audit and example mistakes
A correct detection needs one-to-one overlap of at least 0.50 IoU with a labeled
pothole. Predictions are matched in descending confidence order. Extra boxes are
false positives; unmatched potholes are misses. Save each completed image to Drive
immediately. On reconnection rerun sections 1–4, then 5–6; saved images are skipped.

Green boxes = dataset labels. Red boxes = model predictions at confidence 0.25.
Examples include correct detections, false alarms and misses where available.
''')
code('''
def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0

def match_boxes(truth, predictions, threshold):
    unmatched = set(range(len(truth)))
    tp = 0
    for p in sorted(predictions, key=lambda p: p['confidence'], reverse=True):
        candidates = [(box_iou(p['box'], truth[i]), i) for i in unmatched]
        overlap, best = max(candidates, default=(0.0, -1))
        if best >= 0 and overlap >= threshold:
            unmatched.remove(best)
            tp += 1
    return dict(tp=tp, fp=len(predictions)-tp, fn=len(truth)-tp)

prediction_dir = OUT / 'predictions'
prediction_dir.mkdir(exist_ok=True)
rows = []
for index, record in enumerate(selected):
    path = prediction_dir / f'{index:04d}.json'
    saved = BACKUP / 'predictions' / path.name
    if not path.exists() and saved.exists():
        atomic_copy(saved, path)
    if path.exists():
        row = read_json(path)
        assert row['run_id'] == run_id and row['id'] == record['id'], 'Prediction cache mismatch.'
    else:
        prediction = model.predict(str(ROOT / record['image']), imgsz=IMAGE_SIZE,
                                   conf=CONFIDENCE, iou=0.7, max_det=300,
                                   device=DEVICE, half=False, verbose=False)[0]
        boxes = prediction.boxes.xyxy.cpu().tolist()
        scores = prediction.boxes.conf.cpu().tolist()
        predictions = [dict(box=b, confidence=s) for b, s in zip(boxes, scores)]
        row = dict(run_id=run_id, id=record['id'], predictions=predictions,
                   truth=record['boxes'], **match_boxes(record['boxes'], predictions, MATCH_IOU))
        write_json(path, row)
        atomic_copy(path, saved)
    rows.append(row)
    if (index+1) % 25 == 0:
        print(index+1, '/', len(selected), 'saved')

tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
precision = tp/(tp+fp) if tp+fp else 0.0
recall = tp/(tp+fn) if tp+fn else 0.0
negative_rows = [r for r in rows if not r['truth']]
false_alarm_images = sum(r['fp'] > 0 for r in negative_rows)
report = dict(run_id=run_id, model=model_info, config=config, images=len(rows),
              potholes=sum(len(r['truth']) for r in rows), tp=tp, fp=fp, fn=fn,
              precision=precision, recall=recall,
              f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
              background_images=len(negative_rows), background_false_alarm_images=false_alarm_images,
              background_false_alarm_rate=false_alarm_images/len(negative_rows) if negative_rows else None,
              standard_evaluation=map_report,
              limitations=['Diagnostic enriched sample, not natural prevalence.',
                           'Training overlap and route independence not established.',
                           'RDD labels can miss defects; local camera validation still needed.'])
write_json(OUT / 'summary.json', report)
examples = OUT / 'examples'
examples.mkdir(exist_ok=True)
example_index = []
for kind, predicate in [('miss', lambda r: r['fn'] > 0),
                        ('false_alarm', lambda r: r['fp'] > 0),
                        ('correct', lambda r: r['tp'] > 0 and r['fp'] == 0 and r['fn'] == 0)]:
    indices = [i for i, row in enumerate(rows) if predicate(row)][:4]
    for i in indices:
        row, record = rows[i], selected[i]
        with Image.open(ROOT / record['image']) as source_image:
            image = source_image.convert('RGB')
        draw = ImageDraw.Draw(image)
        for box in row['truth']:
            draw.rectangle(box, outline='lime', width=3)
        for p in row['predictions']:
            draw.rectangle(p['box'], outline='red', width=2)
            draw.text((p['box'][0], p['box'][1]), f"{p['confidence']:.2f}", fill='red')
        name = f'{kind}_{i:04d}.jpg'
        ImageOps.contain(image, (1280, 960)).save(examples / name)
        example_index.append(dict(file=name, id=record['id'], tp=row['tp'], fp=row['fp'], fn=row['fn']))
write_json(OUT / 'example_index.json', example_index)
backup_tree(OUT, BACKUP)
print(f'Precision: {precision:.1%} | Recall: {recall:.1%} | Correct: {tp} | False boxes: {fp} | Missed: {fn}')
print(f'Background images with false alarms: {false_alarm_images}/{len(negative_rows)}')
from IPython.display import display
for item in example_index:
    print(item)
    with Image.open(examples / item['file']) as preview:
        display(ImageOps.contain(preview, (800, 500)))
''')
md('''
## 7. Export and send the results
Download the ZIP and attach it in our chat. It includes summary scores, per-image
predictions, image manifest, settings, package versions, charts and example images.
It does not include the full dataset or model weights.

If the browser download fails, find the ZIP in your Drive run folder. Share the ZIP
and, if any step failed, its section number and error text. We will fix that existing
section instead of adding workaround cells. Scores alone do not establish deployment
readiness; we will review misses, false alarms and local camera examples together.
''')
code('''
assert (OUT / 'summary.json').exists(), 'Complete section 6 before exporting.'
backup_tree(OUT, BACKUP)
archive = Path(shutil.make_archive(str(ROOT / (RUN_NAME + '_results')), 'zip', OUT))
atomic_copy(archive, BACKUP / archive.name)
print('Saved ZIP:', BACKUP / archive.name)
from google.colab import files
files.download(str(archive))
''')
for index, cell in enumerate(cells):
    cell['id'] = f'eval-{index:02d}'
notebook = dict(nbformat=4, nbformat_minor=5, metadata={
    'colab': {'name': 'BANTAI_Pretrained_Pothole_Evaluation.ipynb'},
    'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python'}}, cells=cells)
output = HERE / 'BANTAI_Pretrained_Pothole_Evaluation.ipynb'
output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Built:', output.name, '| cells:', len(cells))
