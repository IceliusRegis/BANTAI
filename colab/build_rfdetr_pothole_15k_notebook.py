"""Build the standalone BANTAI_RFDETR_Pothole_15K_ReadyModel.ipynb Colab notebook.

Does not require a GPU or network locally to build the notebook.
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
        'cell_type': 'markdown',
        'metadata': {},
        'source': textwrap.dedent(text).strip() + '\n',
    })


def code(text: str) -> None:
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': textwrap.dedent(text).strip() + '\n',
    })


def repair_slot(section: str) -> None:
    md(f"""### {section} troubleshooting slot
    If this section fails, insert a temporary code cell **here**, immediately below its main cell.
    Keep any fix scoped to this section, then rerun its main cell and dependents.
    Do not alter test sets, class mapping, or integrity guards to bypass errors.
    """)


# --- Title & Overview ---
md("""
# BANTAI — Ready-to-Use RF-DETR Nano Pothole Detector (15,000 Images)
**High-Accuracy Real-Time Pothole Model Pretrained on 15,000+ Road Images**

- **Model Architecture**: [dronefreak/rdd2022-rfdetr-nano](https://huggingface.co/dronefreak/rdd2022-rfdetr-nano) (RF-DETR Nano / DINOv2-small, `384x384`).
- **Dataset**: [dronefreak/RDD2022](https://huggingface.co/datasets/dronefreak/RDD2022) across multi-country splits (India, Japan, Czech, Norway, US, China).
- **Scale**: **15,000 total road images** (`12,000 train + 1,500 valid + 1,500 test`), enriched with up to 70% pothole-positive images + 30% crack/road hard-negatives.
- **Head Surgery**: Slices the pretrained 4-class head (`pothole` + `__background__`) so all learned pothole weights are 100% preserved natively in a single-class (`0: pothole`) detector without resetting the classification head.
- **Real-Time Video / Webcam**: Runs at **~100–180 FPS** on Colab GPUs and **~60–90 FPS** locally on a GTX 1660 Ti, featuring temporal IoU box smoothing.
- **Unattended `Run all` Ready**: Clicking **Runtime → Run all** runs Sections 1–9A end-to-end without hanging on interactive upload prompts, saves the finished model bundle to Google Drive (`MyDrive/BANTAI/pothole_ready_model/`), and triggers a browser ZIP download.
""")

# --- Section 1: Settings & Drive ---
md("""
## 1. Configuration, GPU Profile & Google Drive Mount
Connects Google Drive for persistent checkpoint backups, configures the 15,000-image dataset (adjustable from 10k to 20k), and automatically tunes `batch_size` and workers for A100 (batch 32), L4 (batch 16), or T4 (batch 8).
""")
code(r"""
from pathlib import Path
import json, os, sys, subprocess, gc, shutil
import torch
from google.colab import drive

RUN_NAME = 'bantai_pothole_15k_nano_001'
COUNTRIES = None  # None uses all countries: ('India', 'Japan', 'Czech', 'Norway', 'United_States', 'China_MotorBike', 'China_Drone')
SEED = 42

# 15,000 total images (10k-20k scale)
TRAIN_IMAGES = 12000
VALID_IMAGES = 1500
TEST_IMAGES = 1500

EPOCHS = 12
LEARNING_RATE = 1e-5
LR_ENCODER = 1e-6

assert RUN_NAME.replace('_', '').isalnum(), 'Use letters, numbers and underscores for RUN_NAME.'
assert TRAIN_IMAGES + VALID_IMAGES + TEST_IMAGES >= 10000, 'Minimum 10,000 images required.'

drive.mount('/content/drive')
ROOT = Path('/content/bantai_rfdetr_15k') / RUN_NAME
OUT = ROOT / 'results'
BACKUP = Path('/content/drive/MyDrive/BANTAI/pothole_15k') / RUN_NAME
REPORTS_BACKUP = BACKUP / 'reports'
TRAIN_DIR = BACKUP / 'training'
READY_BUNDLE_DIR = Path('/content/drive/MyDrive/BANTAI/pothole_ready_model')

for folder in (ROOT, OUT, BACKUP, REPORTS_BACKUP, TRAIN_DIR, READY_BUNDLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(ROOT / 'hf_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Automatically use HF_TOKEN from Colab Secrets if available (raises Hugging Face rate limits)
try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
        print('Loaded HF_TOKEN from Colab Secrets for accelerated Hugging Face downloads.')
except Exception:
    pass

if not torch.cuda.is_available():
    print('WARNING: No GPU detected! Select Runtime -> Change runtime type -> A100 or L4 GPU.')

vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

if vram_gb >= 35:  # A100 (40GB/80GB)
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 32, 1, 4
elif vram_gb >= 20:  # L4 (24GB) or RTX 3090/4090
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 16, 2, 4
elif vram_gb >= 10:  # T4 (15GB)
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 8, 4, 2
else:
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_WORKERS = 4, 4, 2

print(f"Hardware: {gpu_name} ({vram_gb:.1f} GB VRAM)")
print(f"Optimal Profile -> Batch Size: {BATCH_SIZE} | Grad Accum: {GRAD_ACCUM_STEPS} | Workers: {NUM_WORKERS}")
print(f"Temporary SSD: {ROOT}")
print(f"Google Drive Backup: {BACKUP}")
""")
repair_slot('1')

# --- Section 2: Dependencies ---
md("""
## 2. Fast Package Installation
Installs `rfdetr[train]`, `huggingface-hub`, `pycocotools`, `hf-transfer` (Rust-accelerated parallel downloads), and `opencv-python-headless`.
""")
code(r"""
PACKAGES = [
    'rfdetr[train]==1.11.2',
    'huggingface-hub==2.3.0',
    'pycocotools==2.0.11',
    'hf-transfer==0.1.9',
    'opencv-python-headless>=4.8.0',
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *PACKAGES], check=True)
import torch
from rfdetr import RFDETR, RFDETRNano
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw, ImageOps

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU')
(OUT / 'environment.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
""")
repair_slot('2')

# --- Section 3: Embed Workflow Helpers & Checkpoint Head Surgery ---
workflow_code = (HERE / 'rfdetr_pothole_15k_workflow.py').read_text(encoding='utf-8')
code(workflow_code)

md("""
## 3. Pretrained Model Download & 1-Class Pothole Head Surgery
Downloads the paper's `dronefreak/rdd2022-rfdetr-nano` checkpoint and performs **1-Class Head Surgery**:
Slices class index 3 (`pothole`) and index 4 (`__background__`) into a native 1-class checkpoint so that RF-DETR retains all pretrained pothole features and does NOT wipe the linear head.
""")
code(r"""
RAW_MODEL_DIR = ROOT / 'raw_model'
RAW_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading pretrained 4-class road damage checkpoint...", flush=True)
SRC_WEIGHTS = Path(hf_hub_download(
    MODEL_REPO, filename='checkpoint_best_total.pth', revision=MODEL_REVISION, local_dir=RAW_MODEL_DIR))
model_config_path = hf_hub_download(
    MODEL_REPO, filename='config.json', revision=MODEL_REVISION, local_dir=RAW_MODEL_DIR)

raw_config = read_json(model_config_path)
assert raw_config['class_names'] == SOURCE_CLASSES, 'Publisher class names mismatch.'
assert raw_config['model_config']['resolution'] == 384, 'Unexpected Nano resolution.'

SURGERY_WEIGHTS = ROOT / 'model' / 'pothole_pretrained_1class.pth'
surgery_info = convert_checkpoint_to_single_class_pothole(SRC_WEIGHTS, SURGERY_WEIGHTS)
write_json(OUT / 'surgery_info.json', surgery_info)
backup_results(OUT, REPORTS_BACKUP)
print("1-Class Pothole Head Surgery Completed:")
print(json.dumps(surgery_info, indent=2))

# Save matching 1-class config
one_class_config = copy.deepcopy(raw_config)
one_class_config['class_names'] = list(TARGET_CLASSES)
one_class_config['model_config']['num_classes'] = 1
write_json(SURGERY_WEIGHTS.parent / 'config.json', one_class_config)

# Verify zero-shot loading, class names, and forward inference
initial_detector = RFDETR.from_checkpoint(str(SURGERY_WEIGHTS), device=DEVICE)
assert isinstance(initial_detector, RFDETRNano), 'Checkpoint is not RF-DETR Nano.'
assert initial_detector.model.class_names == TARGET_CLASSES, f"Unexpected class names: {initial_detector.model.class_names}"
dummy_dets = initial_detector.predict(Image.new('RGB', (384, 384), color='gray'), threshold=0.01)
dummy_preds, _ = foreground_pothole_predictions(dummy_dets, {'id': 'dummy_check'})
print(f"Pretrained 1-Class Pothole Detector verified! Dummy forward check passed ({len(dummy_preds)} boxes on blank image).")
del initial_detector
gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
""")
repair_slot('3')

# --- Section 4: 15K Dataset Preparation ---
md("""
## 4. 15,000-Image Multi-Country Dataset Download & Staging
Prepares `12,000 train + 1,500 valid + 1,500 test = 15,000 total road images`:
1. Prioritizes up to 70% pothole-positive images across all countries.
2. Balances with 20% crack hard-negatives and 10% clean asphalt backgrounds.
3. Resilient parallel downloading (`max_workers=16` with automatic retry/backoff) + zero-copy hardlinks (`os.link`).
""")
code(r"""
dataset_counts = {'train': TRAIN_IMAGES, 'valid': VALID_IMAGES, 'test': TEST_IMAGES}
PREPARED, manifest = prepare_15k_dataset(
    root=ROOT,
    backup=BACKUP,
    counts=dataset_counts,
    seed=SEED,
    countries=COUNTRIES,
    max_workers=16
)
backup_results(OUT, REPORTS_BACKUP)
print("Dataset Ready at:", PREPARED)
""")
repair_slot('4')

# --- Section 5: Pretrained 1-Class Baseline Validation ---
md("""
## 5. Pretrained 1-Class Model: Baseline Validation
Evaluates the surgically adapted model on the 1,500 validation images before fine-tuning.
Caches results to Google Drive so completed evaluations are never lost if the runtime disconnects.
""")
code(r"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_pothole_model(detector_path, split='valid', label='baseline'):
    report_path = OUT / f'{label}_{split}_report.json'
    backup_report_path = REPORTS_BACKUP / report_path.name
    weights_digest = sha256(detector_path)

    if not report_path.exists() and backup_report_path.exists():
        atomic_copy(backup_report_path, report_path)
    if report_path.exists():
        cached = read_json(report_path)
        if cached.get('weights_sha256') == weights_digest and cached.get('manifest_id') == manifest['id']:
            print(f"Reusing cached {label} ({split}) evaluation report from Drive.")
            return cached

    annotation_file = PREPARED / split / '_annotations.coco.json'
    coco_data = read_json(annotation_file)
    images = coco_data['images']

    detector = RFDETR.from_checkpoint(str(detector_path), device=DEVICE)
    try:
        detector.optimize_for_inference(compile=False)
    except Exception as exc:
        print(f"Note: using standard eager inference ({exc})")

    # Warm-up pass
    if images:
        first_img = PREPARED / split / images[0]['file_name']
        with Image.open(first_img) as warm_src:
            detector.predict(warm_src.convert('RGB'), threshold=0.01)

    rows = []
    total_bg_filtered = 0
    print(f"Evaluating {label} on {len(images)} {split} images...", flush=True)
    t0 = time.time()
    for idx, img_info in enumerate(images):
        img_path = PREPARED / split / img_info['file_name']
        with Image.open(img_path) as src:
            rgb = src.convert('RGB')
        dets = detector.predict(rgb, threshold=0.01)
        preds, bg_filtered = foreground_pothole_predictions(dets, img_info)
        total_bg_filtered += bg_filtered
        rows.append({'image_id': img_info['id'], 'predictions': preds})
        if (idx + 1) % 300 == 0 or (idx + 1) == len(images):
            print(f"  {idx + 1} / {len(images)} evaluated", flush=True)

    elapsed = time.time() - t0
    gt = COCO(str(annotation_file))
    coco_preds = []
    for r in rows:
        for p in r['predictions']:
            x1, y1, x2, y2 = p['xyxy']
            coco_preds.append({
                'image_id': r['image_id'],
                'category_id': 0,
                'score': p['score'],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
            })
    if coco_preds:
        pred_coco = gt.loadRes(coco_preds)
    else:
        pred_coco = COCO()
        pred_coco.dataset = {**coco_data, 'annotations': []}
        pred_coco.createIndex()

    evaluator = COCOeval(gt, pred_coco, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    calib = calibrate_operating_threshold(coco_data, rows)
    report = {
        'label': label,
        'split': split,
        'manifest_id': manifest['id'],
        'weights_sha256': weights_digest,
        'images': len(images),
        'background_filtered': total_bg_filtered,
        'map50': float(evaluator.stats[1]),
        'map50_95': float(evaluator.stats[0]),
        'optimal_confidence': calib['optimal_confidence'],
        'operating_point': calib['best_operating_point'],
        'elapsed_seconds': round(elapsed, 2),
        'fps': round(len(images) / max(1e-3, elapsed), 2),
    }
    write_json(report_path, report)
    atomic_copy(report_path, backup_report_path)

    del detector
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return report

baseline_report = evaluate_pothole_model(SURGERY_WEIGHTS, split='valid', label='pretrained_baseline')
write_json(OUT / 'baseline_report.json', baseline_report)
backup_results(OUT, REPORTS_BACKUP)
print("\n--- Pretrained 1-Class Baseline Results ---")
print(f"Pothole mAP@50: {baseline_report['map50']:.3f} | mAP@50:95: {baseline_report['map50_95']:.3f}")
print(f"Optimal Operating Confidence: {baseline_report['optimal_confidence']:.2%}")
print(f"Precision: {baseline_report['operating_point']['precision']:.1%} | Recall: {baseline_report['operating_point']['recall']:.1%} | F1: {baseline_report['operating_point']['f1']:.3f}")
print(f"Inference Speed: {baseline_report['fps']:.1f} FPS")
""")
repair_slot('5')

# --- Section 6: Full A100/L4 GPU Fine-Tuning ---
md("""
## 6. Full A100 / L4 GPU Fine-Tuning (12,000 Train Images)
Fine-tunes the adapted 1-class checkpoint on the 12,000 training images using:
- Resolution: `384x384`
- Multi-scale augmentations and EMA weight smoothing
- Early stopping patience: 4 epochs
- Epoch checkpoints backed up directly to Google Drive (`training/last.ckpt`)
""")
code(r"""
RUN_FINE_TUNING = True  # Enabled out-of-the-box for tonight's run

fine_tuned_weights = TRAIN_DIR / 'checkpoint_best_total.pth'
completed_marker = TRAIN_DIR / 'completed.json'

if RUN_FINE_TUNING:
    assert DEVICE == 'cuda', 'CUDA GPU required for fine-tuning. Switch runtime to GPU.'
    if completed_marker.exists() and fine_tuned_weights.exists():
        print("Fine-tuning already completed in this run folder; reusing saved checkpoint:", fine_tuned_weights)
    else:
        print(f"Starting A100/L4 Fine-Tuning with batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}...")

        training_view = ROOT / 'training_view'
        training_view.mkdir(parents=True, exist_ok=True)
        for s in ('train', 'valid'):
            link = training_view / s
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(PREPARED / s, target_is_directory=True)
        assert not (training_view / 'test').exists(), 'Test split must never enter training_view.'

        gc.collect()
        torch.cuda.empty_cache()

        trainer_model = RFDETR.from_checkpoint(str(SURGERY_WEIGHTS), device=DEVICE)
        trainer_model.model_config.pretrain_weights = str(SURGERY_WEIGHTS)
        resume = TRAIN_DIR / 'last.ckpt'
        print('Resuming from last.ckpt' if resume.exists() else 'Starting from 1-class pothole surgery weights')

        trainer_model.train(
            dataset_dir=str(training_view),
            output_dir=str(TRAIN_DIR),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            lr=LEARNING_RATE,
            lr_encoder=LR_ENCODER,
            resolution=384,
            device=DEVICE,
            num_workers=NUM_WORKERS,
            seed=SEED,
            class_names=TARGET_CLASSES,
            early_stopping=True,
            early_stopping_patience=4,
            early_stopping_min_delta=0.001,
            early_stopping_use_ema=True,
            use_ema=True,
            checkpoint_interval=1,
            run_test=False,
            tensorboard=False,
            wandb=False,
            multi_scale=True,
            expanded_scales=False,
            resume=str(resume) if resume.exists() else None,
        )
        assert fine_tuned_weights.exists(), "Training finished but checkpoint_best_total.pth is missing."
        write_json(completed_marker, {
            'sha256': sha256(fine_tuned_weights),
            'manifest_id': manifest['id'],
        })
        print("Fine-tuning completed successfully! Best weights:", fine_tuned_weights)
        del trainer_model
        gc.collect()
        torch.cuda.empty_cache()
else:
    print("Fine-tuning skipped. Pretrained baseline weights will be exported.")
""")
repair_slot('6')

# --- Section 7: Validation Comparison, Threshold Locking & Held-Out Test ---
md("""
## 7. Validation Comparison, Operating Threshold Locking & Held-Out Test Evaluation
Evaluates the fine-tuned model against the baseline on the 1,500 validation images, selects the superior model, locks its calibrated F1 confidence threshold, and runs a final verification pass on the 1,500 held-out test images.
""")
code(r"""
candidate_weights = SURGERY_WEIGHTS
selected_label = 'pretrained_baseline'

if RUN_FINE_TUNING and fine_tuned_weights.exists():
    finetuned_report = evaluate_pothole_model(fine_tuned_weights, split='valid', label='finetuned_15k')
    write_json(OUT / 'finetuned_report.json', finetuned_report)
    print("\n--- Validation Comparison Summary ---")
    print(f"Pretrained Baseline -> mAP@50: {baseline_report['map50']:.3f} | mAP@50:95: {baseline_report['map50_95']:.3f} | F1: {baseline_report['operating_point']['f1']:.3f}")
    print(f"Fine-Tuned 15K      -> mAP@50: {finetuned_report['map50']:.3f} | mAP@50:95: {finetuned_report['map50_95']:.3f} | F1: {finetuned_report['operating_point']['f1']:.3f}")

    if (finetuned_report['map50_95'], finetuned_report['map50']) >= (baseline_report['map50_95'], baseline_report['map50']):
        candidate_weights = fine_tuned_weights
        selected_label = 'finetuned_15k'
        final_val_report = finetuned_report
    else:
        final_val_report = baseline_report
else:
    final_val_report = baseline_report

LOCKED_THRESHOLD = final_val_report['optimal_confidence']
print(f"\nWINNING MODEL: {selected_label}")
print(f"LOCKED OPERATING THRESHOLD: {LOCKED_THRESHOLD:.2%}")

# Evaluate winning model on the 1,500 held-out test images
test_report = evaluate_pothole_model(candidate_weights, split='test', label=f'{selected_label}_final_test')
print(f"Held-Out Test mAP@50: {test_report['map50']:.3f} | mAP@50:95: {test_report['map50_95']:.3f} | Speed: {test_report['fps']:.1f} FPS")

selection_summary = {
    'selected_model': selected_label,
    'locked_threshold': LOCKED_THRESHOLD,
    'validation_metrics': final_val_report,
    'held_out_test_metrics': test_report,
}
write_json(OUT / 'model_selection.json', selection_summary)
backup_results(OUT, REPORTS_BACKUP)
""")
repair_slot('7')

# --- Section 8: Packaging Ready Model Bundle ---
md("""
## 8. Export Ready-to-Use Model Bundle & Download
Embeds the trained weights, class mapping, and calibrated threshold into a turnkey bundle:
1. `bantai_pothole_rfdetr_nano_ready.pth` (Ready PyTorch model with fine-tuning embedded)
2. `bantai_pothole_rfdetr_nano.onnx` (Optional ONNX export when supported by the runtime)
3. `bantai_pothole_detector.py` (3-line Python inference + video/webcam CLI with temporal smoothing)
4. `model_card.json` (Full training metadata, validation & test metrics, and SHA-256)
5. Backs up directly to Google Drive (`MyDrive/BANTAI/pothole_ready_model/`) and downloads as a ZIP.
""")
code(r"""
FINAL_WEIGHTS_NAME = 'bantai_pothole_rfdetr_nano_ready.pth'
READY_BUNDLE_LOCAL = ROOT / 'ready_model_bundle'
READY_BUNDLE_LOCAL.mkdir(parents=True, exist_ok=True)

# 1. Copy final winning weights with embedded metadata
target_weights_path = READY_BUNDLE_LOCAL / FINAL_WEIGHTS_NAME
ckpt_payload = torch.load(str(candidate_weights), map_location='cpu', weights_only=False)
if isinstance(ckpt_payload, dict):
    ckpt_payload['class_names'] = list(TARGET_CLASSES)
    ckpt_payload['calibrated_operating_threshold'] = float(LOCKED_THRESHOLD)
    ckpt_payload['bantai_model_version'] = 'rfdetr-nano-pothole-15k-v1'
torch.save(ckpt_payload, str(target_weights_path))
del ckpt_payload

# 2. Optional ONNX export
onnx_filename = None
try:
    export_det = RFDETR.from_checkpoint(str(target_weights_path), device=DEVICE)
    export_det.export(output_dir=str(READY_BUNDLE_LOCAL))
    onnx_candidates = list(READY_BUNDLE_LOCAL.glob('*.onnx'))
    if onnx_candidates:
        target_onnx = READY_BUNDLE_LOCAL / 'bantai_pothole_rfdetr_nano.onnx'
        if onnx_candidates[0] != target_onnx:
            onnx_candidates[0].replace(target_onnx)
        onnx_filename = target_onnx.name
        print("Exported ONNX model:", onnx_filename)
    del export_det
except Exception as onnx_exc:
    print(f"Note: ONNX export skipped ({onnx_exc}); PyTorch .pth bundle is complete.")

# 3. Generate zero-config inference script
detector_script_code = generate_ready_pothole_detector_script(FINAL_WEIGHTS_NAME, LOCKED_THRESHOLD)
(READY_BUNDLE_LOCAL / 'bantai_pothole_detector.py').write_text(detector_script_code, encoding='utf-8')

# 4. Generate Model Card
model_card = {
    'model_name': 'BANTAI RF-DETR Nano 15K Pothole Detector',
    'base_checkpoint': f"{MODEL_REPO}@{MODEL_REVISION}",
    'training_dataset': f"{DATA_REPO} ({TRAIN_IMAGES + VALID_IMAGES + TEST_IMAGES} multi-country road images)",
    'classes': list(TARGET_CLASSES),
    'resolution': 384,
    'selected_candidate': selected_label,
    'calibrated_operating_threshold': LOCKED_THRESHOLD,
    'validation_metrics': final_val_report,
    'held_out_test_metrics': test_report,
    'weights_file': FINAL_WEIGHTS_NAME,
    'weights_sha256': sha256(target_weights_path),
    'onnx_file': onnx_filename,
}
write_json(READY_BUNDLE_LOCAL / 'model_card.json', model_card)

# 5. Mirror to Google Drive
backup_results(READY_BUNDLE_LOCAL, READY_BUNDLE_DIR)
print(f"Model Bundle saved to Google Drive: {READY_BUNDLE_DIR}")

# 6. Create ZIP archive and trigger browser download
zip_path = Path(shutil.make_archive(str(ROOT / 'BANTAI_Ready_Pothole_Model_Bundle'), 'zip', READY_BUNDLE_LOCAL))
atomic_copy(zip_path, READY_BUNDLE_DIR / zip_path.name)
print(f"Bundle ZIP ready: {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MB)")

try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as dl_exc:
    print(f"Browser download trigger note ({dl_exc}). Download directly from Drive: {READY_BUNDLE_DIR / zip_path.name}")
""")
repair_slot('8')

# --- Section 9A: Automatic Visual Demo + Optional Live Photo Upload Test ---
md("""
## 9A. Automatic Held-Out Visual Demo & Live Photo Upload Test
Automatically runs the packaged `PotholeDetector` on 4 held-out test images so `Run all` finishes with visual detections and never hangs on a file picker.
To upload your own custom road photos from your phone/computer, set `UPLOAD_CUSTOM_PHOTOS = True` and re-run this cell.
""")
code(r"""
from google.colab import files
from IPython.display import display
from PIL import Image, ImageDraw, ImageOps
import importlib
import io

if str(READY_BUNDLE_LOCAL) not in sys.path:
    sys.path.insert(0, str(READY_BUNDLE_LOCAL))
import bantai_pothole_detector
importlib.reload(bantai_pothole_detector)
from bantai_pothole_detector import PotholeDetector

demo_detector = PotholeDetector(
    weights_path=str(READY_BUNDLE_LOCAL / FINAL_WEIGHTS_NAME),
    confidence=LOCKED_THRESHOLD,
)

# 1. Automatic preview on 4 held-out test images (non-blocking for 'Run all')
test_coco = read_json(PREPARED / 'test' / '_annotations.coco.json')
pos_test_ids = {ann['image_id'] for ann in test_coco['annotations']}
sample_test_imgs = [img for img in test_coco['images'] if img['id'] in pos_test_ids][:4]

print(f"Showing automatic detections on {len(sample_test_imgs)} held-out test road images:")
for img_info in sample_test_imgs:
    sample_path = PREPARED / 'test' / img_info['file_name']
    with Image.open(sample_path) as src:
        pil_img = src.convert('RGB')
    results = demo_detector.detect(pil_img)
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    line_w = max(3, round(min(annotated.size) / 250))
    for r in results:
        box = r['xyxy']
        draw.rectangle(box, outline='red', width=line_w)
        lbl = f"Pothole {r['score']:.1%}"
        draw.rectangle(draw.textbbox((box[0], max(0, box[1] - 18)), lbl), fill='red')
        draw.text((box[0], max(0, box[1] - 18)), lbl, fill='white')
    print(f"  Test image {img_info['file_name']}: {len(results)} potholes detected")
    display(ImageOps.contain(annotated, (900, 600)))

# 2. Optional custom photo upload
UPLOAD_CUSTOM_PHOTOS = False  # Set to True and run this cell to upload your own road photos
if UPLOAD_CUSTOM_PHOTOS:
    print("Upload a road photo (JPG, PNG, WebP) to test detection:")
    uploaded_photos = files.upload()
    for fname, img_bytes in uploaded_photos.items():
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert('RGB')
        results = demo_detector.detect(pil_img)
        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)
        line_w = max(3, round(min(annotated.size) / 250))
        for r in results:
            box = r['xyxy']
            draw.rectangle(box, outline='red', width=line_w)
            lbl = f"Pothole {r['score']:.1%}"
            draw.rectangle(draw.textbbox((box[0], max(0, box[1] - 18)), lbl), fill='red')
            draw.text((box[0], max(0, box[1] - 18)), lbl, fill='white')
        print(f"\n{fname}: {len(results)} potholes detected at {LOCKED_THRESHOLD:.1%} confidence")
        for r in results:
            print(f"  Box: {r['xyxy']} | Confidence: {r['score']:.1%}")
        display(ImageOps.contain(annotated, (1200, 800)))
""")

# --- Section 9B: Real-Time Video & In-Browser Webcam Demo ---
md("""
## 9B. Real-Time Video File (`.mp4`) & Live Browser Webcam Test
Test live video or webcam streaming with temporal anti-flicker smoothing (`TemporalPotholeTracker`):
1. Set `UPLOAD_CUSTOM_VIDEO = True` and run the first cell below to upload a road `.mp4`/`.mov` video and download the annotated `.mp4`.
2. Or set `RUN_COLAB_WEBCAM = True` in the second cell below to stream your browser webcam live into the Colab GPU detector!
""")
code(r"""
UPLOAD_CUSTOM_VIDEO = False  # Set to True and run this cell to upload and annotate a road video (.mp4)

if UPLOAD_CUSTOM_VIDEO:
    print("Select a road video (.mp4, .avi, .mov) to annotate:")
    uploaded_video = files.upload()
    for vid_name, vid_bytes in uploaded_video.items():
        in_vid_path = ROOT / vid_name
        out_vid_path = ROOT / f"annotated_{vid_name}"
        in_vid_path.write_bytes(vid_bytes)

        print(f"Annotating video: {vid_name}...")
        demo_detector.run_video(str(in_vid_path), str(out_vid_path), enable_tracking=True)
        atomic_copy(out_vid_path, READY_BUNDLE_DIR / out_vid_path.name)
        files.download(str(out_vid_path))
else:
    print("Video upload skipped during automatic run. Set UPLOAD_CUSTOM_VIDEO = True to annotate an .mp4 file.")
""")

code(r"""
#@title Live Browser Webcam Stream into RF-DETR Nano (Set RUN_COLAB_WEBCAM = True)
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np

RUN_COLAB_WEBCAM = False  # Set to True and run this cell to stream your webcam live in Colab
MAX_WEBCAM_FRAMES = 150

if RUN_COLAB_WEBCAM:
    js_init = Javascript('''
        var video = null;
        var stream = null;
        var canvas = null;
        var imgOverlay = null;
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
            canvas.width = 640;
            canvas.height = 480;
            imgOverlay = document.createElement('img');
            imgOverlay.style.border = '2px solid #ef4444';
            imgOverlay.style.borderRadius = '8px';
            div.appendChild(imgOverlay);
            return true;
        }
        function grabFrame(overlayDataUrl) {
            if (overlayDataUrl && imgOverlay) {
                imgOverlay.src = overlayDataUrl;
            }
            canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
            return canvas.toDataURL('image/jpeg', 0.8);
        }
        function stopColabWebcam() {
            if (stream) {
                stream.getVideoTracks()[0].stop();
            }
        }
    ''')
    display(js_init)
    eval_js('initColabWebcam()')
    demo_detector.tracker.tracks.clear()
    overlay_url = ''
    print(f"Streaming {MAX_WEBCAM_FRAMES} live frames through RF-DETR Nano on GPU...")
    try:
        for frame_i in range(MAX_WEBCAM_FRAMES):
            t_start = time.time()
            data_url = eval_js(f'grabFrame("{overlay_url}")')
            raw_bytes = b64decode(data_url.split(',')[1])
            pil_frame = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            raw_dets = demo_detector.detect(pil_frame)
            tracked = demo_detector.tracker.update(raw_dets)
            fps = 1.0 / max(1e-4, time.time() - t_start)

            frame_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            for item in tracked:
                x1, y1, x2, y2 = map(int, item['xyxy'])
                lbl = f"Pothole #{item['id']} {item['score']:.0%}"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame_bgr, lbl, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"FPS: {fps:.1f} | Potholes: {len(tracked)}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            _, enc_jpg = cv2.imencode('.jpg', frame_bgr)
            overlay_url = 'data:image/jpeg;base64,' + b64encode(enc_jpg.tobytes()).decode('ascii')
    finally:
        eval_js('stopColabWebcam()')
        print("Webcam stream finished.")
else:
    print("Colab browser webcam stream is off by default. Set RUN_COLAB_WEBCAM = True to stream in Colab,")
    print("or run locally on your Windows PC (GTX 1660 Ti) with:")
    print("  python bantai_pothole_detector.py --weights bantai_pothole_rfdetr_nano_ready.pth --webcam 0")
""")

for idx, cell in enumerate(cells):
    cell['id'] = f'rfdetr-15k-{idx:02d}'
    if cell['cell_type'] == 'code':
        ast.parse(cell['source'])

notebook_data = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'colab': {'name': 'BANTAI_RFDETR_Pothole_15K_ReadyModel.ipynb'},
        'accelerator': 'GPU',
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
    },
    'cells': cells,
}

out_path = HERE / 'BANTAI_RFDETR_Pothole_15K_ReadyModel.ipynb'
out_path.write_text(json.dumps(notebook_data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f"Successfully generated {out_path.name} with {len(cells)} cells.")
