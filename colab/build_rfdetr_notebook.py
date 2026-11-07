"""Rebuild the standalone Colab notebook. Does not run model code."""
import ast
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': textwrap.dedent(text).strip() + '\n'})


def repair_slot(section):
    markdown(f'''### {section} troubleshooting slot
    If this section fails, insert a code cell **here**, immediately below its main cell.
    Keep the fix limited to this section, then rerun its main cell and dependent sections.
    Record the error and fix in a short Markdown cell. Do not change test selection,
    class mapping or settings to improve test scores. After a fix works, fold it into
    the main cell and clear the temporary cell. Send the section number and error to our chat.
    ''')


markdown('''
# BANTAI — RF-DETR Nano baseline and fine-tuning
**Start with a pothole-trained checkpoint, evaluate it, optionally adapt it, then compare.**

Model: [dronefreak/rdd2022-rfdetr-nano](https://huggingface.co/dronefreak/rdd2022-rfdetr-nano).
Dataset: [publisher's RDD2022 split](https://huggingface.co/datasets/dronefreak/RDD2022),
original [RDD2022 paper](https://doi.org/10.1002/gdj3.260).
Training API: [RF-DETR](https://rfdetr.roboflow.com/learn/train/).

The four model classes are longitudinal crack, transverse crack, alligator crack and
pothole. Potholes are the primary MVP result; we retain all four classes during adaptation.
No application integration is performed here.

## How to run
1. Upload this notebook to Colab. Select a T4 GPU if available.
2. Run sections 1–5 for preparation and the original model's **validation baseline**.
3. In section 6 set `RUN_FINE_TUNING=True` to start the short fine-tuning run.
4. Section 7 compares validation results and chooses a candidate.
5. Once development is finished, enable section 8 for paired final test evaluation.
6. Section 9 exports a results ZIP to send back in our chat.

Data and model caches use `/content` (temporary SSD). Reports and training checkpoints
use Drive. Free GPU access and runtime duration are not guaranteed; training can resume
from a completed epoch, but a partially completed epoch must repeat.

**Evaluation honesty:** preserve the publisher's train/valid/test assignments. Its test
split is a custom held-out portion of publicly labeled RDD images, not the unlabeled
official challenge test. The publisher's split correspondence is documented but not
independently proven here. Near-duplicate/route independence is not guaranteed. Exact
duplicates within the downloaded sample are rejected. Local Philippine camera testing
is still required. Fine-tuning on a small subset of data previously seen by the starting
model is an adaptation experiment; it may not improve performance.

Each numbered section has a troubleshooting slot for a temporary repair cell.
Keep one settings cell. Once a fix works, edit the original section instead of leaving
multiple competing versions. Older notebooks remain separate.
''')
markdown('''## 1. Configuration and Drive
Defaults download **1200 training + 200 validation + 200 test images**, across
India, Japan, and the Czech Republic. Each split is half pothole-positive and half without
an annotated pothole (the latter can contain cracks). This is an enriched diagnostic sample,
not traffic prevalence. All available road-damage annotations in selected images are retained.

Changing settings requires a new RUN_NAME. The final test remains unused for model
selection. Training is deliberately opt-in in section 6.
''')
code(r'''
from pathlib import Path
import json, os, sys, subprocess, gc, shutil
RUN_NAME = 'rfdetr_rdd_multicountry_002'
COUNTRY = ('India', 'Japan', 'Czech')
SEED = 42
TRAIN_IMAGES, VALID_IMAGES, TEST_IMAGES = 1200, 200, 200
EPOCHS = 15
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-5
CONFIDENCE = 0.12
assert RUN_NAME.replace('_', '').isalnum(), 'Use letters, numbers and underscores.'
assert all(n >= 10 for n in (TRAIN_IMAGES, VALID_IMAGES, TEST_IMAGES))
assert 0 < CONFIDENCE < 1 and EPOCHS > 0 and BATCH_SIZE > 0 and GRAD_ACCUM_STEPS > 0
from google.colab import drive
drive.mount('/content/drive')
ROOT = Path('/content/bantai_rfdetr') / RUN_NAME
OUT = ROOT / 'results'
BACKUP = Path('/content/drive/MyDrive/BANTAI/rfdetr') / RUN_NAME
TRAIN_DIR = BACKUP / 'training'
for folder in (ROOT, OUT, BACKUP, TRAIN_DIR):
    folder.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(ROOT / 'hf_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
print('Temporary work:', ROOT)
print('Persistent results:', BACKUP)
''')
repair_slot('1')
markdown('''## 2. Dependencies and workflow helpers
This installs exact top-level versions **inside Colab only**. Running this section is
the installation step; no setup scripts from the model repository are executed.
Colab supplies PyTorch; transitive dependencies are resolved by pip and the full
environment is recorded. If imports request a restart, restart the session and rerun
from section 1. Initial installation needs internet even when inference later runs locally.
''')
code(r'''
PACKAGES = ['rfdetr[train]==1.11.2', 'huggingface-hub==2.3.0', 'pycocotools==2.0.11']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *PACKAGES], check=True)
import torch
from rfdetr import RFDETR, RFDETRNano
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU (evaluation only)')
(OUT / 'environment.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
''')
code((HERE / 'rfdetr_workflow.py').read_text(encoding='utf-8'))
code(r'''
config = dict(version='rfdetr-workflow-v1', packages=PACKAGES, country=COUNTRY, seed=SEED,
              counts={'train': TRAIN_IMAGES, 'valid': VALID_IMAGES, 'test': TEST_IMAGES},
              epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
              lr=LEARNING_RATE, lr_encoder=1e-6, confidence=CONFIDENCE,
              model_repo=MODEL_REPO, model_revision=MODEL_REVISION,
              data_repo=DATA_REPO, data_revision=DATA_REVISION)
guard_json(BACKUP / 'config.json', config)
write_json(OUT / 'config.json', config)
backup_results(OUT, BACKUP / 'reports')
''')
repair_slot('2')
markdown('''## 3. Download and verify the starting checkpoint
Pin both model and dataset revisions so future updates do not silently change this run.
Weights are approximately 121 MB. Load through RF-DETR's checkpoint-aware loader,
which restores saved architecture settings. Restricted checkpoint loading stays enabled;
if it refuses an unfamiliar checkpoint, stop and inspect rather than enabling unsafe loading.
''')
code(r'''
WEIGHTS = Path(hf_hub_download(MODEL_REPO, filename='checkpoint_best_total.pth',
                             revision=MODEL_REVISION, local_dir=ROOT / 'model'))
model_config_path = hf_hub_download(MODEL_REPO, filename='config.json',
                                   revision=MODEL_REVISION, local_dir=ROOT / 'model')
publisher_config = read_json(model_config_path)
assert publisher_config['class_names'] == CLASSES, 'Publisher class mapping changed.'
model_info = dict(repo=MODEL_REPO, revision=MODEL_REVISION, sha256=sha256(WEIGHTS),
                  class_names=CLASSES, resolution=publisher_config['model_config']['resolution'])
assert model_info['resolution'] == 384, 'Review unexpected Nano resolution before continuing.'
guard_json(BACKUP / 'model.json', model_info)
write_json(OUT / 'model.json', model_info)

def load_detector(path):
    detector = RFDETR.from_checkpoint(str(path), device=DEVICE)
    assert isinstance(detector, RFDETRNano), 'Checkpoint is not RF-DETR Nano.'
    assert detector.model_config.num_classes == 4, 'Unexpected class count.'
    names = detector.model.class_names
    assert names == CLASSES, f'Unexpected checkpoint class names: {names}'
    return detector

detector = load_detector(WEIGHTS)
print('Loaded RF-DETR Nano:', model_info)
backup_results(OUT, BACKUP / 'reports')
''')
repair_slot('3')
markdown('''## 4. Download only the chosen images and prepare COCO labels
Metadata is small; only selected images are downloaded, using six concurrent requests.
All image work uses Colab's SSD. Upstream split membership is preserved; test data is
prepared but not used for training or parameter selection. Image dimensions, boxes,
class IDs and exact duplicates are checked. Source metadata and hashes are exported.
Images with EXIF rotation/mirroring are quarantined with an exclusion record rather
than altering pixels or guessing how to transform annotation boxes. Usable counts
may therefore be lower than requested; review the counts and exclusions.
Related scenes crossing the publisher's splits remain a limitation requiring further review.
''')
code(r'''
PREPARED, manifest = prepare_dataset(ROOT, BACKUP, config['counts'], SEED, COUNTRY)
backup_results(OUT, BACKUP / 'reports')
# Smoke test on one TRAIN image; no test image is displayed.
detector = load_detector(WEIGHTS)
first = read_json(PREPARED / 'train/_annotations.coco.json')['images'][0]
smoke = detector.predict(str(PREPARED / 'train' / first['file_name']), threshold=CONFIDENCE)
print('Inference smoke test completed; predicted boxes:', len(smoke))
print('Dataset ready:', PREPARED)
''')
repair_slot('4')
markdown('''## 5. Original model: validation baseline
This checks the starting model without training. mAP uses a low prediction floor;
precision, recall, F1 and false alarms use the fixed operating confidence in section 1.
Per-image results save immediately to Drive and resume after interruptions.
Green example boxes are pothole labels; red boxes are pothole predictions.
''')
code(r'''
baseline_valid = evaluate(detector, WEIGHTS, PREPARED, 'valid', OUT, BACKUP / 'reports',
                          manifest['id'], CONFIDENCE, 'baseline_validation')
print(json.dumps({'pothole_AP': baseline_valid['per_class_ap']['pothole'],
                  'pothole_fixed': baseline_valid['fixed_confidence']['pothole'],
                  'mean_predict_ms': baseline_valid['mean_predict_ms']}, indent=2))
''')
repair_slot('5')
markdown('''## 6. Optional fine-tuning — run after reviewing the baseline
Set RUN_FINE_TUNING=True below to start. GPU is required by this notebook for training.
The default is 15 epochs, early stopping after 4 stagnant validation epochs, batch 2,
gradient accumulation 8, multi-scale zoom training, and EMA weight smoothing.
Improvements are not guaranteed.

Images remain on SSD. RF-DETR writes each epoch's recovery checkpoint directly to
Drive (`training/last.ckpt`), with best weights promoted at successful completion.
Drive writes add overhead but avoid a custom background backup process. Recovery
replays an incomplete epoch. Keep sufficient Drive space for optimizer checkpoints.

Only train and valid data are provided to the training call: a separate training view
contains those two folders, with no test folder. Test evaluation is disabled in training.
Once final testing starts, this run cannot be trained again. Use a new study run for
further development and independently reserve new test data before interpreting it as a fresh test.
''')
code(r'''
RUN_FINE_TUNING = False  # Change to True when ready to spend GPU time.
fine_tuned_path = TRAIN_DIR / 'checkpoint_best_total.pth'
if RUN_FINE_TUNING:
    assert DEVICE == 'cuda', 'GPU unavailable. Baseline evaluation works on CPU; defer training.'
    assert not (BACKUP / 'final_test_started.json').exists() or (TRAIN_DIR / 'completed.json').exists(), 'Final test already opened; training is locked.'
    assert (OUT / 'baseline_validation/summary.json').exists(), 'Run section 5 first.'
    training_identity = dict(config=config, manifest=manifest['id'], model=model_info['sha256'])
    guard_json(TRAIN_DIR / 'run_identity.json', training_identity)
    if (TRAIN_DIR / 'completed.json').exists():
        assert fine_tuned_path.exists(), 'Completed run is missing its best weights.'
        print('Fine-tuning already completed; reusing saved weights.')
    else:
        # Give the trainer train/valid only; it must never see our test images.
        training_view = ROOT / 'training_view'
        training_view.mkdir(exist_ok=True)
        for split in ('train', 'valid'):
            link = training_view / split
            if not link.exists():
                link.symlink_to(PREPARED / split, target_is_directory=True)
        assert not (training_view / 'test').exists()
        if 'detector' in globals():
            del detector
        gc.collect()
        torch.cuda.empty_cache()
        trainer_model = load_detector(WEIGHTS)
        resume = TRAIN_DIR / 'last.ckpt'
        print('Resuming from completed epoch' if resume.exists() else 'Starting from downloaded road-damage weights')
        trainer_model.train(dataset_dir=str(training_view), output_dir=str(TRAIN_DIR),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
                            lr=LEARNING_RATE, lr_encoder=1e-6, resolution=384, device=DEVICE,
                            num_workers=0, seed=SEED, class_names=CLASSES,
                            early_stopping=True, early_stopping_patience=4,
                            early_stopping_min_delta=0.001, early_stopping_use_ema=True,
                            use_ema=True, checkpoint_interval=5,
                            run_test=False, tensorboard=False, wandb=False,
                            multi_scale=True, expanded_scales=False,
                            resume=str(resume) if resume.exists() else None)
        assert fine_tuned_path.exists(), 'Best checkpoint missing; inspect the training output before continuing.'
        write_json(TRAIN_DIR / 'completed.json', {'sha256': sha256(fine_tuned_path)})
        del trainer_model
        gc.collect()
        torch.cuda.empty_cache()
else:
    print('Fine-tuning skipped. Set RUN_FINE_TUNING=True in this cell to enable it.')
''')
repair_slot('6')
markdown('''## 7. Validation comparison and candidate selection
Compare both models on validation data. The candidate is selected by pothole
mAP@50:95, breaking ties in favor of the original model. This does not claim that
overall crack performance improved; those scores are reported separately.
''')
code(r'''
completed = TRAIN_DIR / 'completed.json'
candidate = 'baseline'
comparison = {'baseline': baseline_valid}
if completed.exists():
    assert read_json(completed)['sha256'] == sha256(fine_tuned_path), 'Fine-tuned weights changed.'
    if 'detector' in globals():
        del detector
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    detector = load_detector(fine_tuned_path)
    tuned_valid = evaluate(detector, fine_tuned_path, PREPARED, 'valid', OUT, BACKUP / 'reports',
                           manifest['id'], CONFIDENCE, 'finetuned_validation')
    comparison['fine_tuned'] = tuned_valid
    original_ap = baseline_valid['per_class_ap']['pothole']['map50_95']
    tuned_ap = tuned_valid['per_class_ap']['pothole']['map50_95']
    assert original_ap is not None and tuned_ap is not None, 'No evaluable pothole labels.'
    if tuned_ap > original_ap:
        candidate = 'fine_tuned'
comparison['selected_by_validation'] = candidate
write_json(OUT / 'validation_comparison.json', comparison)
backup_results(OUT, BACKUP / 'reports')
for name in ('baseline', 'fine_tuned'):
    if name in comparison:
        print(name, 'pothole mAP50:95:', comparison[name]['per_class_ap']['pothole']['map50_95'])
print('Validation-selected candidate:', candidate)
''')
repair_slot('7')
markdown('''## 8. Final paired test — enable only when development is finished
Set RUN_FINAL_TEST=True to evaluate both original and fine-tuned weights on the
same reserved test sample. This freezes the candidate selected in section 7. Do not
change training settings or select confidence thresholds using test results.
Rerunning after interruption resumes saved predictions. Scores measure this sample,
not the official challenge leaderboard or local barangay deployment accuracy.
''')
code(r'''
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    frozen = {'manifest': manifest['id'], 'config': identity(config),
              'baseline_sha256': model_info['sha256'], 'candidate': candidate,
              'fine_tuned_sha256': sha256(fine_tuned_path) if completed.exists() else None}
    guard_json(BACKUP / 'final_test_started.json', frozen)
    final = {'selected_before_test': candidate, 'results': {}}
    pairs = [('baseline', WEIGHTS)]
    if completed.exists():
        pairs.append(('fine_tuned', fine_tuned_path))
    for name, weights in pairs:
        if 'detector' in globals():
            del detector
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        detector = load_detector(weights)
        final['results'][name] = evaluate(detector, weights, PREPARED, 'test', OUT, BACKUP / 'reports',
                                          manifest['id'], CONFIDENCE, name + '_test')
    write_json(OUT / 'final_comparison.json', final)
    backup_results(OUT, BACKUP / 'reports')
    for name, report in final['results'].items():
        print(name, 'POTHOLE:', report['per_class_ap']['pothole'], report['fixed_confidence']['pothole'])
else:
    print('Test remains reserved. Enable RUN_FINAL_TEST only after finishing development.')
''')
repair_slot('8')
markdown('''## 8A. Upload a road photo — demonstration
Run after section 7, before section 9. Section 8 may remain disabled. This cell
explicitly reloads the candidate selected in section 7 and verifies its weight hash.
It does not assume the detector left by any earlier cell is the chosen model.
Upload a downloaded JPG, PNG or WebP road image using the file picker. Boxes show all
four road-damage classes with confidence scores. This is an exploratory demonstration;
it does not modify test metrics or train the model. Outputs are included in section 9's ZIP.
''')
code((HERE / 'rfdetr_upload_demo.py').read_text(encoding='utf-8'))
repair_slot('8A')
markdown('''### 8A diagnostic: investigate a missed detection (optional)
Run only to investigate uploaded demonstration images. Reuses the original uploaded
photo and explicitly compares both evaluated checkpoints. Prints the strongest score
for each class and shows up to five boxes at display thresholds 0.25, 0.10 and 0.05.
Low-score boxes are uncertain hypotheses. This cell does not change CONFIDENCE,
select a candidate, retrain weights, or alter final-test reports. Outputs enter the ZIP.
''')
code((HERE / 'rfdetr_diagnostic_demo.py').read_text(encoding='utf-8'))
markdown('''## 9. Export results for review
The ZIP includes completed reports, per-image predictions, manifests, environment
versions and representative pothole mistakes. It excludes the raw dataset, model
weights and large optimizer checkpoints. Training checkpoints remain in your Drive.
You can export after baseline evaluation, after training, or after final testing; the
status file explicitly records which phases finished. Send the ZIP back in our chat.

If Colab disconnects: open the **same notebook**, keep RUN_NAME unchanged, and rerun
sections 1–5. Data is rebuilt, completed predictions are restored, and section 6 resumes
from `last.ckpt` when enabled. Do not create duplicate sections for recovery.
''')
code(r'''
status = {'baseline_validation_completed': (OUT / 'baseline_validation/summary.json').exists(),
          'fine_tuning_completed': (TRAIN_DIR / 'completed.json').exists(),
          'final_test_started': (BACKUP / 'final_test_started.json').exists(),
          'final_test_completed': (OUT / 'final_comparison.json').exists(),
          'limitations': ['Publisher split provenance not independently proven.',
                          'Enriched sample; no natural-prevalence claim.',
                          'Near-duplicate/route independence not guaranteed.',
                          'Local camera performance remains untested.']}
write_json(OUT / 'status.json', status)
# Include lightweight training records, never optimizer state or large weights.
for filename in ('training_config.json', 'completed.json', 'run_identity.json'):
    if (TRAIN_DIR / filename).exists():
        atomic_copy(TRAIN_DIR / filename, OUT / 'training' / filename)
backup_results(OUT, BACKUP / 'reports')
archive = Path(shutil.make_archive(str(ROOT / (RUN_NAME + '_results')), 'zip', OUT))
atomic_copy(archive, BACKUP / archive.name)
print('Saved to Drive:', BACKUP / archive.name)
print(json.dumps(status, indent=2))
from google.colab import files
files.download(str(archive))
''')
repair_slot('9')

for i, cell in enumerate(cells):
    cell['id'] = f'rfdetr-{i:02d}'
    if cell['cell_type'] == 'code':
        ast.parse(cell['source'])
notebook = {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'colab': {'name': 'BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb'},
                         'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}}, 'cells': cells}
path = HERE / 'BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb'
path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Built', path.name, '|', len(cells), 'cells; Python syntax checked.')
