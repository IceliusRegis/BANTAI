# BANTAI: Pothole Detection Models & Workflows

## 1. Flagship Ready-to-Use 15K Model: `BANTAI_RFDETR_Pothole_15K_ReadyModel.ipynb`

Open `BANTAI_RFDETR_Pothole_15K_ReadyModel.ipynb` using **Google Colab → File → Upload notebook**.

- **Pretrained Architecture**: `dronefreak/rdd2022-rfdetr-nano` (RF-DETR Nano from our paper).
- **Scale**: **15,000 total road images** (`12,000 train + 1,500 valid + 1,500 test`) across all 6 countries in RDD2022.
- **Single-Class Head Surgery**: Automatically converts the 4-class checkpoint into a native 1-class (`0: pothole`) checkpoint, protecting and reusing all learned pothole weights.
- **Hardware Optimization**: Auto-scales batch size and parallel data loader workers for Colab **A100 (batch 32)**, **L4 (batch 16)**, or **T4 (batch 8)**. Full run finishes in ~35–45 minutes.
- **Real-Time Video & Webcam**: Runs at **~100–180 FPS** on Colab GPUs and **~60–90 FPS** locally on a GTX 1660 Ti, featuring temporal IoU box smoothing and anti-flicker filtering.
- **Ready Deliverables**: Section 8 packages `bantai_pothole_rfdetr_nano_ready.pth`, embedded `model_card.json`, and the standalone `bantai_pothole_detector.py` inference engine directly to Google Drive (`MyDrive/BANTAI/pothole_ready_model/`) and browser ZIP download.
- **Testing**: Section 9A tests photo uploads; Section 9B tests uploaded `.mp4` road videos and live in-browser webcam streams.

### Local Maintenance & Testing
```powershell
python colab/build_rfdetr_pothole_15k_notebook.py
python -m unittest discover -s colab -p "test_rfdetr_pothole_15k_workflow.py" -v
```

---

## 2. Post-Training Model Enhancer: `BANTAI_Pothole_Model_Enhancer.ipynb`

Open `BANTAI_Pothole_Model_Enhancer.ipynb` using **Google Colab → File → Upload notebook**.

- **Automatic Ingestion**: Automatically discovers and loads `bantai_pothole_rfdetr_nano_ready.pth` directly from Google Drive (`MyDrive/BANTAI/pothole_ready_model/` or `MyDrive/BANTAI/pothole_15k/`).
- **Real-Time Video / Live Webcam Mode (30–60+ FPS)**: Ultra-fast pipeline with **LAB-CLAHE shadow balancing**, **horizon ROI filtering**, **fast WBF box fusion**, and **temporal multi-frame tracking**. Total added overhead is only **~3.2 ms** on GPU.
- **High-Accuracy Photo Mode**: **Road-focused SAHI slicing (2x2 overlapping road crops at native resolution)** + **Horizontal-flip TTA** to recover small/distant potholes ($>15\text{ m}$ ahead) that standard downscaling compresses away.
- **Shadow & Manhole Rejection**: Laplacian edge-roughness discriminator flags and eliminates flat tree shadows and smooth utility covers.
- **Severity Indexing**: Tags each detected pothole as `Low`, `Moderate`, or `Critical` for direct consumption by the BANTAI municipal hotspot dashboard.
- **Automated A/B Benchmark**: Renders side-by-side visual comparisons (`Raw Model` vs. `Enhanced Model`) and packages `BANTAI_Enhanced_Pothole_Model_Bundle.zip` to Drive (`MyDrive/BANTAI/pothole_enhanced_model/`).

### Local Maintenance & Testing
```powershell
python colab/build_pothole_enhancer_notebook.py
python -m unittest discover -s colab -p "test_pothole_enhancer_workflow.py" -v
```

---

## 3. Baseline & Diagnostic Notebooks
- `BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb`: Initial 1,200-image diagnostic baseline.
- `BANTAI_Pretrained_Pothole_Evaluation.ipynb`: YOLOv8s pretrained evaluation on RDD2022 India.
- `BANTAI_Pothole_First_Attempt.ipynb`: YOLO11n country-by-country workflow.

## Storage and recovery

- Archives, original extracted files, prepared images and training run live under
  `/content/pothole_attempt_001`; source archives/images/annotations are not edited.
- Drive destination: `MyDrive/BANTAI/pothole/pothole_attempt_001`. Raw RDD images are
  not copied to Drive. Allow space for model checkpoints and the delivery ZIP.
- Each completed epoch backs up `last.pt`, `best.pt`, settings and training CSV using
  temporary files followed by rename. Copies can still fail if Drive is unavailable;
  errors stop the run rather than claiming a successful backup.
- After runtime loss, rerun Setup and Phases 1–2 with the **same** attempt name and
  review choices to reconstruct the same local paths, then set `RESUME=True`.
- An incomplete archive restarts its download. A completed training attempt is not
  retrained. A completed test call returns its saved report. Interrupted test attempts
  remain recorded and block automatic re-evaluation; inspect the record first.
- Use a new attempt name when deliberately changing data or training choices.

## Data safeguards and limits

Missing/ambiguous XML, unreadable images, rotation mismatches, invalid pothole boxes
and exact duplicates with conflicting labels are excluded with reasons. Other
damage classes are not relabeled as potholes. Images with valid XML and no `D40`
are candidate backgrounds; annotations can still miss real defects. Tiny/difficult
boxes are flagged, retained by default, and can be excluded after review.

Near-duplicate grouping uses 64-bit dHash at Hamming radius 4 plus country-specific
100-number filename blocks and optional manual capture/route groups. It does **not**
prove route independence. Review examples and their entire groups are assigned to
training, so examples already seen by the reviewer do not enter the held-out test.
Ratios are approximate and counts are printed; preparation fails if any split has
no positive examples. Large groups can prevent a valid split and need investigation,
not arbitrary splitting. Background images and camera viewpoints are not automatically
rebalanced; missing variety requires additional representative labeled data.

Only training and validation paths appear in the training configuration. Training
verification does not read test image bytes. Preparation necessarily audits and
groups all candidate images before the split exists. Source XML pixel coordinates
are preserved as given and normalized for YOLO, without silent clipping.

Colab's GPU and package stack can change. The notebook pins `ultralytics==8.3.221`
and records the full resolved environment but does not replace Colab's preinstalled
Torch/CUDA stack or claim a fully locked cloud image. Pretrained weights are obtained
through Ultralytics; their actual SHA-256 is recorded. Archive hashes are locally
computed provenance, not publisher-signed integrity verification. The Figshare
fallback checks the selected country ZIP's CRC against its master-archive entry;
it records the publisher's master MD5 but does not claim to verify the entire file.

Results are an **internal RDD holdout**, not official challenge results. Actual phone
and baby-monitor-camera performance must be tested separately. Detection confidence
does not establish pothole severity, depth, safety or measured accuracy.

## Maintenance and local verification

`pothole_workflow.py` is the canonical implementation. The notebook embeds it so
the user needs to upload only one file. After editing the source, regenerate it:

```powershell
python colab/pothole_workflow.py --build-notebook
python -m unittest discover -s colab -p test_workflow.py -v
```

Tests use generated image/XML fixtures, temporary directories and existing Pillow.
They do not install dependencies, download RDD, load a model or claim GPU validation.

## Sources

- [RDD2022 official country downloads and labels](https://github.com/sekilab/RoadDamageDetector)
- [RDD2022 paper](https://doi.org/10.1002/gdj3.260) — cite Arya et al. in project work.
- [Ultralytics 8.3.221 release](https://pypi.org/project/ultralytics/8.3.221/)
- [Pinned training implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/engine/trainer.py)
- [Colab storage/runtime FAQ](https://research.google.com/colaboratory/faq.html)

Retain the source dataset's terms and citations. Ultralytics is distributed under
AGPL-3.0, with alternative commercial licensing; preserve its applicable obligations.
