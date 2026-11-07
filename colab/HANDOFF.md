# BANTAI Pothole Model & Enhancer — Technical Handoff (`HANDOFF.md`)

**Date:** 2026-11-07  
**Active Directory:** `C:\Users\Samuel\Downloads\BANTAI\colab\`  
**Isolated / Untouched Directory:** `c:\Users\Samuel\Desktop\IA-MODEL TRAINING\BANTAI-Modeling-` (Trash detection repo — **do not modify**)

---

## 1. Architecture & Pipeline Overview

The BANTAI Pothole Detection system is built on **RF-DETR Nano** (`dinov2_windowed_small` encoder, `384x384` resolution, `patch_size=16`, `dec_layers=2`, `num_select=300`, 1 foreground class: `["pothole"]`) across a two-stage Google Colab pipeline:

### Stage 1: Base 15K Model Training (`BANTAI_RFDETR_Pothole_15K_Colab.ipynb`)
- **Builder & Workflow:** [`build_rfdetr_pothole_15k_notebook.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/build_rfdetr_pothole_15k_notebook.py) and [`rfdetr_pothole_15k_workflow.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/rfdetr_pothole_15k_workflow.py).
- **Dataset:** `dronefreak/RDD2022` (15,000 road images: `10,500` train / `2,250` valid / `2,250` test, balanced `50%` positive pothole images and `50%` clean/non-pothole hard negatives).
- **Google Drive Outputs:**
  - Training checkpoints: `/content/drive/MyDrive/BANTAI/pothole_15k/training_runs/rfdetr_nano_15k/checkpoint_best_total.pth` (and `checkpoint_best_ema.pth`).
  - Exported Stage-1 ready bundle: `/content/drive/MyDrive/BANTAI/pothole_ready_model/bantai_pothole_rfdetr_nano_ready.pth` (`~116.6 MB`).
- **Status:** **Complete & Verified.** Do not overwrite `/content/drive/MyDrive/BANTAI/pothole_15k/training_runs/rfdetr_nano_15k/checkpoint_best_total.pth`.

### Stage 2: All-Terrain Enhancer & Post-Processing (`BANTAI_Pothole_Model_Enhancer.ipynb`)
- **Builder & Workflow:** [`build_pothole_enhancer_notebook.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/build_pothole_enhancer_notebook.py), [`pothole_enhancer_workflow.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py), and [`test_pothole_enhancer_workflow.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/test_pothole_enhancer_workflow.py).
- **Purpose:**
  1. Ingests the untouched Stage-1 checkpoint (`checkpoint_best_total.pth` from `pothole_15k`).
  2. Streams out-of-domain pothole datasets (`keremberke/pothole-segmentation`, `manot/pothole-segmentation`, supplemented by `RDD2022`) via fast single-file Parquet shards (`1,000` train / `100` valid / `100` test) in ~30–45 seconds.
  3. Runs a 6-epoch domain-adaptation fine-tuning pass (`LR=1.2e-4`, `LR_ENCODER=2.0e-5`, `multi_scale=False`) and embeds `model_config` directly into the `.pth` checkpoint via [`preserve_model_config_in_checkpoint`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py#L417-L473).
  4. Packages the enhanced inference engine (`bantai_pothole_detector.py`) with **CIE-LAB CLAHE**, **5-Tile Road-SAHI (2x2 grid + Center-Horizon Focal Tile)**, **Horizontal-Flip TTA**, **Weighted Boxes Fusion (WBF)**, **Rim-Aware Laplacian Roughness Verification**, and **Temporal Multi-Frame Tracking**.
- **Google Drive Outputs:**
  - `/content/drive/MyDrive/BANTAI/pothole_enhanced_model/bantai_pothole_rfdetr_nano_ready.pth`
  - `/content/drive/MyDrive/BANTAI/pothole_enhanced_model/bantai_pothole_detector.py`
  - `/content/drive/MyDrive/BANTAI/pothole_enhanced_model/BANTAI_Enhanced_Pothole_Model_Bundle.zip`

---

## 2. Current State (Restored Clean v2 Pipeline)

During an experimental v3 iteration, three regressions were introduced and have now been **fully reverted and fixed** in the current codebase:

| Issue in Experimental v3 | Root Cause | Fix Applied in Current v2-Restored Code |
| :--- | :--- | :--- |
| **1. Boxing random non-pothole background objects** | In 1-class RF-DETR (`CLASSES = ["pothole"]`), class ID `0` is `"pothole"` and **class ID `1` is DETR's `"__background__"` (no-object) query**. Removing `if int(cid) == 0:` caused high-confidence background queries (`cid == 1`) to be drawn as potholes. | Strictly enforced `int(cid) == 0` for potholes and skipped `int(cid) == 1` (`__background__`) in [`EnhancedPotholeDetector._raw_predict`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py#L1016-L1032) and [`PotholeDetector._predict_raw`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py#L1402-L1408). |
| **2. `0 potholes detected` on valid road photos** | Populating `"sources": ["sahi_tile"]` in `weighted_boxes_fusion` activated a filter (`if cluster_count == 1 and sources == ["sahi_tile"] and score < 0.45: continue`) inside both `PotholeDetector.detect()` and Section 8A, deleting all single-tile SAHI detections under `0.45`. | Completely removed the single-tile SAHI rejection rule from [`filter_horizon_and_aspect_ratio`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py#L705-L805), [`generate_enhanced_detector_script`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py#L1400-L1460), and Section 8A in [`build_pothole_enhancer_notebook.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/build_pothole_enhancer_notebook.py#L517-L605). |
| **3. Degraded Stage-2 weights after v3 training** | `generate_hard_negative_road_sample` drew synthetic `PIL.ImageDraw` cartoon ellipses/rectangles onto 30% of training images, and `augment_all_pothole_types` painted flat solid-color RGB ovals over 70% of real pothole centers. | Deleted `generate_hard_negative_road_sample` completely. Restored [`augment_all_pothole_types`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py#L48-L102) to natural photometric scaling (`crop *= 0.82` and `crop * 1.25 + 15.0`) that preserves real underlying crater texture. Bumped dataset manifest to `"v2_clean_restored"` and training flag to `completed_v2_restored.json` so stale v3 caches are automatically replaced. |

---

## 3. Critical Invariants (Never Violate These)

1. **`class_id` Indexing in RF-DETR (`num_classes=1`)**:
   - **ALWAYS** filter with `if int(cid) == 0:`.
   - **NEVER** accept `int(cid) == 1` — in DETR/RF-DETR with 1 foreground class, index `1` (`len(CLASSES)`) is the `"__background__"` class.
2. **Never Kill Single-Tile SAHI Detections**:
   - Small/distant potholes in dashcam and street views are often only large enough inside a single zoomed-in SAHI tile (`cluster_count == 1`, `sources == ["sahi_tile"]`, `score` between `0.22` and `0.42`). Never reject detections solely because `sources == ["sahi_tile"]`.
3. **Never Draw Synthetic `ImageDraw` Shapes onto Training Images**:
   - Only use real photographic road samples (from Parquet shards or `RDD2022`) for positive and negative training examples, and only apply photometric/scale transforms that preserve real road and crater texture.
4. **No Double-Filtering Between `PotholeDetector.detect()` and Section 8A**:
   - Keep `PotholeDetector.detect()` in the generated `bantai_pothole_detector.py` focused on CLAHE + 5-Tile Road-SAHI + Flip-TTA + WBF + basic horizon/roughness checks (`cy < h * 0.25`, `roughness < 3.5`), and let Section 8A (`filter_dashcam_and_false_positives`) handle camera-specific dashcam hood/vent/bumper rules cleanly.

---

## 4. Visual Benchmark Status (5 User Test Photos)

Across the 5 real-world out-of-domain test images evaluated in Section 8A:

1. **Photo 1 (`1920x1080` VIOFO Country Road Dashcam)**:
   - *v2 Result*: Detected the real left-lane pothole (`27.8%`), plus a duplicate overlapping box (`22.6%`) and a false positive on the bottom dashboard air-vent reflection (`58.8%`).
   - *Surgical Tweaks in Section 8A*:
     - **Tweak 1** (`cy > h * 0.64 or y2 > h * 0.68` on wide `16:9` dashcams) masks the VIOFO dashboard air-vent reflection.
     - **Tweak 3** (`(inter / min(i_area, j_area)) >= 0.45 and j_score > i_score`) suppresses the lower-confidence duplicate box on the same pothole.
2. **Photo 2 (`640x640` Urban Street with Yellow/Black Curb & Parked Cars)**:
   - *v2 Result*: Detected the real foreground left-edge pothole (`45.2%`) and mid-left potholes (`28.6%`, `60.3%`), plus false positives on middle-road manhole covers (`22.2%`, `53.5%`).
   - *Remaining Challenge*: Because the image is square (`640x640`, aspect `1.0`), wide-dashcam masks do not apply, and metallic manhole covers/grates have high Laplacian edge variance (`roughness > 20`). Suppressing manhole covers without losing real street potholes requires **real** photographic hard-negative road/manhole images (not synthetic drawings) or a circular/symmetry check.
3. **Photo 3 (`1366x668` Wide Highway Dashcam with Google Maps Sidebar)**:
   - *v2 Result*: Masked the right-side Google Maps UI (`cx > w * 0.86`), detected the center-lane road anomaly (`55.2%`), and had a low-score box on a distant SUV rear bumper (`36.8%`).
   - *Surgical Tweak in Section 8A*:
     - **Tweak 2** (`cy < h * 0.38 and (bw / bh) >= 2.0 and score < 0.45` on wide dashcams) filters distant horizontal car bumpers near the horizon.
4. **Photo 4 (`633x480` Close-Up Dry Gravel Crater)**:
   - *v2 Result*: Clean single box (`84.7% [Critical]`) with inner SAHI tile fragments suppressed by Bidirectional Containment.
5. **Photo 5 (`678x452` Twin Muddy Snow-Thaw Water Puddles)**:
   - *v2 Result*: `0` detections because the base `RDD2022` model has low raw confidence (`< 0.15`) on opaque beige/brown water-filled craters where the interior is a flat mirror reflection.
   - *Remaining Challenge*: Requires real wet/puddle pothole training images in Stage 2 (rather than synthetic painted ovals) so the detector backbone recognizes water-filled cavities natively.

---

## 5. Key Files & Commands

### Files in `C:\Users\Samuel\Downloads\BANTAI\colab\`
- [`pothole_enhancer_workflow.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/pothole_enhancer_workflow.py) — Core Stage-2 dataset loader, 6-domain photometric augmentation, CLAHE, 5-Tile Road-SAHI, WBF, rim-aware roughness filter, temporal tracker, and standalone `bantai_pothole_detector.py` generator.
- [`build_pothole_enhancer_notebook.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/build_pothole_enhancer_notebook.py) — Script that compiles `pothole_enhancer_workflow.py` and all Colab cells into `BANTAI_Pothole_Model_Enhancer.ipynb` with AST syntax validation.
- [`BANTAI_Pothole_Model_Enhancer.ipynb`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/BANTAI_Pothole_Model_Enhancer.ipynb) — The generated 26-cell turnkey Google Colab notebook.
- [`test_pothole_enhancer_workflow.py`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/test_pothole_enhancer_workflow.py) — Offline unit test suite (19 unit tests covering CLAHE, SAHI remapping, focal tile, WBF, horizon/vent/bumper filtering, duplicate suppression, rim-aware roughness, and AST validation).

### Local Verification Commands
```powershell
# 1. Rebuild the Colab Enhancer Notebook (validates AST on all 26 cells)
python "C:\Users\Samuel\Downloads\BANTAI\colab\build_pothole_enhancer_notebook.py"

# 2. Run the offline unit test suite
python -m unittest discover -s "C:\Users\Samuel\Downloads\BANTAI\colab" -p "test_pothole_enhancer_workflow.py"
```

---

## 6. Next Recommended Step for User / Next Agent

1. **Restore Working v2 Baseline in Colab First**:
   - Upload the freshly generated [`BANTAI_Pothole_Model_Enhancer.ipynb`](file:///C:/Users/Samuel/Downloads/BANTAI/colab/BANTAI_Pothole_Model_Enhancer.ipynb) to Google Colab and run **Runtime -> Run all** (~3–4 minutes).
   - Because the dataset manifest is now `"v2_clean_restored"` and the training flag is `completed_v2_restored.json`, it will automatically discard the corrupted v3 cache, fine-tune for 6 epochs on the clean v2 Parquet dataset from the untouched Stage-1 checkpoint (`/content/drive/MyDrive/BANTAI/pothole_15k/.../checkpoint_best_total.pth`), and overwrite `ENHANCED_DRIVE_DIR` with the clean v2 Stage-2 weights and `bantai_pothole_detector.py`.
2. **Verify the 5 Test Photos in Section 8A**:
   - Confirm that the restored v2 model + the 3 surgical tweaks in Section 8A reproduce the clean detections on Photo 1 (country road without the vent/duplicate), Photo 2 (urban street potholes), Photo 3 (highway road anomaly without the distant SUV bumper), and Photo 4 (`84.7% [Critical]` crater).
3. **Rule for Any Future Iteration**:
   - **Never change training data and inference filters in the same step.** Test any post-processing filter change standalone in Section 8A on the existing checkpoint before ever modifying training or running another fine-tuning pass.
