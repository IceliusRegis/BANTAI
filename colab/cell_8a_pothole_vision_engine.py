# @title 8A. Manual Photo & Video (.mp4) Upload — Master Combined Engine (Best of V1–V6)
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
    '''Reject colorful roadside trash wrappers, plastic cups, dry leaves, and weathered white paint specks (V1-V6 combined).'''
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
        # From V3-V6: Reject weathered white lane stripes / crosswalk paint specks on flat asphalt (no dark shadow cavity)
        val_p10 = float(np.percentile(hsv[..., 2], 10))
        val_p85 = float(np.percentile(hsv[..., 2], 85))
        if area_ratio < 0.015 and score < 0.48:
            if val_p85 >= 210.0 and val_p10 >= 100.0:
                return True
    return False


def _is_flat_intact_concrete_or_road_slab(gray_arr, box, score, is_video_stream=False):
    '''Reject uniform flat concrete slabs or smooth road surface boxes that lack a dark pothole cavity or broken rim.'''
    gh, gw = gray_arr.shape
    x1, y1, x2, y2 = [max(0, int(round(v))) for v in box]
    x2, y2 = min(gw, x2), min(gh, y2)
    if x2 <= x1 + 6 or y2 <= y1 + 6:
        return True
    patch = gray_arr[y1:y2, x1:x2]
    p_std = float(np.std(patch))
    p10 = float(np.percentile(patch, 10))
    p90 = float(np.percentile(patch, 90))
    contrast_span = p90 - p10
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, gw * gh))

    # On video streams, suppress low/mid-confidence boxes on flat concrete slabs without a dark shadow/cavity
    if is_video_stream:
        if score < 0.34 and p_std < 24.0 and contrast_span < 58.0:
            return True
        if area_ratio > 0.055 and score < 0.48 and p10 > 82.0 and p_std < 30.0:
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
    '''OpenCV Canny + Contour Solidity + cv2.fitEllipse check for flush circular/elliptical manhole lids (V1-V6).'''
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
    '''Resolution-invariant detector for circular/elliptical cast-iron manhole covers, valve caps & concrete collars (V1-V6 combined).'''
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

    dy = np.diff(patch, axis=0)
    flip_y = float(np.mean((dy[:-1, :] * dy[1:, :]) < -2.0)) if dy.shape[0] > 2 else 0.0
    strong_flip_y = float(np.mean((dy[:-1, :] * dy[1:, :]) < -12.0)) if dy.shape[0] > 2 else 0.0
    lr_diff = float(np.mean(np.abs(patch - patch[:, ::-1]))) / (p_std + 1e-4)
    tb_diff = float(np.mean(np.abs(patch - patch[::-1, :]))) / (p_std + 1e-4)

    # 1. Corner-cropped foreground waffle manhole cover (V2 border margin + V3-V6 p5 > 85 dark-cavity protection)
    touches_bottom = cy2 >= ch - 12
    touches_side = (cx1 <= 14) or (cx2 >= cw - 14)
    if touches_bottom and touches_side and area_ratio <= 0.050 and aspect <= 1.95:
        p5 = float(np.percentile(patch, 5))
        if p_std < 25.0 and p5 > 85.0 and flip_y >= 0.16:
            return True

    # 2. Video-mode approaching manhole check
    if is_video_stream and 1.22 <= aspect <= 4.5 and 0.0008 <= area_ratio <= 0.060 and cy_ratio <= 0.82:
        if min(lr_diff, tb_diff) <= 1.05 and flip_y >= 0.21:
            if area_ratio < 0.008 or (strong_flip_y >= 0.28 and p_std >= 27.0):
                return True
        if _has_clean_ellipse_or_convex_manhole_rim(np.clip(patch, 0, 255).astype(np.uint8), aspect) and min(lr_diff, tb_diff) <= 1.08:
            return True

    # 3. Standard mid/upper road foreshortened elliptical manhole cover / valve cap / collar
    min_aspect = 1.25 if area_ratio < 0.0065 else 1.88
    max_cy = 0.74 if area_ratio < 0.0065 else 0.65
    if aspect < min_aspect or aspect > 5.2:
        return False
    if area_ratio < 0.0008 or area_ratio >= 0.033:
        return False
    if cy_ratio > max_cy:
        return False

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
    '''DBSCAN density clustering on single-crater overlap metric with Crater Rim Context Dilation (V4-V6).'''
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
        tight_members = [ordered[j] for j in neighbors if _box_iou(core["xyxy"], ordered[j]["xyxy"]) >= 0.45]
        if len(tight_members) > 1:
            weights = np.array([float(m.get("score", 0.2)) ** 2 for m in tight_members], dtype=np.float64)
            weights /= max(1e-6, weights.sum())
            coords = np.array([m["xyxy"] for m in tight_members], dtype=np.float64)
            refined = (coords * weights[:, None]).sum(axis=0)
            core["xyxy"] = [round(float(v), 2) for v in refined]

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
    '''Master All-Terrain Smart Pothole Detector combining the best of V1-V6.'''
    w, h = pil_img.size
    total_area = float(max(1, w * h))
    rgb_arr = np.array(pil_img, dtype=np.float32)
    gray = np.array(pil_img.convert("L"), dtype=np.float32)

    canon_pil = ImageOps.contain(pil_img.convert("L"), (980, 640))
    cw, ch = canon_pil.size
    canon_gray = np.array(canon_pil, dtype=np.float32)
    rx, ry = cw / float(max(1, w)), ch / float(max(1, h))

    is_cabin_dashcam = _is_true_cabin_dashcam(gray)
    is_portrait_walk = h > w * 1.15
    is_square_dashcam = abs(w - h) <= max(w, h) * 0.05

    # 1. Full-Frame pass
    full_dets = detector.detect(pil_img, mode="video", confidence=conf_threshold)
    strong_full_craters = [
        d for d in full_dets
        if float(d.get("score", 0.0)) >= 0.45
        and ((d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]) / total_area) >= 0.012
        and not _is_manhole_or_utility_cover(
            canon_gray,
            [d["xyxy"][0] * rx, d["xyxy"][1] * ry, d["xyxy"][2] * rx, d["xyxy"][3] * ry],
            is_cabin_dashcam=is_cabin_dashcam,
            is_video_stream=is_video_stream,
        )
    ]

    # 2. Road pass (V6 Dual Portrait Windows + Center Trench Window on Video; Multi-Tile SAHI on Photos)
    if is_video_stream:
        road_conf = max(0.18, conf_threshold * 0.85)
        road_crops = []
        if is_portrait_walk:
            # Window 1 (Upper-Mid Road: 8% to 66% height) captures approaching/mid-road vertical trenches early
            # Window 2 (Lower Road: 35% to 94% height) captures close-range potholes
            road_crops.append((0, int(h * 0.08), w, int(h * 0.66)))
            road_crops.append((0, int(h * 0.35), w, int(h * 0.94)))
        else:
            road_crops.append((0, int(h * 0.24), w, h))

        road_dets = []
        for rx1_c, ry1_c, rx2_c, ry2_c in road_crops:
            crop_part = pil_img.crop((rx1_c, ry1_c, rx2_c, ry2_c))
            if hasattr(detector, "_predict_raw"):
                raw_part = detector._predict_raw(crop_part, road_conf)
            elif hasattr(detector, "_raw_predict"):
                raw_part = detector._raw_predict(crop_part, road_conf)
            else:
                raw_part = detector.detect(crop_part, mode="video", confidence=road_conf)
            for rd in raw_part:
                b = rd["xyxy"]
                rd_copy = dict(rd)
                rd_copy["xyxy"] = [b[0] + rx1_c, b[1] + ry1_c, b[2] + rx1_c, b[3] + ry1_c]
                road_dets.append(rd_copy)
        sahi_dets = road_dets
    else:
        sahi_dets = detector.detect(pil_img, mode="photo", confidence=max(0.18, conf_threshold * 0.88))

    # From V2/V3: If a strong crater (>=0.45) is already locked on in video, raise noise floor to 0.32 so bottom-edge 27% concrete specks are ignored
    if is_video_stream and strong_full_craters and not is_cabin_dashcam:
        candidates = [
            d for d in (list(full_dets) + list(sahi_dets))
            if float(d.get("score", 0.0)) >= 0.32
        ]
    else:
        candidates = list(sahi_dets) + list(full_dets)

    max_crater_area = max(
        [((d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1])) for d in candidates if float(d.get("score", 0.0)) >= 0.36]
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

        # 2. Global Foreground Shoulder / Full-Width Slab Gate & Vertical Gate (V6: 0.22 on video for vertical trenches, V1-V5: 0.75 on photos)
        if (bw > w * 0.70 or area_ratio > 0.25) and y2 > h * 0.93 and score < 0.60:
            continue
        min_vert_aspect = 0.22 if is_video_stream else 0.65
        if (bw / bh) < min_vert_aspect and area_ratio > 0.035 and score < 0.52:
            continue

        # 3. Universal Manhole Cover, Water-Valve Cap & Concrete-Collar Rejector
        cbox = [x1 * rx, y1 * ry, x2 * rx, y2 * ry]
        if _is_manhole_or_utility_cover(canon_gray, cbox, is_cabin_dashcam=is_cabin_dashcam, is_video_stream=is_video_stream):
            rejected_manhole_boxes.append([x1, y1, x2, y2])
            continue

        # 3B. Micro-Speck & Smooth Flush Slab Gate
        if not is_cabin_dashcam:
            if area_ratio < 0.0022:
                continue
            if not is_video_stream and area_ratio < 0.015 and score < 0.27:
                c_patch = canon_gray[max(0, int(cbox[1])) : min(ch, int(cbox[3]) + 1), max(0, int(cbox[0])) : min(cw, int(cbox[2]) + 1)]
                if c_patch.size > 0 and float(np.std(c_patch)) < 26.0:
                    continue
            if is_video_stream and _is_flat_intact_concrete_or_road_slab(gray, [x1, y1, x2, y2], score, is_video_stream=True):
                continue

        # 4. HSV Chromatic Roadside Litter, Leaf & White Paint Speck Filter
        if _is_roadside_litter_or_leaf(rgb_arr, [x1, y1, x2, y2], score):
            continue

        # 5. Relative Size Outlier Gate (V1/V3: drops small <15% low-conf specks when a real crater is present)
        if max_crater_area > (total_area * 0.018) and box_area < (max_crater_area * 0.15) and score < 0.35:
            continue

        # 6. Video Stream Stability Floor (prevents 20-27% bottom-edge noise boxes while keeping >=28% approaching trenches)
        if is_video_stream and not is_cabin_dashcam and score < 0.26:
            continue

        if is_cabin_dashcam:
            if y2 > h * 0.60 or cy > h * 0.56:
                continue
            if cx < w * 0.28 or cx > w * 0.62 or x2 > w * 0.66:
                continue
            if area_ratio > 0.015 and score < 0.50:
                continue
        elif is_square_dashcam and not strong_full_craters:
            if x1 < w * 0.165 or y2 > h * 0.80:
                continue

        kept.append(dict(d))

    # Poison-Pill Manhole Overlap Suppression (still photos)
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

    # Concrete-Slab Recovery for still photos
    if not kept and not is_video_stream and not is_cabin_dashcam and not is_square_dashcam:
        kept.extend(_recover_concrete_slab_damage(canon_gray, rx, ry, rejected_manhole_boxes))

    merged = dbscan_single_crater_fusion(kept, gray_arr=gray)
    for r in merged:
        r["severity"] = estimate_pothole_severity(r["xyxy"], w, h, r["score"])
    if return_rejected_manholes:
        return merged, rejected_manhole_boxes
    return merged


def _box_perspective_similarity(track_box, det_box, max_scale=3.4):
    '''Perspective-aware similarity for forward-moving video cameras (V4-V6).'''
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

    if x_dist <= 0.45 and -0.35 <= y_dist <= 1.25 and scale_ratio <= max_scale:
        persp_score = 1.0 - (0.40 * x_dist + 0.30 * abs(y_dist) + 0.15 * (scale_ratio - 1.0) / max_scale)
        return max(iou, max(iomin, persp_score))
    return max(iou, iomin)


class SmoothIoUVideoTracker:
    '''Combined V1-V6 Video Tracker: Perspective ByteTrack Association + Anti-Ghost Coasting Guard.'''

    def __init__(self, iou_threshold=0.20, min_hits=2, max_age=6, alpha=0.80):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.next_id = 1
        self.tracks = []
        self.blacklisted_manholes = []

    def update(self, detections, rejected_manholes=None, frame_gray=None):
        if rejected_manholes:
            for mh_box in rejected_manholes:
                if any(_box_iou(mh_box, t["xyxy"]) >= 0.15 or _box_iomin(mh_box, t["xyxy"]) >= 0.25 for t in self.tracks if t["hits"] >= 2 or t.get("peak_score", 0.0) >= 0.38):
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
        self.blacklisted_manholes = [bz for bz in self.blacklisted_manholes if bz["age"] <= 8 and bz["hits"] >= 2]

        if self.blacklisted_manholes and detections:
            filtered_dets = []
            for d in detections:
                db = d["xyxy"]
                if any(_box_iomin(db, bz["xyxy"]) >= 0.35 or _box_iou(db, bz["xyxy"]) >= 0.22 for bz in self.blacklisted_manholes):
                    continue
                filtered_dets.append(d)
            detections = filtered_dets

        detections = dbscan_single_crater_fusion(detections)

        # 1. Predict forward position via damped velocity (no runaway corner expansion)
        for t in self.tracks:
            t["time_since_update"] += 1
            vx, vy = t.get("vel", [0.0, 0.0])
            t["pred_box"] = [
                t["xyxy"][0] + 0.80 * vx,
                t["xyxy"][1] + 0.80 * vy,
                t["xyxy"][2] + 0.80 * vx,
                t["xyxy"][3] + 0.80 * vy,
            ]

        # 2. Split detections into high-conf (>=0.30) and low-conf (0.20..0.30)
        high_dets = [d for d in detections if float(d.get("score", 0.0)) >= 0.30]
        low_dets = [d for d in detections if 0.20 <= float(d.get("score", 0.0)) < 0.30]

        matched_tracks = set()
        matched_high_dets = set()

        # Stage 1: Match high-confidence detections with Perspective Scale Matching
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

        # Stage 2: Match low-confidence detections against remaining confirmed tracks
        matched_low_dets = set()
        remaining_confirmed = [
            (idx, t) for idx, t in enumerate(self.tracks)
            if idx not in matched_tracks and (t["hits"] >= self.min_hits or t.get("peak_score", 0.0) >= 0.36)
        ]
        if remaining_confirmed and low_dets:
            pairs = []
            for t_idx, track in remaining_confirmed:
                for d_idx, det in enumerate(low_dets):
                    sim = _box_perspective_similarity(track["pred_box"], det["xyxy"])
                    if sim >= (self.iou_threshold * 0.80):
                        pairs.append((sim, t_idx, d_idx))
            pairs.sort(key=lambda x: x[0], reverse=True)
            for _, t_idx, d_idx in pairs:
                if t_idx in matched_tracks or d_idx in matched_low_dets:
                    continue
                matched_tracks.add(t_idx)
                matched_low_dets.add(d_idx)
                self._apply_match(self.tracks[t_idx], low_dets[d_idx])

        # Coast unmatched tracks briefly (with texture check so boxes NEVER coast onto empty flat concrete!)
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                vx, vy = track.get("vel", [0.0, 0.0])
                track["xyxy"] = list(track["pred_box"])
                track["vel"] = [0.75 * vx, 0.75 * vy]
                track["score"] = max(0.18, track["score"] * 0.88)
                if frame_gray is not None and _is_flat_intact_concrete_or_road_slab(frame_gray, track["xyxy"], track["score"], is_video_stream=True):
                    track["time_since_update"] = self.max_age + 1

        # Create new candidate tracks from unmatched detections with score >= 0.28
        for d_idx, det in enumerate(high_dets):
            if d_idx not in matched_high_dets and float(det.get("score", 0.0)) >= 0.28:
                self.tracks.append({
                    "id": self.next_id,
                    "xyxy": list(det["xyxy"]),
                    "score": float(det["score"]),
                    "peak_score": float(det["score"]),
                    "severity": det.get("severity", "Moderate"),
                    "hits": 1,
                    "time_since_update": 0,
                    "vel": [0.0, 0.0],
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]

        active = []
        for t in self.tracks:
            # Confirmed if hits >= 2 (peak >= 0.32) OR instant lock-on if score >= 0.42; max coasting display = 3 frames
            is_confirmed = (t["hits"] >= self.min_hits and t["peak_score"] >= 0.32) or t.get("peak_score", 0.0) >= 0.42
            if is_confirmed and t["time_since_update"] <= 3:
                active.append({
                    "id": t["id"],
                    "xyxy": [round(v, 2) for v in t["xyxy"]],
                    "score": round(t["score"], 3),
                    "severity": t["severity"],
                    "class_name": "pothole",
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
        track["score"] = 0.72 * float(det["score"]) + 0.28 * float(track["score"])
        track["peak_score"] = max(float(track.get("peak_score", 0.0)), float(det["score"]))
        track["severity"] = det.get("severity", track.get("severity", "Moderate"))
        track["hits"] += 1
        track["time_since_update"] = 0


def process_uploaded_video(vname, vbytes, conf_threshold=0.20):
    '''Annotate an uploaded .mp4/.mov/.avi video with Combined V1-V6 Vision Engine.'''
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

    tracker = SmoothIoUVideoTracker(iou_threshold=0.20, min_hits=2, max_age=6, alpha=0.80)
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
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        pil_frame = Image.fromarray(frame_rgb)

        smart_dets, rej_manholes = smart_pothole_detect(
            pil_frame,
            conf_threshold=conf_threshold,
            is_video_stream=True,
            return_rejected_manholes=True,
        )
        active = tracker.update(smart_dets, rejected_manholes=rej_manholes, frame_gray=frame_gray)

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
