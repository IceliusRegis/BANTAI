"""Pothole Detection Service for BANTAI.

Directly integrates the verified Pothole Model Bundle:
C:\\Users\\Samuel\\Downloads\\BANTAI\\BANTAI_Enhanced_Pothole_Model_Bundle_v5_20261107_061616
(and auto-discovers any future BANTAI_Enhanced_Pothole_Model_Bundle_* drop-in folders).

Features:
- Real RF-DETR Nano checkpoint inference + Cell 8A Vision Engine
- DBSCAN Single-Crater Fusion & Manhole Cover Rejection
- Dual-Window Road Analysis for portrait (8%-65% & 36%-94%) & dashcam media
- Normalized 0-100% bounding box output (box_pct / boxPct) for exact UI rendering
- Annotated MP4 Video Preview generator with bounding boxes & HUD counters
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def discover_pothole_bundle() -> Tuple[Path, Optional[Path], float, str]:
    """Auto-discovers the newest BANTAI_Enhanced_Pothole_Model_Bundle directory."""
    bundle_pattern = str(PROJECT_ROOT / "BANTAI_Enhanced_Pothole_Model_Bundle*")
    candidates = [Path(p) for p in sorted(glob.glob(bundle_pattern), reverse=True) if Path(p).is_dir()]

    target_dir: Optional[Path] = None
    if candidates:
        target_dir = candidates[0]
    else:
        colab_dir = PROJECT_ROOT / "colab"
        if colab_dir.is_dir():
            target_dir = colab_dir

    if not target_dir or not target_dir.is_dir():
        target_dir = PROJECT_ROOT / "BANTAI_Enhanced_Pothole_Model_Bundle_v5_20261107_061616"

    model_card = target_dir / "model_card.json"
    weights_path: Optional[Path] = None
    calibrated_conf = 0.20
    bundle_name = target_dir.name

    if model_card.is_file():
        try:
            with open(model_card, "r", encoding="utf-8") as f:
                meta = json.load(f)
                calibrated_conf = float(meta.get("calibrated_operating_threshold", 0.20))
                w_name = meta.get("weights_filename")
                if w_name and (target_dir / w_name).is_file():
                    weights_path = target_dir / w_name
                bundle_name = meta.get("model_name", bundle_name)
        except Exception:
            pass

    if not weights_path and target_dir.is_dir():
        pths = list(target_dir.glob("*.pth"))
        if pths:
            weights_path = sorted(pths, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return target_dir, weights_path, calibrated_conf, bundle_name


class BantaiPotholeService:
    """Production service adapter for BANTAI Enhanced Pothole Detection (Bundle v5+)."""

    def __init__(self, bundle_dir: Optional[str] = None):
        self.bundle_dir, self.weights_path, self.calibrated_conf, self.bundle_name = discover_pothole_bundle()
        if bundle_dir:
            self.bundle_dir = Path(bundle_dir)

        if str(self.bundle_dir) not in sys.path:
            sys.path.insert(0, str(self.bundle_dir))

        self.device = "cpu"
        self.detector = None
        self.tracker = None
        self._init_detector()

    def _init_detector(self):
        """Loads PotholeDetector from bantai_pothole_detector.py with real RF-DETR weights."""
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            from bantai_pothole_detector import PotholeDetector, SmoothIoUVideoTracker
            w_str = str(self.weights_path) if self.weights_path else None
            self.detector = PotholeDetector(
                weights_path=w_str,
                confidence=self.calibrated_conf,
                device=self.device,
            )
            self.tracker = SmoothIoUVideoTracker()
            print(f"[BantaiPotholeService] Loaded {self.bundle_name} on {self.device} from {self.bundle_dir}", flush=True)
        except Exception as exc:
            print(f"[BantaiPotholeService] ERROR loading PotholeDetector: {exc}", flush=True)

    def is_weights_present(self) -> bool:
        return self.weights_path is not None and self.weights_path.is_file()

    def is_model_loaded(self) -> bool:
        return self.detector is not None

    def run_on_frames(
        self,
        frame_entries: List[Dict[str, Any]],
        frames_dir: Path,
        is_video: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run Colab Cell 8A PotholeDetector on extracted frames."""
        frames_dir = Path(frames_dir)
        mode = "video" if is_video else "photo"

        if self.detector is None:
            self._init_detector()

        results = []
        for entry in frame_entries:
            img_path = frames_dir / entry["file_name"]
            w = entry.get("width", 1280)
            h = entry.get("height", 720)

            if self.detector is not None:
                try:
                    with Image.open(img_path) as src:
                        im = ImageOps.exif_transpose(src).convert("RGB")
                        w, h = im.size
                        # Run full Cell 8A Vision Engine (Dual-Window portrait crops, LAB-CLAHE,
                        # Manhole cover rejection, Litter rejection, and DBSCAN single-crater fusion)
                        raw_dets = self.detector.detect(
                            im,
                            mode=mode,
                            confidence=self.calibrated_conf,
                        )
                    results.append(self._format_defects(raw_dets, w, h))
                    continue
                except Exception as exc:
                    print(f"[BantaiPotholeService] Detector error on {entry['file_name']}: {exc}", flush=True)

            # If detector could not run, return empty defects (NEVER return fake hardcoded boxes)
            results.append(self._format_defects([], w, h))

        return results

    def _format_defects(self, detections: List[Dict[str, Any]], img_w: int, img_h: int) -> Dict[str, Any]:
        """Formats output with normalized 0-100% box coordinates."""
        defects = []
        for idx, d in enumerate(detections):
            x1, y1, x2, y2 = d["xyxy"]
            x1 = max(0.0, min(float(img_w), float(x1)))
            y1 = max(0.0, min(float(img_h), float(y1)))
            x2 = max(0.0, min(float(img_w), float(x2)))
            y2 = max(0.0, min(float(img_h), float(y2)))

            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            score = float(d.get("score", 0.0))
            raw_sev = str(d.get("severity", "Moderate")).lower()
            severity = "high" if raw_sev in ("critical", "high") else raw_sev

            lane_ratio = bw / float(max(1, img_w))
            est_diameter_cm = round(max(15.0, min(120.0, lane_ratio * 320.0)), 1)

            box_pct = {
                "x": round((x1 / float(max(1, img_w))) * 100.0, 2),
                "y": round((y1 / float(max(1, img_h))) * 100.0, 2),
                "w": round((bw / float(max(1, img_w))) * 100.0, 2),
                "h": round((bh / float(max(1, img_h))) * 100.0, 2),
            }

            defects.append({
                "defect_id": idx + 1,
                "type": "pothole",
                "label": f"Pothole ({raw_sev.title()})",
                "severity": severity,
                "estimated_diameter_cm": est_diameter_cm,
                "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                "box_pct": box_pct,
                "boxPct": box_pct,
                "confidence": round(score, 3),
                "confidence_pct": round(score * 100),
                "confidencePct": round(score * 100),
            })

        return {
            "model_version": f"BANTAI-Pothole-Bundle-{self.bundle_name}",
            "total_defects": len(defects),
            "defects": defects,
        }

    def render_annotated_video(
        self,
        video_path: Path,
        output_path: Path,
        frame_entries: List[Dict[str, Any]],
        pothole_findings_by_frame: List[Dict[str, Any]],
        trash_findings_by_frame: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Renders an annotated MP4 video with live pothole bounding boxes and HUD counters."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video for annotation: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"[BantaiPotholeService] VideoWriter could not be opened for {output_path}", flush=True)
            cap.release()
            return output_path

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = max(1, len(frame_entries))

        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            progress = current_frame / float(max(1, total_frames))
            entry_idx = min(frame_count - 1, int(round(progress * (frame_count - 1))))

            pothole_defects = (
                pothole_findings_by_frame[entry_idx].get("defects", [])
                if entry_idx < len(pothole_findings_by_frame)
                else []
            )

            # Draw Pothole Bounding Boxes
            for d in pothole_defects:
                x, y, bw, bh = [int(v) for v in d["bbox"]]
                sev = d.get("severity", "moderate").lower()
                color = (0, 0, 230) if sev == "high" else ((0, 140, 255) if sev == "moderate" else (0, 215, 255))
                conf_pct = d.get("confidence_pct", int(d.get("confidence", 0.8) * 100))
                label = f"Pothole {conf_pct}% [{sev.upper()}]"

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x, max(0, y - th - 10)), (x + tw + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Top HUD Banner (Potholes only)
            hud_text = f"BANTAI POTHOLE MONITOR | Potholes Detected: {len(pothole_defects)}"
            cv2.rectangle(frame, (15, 15), (min(width - 15, 480), 55), (20, 24, 28), -1)
            cv2.putText(frame, hud_text, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 180), 2, cv2.LINE_AA)

            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()
        return output_path
