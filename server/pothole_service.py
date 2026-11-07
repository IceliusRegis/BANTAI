"""Model B (Road Defect / Pothole Detection) Service for BANTAI.

Integrates BANTAI RF-DETR Nano All-Terrain Pothole Detector (v5_20261107_061616)
and its Cell 8A Vision Engine from BANTAI_Enhanced_Pothole_Model_Bundle_v5_20261107_061616:
- Multi-Window Road Crops / SAHI Tiling
- Universal Manhole Cover & Utility Valve Cap Rejection (OpenCV Contour/Ellipse + Waffle Frequency)
- Roadside Litter, Leaf & Sunlit Paint Filtering (HSV saturation/value rejection)
- Concrete Slab Dark Patch & Spall Cavity Recovery
- DBSCAN Single-Crater Fusion with 18% Adaptive Crater Rim Context Dilation
- Severity Estimation ("Low", "Moderate", "Critical" -> normalized "low", "moderate", "high")
- Fail-closed contract: zero mock or hardcoded fallback boxes.
"""
from __future__ import annotations

import gc
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageOps

BANTAI_ROOT = Path(__file__).resolve().parent.parent


def discover_pothole_bundle(search_root: Path = BANTAI_ROOT) -> Tuple[Path, Optional[Path], float, str, str]:
    """Auto-discovers the BANTAI_Enhanced_Pothole_Model_Bundle directory and reads model_card.json."""
    env_bundle = os.environ.get("BANTAI_POTHOLE_BUNDLE_DIR")
    if env_bundle and Path(env_bundle).is_dir():
        candidates = [Path(env_bundle).resolve()]
    else:
        bundle_pattern = str(search_root / "BANTAI_Enhanced_Pothole_Model_Bundle*")
        candidates = [Path(p) for p in sorted(glob.glob(bundle_pattern), reverse=True) if Path(p).is_dir()]
        if not candidates:
            dl_candidates = [
                Path(p)
                for p in sorted(
                    glob.glob(r"C:\Users\Justine\Downloads\BANTAI_Enhanced_Pothole_Model_Bundle*"),
                    reverse=True,
                )
                if Path(p).is_dir()
            ]
            candidates.extend(dl_candidates)

    target_dir = (
        candidates[0]
        if candidates
        else (search_root / "BANTAI_Enhanced_Pothole_Model_Bundle_v5_20261107_061616")
    )
    model_card = target_dir / "model_card.json"
    weights_path: Optional[Path] = None
    calibrated_conf = 0.20
    bundle_name = target_dir.name
    vision_engine = "Cell 8A All-Terrain Vision Engine"

    if model_card.is_file():
        try:
            with open(model_card, "r", encoding="utf-8") as f:
                meta = json.load(f)
                calibrated_conf = float(meta.get("calibrated_operating_threshold", 0.20))
                w_name = meta.get("weights_filename")
                if w_name and (target_dir / w_name).is_file():
                    weights_path = target_dir / w_name
                bundle_name = meta.get("model_name", bundle_name)
                vision_engine = meta.get("vision_engine", vision_engine)
        except Exception:
            pass

    if not weights_path and target_dir.is_dir():
        pths = list(target_dir.glob("*.pth"))
        if pths:
            weights_path = sorted(pths, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return target_dir, weights_path, calibrated_conf, bundle_name, vision_engine


class BantaiPotholeService:
    """Singleton production service adapter for Model B (Pothole & Road Defect Detection)."""

    def __init__(
        self,
        bundle_dir: Optional[str | Path] = None,
        weights_path: Optional[str | Path] = None,
        device: str = "auto",
    ):
        if bundle_dir:
            b_path = Path(bundle_dir).resolve()
            self.bundle_dir, disc_weights, self.calibrated_conf, self.bundle_name, self.vision_engine = (
                discover_pothole_bundle(b_path.parent)
            )
            self.bundle_dir = b_path
        else:
            self.bundle_dir, disc_weights, self.calibrated_conf, self.bundle_name, self.vision_engine = (
                discover_pothole_bundle()
            )

        if weights_path:
            self.weights_path: Optional[Path] = Path(weights_path).resolve()
        else:
            self.weights_path = disc_weights

        if self.bundle_dir.is_dir() and str(self.bundle_dir) not in sys.path:
            sys.path.insert(0, str(self.bundle_dir))

        self.requested_device = device
        self.device = "cpu"
        self.detector = None
        self.cache_file = Path(__file__).resolve().parent / "pothole_inference_cache.json"
        self._cache = self._load_cache()
        self._init_detector()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            print(f"[BantaiPotholeService] Warning: failed to save cache ({exc})", flush=True)

    def is_weights_present(self) -> bool:
        return bool(
            self.weights_path
            and self.weights_path.is_file()
            and self.weights_path.stat().st_size > 10 * 1024 * 1024
        )

    def is_fully_available(self) -> bool:
        return self.detector is not None and self.is_weights_present()

    def _init_detector(self) -> None:
        if not self.is_weights_present():
            print(
                f"[BantaiPotholeService] ERROR: Pothole weights missing or invalid at {self.weights_path}. Failing closed.",
                flush=True,
            )
            self.detector = None
            return

        try:
            import torch
            try:
                import transformers
                if not hasattr(transformers, "BackboneConfigMixin"):
                    from transformers.utils.backbone_utils import BackboneConfigMixin, BackboneMixin
                    _orig_init_tb = BackboneMixin._init_transformers_backbone
                    if not getattr(_orig_init_tb, "_bantai_v5_compat", False):
                        def _compat_init_tb(self_bb, config=None):
                            return _orig_init_tb(self_bb, config if config is not None else self_bb.config)
                        _compat_init_tb._bantai_v5_compat = True  # type: ignore[attr-defined]
                        BackboneMixin._init_transformers_backbone = _compat_init_tb
                    transformers.BackboneConfigMixin = BackboneConfigMixin  # type: ignore[attr-defined]
                    transformers.BackboneMixin = BackboneMixin  # type: ignore[attr-defined]
            except Exception:
                pass
            if self.requested_device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.requested_device

            from bantai_pothole_detector import PotholeDetector
            self.detector = PotholeDetector(
                weights_path=str(self.weights_path),
                confidence=self.calibrated_conf,
                device=self.device,
            )
            print(
                f"[BantaiPotholeService] Loaded {self.bundle_name} ({self.vision_engine}) on {self.device} "
                f"(threshold={self.calibrated_conf:.2f}).",
                flush=True,
            )
        except Exception as exc:
            print(f"[BantaiPotholeService] ERROR loading PotholeDetector: {exc}", flush=True)
            self.detector = None

    def detect_image(self, image_path: str | Path, is_video_frame: bool = False) -> Dict[str, Any]:
        """Runs pothole detection on a single image/frame and returns normalized UI-ready JSON."""
        mode = "video" if is_video_frame else "photo"
        if self.detector is None and self.is_weights_present():
            self._init_detector()

        with Image.open(image_path) as src:
            im = ImageOps.exif_transpose(src).convert("RGB")
            w, h = im.size
            if self.detector is None:
                return self._format_defects([], w, h)
            try:
                raw_dets = self.detector.detect(im, mode=mode, confidence=self.calibrated_conf)
                return self._format_defects(raw_dets, w, h)
            except Exception as exc:
                print(f"[BantaiPotholeService] ERROR during detect_image on {image_path}: {exc}", flush=True)
                return self._format_defects([], w, h)

    def run_on_frames(
        self,
        frame_entries: List[Dict[str, Any]],
        frames_dir: Path,
        is_video: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run pothole detection across shared extracted frames (1.0 FPS video frames or single still image).

        Per AGENT_INTEGRATION_GUIDE.md Section 4.3:
        For 1.0 FPS sampled frames, call detector.detect(pil_img, mode='video') directly on each frame
        rather than SmoothIoUVideoTracker (which is designed for continuous 24-60 FPS streams).
        Fail closed: if detector is unavailable or throws an error, return an empty defects list [].
        """
        frames_dir = Path(frames_dir)
        mode = "video" if is_video else "photo"

        if self.detector is None and self.is_weights_present():
            self._init_detector()

        results: List[Dict[str, Any]] = []
        cache_dirty = False

        for entry in frame_entries:
            img_path = frames_dir / entry["file_name"]
            w, h = entry.get("width", 1280), entry.get("height", 720)
            sha = entry.get("file_sha256", "")
            cache_key = f"{sha}:{mode}:{self.calibrated_conf:.2f}" if sha else ""

            if cache_key and cache_key in self._cache:
                cached_raw = self._cache[cache_key]
                results.append(self._format_defects(cached_raw, w, h))
                continue

            if self.detector is None:
                print(
                    f"[BantaiPotholeService] Detector unavailable for {entry.get('file_name')}; returning empty defects list.",
                    flush=True,
                )
                results.append(self._format_defects([], w, h))
                continue

            try:
                with Image.open(img_path) as src:
                    im = ImageOps.exif_transpose(src).convert("RGB")
                    w, h = im.size
                    raw_dets = self.detector.detect(im, mode=mode, confidence=self.calibrated_conf)

                if cache_key:
                    self._cache[cache_key] = raw_dets
                    cache_dirty = True

                results.append(self._format_defects(raw_dets, w, h))
            except Exception as exc:
                print(
                    f"[BantaiPotholeService] Inference error on {entry.get('file_name')} ({exc}). Failing closed with [].",
                    flush=True,
                )
                results.append(self._format_defects([], w, h))

        if cache_dirty:
            self._save_cache()

        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()

        return results

    def _format_defects(self, detections: List[Dict[str, Any]], img_w: int, img_h: int) -> Dict[str, Any]:
        """Formats raw xyxy detections into pixel [x, y, w, h] and 0-100% normalized box_pct."""
        defects: List[Dict[str, Any]] = []
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

            # Estimate diameter in cm assuming a typical lane is ~320 cm wide
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
                "severity": severity,                  # "low" | "moderate" | "high"
                "raw_severity": raw_sev.title(),       # "Low" | "Moderate" | "Critical"
                "estimated_diameter_cm": est_diameter_cm,
                "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],  # [x, y, w, h] in pixels
                "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "box_pct": box_pct,                    # {x, y, w, h} in 0-100% of image dimensions
                "boxPct": box_pct,                     # camelCase alias for TypeScript frontends
                "confidence": round(score, 3),         # float in [0.0, 1.0]
                "confidence_pct": round(score * 100),  # int in [0, 100]
                "confidencePct": round(score * 100),   # camelCase alias for TypeScript frontends
            })

        return {
            "model_version": "BANTAI-Pothole-RFDETR-Nano-v5_20261107_061616",
            "total_defects": len(defects),
            "defects": defects,
        }
