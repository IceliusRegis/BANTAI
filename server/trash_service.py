"""Model A (Trash Detection & Spatial Grouping) Service for BANTAI.

Strictly enforces:
- 9 active supervised classes locked to 0..8
- 5 canonical BANTAI categories (Plastic, Paper, Organic, Metal, Potentially Hazardous)
- 3-Model Consensus (Baseline, Stage 5 Run 1, D-095 Champion)
- Calibrated Stage 5 consensus parameters:
    delta=0.016, coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score",
    enable_rule1..7=True
- Calibrated score threshold 0.15, top_k 25, containment NMS
- D-112 Disjoint Spatial Grouping:
    k = 1 -> individual_litter (model_score = float)
    2 <= k <= 7 -> group_of_individual_litter (model_score = None)
    k >= 8 -> garbage_pile (model_score = None)
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Add BANTAI-Modeling- root to sys.path
MODELING_ROOT = Path(os.environ.get(
    "BANTAI_MODELING_ROOT",
    r"c:\Users\Justine\Documents\Python\Project Image-object Detection"
))
if MODELING_ROOT.is_dir() and str(MODELING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELING_ROOT))

# Strict Taxonomy
SUPERVISED_LABELS = {
    0: "cigarette",
    1: "plastic_bottle",
    2: "plastic_bag",
    3: "plastic_wrapper",
    4: "plastic_film",
    5: "drink_can",
    6: "plastic_cup",
    7: "styrofoam_piece",
    8: "paper",
}

CANONICAL_CATEGORIES = {
    0: "Potentially Hazardous",
    1: "Plastic",
    2: "Plastic",
    3: "Plastic",
    4: "Plastic",
    5: "Metal",
    6: "Plastic",
    7: "Potentially Hazardous",
    8: "Paper",
}


class BantaiTrashService:
    """Production service adapter for Model A (Trash Detection & Spatial Grouping)."""

    def __init__(self, repo_root: Optional[Path] = None, device: Optional[str] = None):
        self.repo_root = Path(repo_root or MODELING_ROOT)
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.checkpoints_ready = self._verify_checkpoints()
        self.cache_file = Path(__file__).resolve().parent / "trash_inference_cache.json"
        self._cache = self._load_cache()
        self._showcase_data = self._load_showcase()

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _verify_checkpoints(self) -> Dict[str, bool]:
        """Check presence of the 3 preserved production checkpoints."""
        runs = self.repo_root / "runs/training"
        champion = runs / "stage5-d095-box-contrastive-20261102-a/checkpoints/best-inference/model.safetensors"
        run1 = runs / "stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/checkpoints/epoch-02/model.safetensors"
        baseline_step = runs / "taco-896e2f44146d4d54aee518e6087908ca/checkpoints/step-000350/model.safetensors"
        baseline_epoch = runs / "taco-896e2f44146d4d54aee518e6087908ca/checkpoints/epoch-02/model.safetensors"
        scale576 = runs / "scale576-integration-c70249cde2f2422cb89b22232fd2b525/record.json"

        return {
            "d095_champion": champion.is_file(),
            "stage5_run1": run1.is_file(),
            "warmstart_baseline": baseline_step.is_file() and baseline_epoch.is_file() and scale576.is_file(),
        }

    def is_fully_available(self) -> bool:
        return all(self.checkpoints_ready.values())

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
        except Exception as e:
            print(f"[BantaiTrashService] Warning: failed to save cache ({e})", flush=True)

    def _load_showcase(self) -> Optional[Dict[str, Any]]:
        """Load precomputed showcase from modeling repo for zero-latency video/still replay."""
        sc_path = self.repo_root / "runs/video_detection/motion_extract_testing/video-showcase.json"
        if sc_path.is_file():
            try:
                with open(sc_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def run_on_frames(
        self,
        frame_entries: List[Dict[str, Any]],
        frames_dir: Path,
        score_threshold: float = 0.15,
        top_k: int = 25,
    ) -> List[Dict[str, Any]]:
        """Run 3-model detection, scene consensus, and D-112 spatial grouping on extracted frames."""
        frames_dir = Path(frames_dir)
        from scripts.taco_stage5_display_profiles import visible_rows
        from scripts.taco_stage5_spatial_grouping import compute_spatial_grouping, DEFAULT_MIN_PILE_ITEMS

        # 1. Fast Cache / Showcase Resolution
        frame_results: List[Optional[Dict[str, Any]]] = [None] * len(frame_entries)
        unresolved_entries: List[Dict[str, Any]] = []
        unresolved_indices: List[int] = []

        for idx, entry in enumerate(frame_entries):
            sha = entry.get("file_sha256", "")
            fname = entry.get("file_name", "")
            img_id_str = str(entry.get("id", ""))
            w, h = entry.get("width", 1280), entry.get("height", 720)

            # Check persistent cache by SHA-256
            if sha and sha in self._cache:
                cached_data = self._cache[sha]
                frame_results[idx] = self._format_frame_findings(entry, cached_data["detections"], cached_data["grouping"])
                continue

            # Check video-showcase.json by sha or filename/id
            matched_pred = None
            if self._showcase_data and "predictions" in self._showcase_data:
                consensus_dict = self._showcase_data["predictions"].get("D-095 + Consensus", {})
                images_dict = self._showcase_data.get("images", {})

                # Match by image sha in showcase images
                for sc_id, sc_img in images_dict.items():
                    if sc_img.get("file_sha256") == sha or sc_img.get("file_name") == fname:
                        if sc_id in consensus_dict:
                            matched_pred = consensus_dict[sc_id]
                            break

                # Match by direct id if not matched by sha
                if not matched_pred and img_id_str in consensus_dict:
                    matched_pred = consensus_dict[img_id_str]

            if matched_pred is not None:
                vis = visible_rows(
                    matched_pred,
                    scheme="9-specific",
                    nms_mode="agnostic_iou_0.50_containment_0.80",
                    threshold=score_threshold,
                )[:top_k]
                grp = compute_spatial_grouping(vis, image_width=w, image_height=h, min_pile_items=DEFAULT_MIN_PILE_ITEMS)
                # Store in persistent cache
                if sha:
                    self._cache[sha] = {"detections": vis, "grouping": grp}
                    self._save_cache()
                frame_results[idx] = self._format_frame_findings(entry, vis, grp)
                continue

            unresolved_entries.append(entry)
            unresolved_indices.append(idx)

        # 2. If all frames were resolved from cache/showcase, return immediately
        if not unresolved_entries:
            return [res for res in frame_results if res is not None]

        # 3. Live 3-Model GPU Inference on Unresolved Entries
        if self.is_fully_available():
            try:
                live_findings = self._run_live_inference(unresolved_entries, frames_dir, score_threshold, top_k)
                for pos, res in zip(unresolved_indices, live_findings):
                    frame_results[pos] = res
            except Exception as exc:
                print(f"[BantaiTrashService] Live inference error: {exc}", flush=True)
                for pos in unresolved_indices:
                    frame_results[pos] = self._run_empty_findings(frame_entries[pos])
        else:
            print("[BantaiTrashService] Warning: Checkpoints not fully available. Returning empty findings.", flush=True)
            for pos in unresolved_indices:
                frame_results[pos] = self._run_empty_findings(frame_entries[pos])

        return [res for res in frame_results if res is not None]

    def _run_live_inference(
        self,
        frame_entries: List[Dict[str, Any]],
        frames_dir: Path,
        score_threshold: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Executes actual Grounding DINO 3-model consensus & spatial grouping."""
        from scripts.taco_stage5_video_extract_and_detect import execute_three_model_detection
        from scripts.taco_stage5_consensus_tiebreak import apply_scene_consensus
        from scripts.taco_stage5_display_profiles import visible_rows
        from scripts.taco_stage5_spatial_grouping import compute_spatial_grouping, DEFAULT_MIN_PILE_ITEMS

        inference_results = execute_three_model_detection(frame_entries, frames_dir, device=self.device)

        frame_results = []
        for entry in frame_entries:
            img_id = entry["id"]
            w, h = entry["width"], entry["height"]
            sha = entry.get("file_sha256", "")

            d095_rows = inference_results["D-095 Champion (soup 50/50)"][img_id]
            d095_svec = inference_results["d095_scores"][img_id]
            base_rows = inference_results["Baseline"][img_id]
            run1_rows = inference_results["Stage 5 (256/384)"][img_id]

            # Exact Stage 5 D-111/D-112 Consensus Parameters
            c_rows, _, _ = apply_scene_consensus(
                d095_rows,
                d095_svec,
                image_width=w,
                image_height=h,
                delta=0.016,
                coactivation="3 plastic subtypes >= 0.20",
                confidence_mode="top1_score",
                enable_rule1=True,
                enable_rule2=True,
                enable_rule3=True,
                enable_rule4=True,
                enable_rule5=True,
                enable_rule6=True,
                enable_rule7=True,
                baseline_rows=base_rows,
                run1_rows=run1_rows,
            )

            # Strict 9-specific scheme, calibrated score >= 0.15, containment NMS
            filtered = visible_rows(
                c_rows,
                scheme="9-specific",
                nms_mode="agnostic_iou_0.50_containment_0.80",
                threshold=score_threshold,
            )[:top_k]

            # D-112 Disjoint Spatial Grouping
            grouping = compute_spatial_grouping(filtered, image_width=w, image_height=h, min_pile_items=DEFAULT_MIN_PILE_ITEMS)

            # Cache the result
            if sha:
                self._cache[sha] = {"detections": filtered, "grouping": grouping}
                self._save_cache()

            frame_results.append(self._format_frame_findings(entry, filtered, grouping))

        # Clear VRAM after Model A finishes
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()

        return frame_results

    def _format_frame_findings(
        self,
        entry: Dict[str, Any],
        detections: List[Dict[str, Any]],
        grouping: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Formats output to match ARGON_WEBSITE_INTEGRATION_GUIDE.md Section 6.1 and 6.2."""
        formatted_detections = []
        for idx, row in enumerate(detections):
            cid = int(row.get("category_id", 0))
            cid = cid if cid in SUPERVISED_LABELS else 1
            formatted_detections.append({
                "detection_id": idx + 1,
                "category_id": cid,
                "label": SUPERVISED_LABELS.get(cid, "plastic_bottle"),
                "bantai_category": CANONICAL_CATEGORIES.get(cid, "Plastic"),
                "model_score": round(float(row.get("score", 0.0)), 4),
                "bbox": [round(float(c), 2) for c in row.get("bbox", [0, 0, 0, 0])],
            })

        formatted_regions = []
        for reg in grouping.get("regions", []):
            cat = reg.get("category", "individual_litter")
            k = int(reg.get("item_count", 1))
            member_indices = reg.get("member_indices", [])

            # Compute category breakdown for multi-item regions
            category_breakdown: Dict[str, int] = {}
            for m_idx in member_indices:
                if 0 <= m_idx < len(formatted_detections):
                    bcat = formatted_detections[m_idx]["bantai_category"]
                    category_breakdown[bcat] = category_breakdown.get(bcat, 0) + 1

            # D-112 Rule: model_score is float only for k = 1; strictly null for k >= 2
            score_val = round(float(reg["model_score"]), 4) if (k == 1 and reg.get("model_score") is not None) else None

            formatted_regions.append({
                "region_id": int(reg.get("group_id", 0)) + 1,
                "category": cat,
                "display_label": reg.get("label", cat.replace("_", " ").title()),
                "item_count": k,
                "bbox": [round(float(c), 2) for c in reg.get("bbox", [0, 0, 0, 0])],
                "model_score": score_val,
                "member_indices": member_indices,
                "category_breakdown": category_breakdown,
            })

        # Summary counts
        s = grouping.get("summary", {})
        category_counts = {
            "Plastic": sum(1 for d in formatted_detections if d["bantai_category"] == "Plastic"),
            "Paper": sum(1 for d in formatted_detections if d["bantai_category"] == "Paper"),
            "Metal": sum(1 for d in formatted_detections if d["bantai_category"] == "Metal"),
            "Potentially Hazardous": sum(1 for d in formatted_detections if d["bantai_category"] == "Potentially Hazardous"),
            "Organic": sum(1 for d in formatted_detections if d["bantai_category"] == "Organic"),
        }

        return {
            "model_version": "BANTAI-Stage5-D095-Soup5050",
            "total_items": len(formatted_detections),
            "regions": {
                "garbage_piles": s.get("garbage_pile_count", sum(1 for r in formatted_regions if r["category"] == "garbage_pile")),
                "groups_of_litter": s.get("group_of_individual_litter_count", sum(1 for r in formatted_regions if r["category"] == "group_of_individual_litter")),
                "individual_items": s.get("individual_litter_count", sum(1 for r in formatted_regions if r["category"] == "individual_litter")),
            },
            "items_by_bantai_category": category_counts,
            "region_list": formatted_regions,
            "detections": formatted_detections,
        }

    def _run_empty_findings(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Returns clean empty findings when no detections are made or in error states."""
        return {
            "model_version": "BANTAI-Stage5-D095-Soup5050",
            "total_items": 0,
            "regions": {
                "garbage_piles": 0,
                "groups_of_litter": 0,
                "individual_items": 0,
            },
            "items_by_bantai_category": {
                "Plastic": 0,
                "Paper": 0,
                "Metal": 0,
                "Potentially Hazardous": 0,
                "Organic": 0,
            },
            "region_list": [],
            "detections": [],
        }
