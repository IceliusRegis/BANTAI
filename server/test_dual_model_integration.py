"""Unit test suite for BANTAI dual-model integration and contracts.

Verifies:
1. Single-pass media ingestion contracts (image staging & frame entry schemas)
2. Model A D-112 Spatial Grouping schema contracts:
   - k = 1 -> individual_litter with float model_score
   - 2 <= k <= 7 -> group_of_individual_litter with model_score = None and category_breakdown
   - k >= 8 -> garbage_pile with model_score = None and category_breakdown
3. Model B Pothole detector schema contracts ([x, y, w, h] conversion, estimated_diameter_cm)
4. Section 6.3 Hotspot Prioritization score formula and priority level thresholds
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from server.media_ingestion import stage_single_image, sha256_file
from server.trash_service import BantaiTrashService, SUPERVISED_LABELS, CANONICAL_CATEGORIES
from server.pothole_service import BantaiPotholeService
from server.fusion import calculate_hotspot_score, determine_priority_and_dispatch, fuse_dual_model_incident


class TestDualModelIntegration(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_single_pass_media_ingestion(self):
        """Test staging single still image into shared extracted_frames/."""
        test_img_path = self.tmp_path / "test_street.jpg"
        im = Image.new("RGB", (640, 480), color=(128, 128, 128))
        im.save(test_img_path)

        frames_dir = self.tmp_path / "extracted_frames"
        meta, entries = stage_single_image(test_img_path, frames_dir, start_id=10000)

        self.assertEqual(meta["type"], "image")
        self.assertEqual(meta["width"], 640)
        self.assertEqual(meta["height"], 480)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], 10000)
        self.assertEqual(entries[0]["width"], 640)
        self.assertEqual(entries[0]["height"], 480)
        self.assertTrue(Path(entries[0]["file_path"]).is_file())

    def test_trash_service_d112_cardinality_and_breakdown(self):
        """Verify D-112 Disjoint Spatial Grouping null-score rules for multi-item regions."""
        service = BantaiTrashService()

        # Case 1: Individual Litter (k = 1) -> float score
        single_entry = {"id": 10000, "width": 1000, "height": 1000}
        single_dets = [{"category_id": 1, "score": 0.88, "bbox": [100, 100, 50, 50]}]
        single_grouping = {
            "regions": [
                {
                    "group_id": 0,
                    "category": "individual_litter",
                    "label": "Individual Litter",
                    "bbox": [100, 100, 50, 50],
                    "item_count": 1,
                    "model_score": 0.88,
                    "member_indices": [0],
                }
            ],
            "summary": {"individual_litter_count": 1, "group_of_individual_litter_count": 0, "garbage_pile_count": 0},
        }
        f1 = service._format_frame_findings(single_entry, single_dets, single_grouping)
        self.assertEqual(f1["total_items"], 1)
        self.assertEqual(f1["regions"]["individual_items"], 1)
        self.assertEqual(f1["region_list"][0]["category"], "individual_litter")
        self.assertEqual(f1["region_list"][0]["model_score"], 0.88)

        # Case 2: Group of Individual Litter (k = 3) -> model_score must be None, category_breakdown populated
        group_dets = [
            {"category_id": 1, "score": 0.85, "bbox": [100, 100, 40, 40]},  # Plastic
            {"category_id": 8, "score": 0.72, "bbox": [120, 110, 30, 30]},  # Paper
            {"category_id": 5, "score": 0.68, "bbox": [115, 130, 35, 35]},  # Metal
        ]
        group_grouping = {
            "regions": [
                {
                    "group_id": 0,
                    "category": "group_of_individual_litter",
                    "label": "Group of Individual Litter",
                    "bbox": [100, 100, 60, 65],
                    "item_count": 3,
                    "model_score": None,
                    "member_indices": [0, 1, 2],
                }
            ],
            "summary": {"individual_litter_count": 0, "group_of_individual_litter_count": 1, "garbage_pile_count": 0},
        }
        f2 = service._format_frame_findings(single_entry, group_dets, group_grouping)
        reg = f2["region_list"][0]
        self.assertEqual(reg["category"], "group_of_individual_litter")
        self.assertIsNone(reg["model_score"])  # Strict D-112 rule: Null for multi-item regions
        self.assertEqual(reg["item_count"], 3)
        self.assertEqual(reg["category_breakdown"]["Plastic"], 1)
        self.assertEqual(reg["category_breakdown"]["Paper"], 1)
        self.assertEqual(reg["category_breakdown"]["Metal"], 1)

        # Case 3: Garbage Pile (k = 8) -> model_score must be None
        pile_dets = [{"category_id": 1, "score": 0.80, "bbox": [100 + i*5, 100, 20, 20]} for i in range(8)]
        pile_grouping = {
            "regions": [
                {
                    "group_id": 0,
                    "category": "garbage_pile",
                    "label": "Garbage Pile",
                    "bbox": [100, 100, 100, 60],
                    "item_count": 8,
                    "model_score": None,
                    "member_indices": list(range(8)),
                }
            ],
            "summary": {"individual_litter_count": 0, "group_of_individual_litter_count": 0, "garbage_pile_count": 1},
        }
        f3 = service._format_frame_findings(single_entry, pile_dets, pile_grouping)
        reg3 = f3["region_list"][0]
        self.assertEqual(reg3["category"], "garbage_pile")
        self.assertIsNone(reg3["model_score"])
        self.assertEqual(reg3["item_count"], 8)
        self.assertEqual(reg3["category_breakdown"]["Plastic"], 8)

    def test_pothole_service_defects_formatting(self):
        """Verify Model B road defect schema and [x,y,w,h] conversion."""
        service = BantaiPotholeService()
        raw_dets = [
            {"xyxy": [100.0, 150.0, 250.0, 230.0], "score": 0.892, "severity": "Critical"},
        ]
        res = service._format_defects(raw_dets, img_w=1000, img_h=1000)
        self.assertEqual(res["total_defects"], 1)
        d = res["defects"][0]
        self.assertEqual(d["type"], "pothole")
        self.assertEqual(d["severity"], "high")
        self.assertEqual(d["bbox"], [100.0, 150.0, 150.0, 80.0])  # [x, y, w, h]
        self.assertEqual(d["confidence"], 0.892)
        self.assertGreater(d["estimated_diameter_cm"], 0)
        self.assertEqual(d["confidence_pct"], 89)
        self.assertIn("box_pct", d)
        self.assertFalse(hasattr(service, "_mock_defects"))
        self.assertTrue(service.is_weights_present())
        self.assertTrue(service.is_fully_available())

    def test_hotspot_prioritization_formula(self):
        """Verify calibrated Section 6.3 Hotspot formula."""
        # 1 pile (35), 1 group (15), 2 litter (6), 1 high pothole (40) -> 35 + 15 + 6 + 40 = 96.0
        score = calculate_hotspot_score(
            piles=1,
            groups=1,
            litter=2,
            potholes_by_severity={"high": 1, "moderate": 0, "low": 0},
        )
        self.assertEqual(score, 96.0)
        priority, dispatch = determine_priority_and_dispatch(score, has_pile=True, has_high_pothole=True)
        self.assertEqual(priority, "CRITICAL")
        self.assertIn("garbage pile hazard", dispatch)
        self.assertIn("severe pothole hazard", dispatch)

        # Elevated case: 1 group (15) + 1 low pothole (25) = 40.0
        score2 = calculate_hotspot_score(
            piles=0,
            groups=1,
            litter=0,
            potholes_by_severity={"low": 1},
        )
        self.assertEqual(score2, 40.0)
        priority2, _ = determine_priority_and_dispatch(score2, has_pile=False, has_high_pothole=False)
        self.assertEqual(priority2, "ELEVATED")

        # Routine case: 1 litter (3.0)
        score3 = calculate_hotspot_score(
            piles=0,
            groups=0,
            litter=1,
            potholes_by_severity={},
        )
        self.assertEqual(score3, 3.0)
        priority3, _ = determine_priority_and_dispatch(score3, has_pile=False, has_high_pothole=False)
        self.assertEqual(priority3, "ROUTINE")


if __name__ == "__main__":
    unittest.main()
