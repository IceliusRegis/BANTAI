"""End-to-End verification test on Motion_Extract_Testing_2.mp4 and For_testing.jpg.

Verifies:
1. Health endpoint reports model_a_trash ready and model_b disabled.
2. Motion_Extract_Testing_2.mp4 returns exactly 3 frames (t=4.0s, 5.0s, 6.0s),
   each with 4 Paper detections and 1 Group of Individual Litter (4 items).
3. For_testing.jpg returns 1 frame (5136x3984) with 6 detections and 3 regions:
   - Top: Plastic Individual Litter
   - Middle: Group of Individual Litter (4 items)
   - Bottom: Potentially Hazardous Individual Litter (cigarette butt)
"""
import asyncio
import io
import os
from pathlib import Path
import sys
import unittest

BANTAI_ROOT = Path(__file__).resolve().parent.parent
MODELING_ROOT = Path(r"c:\Users\Justine\Documents\Python\Project Image-object Detection")
sys.path.insert(0, str(MODELING_ROOT))
sys.path.insert(0, str(BANTAI_ROOT))

from starlette.testclient import TestClient
from server.app import app


class TestEndToEndVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.video_path = MODELING_ROOT / "data/input/Motion_Extract_Testing_2.mp4"
        cls.still_path = Path(r"C:\Users\Justine\Downloads\For_testing.jpg")
        cls.pothole_path = Path(r"C:\Users\Justine\Downloads\test#3.jpg")

    def test_01_health_check(self):
        """GET /api/v1/health returns both Model A and Model B ready."""
        resp = self.client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_a_trash"]["fully_ready"])
        self.assertEqual(data["model_b_pothole"]["status"], "ready")
        self.assertTrue(data["model_b_pothole"]["fully_ready"])
        self.assertTrue(data["model_b_pothole"]["weights_present"])
        print("\n[OK] Health check verified: Both Model A (Trash) and Model B (Pothole Cell 8A) are ready.")

    def test_02_video_motion_extract_testing_2(self):
        """POST /api/v1/analyze with Motion_Extract_Testing_2.mp4 returns only the 3 paper towel frames."""
        if not self.video_path.is_file():
            self.skipTest(f"Video file not found at {self.video_path}")

        print(f"\nAnalyzing video: {self.video_path}...")
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/v1/analyze",
                data={"location": "J. P. Rizal St. near Market", "category": "Trash", "sample_fps": "1.0"},
                files={"file": ("Motion_Extract_Testing_2.mp4", f, "video/mp4")},
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()

        # Check source media
        self.assertEqual(data["source_media"]["type"], "video")

        # Verify ONLY the 3 frames (t=4.0s, 5.0s, 6.0s) are returned
        frames = data["frames"]
        self.assertEqual(len(frames), 3, f"Expected exactly 3 frames, got {len(frames)}")

        timestamps = [round(f["timestamp_sec"], 1) for f in frames]
        self.assertEqual(timestamps, [4.0, 5.0, 6.0], f"Expected [4.0, 5.0, 6.0], got {timestamps}")

        # Each frame must contain 4 Paper detections and 1 Group of Individual Litter (4 items)
        for f in frames:
            self.assertEqual(f["road_defect_findings"]["model_version"], "skipped_by_category")
            tf = f["trash_findings"]
            self.assertEqual(tf["total_items"], 4)
            self.assertEqual(tf["regions"]["groups_of_litter"], 1)
            self.assertEqual(tf["regions"]["garbage_piles"], 0)
            self.assertEqual(tf["regions"]["individual_items"], 0)

            # All 4 detections must be Paper
            for det in tf["detections"]:
                self.assertEqual(det["bantai_category"], "Paper")

            # Region must be Group of Individual Litter (4 items) with null model_score
            reg = tf["region_list"][0]
            self.assertEqual(reg["category"], "group_of_individual_litter")
            self.assertEqual(reg["item_count"], 4)
            self.assertIsNone(reg["model_score"])
            self.assertEqual(reg["category_breakdown"]["Paper"], 4)

        print(f"[OK] Motion_Extract_Testing_2.mp4 verified: Exactly 3 frames ({timestamps}) with 4 Paper detections in 1 Group of Individual Litter.")

    def test_03_still_image_for_testing(self):
        """POST /api/v1/analyze with For_testing.jpg returns 1 frame with 6 items in 3 distinct regions."""
        if not self.still_path.is_file():
            self.skipTest(f"Still image not found at {self.still_path}")

        print(f"\nAnalyzing still image: {self.still_path}...")
        with open(self.still_path, "rb") as f:
            resp = self.client.post(
                "/api/v1/analyze",
                data={"location": "Barangay Santa Elena Curb", "category": "Trash", "sample_fps": "1.0"},
                files={"file": ("For_testing.jpg", f, "image/jpeg")},
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()

        # Check source media
        self.assertEqual(data["source_media"]["type"], "image")

        # Verify exactly 1 frame
        frames = data["frames"]
        self.assertEqual(len(frames), 1)

        f0 = frames[0]
        self.assertEqual(f0["road_defect_findings"]["model_version"], "skipped_by_category")
        self.assertEqual(f0["width"], 5136)
        self.assertEqual(f0["height"], 3984)

        tf = f0["trash_findings"]
        self.assertEqual(tf["total_items"], 6, f"Expected 6 detections, got {tf['total_items']}")
        self.assertEqual(tf["regions"]["groups_of_litter"], 1)
        self.assertEqual(tf["regions"]["garbage_piles"], 0)
        self.assertEqual(tf["regions"]["individual_items"], 2)

        # Verify the 3 vertical regions
        regs = tf["region_list"]
        self.assertEqual(len(regs), 3)

        # 1. Middle: Group of Individual Litter (4 items)
        grp = next(r for r in regs if r["category"] == "group_of_individual_litter")
        self.assertEqual(grp["item_count"], 4)
        self.assertIsNone(grp["model_score"])
        self.assertTrue(1500 <= grp["bbox"][1] <= 2500)

        # 2. Top: Individual Litter (Plastic, y < 1000)
        top_ind = next(r for r in regs if r["category"] == "individual_litter" and r["bbox"][1] < 1000)
        self.assertEqual(top_ind["item_count"], 1)
        self.assertIsNotNone(top_ind["model_score"])
        self.assertIn("Plastic", top_ind["category_breakdown"])

        # 3. Bottom: Individual Litter (Potentially Hazardous cigarette butt, y > 3000)
        bot_ind = next(r for r in regs if r["category"] == "individual_litter" and r["bbox"][1] > 3000)
        self.assertEqual(bot_ind["item_count"], 1)
        self.assertIsNotNone(bot_ind["model_score"])
        self.assertIn("Potentially Hazardous", bot_ind["category_breakdown"])

        print("[OK] For_testing.jpg verified: Exactly 1 frame (5136x3984), 6 detections, 3 regions (Top Plastic, Middle Group of 4, Bottom Potentially Hazardous).")

    def test_04_pothole_category_selective_routing(self):
        """POST /api/v1/analyze with category='Pothole' routes to Model B and skips Model A."""
        if not self.pothole_path.is_file():
            self.skipTest(f"Pothole image not found at {self.pothole_path}")

        print(f"\nAnalyzing pothole image: {self.pothole_path}...")
        with open(self.pothole_path, "rb") as f:
            resp = self.client.post(
                "/api/v1/analyze",
                data={"location": "Shoe Avenue", "category": "Pothole", "sample_fps": "1.0"},
                files={"file": ("test3.jpg", f, "image/jpeg")},
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()

        # Check source media
        self.assertEqual(data["source_media"]["type"], "image")

        frames = data["frames"]
        self.assertEqual(len(frames), 1)

        f0 = frames[0]
        # Trash Model A must be skipped
        self.assertEqual(f0["trash_findings"]["model_version"], "skipped_by_category")
        self.assertEqual(f0["trash_findings"]["total_items"], 0)

        # Pothole Model B must be active
        self.assertEqual(f0["road_defect_findings"]["model_version"], "BANTAI-Pothole-RFDETR-Nano-v5_20261107_061616")
        self.assertGreaterEqual(f0["road_defect_findings"]["total_defects"], 1)

        defects = f0["road_defect_findings"]["defects"]
        print(f"[OK] Pothole category routing verified: Model A skipped, Model B detected {len(defects)} pothole(s).")


if __name__ == "__main__":
    unittest.main()
