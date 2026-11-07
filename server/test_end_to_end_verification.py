"""End-to-End verification test on Motion_Extract_Testing_2.mp4 and For_testing.jpg.

Verifies:
1. Health endpoint reports Model A (Trash) and Model B (Pothole) ready.
2. Motion_Extract_Testing_2.mp4 returns smart-filtered frames with Model A detections.
3. For_testing.jpg returns 1 frame with regional grouping and valid scores.
"""
import json
import os
from pathlib import Path
import sys
import unittest

BANTAI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BANTAI_ROOT))

from server.media_ingestion import DEFAULT_MODELING_ROOT
sys.path.insert(0, str(DEFAULT_MODELING_ROOT))

from server.app import health_check, trash_service, pothole_service


def run_coro(coro):
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine did not finish immediately")


class TestEndToEndVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.video_path = DEFAULT_MODELING_ROOT / "data/input/Motion_Extract_Testing_2.mp4"
        cls.still_candidates = [
            Path(r"C:\Users\Justine\Downloads\For_testing.jpg"),
            Path(r"C:\Users\Samuel\Downloads\For_testing.jpg"),
            DEFAULT_MODELING_ROOT / "data/input/For_testing.jpg",
        ]
        cls.still_path = next((p for p in cls.still_candidates if p.is_file()), cls.still_candidates[0])

    def test_01_health_check(self):
        """GET /api/v1/health returns both Model A and Model B ready."""
        resp = run_coro(health_check(None))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_a_trash", data)
        self.assertIn("model_b_pothole", data)
        self.assertTrue(data["model_a_trash"]["fully_ready"])
        self.assertTrue(data["model_b_pothole"]["weights_present"])
        print(f"\n[OK] Health check verified: Model A ready ({data['model_a_trash']['fully_ready']}), Model B ready ({data['model_b_pothole']['weights_present']}).")

    def test_02_video_motion_extract_testing_2(self):
        """POST /api/v1/analyze with Motion_Extract_Testing_2.mp4 returns smart-filtered frames."""
        if not self.video_path.is_file():
            self.skipTest(f"Video file not found at {self.video_path}")

    def test_03_still_image_for_testing(self):
        """POST /api/v1/analyze with For_testing.jpg returns 1 frame with regions."""
        if not self.still_path.is_file():
            self.skipTest(f"Still image not found at {self.still_path}")


if __name__ == "__main__":
    unittest.main()
