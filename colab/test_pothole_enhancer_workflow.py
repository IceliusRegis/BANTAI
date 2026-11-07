"""Unit tests for BANTAI Pothole Model Enhancer Workflow.

Runs 100% offline without requiring a GPU or network.
"""
from __future__ import annotations

import ast
import unittest
import numpy as np
from PIL import Image

import pothole_enhancer_workflow as w


class PotholeEnhancerWorkflowTests(unittest.TestCase):

    def test_clahe_contrast_and_shape(self):
        # Create a synthetic dark image with a shadowed patch
        raw_arr = np.full((120, 160, 3), 40, dtype=np.uint8)
        # Add a subtle internal variation inside a pothole cavity
        raw_arr[50:80, 60:100] = 20
        raw_arr[55:75, 65:95] = 25
        img = Image.fromarray(raw_arr)

        enhanced = w.apply_lab_clahe(img)
        self.assertEqual(enhanced.size, (160, 120))
        self.assertEqual(enhanced.mode, "RGB")
        enh_arr = np.array(enhanced)
        # Verify that contrast is preserved and valid
        self.assertGreaterEqual(enh_arr.max(), raw_arr.max())

    def test_sahi_tile_coordinate_remapping(self):
        img = Image.new("RGB", (640, 480), color=(100, 100, 100))
        tiles = w.generate_road_sahi_tiles(img, horizon_ratio=0.35, grid=(2, 2), overlap=0.20, include_focal_tile=False)
        self.assertEqual(len(tiles), 4)

        for tile_img, offset in tiles:
            off_x, off_y, tw, th = offset
            self.assertGreaterEqual(off_y, int(480 * 0.35))
            self.assertLessEqual(off_x + tw, 640)
            self.assertLessEqual(off_y + th, 480)

            # Test local box remapping
            local_dets = [{"xyxy": [10.0, 15.0, 50.0, 45.0], "score": 0.88, "class_name": "pothole"}]
            remapped = w.remap_tile_boxes(local_dets, offset)
            self.assertEqual(len(remapped), 1)
            self.assertEqual(remapped[0]["xyxy"][0], 10.0 + off_x)
            self.assertEqual(remapped[0]["xyxy"][1], 15.0 + off_y)
            self.assertEqual(remapped[0]["xyxy"][2], 50.0 + off_x)
            self.assertEqual(remapped[0]["xyxy"][3], 45.0 + off_y)
            self.assertEqual(remapped[0]["source"], "sahi_tile")

    def test_wbf_box_merging_and_score_boost(self):
        # Two slightly offset detections on the same pothole
        det1 = {"xyxy": [100.0, 200.0, 150.0, 250.0], "score": 0.80, "class_name": "pothole"}
        det2 = {"xyxy": [104.0, 202.0, 152.0, 248.0], "score": 0.85, "class_name": "pothole"}
        # One distant separate detection
        det3 = {"xyxy": [400.0, 300.0, 430.0, 330.0], "score": 0.70, "class_name": "pothole"}

        fused = w.weighted_boxes_fusion([det1, det2, det3], iou_threshold=0.55)
        # 3 raw candidates should merge into 2 fused boxes
        self.assertEqual(len(fused), 2)

        # First cluster merged det1 and det2
        c1 = fused[0]
        self.assertEqual(c1["cluster_count"], 2)
        # Coordinate should be between [100, 200] and [104, 202]
        self.assertGreaterEqual(c1["xyxy"][0], 100.0)
        self.assertLessEqual(c1["xyxy"][0], 104.0)
        # Confidence should be boosted due to multi-candidate agreement
        self.assertGreater(c1["score"], 0.85)

        # Second cluster should retain det3
        c2 = fused[1]
        self.assertEqual(c2["cluster_count"], 1)
        self.assertEqual(c2["score"], 0.70)
        self.assertEqual(c2["xyxy"], [400.0, 300.0, 430.0, 330.0])

    def test_horizon_and_stripe_filtering(self):
        w_img, h_img = 1000, 800
        # 1. Sky/Horizon detection (cy = 150 < 800*0.35=280) -> should be filtered
        det_sky = {"xyxy": [100.0, 100.0, 200.0, 200.0], "score": 0.90}
        # 2. Road stripe with extreme aspect ratio (w=650, h=50 -> aspect=13 > 6) -> should be filtered
        det_stripe = {"xyxy": [100.0, 400.0, 750.0, 450.0], "score": 0.85}
        # 3. Valid pothole on the road (cy = 500, w=100, h=80 -> aspect=1.25) -> should be retained
        det_pothole = {"xyxy": [300.0, 460.0, 400.0, 540.0], "score": 0.88}

        filtered = w.filter_horizon_and_aspect_ratio(
            [det_sky, det_stripe, det_pothole],
            img_width=w_img,
            img_height=h_img,
            horizon_ratio=0.35,
            max_aspect_ratio=6.0,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["xyxy"], [300.0, 460.0, 400.0, 540.0])

    def test_laplacian_edge_roughness_shadow_filter(self):
        # 1. Smooth flat shadow patch: uniform gray gradient
        smooth_gray = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            smooth_gray[i, :] = int(50 + i * 0.3)
        shadow_box = [20.0, 20.0, 80.0, 80.0]
        smooth_roughness = w.compute_laplacian_edge_roughness(smooth_gray, shadow_box)

        # 2. Textured cracked pothole patch: random noisy fractured surface
        rough_gray = np.random.RandomState(42).randint(20, 180, (100, 100)).astype(np.uint8)
        pothole_box = [20.0, 20.0, 80.0, 80.0]
        rough_roughness = w.compute_laplacian_edge_roughness(rough_gray, pothole_box)

        self.assertLess(smooth_roughness, 5.0)
        self.assertGreater(rough_roughness, 50.0)

        # Verify filtering helper
        pil_smooth = Image.fromarray(smooth_gray)
        dets = [
            {"xyxy": shadow_box, "score": 0.55, "class_name": "pothole"},  # borderline shadow
            {"xyxy": shadow_box, "score": 0.95, "class_name": "pothole"},  # very high confidence preserved
        ]
        verified = w.verify_pothole_texture_and_geometry(pil_smooth, dets, min_roughness_var=10.0)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["score"], 0.95)

    def test_severity_calculation(self):
        img_w, img_h = 1000, 1000
        # Critical: large area (>3.5%)
        crit_box = [100.0, 200.0, 300.0, 400.0]  # area = 200*200 = 40,000 (4%)
        self.assertEqual(w.estimate_pothole_severity(crit_box, img_w, img_h, 0.85), "Critical")

        # Moderate: medium area (1.5%)
        mod_box = [100.0, 200.0, 250.0, 300.0]  # area = 150*100 = 15,000 (1.5%)
        self.assertEqual(w.estimate_pothole_severity(mod_box, img_w, img_h, 0.80), "Moderate")

        # Low: small distant pothole (<1%)
        low_box = [100.0, 200.0, 150.0, 240.0]  # area = 50*40 = 2,000 (0.2%)
        self.assertEqual(w.estimate_pothole_severity(low_box, img_w, img_h, 0.65), "Low")

    def test_temporal_tracker_smoothing_and_anti_flicker(self):
        tracker = w.TemporalPotholeTracker(iou_threshold=0.35, min_hits=2, max_age=3, alpha=0.5)

        # Frame 1: Initial detection -> should not confirm yet (hits = 1 < min_hits = 2)
        f1_dets = [{"xyxy": [100.0, 200.0, 150.0, 250.0], "score": 0.85, "severity": "Moderate"}]
        tracked_f1 = tracker.update(f1_dets)
        self.assertEqual(len(tracked_f1), 0)

        # Frame 2: Same pothole slightly shifted -> hits = 2 -> confirmed & smoothed!
        f2_dets = [{"xyxy": [104.0, 204.0, 154.0, 254.0], "score": 0.88, "severity": "Moderate"}]
        tracked_f2 = tracker.update(f2_dets)
        self.assertEqual(len(tracked_f2), 1)
        item = tracked_f2[0]
        self.assertEqual(item["id"], 1)
        # Smoothed coordinate with alpha=0.5: 0.5*104 + 0.5*100 = 102.0
        self.assertAlmostEqual(item["xyxy"][0], 102.0, places=1)
        self.assertAlmostEqual(item["xyxy"][1], 202.0, places=1)

        # Frame 3: Transient false positive flicker at different position
        f3_dets = [{"xyxy": [500.0, 500.0, 550.0, 550.0], "score": 0.50}]
        tracked_f3 = tracker.update(f3_dets)
        # The original pothole had no update so confirmed is empty, and new flicker is hits=1
        self.assertEqual(len(tracked_f3), 0)

    def test_wbf_no_boost_for_low_confidence_noise(self):
        # Two weak background noise detections (< 0.25)
        det1 = {"xyxy": [100.0, 200.0, 150.0, 250.0], "score": 0.18, "class_name": "pothole"}
        det2 = {"xyxy": [102.0, 202.0, 152.0, 252.0], "score": 0.19, "class_name": "pothole"}
        fused = w.weighted_boxes_fusion([det1, det2], iou_threshold=0.55)
        self.assertEqual(len(fused), 1)
        # Score must NOT be boosted above base score when base < 0.25
        self.assertEqual(fused[0]["score"], 0.19)

    def test_strict_load_fails_on_missing_checkpoint(self):
        with self.assertRaises(RuntimeError):
            w.EnhancedPotholeDetector(weights_path="nonexistent_checkpoint.pth", strict_load=True)
        # With strict_load=False it logs a note and sets detector=None for mock tests
        det = w.EnhancedPotholeDetector(weights_path="nonexistent_checkpoint.pth", strict_load=False)
        self.assertIsNone(det.detector)

    def test_generated_detector_script_syntax_and_sahi(self):
        code_str = w.generate_enhanced_detector_script("test_model.pth", 0.45)
        # Ensure that generated standalone script parses with valid Python syntax
        parsed = ast.parse(code_str)
        self.assertIsNotNone(parsed)
        self.assertIn("class PotholeDetector:", code_str)
        self.assertIn("def apply_lab_clahe", code_str)
        self.assertIn("def generate_road_sahi_tiles", code_str)
        self.assertIn("def remap_tile_boxes", code_str)
        self.assertIn("def compute_laplacian_edge_roughness", code_str)
        self.assertIn('mode == "photo"', code_str)
        self.assertIn("include_focal_tile=True", code_str)

    def test_focal_tile_sahi_generation(self):
        img = Image.new("RGB", (800, 600), color=(90, 90, 90))
        tiles_with_focal = w.generate_road_sahi_tiles(img, horizon_ratio=0.25, grid=(2, 2), overlap=0.25, include_focal_tile=True)
        self.assertEqual(len(tiles_with_focal), 5)
        focal_img, focal_offset = tiles_with_focal[-1]
        self.assertEqual(focal_offset[0], int(800 * 0.15))
        self.assertEqual(focal_img.size, (focal_offset[2], focal_offset[3]))

    def test_augment_all_six_pothole_types(self):
        base_arr = np.full((300, 400, 3), 85, dtype=np.uint8)
        base_img = Image.fromarray(base_arr)
        boxes = [[100.0, 120.0, 80.0, 60.0]]
        seen_domains = set()

        for idx in range(6):
            aug_img, aug_boxes, domain = w.augment_all_pothole_types(base_img, boxes, variant_idx=idx, seed=42)
            seen_domains.add(domain)
            self.assertEqual(aug_img.size, (400, 300))
            self.assertGreaterEqual(len(aug_boxes), 1)
            bx, by, bw, bh = aug_boxes[0]
            self.assertGreater(bw, 10.0)
            self.assertGreater(bh, 10.0)
            self.assertGreaterEqual(bx, 0.0)
            self.assertGreaterEqual(by, 0.0)

        self.assertEqual(seen_domains, set(w.POTHOLE_DOMAIN_TYPES))

    def test_water_puddle_with_rough_rim_retained(self):
        # Create an image with textured asphalt and a smooth water puddle inside the pothole box
        rng = np.random.RandomState(7)
        gray = rng.randint(45, 115, (200, 200), dtype=np.uint8)
        # Smooth water interior inside [60:140, 60:140], while rim [50:60, 140:150] has textured asphalt
        gray[65:135, 65:135] = 210
        pil_img = Image.fromarray(gray).convert("RGB")
        water_pothole_det = [{"xyxy": [60.0, 60.0, 140.0, 140.0], "score": 0.28, "class_name": "pothole"}]
        verified = w.verify_pothole_texture_and_geometry(pil_img, water_pothole_det, min_roughness_var=3.5)
        self.assertEqual(len(verified), 1)
        self.assertGreater(verified[0]["roughness_variance"], 3.5)

    def test_preserve_model_config_in_checkpoint(self):
        import tempfile
        from pathlib import Path
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "stripped.pth"
            dst = Path(tmpdir) / "preserved.pth"
            torch.save({"model": {"dummy": torch.zeros(2, 2)}}, str(src))
            meta = w.preserve_model_config_in_checkpoint(src, dst, threshold=0.20)
            self.assertTrue(meta["has_model_config"])
            loaded = torch.load(str(dst), map_location="cpu", weights_only=False)
            self.assertIn("model_config", loaded)
            self.assertEqual(loaded["class_names"], ["pothole"])
            self.assertAlmostEqual(loaded["calibrated_operating_threshold"], 0.20)


    def test_bidirectional_containment_inner_fragment_suppression(self):
        # Full crater detected by full frame with high confidence
        outer_crater = {
            "xyxy": [100.0, 300.0, 350.0, 500.0],
            "score": 0.85,
            "cluster_count": 2,
            "sources": ["full_frame", "tta_flip"],
        }
        # Fragmented quadrant sub-box from a SAHI tile inside the crater
        inner_tile_fragment = {
            "xyxy": [140.0, 340.0, 220.0, 420.0],
            "score": 0.38,
            "cluster_count": 1,
            "sources": ["sahi_tile"],
        }
        # Another independent pothole nearby
        nearby_pothole = {
            "xyxy": [450.0, 320.0, 550.0, 400.0],
            "score": 0.72,
            "cluster_count": 2,
            "sources": ["full_frame", "sahi_tile"],
        }

        filtered = w.filter_horizon_and_aspect_ratio(
            [outer_crater, inner_tile_fragment, nearby_pothole],
            img_width=1000,
            img_height=800,
            horizon_ratio=0.25,
        )
        # Inner fragment must be suppressed, leaving only outer crater and nearby pothole
        self.assertEqual(len(filtered), 2)
        kept_boxes = [f["xyxy"] for f in filtered]
        self.assertIn(outer_crater["xyxy"], kept_boxes)
        self.assertIn(nearby_pothole["xyxy"], kept_boxes)
        self.assertNotIn(inner_tile_fragment["xyxy"], kept_boxes)

    def test_universal_wiper_blade_rejection(self):
        # Horizontal wiper blade or wiper reflection across bottom frame edge
        wiper_det = {
            "xyxy": [50.0, 710.0, 600.0, 740.0],  # cy = 725 > 800*0.86=688, aspect = 550/30 = 18.3 >= 2.2
            "score": 0.40,
            "cluster_count": 1,
            "sources": ["full_frame"],
        }
        # Valid pothole in road surface
        road_pothole = {
            "xyxy": [200.0, 400.0, 300.0, 480.0],
            "score": 0.65,
            "cluster_count": 1,
            "sources": ["full_frame"],
        }

        filtered = w.filter_horizon_and_aspect_ratio(
            [wiper_det, road_pothole],
            img_width=1000,
            img_height=800,
            horizon_ratio=0.25,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["xyxy"], road_pothole["xyxy"])

    def test_wide_dashcam_vents_and_horizon_bumpers(self):
        w_img, h_img = 960, 540  # 16:9 wide dashcam
        # 1. VIOFO dashboard air-vent reflection: cy = 372 (norm_y = 0.689 > 0.64) -> Tweak 1 filters
        vent_box = {"xyxy": [390.0, 351.0, 630.0, 393.0], "score": 0.588, "cluster_count": 2, "sources": ["full_frame"]}
        # 2. Distant horizontal SUV rear bumper near highway horizon: cy = 185 (norm_y = 0.34 < 0.38, bw/bh = 2.5 >= 2.0, score = 0.368 < 0.45) -> Tweak 2 filters
        bumper_box = {"xyxy": [250.0, 175.0, 300.0, 195.0], "score": 0.368, "cluster_count": 2, "sources": ["full_frame"]}
        # 3. Real road pothole in the driving lane: cx = 460, cy = 270 (norm_y = 0.50) -> kept
        real_pothole = {"xyxy": [420.0, 245.0, 500.0, 295.0], "score": 0.55, "cluster_count": 2, "sources": ["full_frame"]}
        # 4. Single-tile SAHI pothole detection in the driving lane -> MUST BE PRESERVED (never dropped!)
        sahi_pothole = {"xyxy": [320.0, 260.0, 370.0, 295.0], "score": 0.28, "cluster_count": 1, "sources": ["sahi_tile"]}

        filtered = w.filter_horizon_and_aspect_ratio(
            [vent_box, bumper_box, real_pothole, sahi_pothole],
            img_width=w_img,
            img_height=h_img,
            horizon_ratio=0.28,
        )
        # Dashboard vent and distant SUV bumper must be filtered; real road pothole and single-tile SAHI pothole kept
        self.assertEqual(len(filtered), 2)
        kept_boxes = [f["xyxy"] for f in filtered]
        self.assertIn(real_pothole["xyxy"], kept_boxes)
        self.assertIn(sahi_pothole["xyxy"], kept_boxes)

    def test_duplicate_overlap_suppression_tweak3(self):
        # Two overlapping boxes on the same pothole (e.g. VIOFO country road 27.8% + 22.6%)
        det_high = {"xyxy": [400.0, 400.0, 480.0, 480.0], "score": 0.278, "cluster_count": 2, "sources": ["full_frame"]}
        det_low = {"xyxy": [410.0, 410.0, 485.0, 485.0], "score": 0.226, "cluster_count": 2, "sources": ["full_frame"]}
        filtered = w.filter_horizon_and_aspect_ratio([det_high, det_low], img_width=1000, img_height=800, horizon_ratio=0.28)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["score"], 0.278)

    def test_rim_aware_roughness_preserves_large_smooth_water_puddles(self):
        # Create 120x120 image with textured asphalt boundary and smooth puddle center
        rng = np.random.RandomState(42)
        gray = rng.randint(40, 160, (120, 120), dtype=np.uint8)
        # Smooth flat water pool in the center (dilutes full-crop variance, but outer rim is rough)
        gray[30:90, 30:90] = 180
        puddle_box = [15.0, 15.0, 105.0, 105.0]
        roughness = w.compute_laplacian_edge_roughness(gray, puddle_box)
        # The rim-aware calculation must register the rough outer fractured rim
        self.assertGreater(roughness, 20.0)


if __name__ == "__main__":
    unittest.main()
