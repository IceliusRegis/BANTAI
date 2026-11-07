"""Unit tests for BANTAI RF-DETR Nano 15K Pothole Workflow and Video Engine.

Does not require a GPU or network.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import rfdetr_pothole_15k_workflow as w

HERE = Path(__file__).resolve().parent


class DummyTensor:
    """Mock tensor for PyTorch tensor slicing tests without requiring PyTorch install."""
    def __init__(self, data):
        self.data = data
        self.shape = (len(data),) if not hasattr(data[0], '__len__') else (len(data), len(data[0]))

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return DummyTensor([self.data[i] for i in indices])
        return self.data[indices]

    def contiguous(self):
        return self

    def clone(self):
        return DummyTensor(list(self.data))

    def tolist(self):
        return self.data


class Pothole15KWorkflowTests(unittest.TestCase):

    def test_head_surgery_slicing_5_classes(self):
        # 5 rows: 0, 1, 2 = cracks, 3 = pothole, 4 = background
        weights = DummyTensor([[float(r)] * 8 for r in range(5)])
        bias = DummyTensor([float(r) for r in range(5)])
        state_dict = {
            'class_embed.0.weight': weights,
            'class_embed.0.bias': bias,
            'backbone.layer.weight': DummyTensor([[1.0] * 8]),
        }
        new_state, modified, orig_rows, new_rows = w.perform_head_surgery(state_dict)
        self.assertEqual(modified, 2)
        self.assertEqual(orig_rows, 5)
        self.assertEqual(new_rows, 2)
        # Verify num_classes offset preservation (both 4->1 and 5->2)
        self.assertEqual(w._compute_target_num_classes(4, orig_rows, new_rows), 1)
        self.assertEqual(w._compute_target_num_classes(5, orig_rows, new_rows), 2)

        # Sliced to 2 rows: pothole (row 3) and background (row 4)
        sliced_w = new_state['class_embed.0.weight']
        self.assertEqual(sliced_w.shape[0], 2)
        self.assertEqual(sliced_w.data[0], [3.0] * 8)
        self.assertEqual(sliced_w.data[1], [4.0] * 8)

        sliced_b = new_state['class_embed.0.bias']
        self.assertEqual(sliced_b.shape[0], 2)
        self.assertEqual(sliced_b.data[0], 3.0)
        self.assertEqual(sliced_b.data[1], 4.0)

        # Backbone should be untouched
        self.assertEqual(new_state['backbone.layer.weight'].shape[0], 1)

    def test_head_surgery_slicing_4_classes(self):
        # 4 rows without separate background slot (sigmoid focal loss)
        weights = DummyTensor([[float(r)] * 8 for r in range(4)])
        bias = DummyTensor([float(r) for r in range(4)])
        state_dict = {
            'enc_out_class_embed.weight': weights,
            'enc_out_class_embed.bias': bias,
        }
        new_state, modified, orig_rows, new_rows = w.perform_head_surgery(state_dict)
        self.assertEqual(modified, 2)
        self.assertEqual(orig_rows, 4)
        self.assertEqual(new_rows, 1)
        self.assertEqual(w._compute_target_num_classes(4, orig_rows, new_rows), 1)
        sliced_w = new_state['enc_out_class_embed.weight']
        self.assertEqual(sliced_w.shape[0], 1)
        self.assertEqual(sliced_w.data[0], [3.0] * 8)

    def test_foreground_pothole_predictions_filters_background(self):
        class Array:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        dets = types.SimpleNamespace(
            xyxy=Array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            class_id=Array([0, 1]),
            confidence=Array([0.85, 0.99]),
            data={'class_name': ['pothole', '__background__']},
        )
        preds, bg_filtered = w.foreground_pothole_predictions(dets, {'id': 1})
        self.assertEqual(len(preds), 1)
        self.assertEqual(bg_filtered, 1)
        self.assertEqual(preds[0]['category_id'], 0)
        self.assertEqual(preds[0]['score'], 0.85)

    def test_single_class_annotation_filtering_and_remapping(self):
        objects = {
            'bbox': [[10, 10, 50, 50], [20, 20, 30, 30], [5, 5, 10, 10]],
            'categories': [0, 3, 1],  # crack 0, pothole 3, crack 1
        }
        anns = w.validate_objects_single_class(objects, width=100, height=100)
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0]['category_id'], 0)
        self.assertEqual(anns[0]['bbox'], [20, 20, 30, 30])

    def test_crack_only_image_becomes_empty_hard_negative(self):
        objects = {
            'bbox': [[10, 10, 50, 50], [5, 5, 10, 10]],
            'categories': [0, 1],  # cracks only, no pothole
        }
        anns = w.validate_objects_single_class(objects, width=100, height=100)
        self.assertEqual(len(anns), 0)

    def test_pothole_enriched_sampling_balance(self):
        records = []
        for i in range(100):
            if i < 40:
                cats = [3]  # positive pothole
            elif i < 70:
                cats = [0, 1]  # crack hard-negative
            else:
                cats = []  # clean background
            records.append({'path': f'img_{i:03d}.jpg', 'objects': {'bbox': [[1, 1, 10, 10]], 'categories': cats}})

        selected = w.select_pothole_enriched_records(records, target_count=50, seed=42)
        self.assertEqual(len(selected), 50)
        pos_count = sum(3 in r['objects']['categories'] for r in selected)
        crack_count = sum(3 not in r['objects']['categories'] and len(r['objects']['categories']) > 0 for r in selected)
        bg_count = sum(len(r['objects']['categories']) == 0 for r in selected)
        self.assertGreaterEqual(pos_count, 25)
        self.assertGreaterEqual(crack_count, 10)
        self.assertGreaterEqual(bg_count, 5)

    def test_prepare_15k_dataset_quarantines_duplicates_and_keeps_contiguous_ids(self):
        class FakeImage:
            size = (100, 100)
            def __init__(self, path): self.path = Path(path)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def load(self): pass
            def getexif(self): return {274: 6} if 'rot' in self.path.name else {}

        image_api = types.SimpleNamespace(open=lambda path: FakeImage(path))
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp) / 'root'
            cache = root / 'download_cache'
            backup = Path(temp) / 'backup'
            for split in ('train', 'valid', 'test'):
                folder = cache / 'data/images' / split / 'shard_000'
                folder.mkdir(parents=True)
                records = []
                for i in range(10):
                    name = f'India_{split}_{i}{"_rot" if i == 1 else ""}.jpg'
                    labels = [3] if i < 6 else [0]
                    records.append({
                        'file_name': name,
                        'objects': {'bbox': [[1, 2, 10, 10]] * len(labels), 'categories': labels},
                    })
                    # Make i==2 an exact byte duplicate of i==0 in train
                    content = f'{split}_0'.encode() if (split == 'train' and i == 2) else f'{split}_{i}'.encode()
                    (folder / name).write_bytes(content)
                (folder / 'metadata.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            hf = types.SimpleNamespace(
                snapshot_download=lambda *args, **kwargs: str(cache),
                hf_hub_download=lambda *args, **kwargs: str(cache / kwargs['filename']),
            )
            with patch.dict('sys.modules', {'huggingface_hub': hf, 'PIL': types.SimpleNamespace(Image=image_api)}):
                prepared, manifest = w.prepare_15k_dataset(
                    root, backup, {'train': 10, 'valid': 10, 'test': 10}, 42, ('India',), max_workers=2
                )
                train_coco = w.read_json(prepared / 'train' / '_annotations.coco.json')
                # 10 requested minus 1 rotated minus 1 duplicate = 8 usable images with contiguous IDs 0..7
                self.assertEqual(len(train_coco['images']), 8)
                self.assertEqual([img['id'] for img in train_coco['images']], list(range(8)))
                exclusions = w.read_json(backup / 'data_exclusions.json')
                self.assertGreaterEqual(len(exclusions), 2)

    def test_box_matching_and_calibration(self):
        coco = {
            'annotations': [
                {'image_id': 1, 'category_id': 0, 'bbox': [10, 10, 20, 20]},
                {'image_id': 2, 'category_id': 0, 'bbox': [50, 50, 30, 30]},
            ]
        }
        rows = [
            {'image_id': 1, 'predictions': [{'score': 0.85, 'xyxy': [10, 10, 30, 30]}]},
            {'image_id': 2, 'predictions': [{'score': 0.70, 'xyxy': [50, 50, 80, 80]}]},
            {'image_id': 3, 'predictions': [{'score': 0.40, 'xyxy': [0, 0, 10, 10]}]},
        ]
        calib = w.calibrate_operating_threshold(coco, rows, iou_thresh=0.5)
        self.assertIn('optimal_confidence', calib)
        self.assertGreaterEqual(calib['best_operating_point']['f1'], 0.8)
        self.assertEqual(calib['best_operating_point']['tp'], 2)

    def test_temporal_tracker_smoothing_and_anti_flicker(self):
        tracker = w.TemporalPotholeTracker(iou_threshold=0.35, min_hits=2, max_age=3, alpha=0.5)
        f1_dets = [{'xyxy': [10.0, 10.0, 50.0, 50.0], 'score': 0.8}]
        self.assertEqual(len(tracker.update(f1_dets)), 0)

        f2_dets = [{'xyxy': [12.0, 12.0, 52.0, 52.0], 'score': 0.85}]
        confirmed_f2 = tracker.update(f2_dets)
        self.assertEqual(len(confirmed_f2), 1)
        self.assertEqual(confirmed_f2[0]['id'], 1)
        self.assertAlmostEqual(confirmed_f2[0]['xyxy'][0], 11.0, places=1)

        self.assertEqual(len(tracker.update([])), 0)
        self.assertEqual(len(tracker.tracks), 1)

        f4_dets = [{'xyxy': [11.0, 11.0, 51.0, 51.0], 'score': 0.9}]
        confirmed_f4 = tracker.update(f4_dets)
        self.assertEqual(len(confirmed_f4), 1)
        self.assertEqual(confirmed_f4[0]['id'], 1)

    def test_generated_detector_script_syntax(self):
        code_str = w.generate_ready_pothole_detector_script('test_weights.pth', 0.25)
        tree = ast.parse(code_str)
        self.assertIsInstance(tree, ast.Module)
        self.assertIn('class PotholeDetector', code_str)
        self.assertIn('def run_webcam', code_str)
        self.assertIn('def run_video', code_str)

    def test_notebook_generation_and_integrity(self):
        notebook_path = HERE / 'BANTAI_RFDETR_Pothole_15K_ReadyModel.ipynb'
        self.assertTrue(notebook_path.exists())
        nb = w.read_json(notebook_path)
        self.assertEqual(nb['nbformat'], 4)
        self.assertEqual(len({c['id'] for c in nb['cells']}), len(nb['cells']))
        self.assertGreaterEqual(
            sum(c['source'].startswith('### ') and 'troubleshooting slot' in c['source'] for c in nb['cells']),
            8,
        )
        all_code = '\n'.join(c['source'] for c in nb['cells'])
        self.assertIn('run_video', all_code)
        self.assertIn('initColabWebcam', all_code)
        self.assertIn('TRAIN_IMAGES = 12000', all_code)
        self.assertIn('convert_checkpoint_to_single_class_pothole', all_code)


if __name__ == '__main__':
    unittest.main()
