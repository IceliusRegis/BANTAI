"""Offline scoring/data tests. No model packages, network or GPU needed."""
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import rfdetr_workflow as w

HERE = Path(__file__).resolve().parent


class WorkflowTests(unittest.TestCase):
    def test_notebook_syntax_and_repair_slots(self):
        notebook = w.read_json(HERE / 'BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb')
        self.assertEqual(notebook['nbformat'], 4)
        self.assertEqual(len({c['id'] for c in notebook['cells']}), len(notebook['cells']))
        self.assertEqual(sum(c['source'].startswith('### ') and 'troubleshooting slot' in c['source']
                             for c in notebook['cells']), 10)

    def test_background_filter_requires_explicit_name(self):
        class Array:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        def detections(ids, names):
            return types.SimpleNamespace(xyxy=Array([[0, 0, 10, 10]] * len(ids)),
                                         class_id=Array(ids), confidence=Array([.9] * len(ids)),
                                         data={} if names is None else {'class_name': names})
        foreground, filtered = w.foreground_predictions(
            detections([0, 1, 2, 3, 4], w.CLASSES + ['__background__']), {'id': 1})
        self.assertEqual([p['category_id'] for p in foreground], [0, 1, 2, 3])
        self.assertEqual(filtered, 1)
        for ids, names in [([4], None), ([4], ['pothole']), ([5], ['__background__']),
                           ([-1], ['pothole']), ([3.5], ['pothole'])]:
            with self.assertRaises(ValueError):
                w.foreground_predictions(detections(ids, names), {'id': 1})

    def test_demo_is_embedded_and_loads_selected_checkpoint(self):
        notebook = w.read_json(HERE / 'BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb')
        demo = next(c['source'] for c in notebook['cells'] if '# Paste this entire cell' in c['source'])
        self.assertEqual(demo.strip(), (HERE / 'rfdetr_upload_demo.py').read_text().strip())
        self.assertIn("selected_candidate = validation_report['selected_by_validation']", demo)
        self.assertIn('demo_detector = load_detector(demo_weights)', demo)
        self.assertTrue(demo.startswith('#@title 8A.'))
        self.assertNotIn('final_comparison.json', demo)
        self.assertNotIn('final_test_started.json', demo)
        self.assertIn('sha256(demo_weights) == expected_sha256', demo)

    def test_diagnostic_preserves_experiment_settings(self):
        notebook = w.read_json(HERE / 'BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb')
        diagnostic = (HERE / 'rfdetr_diagnostic_demo.py').read_text()
        self.assertTrue(any(c['source'].strip() == diagnostic.strip() for c in notebook['cells']))
        tree = ast.parse(diagnostic)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, 'train')
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.assertNotIn(node.id, ('CONFIDENCE', 'candidate', 'final', 'comparison'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(cell['source'])
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_generated_helpers_match_source(self):
        notebook = w.read_json(HERE / 'BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb')
        embedded = next(c['source'] for c in notebook['cells'] if 'def prepare_dataset(' in c['source'])
        self.assertEqual(embedded.strip(), (HERE / 'rfdetr_workflow.py').read_text().strip())

    def test_deterministic_sampling_and_shortage(self):
        rows = [{'path': str(i), 'objects': {'bbox': [[1, 2, 3, 4]], 'categories': [3 if i < 10 else 0]}}
                for i in range(20)]
        a = w.select_records(rows, 10, 42)
        self.assertEqual(a, w.select_records(list(reversed(rows)), 10, 42))
        self.assertEqual(sum(3 in r['objects']['categories'] for r in a), 5)
        with self.assertRaises(ValueError):
            w.select_records(rows, 22, 42)

    def test_labels_retain_four_classes_and_validate_bounds(self):
        objects = {'bbox': [[1, 2, 3, 4]] * 4, 'categories': [0, 1, 2, 3]}
        self.assertEqual([a['category_id'] for a in w.validate_objects(objects, 10, 10)], [0, 1, 2, 3])
        for bad in ({'bbox': [[0, 0, 0, 2]], 'categories': [3]},
                    {'bbox': [[0, 0, 2, 2]], 'categories': [4]},
                    {'bbox': [[0, 0, 20, 2]], 'categories': [3]},
                    {'bbox': [[0, 0, 2, 2]], 'categories': []}):
            with self.assertRaises(ValueError):
                w.validate_objects(bad, 10, 10)

    def test_duplicate_boxes_are_not_double_correct(self):
        pred = [{'xyxy': [0, 0, 10, 10], 'score': .9}, {'xyxy': [0, 0, 10, 10], 'score': .8}]
        self.assertEqual(w.match_boxes([[0, 0, 10, 10]], pred), {'tp': 1, 'fp': 1, 'fn': 0})
        self.assertEqual(w.match_boxes([[0, 0, 10, 10]], []), {'tp': 0, 'fp': 0, 'fn': 1})

    def test_fixed_confidence_is_class_aware_and_counts_false_alarms(self):
        coco = {'annotations': [{'image_id': 1, 'category_id': 3, 'bbox': [0, 0, 10, 10]}]}
        rows = [{'image_id': 1, 'predictions': [
                    {'category_id': 0, 'score': .9, 'xyxy': [0, 0, 10, 10]},
                    {'category_id': 3, 'score': .1, 'xyxy': [0, 0, 10, 10]}]},
                {'image_id': 2, 'predictions': [{'category_id': 3, 'score': .9, 'xyxy': [0, 0, 10, 10]}]}]
        p = w.fixed_metrics(coco, rows, .25)['pothole']
        self.assertEqual((p['tp'], p['fp'], p['fn']), (0, 1, 1))
        self.assertEqual(p['image_false_alarm_rate'], 1.)

    def test_immutable_run_configuration(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / 'config.json'
            w.guard_json(path, {'seed': 42})
            w.guard_json(path, {'seed': 42})
            with self.assertRaises(ValueError):
                w.guard_json(path, {'seed': 43})
            self.assertEqual(w.read_json(path), {'seed': 42})

    def test_unsafe_paths(self):
        for value in ('../outside', '/absolute', 'C:/outside', 'folder\\outside'):
            with self.assertRaises(ValueError):
                w.safe_relative(value)

    def test_prepare_coco_preserves_upstream_split_and_detects_duplicates(self):
        rotated_paths = set()
        class FakeImage:
            size = (100, 100)
            def __init__(self, path): self.path = Path(path)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def load(self): pass
            def getexif(self): return {274: 6} if self.path.name in rotated_paths else {}
        image_api = types.SimpleNamespace(open=lambda path: FakeImage(path))
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp) / 'root'
            cache = root / 'download_cache'
            backup = Path(temp) / 'backup'
            for split in ('train', 'valid', 'test'):
                folder = cache / 'data/images' / split / 'shard_000'
                folder.mkdir(parents=True)
                records = []
                for i in range(12):
                    name = f'India_{split}_{i}.jpg'
                    labels = [0, 1, 2, 3] if i < 6 else [0, 1, 2]
                    records.append({'file_name': name, 'objects': {'bbox': [[1, 2, 3, 4]] * len(labels),
                                                                 'categories': labels}})
                    (folder / name).write_bytes(name.encode())
                (folder / 'metadata.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            hf = types.SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(cache),
                                       hf_hub_download=lambda *args, **kwargs: str(cache / kwargs['filename']))
            with patch.dict('sys.modules', {'huggingface_hub': hf, 'PIL': types.SimpleNamespace(Image=image_api)}):
                prepared, manifest = w.prepare_dataset(root, backup, {'train': 10, 'valid': 10, 'test': 10}, 42, 'India')
                for split in ('train', 'valid', 'test'):
                    coco = w.read_json(prepared / split / '_annotations.coco.json')
                    self.assertEqual(len(coco['images']), 10)
                    self.assertEqual([c['id'] for c in coco['categories']], [0, 1, 2, 3])
                    self.assertTrue(all('/' + split + '/' in r['path'] for r in manifest['splits'][split]))
                # Rebuilding the same run has an identical manifest and selection.
                _, again = w.prepare_dataset(root, backup, {'train': 10, 'valid': 10, 'test': 10}, 42, 'India')
                self.assertEqual(manifest, again)
                rotated = manifest['splits']['train'][0]
                rotated_paths.add(rotated['file_name'])
                rotated_backup = Path(temp) / 'rotation_review'
                _, quarantined = w.prepare_dataset(root, rotated_backup, {'train': 10, 'valid': 10, 'test': 10}, 42, 'India')
                self.assertEqual(len(quarantined['splits']['train']), 9)
                self.assertEqual(len(quarantined['splits']['test']), 10)
                excluded = w.read_json(rotated_backup / 'data_exclusions.json')
                self.assertEqual(excluded[0]['path'], rotated['path'])
                self.assertEqual(excluded[0]['orientation'], 6)
                self.assertEqual(w.sha256(cache / rotated['path']), rotated['sha256'])
                self.assertEqual(len(w.read_json(prepared / 'train/_annotations.coco.json')['images']), 9)
                rotated_paths.clear()
                first_train = manifest['splits']['train'][0]
                first_valid = manifest['splits']['valid'][0]
                (cache / first_valid['path']).write_bytes((cache / first_train['path']).read_bytes())
                with self.assertRaisesRegex(ValueError, 'Exact duplicate'):
                    w.prepare_dataset(root, backup, {'train': 10, 'valid': 10, 'test': 10}, 42, 'India')

    def test_multi_country_filtering_and_notebook_enhancements(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            cache = Path(temp) / 'cache'
            for split in ('train', 'valid', 'test'):
                folder = cache / 'data/images' / split / 'shard_000'
                folder.mkdir(parents=True)
                records = [
                    {'file_name': f'{c}_{split}_0.jpg', 'objects': {'bbox': [[1, 2, 3, 4]], 'categories': [3]}}
                    for c in ('India', 'Japan', 'Czech', 'Norway')
                ]
                (folder / 'metadata.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            subset = w.metadata_records(cache, ('India', 'Japan', 'Czech'))
            self.assertEqual(len(subset['train']), 3)
            all_countries = w.metadata_records(cache, None)
            self.assertEqual(len(all_countries['train']), 4)
            # Verify tuple round-trips cleanly through guard_json
            guard_path = Path(temp) / 'tuple_guard.json'
            w.guard_json(guard_path, {'country': ('India', 'Japan', 'Czech')})
            w.guard_json(guard_path, {'country': ('India', 'Japan', 'Czech')})
        notebook = w.read_json(HERE / 'BANTAI_RFDETR_Nano_Baseline_Finetuning.ipynb')
        sources = '\n'.join(c['source'] for c in notebook['cells'])
        self.assertIn("COUNTRY = ('India', 'Japan', 'Czech')", sources)
        self.assertIn('TRAIN_IMAGES, VALID_IMAGES, TEST_IMAGES = 1200, 200, 200', sources)
        self.assertIn('multi_scale=True', sources)
        self.assertIn('use_ema=True', sources)


if __name__ == '__main__':
    unittest.main()
