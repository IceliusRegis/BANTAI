"""Offline fixtures only: no RDD download, package install, model load or GPU use."""
import json
import io
from pathlib import Path
import random
import shutil
import unittest
import uuid
import zipfile
from types import SimpleNamespace
from unittest import mock
import urllib.error

from PIL import Image

import pothole_workflow as workflow


class WorkspaceTemp:
    """Keep generated fixtures inside the repository, including on Windows."""
    def __init__(self):
        self.base = Path(__file__).resolve().parent / ".test-tmp"
        self.path = self.base / uuid.uuid4().hex
        self.path.mkdir(parents=True)
        self.name = str(self.path)

    def cleanup(self):
        if self.path.resolve().parent != self.base.resolve():
            raise ValueError("Fixture cleanup escaped its dedicated directory")
        shutil.rmtree(self.path)

    def __enter__(self):
        return self.name

    def __exit__(self, *_):
        self.cleanup()


def xml(boxes, width=40, height=30, name="D40"):
    objects = "".join("<object><name>" + name + "</name><bndbox>" +
                      "".join("<" + key + ">" + str(value) + "</" + key + ">" for key, value in
                              zip(("xmin", "ymin", "xmax", "ymax"), box)) +
                      "</bndbox></object>" for box in boxes)
    return f"<annotation><size><width>{width}</width><height>{height}</height></size>{objects}</annotation>"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = WorkspaceTemp()
        self.root = Path(self.temporary.name) / "runtime"
        self.root.mkdir()
        self.backup = Path(self.temporary.name) / "backup"

    def tearDown(self):
        self.temporary.cleanup()

    def annotation(self, content):
        path = self.root / "annotation.xml"
        path.write_text(content, encoding="utf-8")
        return path

    def record(self, index, positive=True):
        return {"id": str(index), "country": "India", "image": f"image_{index * 100}.jpg",
                "dhash": random.Random(index).getrandbits(64), "width": 40, "height": 30,
                "boxes": [[4, 3, 20, 15]] if positive else []}

    def test_boxes_convert_and_other_classes_do_not_become_potholes(self):
        boxes, _ = workflow.parse_annotation(self.annotation(xml([[4, 3, 20, 15]])), 40, 30)
        record = {"boxes": boxes, "width": 40, "height": 30}
        self.assertEqual(workflow.yolo_labels(record), "0 0.300000000 0.300000000 0.400000000 0.400000000\n")
        boxes, _ = workflow.parse_annotation(self.annotation(xml([[4, 3, 20, 15]], name="D00")), 40, 30)
        self.assertEqual(boxes, [])

    def test_bad_boxes_and_dimensions_fail_closed(self):
        for box in [[-1, 0, 20, 15], [4, 3, 45, 15], [20, 3, 4, 15], [0, 0, float("nan"), 10]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                workflow.parse_annotation(self.annotation(xml([box])), 40, 30)
        with self.assertRaises(ValueError):
            workflow.parse_annotation(self.annotation(xml([], width=50)), 40, 30)

    def test_entity_declarations_rejected(self):
        with self.assertRaises(ValueError):
            workflow.parse_annotation(self.annotation('<!DOCTYPE a [<!ENTITY a "x">]><annotation/>'), 40, 30)

    def test_archive_ignores_official_test_and_rejects_traversal(self):
        archive = self.root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("India/train/images/a.jpg", b"fixture")
            bundle.writestr("India/test/images/b.jpg", b"do not extract")
        destination = self.root / "raw"
        self.assertEqual(workflow.extract_training(archive, destination), 1)
        self.assertFalse((destination / "India/test/images/b.jpg").exists())
        self.assertEqual(workflow.extract_training(archive, destination), 1)
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../train/escape.jpg", b"escape")
        with self.assertRaises(ValueError):
            workflow.extract_training(archive, destination)

    def range_response(self, payload, request):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        response = io.BytesIO(payload[start:end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        return response

    def test_range_reader_seeks_caches_and_checks_partial_responses(self):
        payload = bytes(range(100))
        with mock.patch.object(workflow.urllib.request, "urlopen", side_effect=lambda r, **_: self.range_response(payload, r)) as opened:
            reader = workflow.RangeDownload("https://api.figshare.com/example", len(payload), block_size=16)
            self.assertEqual(reader.read(10), payload[:10])
            self.assertEqual(reader.read(2), payload[10:12])
            self.assertEqual(opened.call_count, 1)
            reader.seek(-6, 2)
            self.assertEqual(reader.read(), payload[-6:])
            reader.seek(10)
            self.assertEqual(reader.read(30), payload[10:40])
        ignored = io.BytesIO(payload)
        ignored.status, ignored.headers = 200, {}
        with mock.patch.object(workflow.urllib.request, "urlopen", return_value=ignored), self.assertRaises(IOError):
            workflow.RangeDownload("https://api.figshare.com/example", 100).read(1)

    def test_forbidden_country_download_uses_original_nested_zip(self):
        country = io.BytesIO()
        with zipfile.ZipFile(country, "w") as bundle:
            bundle.writestr("India/train/images/a.jpg", b"image fixture")
            bundle.writestr("India/test/images/b.jpg", b"unused test fixture")
        master = io.BytesIO()
        with zipfile.ZipFile(master, "w", compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr("RDD2022/India.zip", country.getvalue())
            bundle.writestr("RDD2022/Japan.zip", b"not selected")
        payload = master.getvalue()
        metadata = {"files": [{"name": workflow.FIGSHARE_FILENAME, "id": 38030910,
                               "size": len(payload), "computed_md5": "publisher metadata"}]}
        def open_request(request, **_):
            if request == workflow.FIGSHARE_ARTICLE_API:
                return io.BytesIO(json.dumps(metadata).encode())
            if request.full_url.startswith(workflow.URLS["India"]):
                raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)
            return self.range_response(payload, request)
        archive = self.root / "India.zip"
        with mock.patch.object(workflow.urllib.request, "urlopen", side_effect=open_request), mock.patch("builtins.print"):
            origin = workflow.download_country("India", workflow.URLS["India"], archive)
        self.assertEqual(archive.read_bytes(), country.getvalue())
        self.assertEqual(origin["source_member"], "RDD2022/India.zip")
        self.assertTrue(archive.with_suffix(".origin.json").exists())
        self.assertEqual(workflow.extract_training(archive, self.root / "extracted"), 1)

    def test_non_access_http_error_does_not_silently_switch_source(self):
        error = urllib.error.HTTPError(workflow.URLS["India"], 429, "Rate limited", {}, None)
        with mock.patch.object(workflow.urllib.request, "urlopen", side_effect=error), \
                mock.patch.object(workflow, "mirror_country") as fallback, mock.patch("builtins.print"):
            with self.assertRaises(urllib.error.HTTPError):
                workflow.download_country("India", workflow.URLS["India"], self.root / "India.zip")
        fallback.assert_not_called()
        error.close()

    def test_changed_extracted_original_is_not_overwritten(self):
        archive = self.root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("train/images/a.jpg", b"original")
        workflow.extract_training(archive, self.root / "raw")
        original = self.root / "raw/train/images/a.jpg"
        original.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            workflow.extract_training(archive, self.root / "raw")
        self.assertEqual(original.read_bytes(), b"changed")

    def test_related_images_share_groups_and_manual_groups_merge(self):
        records = [self.record(i) for i in range(5)]
        records[1]["dhash"] = records[0]["dhash"] ^ 3
        records[3]["image"] = "image_201.jpg"  # same filename block as record 2
        groups = workflow.group_related(records, {"0": "route", "4": "route"})
        sets = [{r["id"] for r in group} for group in groups]
        self.assertIn({"0", "1", "4"}, sets)
        self.assertIn({"2", "3"}, sets)

    def test_grouped_splits_deterministic_balanced_and_review_is_train_only(self):
        groups = [[self.record(i, positive=i % 2 == 0)] for i in range(100)]
        first = workflow.split_groups(groups, {"0", "1"})
        self.assertEqual(first, workflow.split_groups(groups, {"0", "1"}))
        self.assertEqual(first["0"], "train")
        self.assertEqual(first["1"], "train")
        for split, ratio in workflow.RATIOS.items():
            self.assertLessEqual(abs(sum(s == split for s in first.values()) - 100 * ratio), 2)

    def test_split_fails_if_one_group_cannot_be_separated(self):
        with self.assertRaises(ValueError):
            workflow.split_groups([[self.record(i) for i in range(20)]])

    def test_operating_metrics_handle_duplicate_predictions_backgrounds_and_misses(self):
        examples = [
            {"id": "a", "truth": [[0, 0, 10, 10], [20, 20, 30, 30]],
             "predictions": [{"box": [0, 0, 10, 10], "confidence": .9},
                             {"box": [0, 0, 10, 10], "confidence": .8}]},
            {"id": "b", "truth": [], "predictions": [{"box": [0, 0, 10, 10], "confidence": .6}]},
        ]
        result = workflow.operating_counts(examples, .5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 1))
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], .5)
        self.assertEqual(workflow.operating_counts([], .5)["f1"], 0)

    def test_phase_gates_do_not_import_training_or_read_data(self):
        with mock.patch.object(workflow, "require_ultralytics", side_effect=AssertionError("GPU imported")):
            with self.assertRaises(RuntimeError):
                workflow.prepare_dataset(self.root, self.backup)
            with self.assertRaises(RuntimeError):
                workflow.train_baseline(self.root, self.backup)
            with self.assertRaises(RuntimeError):
                workflow.evaluate_once(self.root, self.backup)

    def create_sources(self):
        provenance = {country: {"url": workflow.URLS[country], "sha256": country} for country in workflow.COUNTRIES}
        workflow.write_json(self.root / "source_provenance.json", provenance)
        for country in workflow.COUNTRIES:
            for index in range(30):
                folder = self.root / "raw" / country / country / "train"
                image = folder / "images" / f"{country}_{index * 100:06d}.png"
                annotation = folder / "annotations" / "xmls" / (image.stem + ".xml")
                image.parent.mkdir(parents=True, exist_ok=True)
                annotation.parent.mkdir(parents=True, exist_ok=True)
                pixels = random.Random(country + str(index)).randbytes(40 * 30 * 3)
                Image.frombytes("RGB", (40, 30), pixels).save(image)
                annotation.write_text(xml([[4, 3, 20, 15]] if index % 2 == 0 else []), encoding="utf-8")

    def prepared_fixture(self):
        self.create_sources()
        with mock.patch("builtins.print"):
            audit = workflow.audit_sources(self.root)
            manifest = workflow.prepare_dataset(self.root, self.backup, approved=True)
        return audit, manifest

    def test_audit_prepare_roundtrip_and_backups(self):
        audit, manifest = self.prepared_fixture()
        self.assertEqual(len(audit["records"]), 90)
        self.assertEqual(workflow.load_manifest(self.root), manifest)
        self.assertTrue((self.backup / "preparation/manifest.json").exists())
        self.assertNotIn("test:", (self.root / "preparation/train.yaml").read_text())
        for group in manifest["groups"]:
            splits = {r["split"] for r in manifest["records"] if r["id"] in group}
            self.assertEqual(len(splits), 1)
        workflow.verify_prepared(self.root, manifest, {"train", "val", "test"})
        with mock.patch("builtins.print"):
            second = workflow.prepare_dataset(self.root, self.backup, approved=True)
        self.assertEqual(second["dataset_id"], manifest["dataset_id"])

    def test_test_bytes_not_read_by_training_data_verification(self):
        _, manifest = self.prepared_fixture()
        test_images = {str((self.root / r["prepared_image"]).resolve()) for r in manifest["records"] if r["split"] == "test"}
        original = workflow.sha256
        def guarded(path):
            if str(Path(path).resolve()) in test_images:
                raise AssertionError("Test image read")
            return original(path)
        with mock.patch.object(workflow, "sha256", side_effect=guarded):
            workflow.verify_prepared(self.root, manifest, {"train", "val"})

    def test_prepared_label_tampering_rejected(self):
        _, manifest = self.prepared_fixture()
        record = manifest["records"][0]
        image = self.root / record["prepared_image"]
        label = image.parent.parent.parent / "labels" / record["split"] / (image.stem + ".txt")
        label.write_text("", encoding="utf-8")
        if record["boxes"]:
            with self.assertRaises(ValueError):
                workflow.verify_prepared(self.root, manifest, {record["split"]})
        else:
            label.write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                workflow.verify_prepared(self.root, manifest, {record["split"]})

    def test_changed_review_does_not_overwrite_saved_attempt(self):
        audit, _ = self.prepared_fixture()
        with self.assertRaises(ValueError), mock.patch("builtins.print"):
            workflow.prepare_dataset(self.root, self.backup, approved=True, exclude_ids=[audit["records"][0]["id"]])

    def completed_fixture(self):
        _, manifest = self.prepared_fixture()
        weights = self.backup / "run/weights/best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"fixture model; never load with torch")
        completion = {"dataset_id": manifest["dataset_id"], "best_sha256": workflow.sha256(weights)}
        workflow.write_json(self.backup / "completed.json", completion)
        return manifest, completion

    def test_validation_selection_does_not_predict_test_and_is_reused(self):
        _, completed = self.completed_fixture()
        seen = []
        def predictions(root, model, records):
            self.assertTrue(all(r["split"] == "val" for r in records))
            seen.extend(records)
            return [{"id": r["id"], "truth": r["boxes"], "predictions": [
                {"box": box, "confidence": .8} for box in r["boxes"]]} for r in records]
        with mock.patch.object(workflow, "require_ultralytics", return_value=(None, mock.Mock())), \
                mock.patch.object(workflow, "predict_split", side_effect=predictions), mock.patch("builtins.print"):
            result = workflow.select_threshold(self.root, self.backup)
        self.assertTrue(seen)
        self.assertEqual(result["validation"]["f1"], 1.)
        with mock.patch.object(workflow, "require_ultralytics", side_effect=AssertionError("Reran selection")):
            self.assertEqual(workflow.select_threshold(self.root, self.backup), result)

    def test_test_evaluation_cache_and_interrupted_attempt_guard(self):
        _, completion = self.completed_fixture()
        workflow.write_json(self.backup / "threshold.json", {**completion, "threshold": .25})
        evaluation = self.backup / "test_evaluation"
        evaluation.mkdir()
        workflow.write_json(evaluation / "started.json", completion)
        with self.assertRaises(RuntimeError):
            workflow.evaluate_once(self.root, self.backup, approved=True)
        saved = {**completion, "threshold": .25, "mAP50": .1}
        workflow.write_json(evaluation / "report.json", saved)
        with mock.patch.object(workflow, "require_ultralytics", side_effect=AssertionError("Test reran")), \
                mock.patch("builtins.print"):
            self.assertEqual(workflow.evaluate_once(self.root, self.backup, approved=True), saved)

    def test_fresh_evaluation_uses_selected_threshold_and_saves_metrics(self):
        _, completion = self.completed_fixture()
        workflow.write_json(self.backup / "threshold.json", {**completion, "threshold": .45})
        metrics = SimpleNamespace(box=SimpleNamespace(map50=.2, map=.1), speed={"inference": 2.5})
        model = mock.Mock()
        def validate(**settings):
            self.assertEqual(settings["split"], "test")
            self.assertEqual(settings["conf"], .001)
            folder = Path(settings["project"]) / settings["name"]
            folder.mkdir(parents=True)
            (folder / "fixture.txt").write_text("plot placeholder", encoding="utf-8")
            return metrics
        model.val.side_effect = validate
        def predictions(root, detector, records):
            self.assertTrue(all(r["split"] == "test" for r in records))
            return [{"id": r["id"], "truth": r["boxes"], "predictions": []} for r in records]
        with mock.patch.object(workflow, "require_ultralytics", return_value=(None, mock.Mock(return_value=model))), \
                mock.patch.object(workflow, "predict_split", side_effect=predictions), mock.patch("builtins.print"):
            report = workflow.evaluate_once(self.root, self.backup, approved=True)
        self.assertEqual(report["threshold"], .45)
        self.assertEqual(report["mAP50"], .2)
        self.assertTrue((self.backup / "test_evaluation/plots/fixture.txt").exists())

    def test_training_checkpoint_callback_and_completed_attempt_reuse(self):
        _, manifest = self.prepared_fixture()
        initial = self.root / "initial.pt"
        initial.write_bytes(b"fixture pretrained")
        model = mock.Mock()
        model.ckpt_path = str(initial)
        def fake_train(**settings):
            self.assertEqual(settings["batch"], .60)
            self.assertEqual(settings["epochs"], 30)
            self.assertNotIn("test:", Path(settings["data"]).read_text())
            run = self.root / "runs/baseline"
            (run / "weights").mkdir(parents=True)
            (run / "weights/last.pt").write_bytes(b"resumable fixture")
            (run / "weights/best.pt").write_bytes(b"best fixture")
            callback = model.add_callback.call_args.args[1]
            callback(SimpleNamespace(save_dir=run, epoch=0))
            self.assertTrue((self.backup / "run/weights/last.pt").exists())
        model.train.side_effect = fake_train
        torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="fixture"),
                                cuda=SimpleNamespace(get_device_name=lambda _: "fixture GPU"))
        with mock.patch.object(workflow, "require_ultralytics", return_value=(torch, mock.Mock(return_value=model))):
            best = workflow.train_baseline(self.root, self.backup, start=True)
        self.assertTrue(best.exists())
        self.assertEqual(workflow.read_json(self.backup / "progress.json")["completed_epochs"], 1)
        with mock.patch.object(workflow, "require_ultralytics", side_effect=AssertionError("Training reran")), \
                mock.patch("builtins.print"):
            self.assertEqual(workflow.train_baseline(self.root, self.backup, start=True), best)

    def test_resuming_requires_matching_contract_and_checkpoint(self):
        _, manifest = self.prepared_fixture()
        workflow.write_json(self.backup / "training_contract.json", workflow.training_contract(manifest))
        torch = SimpleNamespace()
        with mock.patch.object(workflow, "require_ultralytics", return_value=(torch, mock.Mock())):
            with self.assertRaises(FileNotFoundError):
                workflow.train_baseline(self.root, self.backup, start=True, resume=True)
        contract = workflow.read_json(self.backup / "training_contract.json")
        contract["imgsz"] = 320
        workflow.write_json(self.backup / "training_contract.json", contract)
        with self.assertRaises(ValueError):
            workflow.train_baseline(self.root, self.backup, start=True, resume=True)

    def test_delivery_has_metadata_weights_and_no_raw_images(self):
        _, completion = self.completed_fixture()
        workflow.write_json(self.backup / "threshold.json", {**completion, "threshold": .25})
        workflow.write_json(self.backup / "test_evaluation/report.json", {**completion, "mAP50": .1})
        archive = workflow.package_model(self.root, self.backup)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            self.assertIn("pothole_yolo11n.pt", names)
            self.assertIn("metadata.json", names)
            self.assertIn("preparation/manifest.json", names)
            self.assertFalse(any(name.endswith((".jpg", ".jpeg")) for name in names))
            self.assertEqual(json.loads(bundle.read("metadata.json"))["threshold"], .25)

    def test_missing_annotation_not_a_negative_and_conflicts_quarantined(self):
        self.create_sources()
        folder = self.root / "raw/India/India/train"
        original = folder / "images/India_000000.png"
        shutil.copyfile(original, folder / "images/India_999900.png")
        (folder / "annotations/xmls/India_999900.xml").write_text(xml([]), encoding="utf-8")
        (folder / "annotations/xmls/India_000100.xml").unlink()
        with mock.patch("builtins.print"):
            report = workflow.audit_sources(self.root)
        reasons = [item["reason"] for item in report["exclusions"]]
        self.assertEqual(reasons.count("Duplicate image with conflicting annotations"), 2)
        self.assertTrue(any("Missing" in reason for reason in reasons))

    def test_notebook_is_current_and_all_cells_compile(self):
        notebook_path = Path(workflow.__file__).with_name("BANTAI_Pothole_First_Attempt.ipynb")
        with WorkspaceTemp() as directory:
            generated = workflow.build_notebook(Path(directory) / notebook_path.name)
            self.assertEqual(generated.read_bytes(), notebook_path.read_bytes())
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "notebook:" + cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
