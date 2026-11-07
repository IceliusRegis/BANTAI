"""BANTAI pothole baseline. Build the standalone notebook with --build-notebook.

Local imports require only Pillow; GPU libraries are imported only for training.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import sys
import urllib.request
import urllib.error
import uuid
import xml.etree.ElementTree as ET
import zipfile

VERSION = "pothole-baseline-v1"
ULTRALYTICS_VERSION = "8.3.221"
COUNTRIES = ("India", "Japan", "Czech")
BASE_URL = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/"
URLS = {c: BASE_URL + "RDD2022_" + c + ".zip" for c in COUNTRIES}
FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/21431547"
FIGSHARE_FILENAME = "RDD2022_released_through_CRDDC2022.zip"
SEED = 42
RATIOS = {"train": .70, "val": .15, "test": .15}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def backup_tree(source, destination):
    source = Path(source)
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.name.endswith(".partial"):
            atomic_copy(path, Path(destination) / path.relative_to(source))


def extract_training(archive, destination):
    """Never extract unlabeled official test images, links, or escaping paths."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            relative = PurePosixPath(entry.filename)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in entry.filename
                    or any(":" in part for part in relative.parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("Unsafe archive member")
            if entry.is_dir() or "train" not in relative.parts:
                continue
            if relative.suffix.lower() not in {".jpg", ".jpeg", ".png", ".xml"}:
                continue
            target = destination.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes extraction directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # Re-extraction after interruption verifies bytes without replacing originals.
                with bundle.open(entry) as stream:
                    expected = hashlib.sha256(stream.read()).hexdigest()
                if sha256(target) != expected:
                    raise ValueError("Extracted source changed: " + target.name)
            else:
                temporary = target.with_name(target.name + ".partial")
                with bundle.open(entry) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
            count += 1
    if count == 0:
        raise ValueError("Archive contains no recognized training files")
    return count


class RangeDownload(io.RawIOBase):
    """Seekable HTTP ZIP reader; cache one 8 MiB block, never fetch the full archive."""
    def __init__(self, url, size, block_size=8 * 1024**2):
        self.url, self.size, self.block_size = url, size, block_size
        self.position, self.block_start, self.block = 0, -1, b""
        self.transferred_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = min(self.size - self.position, size if size >= 0 else self.size)
        output = bytearray()
        while remaining > 0:
            start = self.position // self.block_size * self.block_size
            if start != self.block_start:
                end = min(start + self.block_size, self.size) - 1
                # Fresh API redirects avoid expired cached signed storage URLs.
                url = self.url + "?request=" + uuid.uuid4().hex
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                               "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise IOError("Mirror did not honor a partial request; full archive download refused")
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise IOError("Incomplete archive range; rerun Phase 1")
                self.block_start, self.block = start, data
                self.transferred_bytes += len(data)
            offset = self.position - start
            count = min(remaining, len(self.block) - offset)
            output.extend(self.block[offset:offset + count])
            self.position += count
            remaining -= count
        return bytes(output)


def mirror_country(country, archive):
    """Stream the original country ZIP nested inside Figshare's full RDD archive."""
    with urllib.request.urlopen(FIGSHARE_ARTICLE_API, timeout=60) as response:
        article = json.load(response)
    files = [file for file in article["files"] if file["name"] == FIGSHARE_FILENAME]
    if len(files) != 1:
        raise ValueError("Official mirror archive not uniquely identified")
    descriptor = files[0]
    file_id = int(descriptor["id"])
    size = int(descriptor["size"])
    if size <= 0:
        raise ValueError("Invalid mirror archive size")
    url = "https://api.figshare.com/v2/file/download/" + str(file_id)
    temporary = Path(archive).with_suffix(".partial")
    print("Using official Figshare mirror for", country, "(country ZIP only)", flush=True)
    with RangeDownload(url, size) as remote, zipfile.ZipFile(remote) as source:
        selected = [entry for entry in source.infolist()
                    if PurePosixPath(entry.filename).name == country + ".zip" and not entry.is_dir()]
        if len(selected) != 1:
            raise ValueError("Mirror archive country ZIP not uniquely identified")
        entry = selected[0]
        if entry.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Mirror layout changed; expected directly range-readable country ZIP")
        downloaded = 0
        with source.open(entry) as stream, temporary.open("wb") as output:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded // (64 * 1024**2) != (downloaded - len(chunk)) // (64 * 1024**2):
                    print(country, round(downloaded / 1024**2), "MiB from mirror", flush=True)
        if downloaded != entry.file_size or not zipfile.is_zipfile(temporary):
            raise IOError("Incomplete country ZIP from mirror; rerun Phase 1")
        origin = {"url": url, "article_api": FIGSHARE_ARTICLE_API, "file_id": file_id,
                  "source_filename": FIGSHARE_FILENAME, "source_archive_bytes": size,
                  "source_archive_md5": descriptor.get("computed_md5"),
                  "integrity": "Original member CRC verified; full source archive MD5 not verified",
                  "local_archive": "Original nested country ZIP; image/XML contents unchanged",
                  "source_member": entry.filename, "downloaded_range_bytes": remote.transferred_bytes}
    write_json(Path(archive).with_suffix(".origin.json"), origin)
    temporary.replace(archive)
    return origin


def download_country(country, url, archive):
    print("Downloading", country, "directly to runtime storage", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BANTAI-academic-baseline/1"})
    temporary = archive.with_suffix(".partial")
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code not in {403, 404, 410}:
            raise
        error.close()
        return mirror_country(country, archive)
    with response, temporary.open("wb") as output:
        expected = response.headers.get("Content-Length")
        downloaded = 0
        for chunk in iter(lambda: response.read(4 * 1024**2), b""):
            output.write(chunk)
            downloaded += len(chunk)
            if downloaded // (64 * 1024**2) != (downloaded - len(chunk)) // (64 * 1024**2):
                print(country, round(downloaded / 1024**2), "MiB", flush=True)
        if expected and downloaded != int(expected):
            raise IOError("Incomplete download; rerun to retry")
    if not zipfile.is_zipfile(temporary):
        raise ValueError("Download was not a ZIP archive")
    origin = {"url": url}
    write_json(archive.with_suffix(".origin.json"), origin)
    temporary.replace(archive)
    return origin


def download_sources(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 8 * 1024**3:
        raise RuntimeError("Need at least 8 GiB free for archives and extracted training images")
    previous_path = root / "source_provenance.json"
    previous = read_json(previous_path) if previous_path.exists() else {}
    provenance = dict(previous)
    for country, url in URLS.items():
        archive = root / "archives" / (country + ".zip")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            origin = download_country(country, url, archive)
        else:
            origin_path = archive.with_suffix(".origin.json")
            origin = read_json(origin_path) if origin_path.exists() else {"url": url}
        digest = sha256(archive)
        if country in previous:
            if previous[country]["sha256"] != digest:
                raise ValueError("Cached archive changed; use a new attempt directory")
        count = extract_training(archive, root / "raw" / country)
        provenance[country] = {**origin, "sha256": digest, "bytes": archive.stat().st_size,
                               "extracted_training_files": count}
        write_json(root / "source_provenance.json", provenance)
    return provenance


def difference_hash(image):
    from PIL import Image
    pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def parse_annotation(path, width, height):
    data = Path(path).read_bytes()
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML declarations are not allowed")
    tree = ET.fromstring(data)
    if int(tree.findtext("size/width", "0")) != width or int(tree.findtext("size/height", "0")) != height:
        raise ValueError("Annotation/image dimensions differ")
    boxes, flags = [], []
    for obj in tree.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name != "D40":
            continue
        values = [float(obj.findtext("bndbox/" + field, "nan"))
                  for field in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = values
        if not all(math.isfinite(v) for v in values) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Invalid pothole box; entire image quarantined")
        if obj.findtext("difficult", "0") == "1":
            flags.append("difficult pothole")
        if min(x2 - x1, y2 - y1) < 8:
            flags.append("small pothole box")
        boxes.append(values)
    # RDD pixel edges are retained as provided; no silent clipping or box enlargement.
    return sorted(boxes), sorted(set(flags))


def audit_sources(root):
    from PIL import Image, ImageOps
    root = Path(root)
    records, exclusions = [], []
    for country in COUNTRIES:
        folder = root / "raw" / country
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        xmls = collections.defaultdict(list)
        for xml in sorted(folder.rglob("*.xml")):
            xmls[xml.stem].append(xml)
        for index, path in enumerate(images):
            identifier = country + "/" + path.relative_to(folder).as_posix()
            try:
                matches = xmls[path.stem]
                if len(matches) != 1:
                    raise ValueError("Missing or ambiguous annotation; never assume background")
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF rotation requires annotation review")
                    boxes, flags = parse_annotation(matches[0], width, height)
                    perceptual = difference_hash(ImageOps.grayscale(image))
                records.append({"id": identifier, "country": country,
                                "image": path.relative_to(root).as_posix(),
                                "xml": matches[0].relative_to(root).as_posix(),
                                "sha256": sha256(path), "xml_sha256": sha256(matches[0]),
                                "dhash": perceptual, "width": width, "height": height,
                                "boxes": boxes, "flags": flags})
            except (ValueError, OSError, ET.ParseError, Image.DecompressionBombError) as error:
                exclusions.append({"id": identifier, "reason": str(error)})
            if index % 1000 == 0:
                print(country, index, "/", len(images), "audited", flush=True)
        if not images:
            raise ValueError("No images for " + country)
    # Same image bytes with incompatible labels are never treated as reliable negatives.
    by_digest = collections.defaultdict(list)
    for record in records:
        by_digest[record["sha256"]].append(record)
    kept = []
    for group in by_digest.values():
        if len({identity(r["boxes"]) for r in group}) > 1:
            exclusions.extend({"id": r["id"], "reason": "Duplicate image with conflicting annotations"} for r in group)
            continue
        kept.append(group[0])
        exclusions.extend({"id": r["id"], "reason": "Exact duplicate", "retained": group[0]["id"]} for r in group[1:])
    records = sorted(kept, key=lambda r: r["id"])
    summary = {}
    for country in COUNTRIES:
        subset = [r for r in records if r["country"] == country]
        summary[country] = {"usable_images": len(subset), "positive_images": sum(bool(r["boxes"]) for r in subset),
                            "candidate_backgrounds": sum(not r["boxes"] for r in subset),
                            "pothole_boxes": sum(len(r["boxes"]) for r in subset)}
    report = {"version": VERSION, "records": records, "exclusions": exclusions, "summary": summary,
              "sources": read_json(root / "source_provenance.json"),
              "limitations": "Background labels may be incomplete. Phone/CCTV performance is unverified."}
    report["audit_id"] = identity(report)
    write_json(root / "inspection" / "audit.json", report)
    print(json.dumps(summary, indent=2))
    print("Quarantined/deduplicated:", len(exclusions))
    return report


def show_samples(root, report, per_kind=6):
    """Show only pre-split review samples; never display the later held-out test set."""
    from PIL import Image, ImageDraw, ImageOps
    from IPython.display import display
    rng = random.Random(SEED)
    for country in COUNTRIES:
        for positive in (True, False):
            choices = [r for r in report["records"] if r["country"] == country and bool(r["boxes"]) == positive]
            rng.shuffle(choices)
            chosen = choices[:per_kind]
            sheet = Image.new("RGB", (960, 260 * math.ceil(max(1, len(chosen)) / 3)), "white")
            for index, record in enumerate(chosen):
                with Image.open(Path(root) / record["image"]) as source:
                    image = source.convert("RGB")
                draw = ImageDraw.Draw(image)
                for box in record["boxes"]:
                    draw.rectangle(box, outline="red", width=3)
                thumb = ImageOps.contain(image, (310, 220))
                x, y = (index % 3) * 320, (index // 3) * 260
                sheet.paste(thumb, (x, y))
                ImageDraw.Draw(sheet).text((x, y + 222), Path(record["image"]).name[:37], fill="black")
            kind = "potholes" if positive else "candidate backgrounds"
            print(country, kind, "— inspect missed boxes and uncertain labels")
            output = Path(root) / "inspection" / (country + ("_positive.jpg" if positive else "_background.jpg"))
            sheet.save(output)
            display(sheet)


def group_related(records, extra_groups=None):
    """dHash radius 4 plus conservative filename blocks; route identity is not known."""
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[find(left)] = find(right)

    # BK-tree avoids all-pairs perceptual comparison on tens of thousands of images.
    tree = None
    blocks, manual = {}, {}
    extra_groups = extra_groups or {}
    for index, record in enumerate(records):
        value = record["dhash"]
        if tree is None:
            tree = [value, index, {}]
        else:
            queue = [tree]
            while queue:
                node = queue.pop()
                distance = (value ^ node[0]).bit_count()
                if distance <= 4:
                    union(index, node[1])
                queue.extend(child for edge, child in node[2].items() if distance - 4 <= edge <= distance + 4)
            node = tree
            while True:
                distance = (value ^ node[0]).bit_count()
                if distance == 0:
                    break
                if distance not in node[2]:
                    node[2][distance] = [value, index, {}]
                    break
                node = node[2][distance]
        match = re.fullmatch(r"(.+?)(\d+)", Path(record["image"]).stem)
        if match:
            block = (record["country"], match[1], int(match[2]) // 100)
            if block in blocks:
                union(index, blocks[block])
            else:
                blocks[block] = index
        if record["id"] in extra_groups:
            key = extra_groups[record["id"]]
            if key in manual:
                union(index, manual[key])
            else:
                manual[key] = index
    groups = collections.defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    return sorted(groups.values(), key=lambda group: min(r["id"] for r in group))


def split_groups(groups, review_ids=frozenset()):
    """Approximate ratios while balancing country, positive and background counts."""
    totals = collections.Counter((r["country"], bool(r["boxes"])) for group in groups for r in group)
    counts = {split: collections.Counter() for split in RATIOS}
    assignments = {}
    ordered = []
    for group in groups:
        if any(r["id"] in review_ids for r in group):
            counts["train"].update((r["country"], bool(r["boxes"])) for r in group)
            assignments.update({r["id"]: "train" for r in group})
        else:
            ordered.append(group)
    random.Random(SEED).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    for group in ordered:
        addition = collections.Counter((r["country"], bool(r["boxes"])) for r in group)
        def cost(split):
            return sum(((counts[split][key] + number - total * RATIOS[split]) ** 2
                        - (counts[split][key] - total * RATIOS[split]) ** 2) / max(1, total)
                       for key, number in addition.items() for total in [totals[key]])
        chosen = min(RATIOS, key=cost)
        counts[chosen].update(addition)
        for record in group:
            assignments[record["id"]] = chosen
    for split in RATIOS:
        subset = [r for group in groups for r in group if assignments[r["id"]] == split]
        if not subset or not any(r["boxes"] for r in subset):
            raise ValueError("Split lacks pothole examples; review grouping/data before training")
    return assignments


def yolo_labels(record):
    lines = []
    for x1, y1, x2, y2 in record["boxes"]:
        values = ((x1 + x2) / (2 * record["width"]), (y1 + y2) / (2 * record["height"]),
                  (x2 - x1) / record["width"], (y2 - y1) / record["height"])
        lines.append("0 " + " ".join(f"{value:.9f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def prepare_dataset(root, backup, approved=False, exclude_ids=(), extra_groups=None):
    if not approved:
        raise RuntimeError("STOP: review Phase 1 summaries and samples, then approve Phase 2")
    root, backup = Path(root), Path(backup)
    report = read_json(root / "inspection" / "audit.json")
    original_id = report.pop("audit_id")
    if identity(report) != original_id:
        raise ValueError("Audit was edited; rerun inspection instead")
    report["audit_id"] = original_id
    known = {r["id"] for r in report["records"]}
    if not set(exclude_ids).issubset(known) or not set(extra_groups or {}).issubset(known):
        raise ValueError("Unknown review exclusion/group ID")
    # All reviewed samples stay in training, so later test metrics remain unseen.
    rng = random.Random(SEED)
    review_ids = set()
    for country in COUNTRIES:
        for positive in (True, False):
            choices = [r for r in report["records"] if r["country"] == country and bool(r["boxes"]) == positive]
            rng.shuffle(choices)
            review_ids.update(r["id"] for r in choices[:6])
    records = [r for r in report["records"] if r["id"] not in exclude_ids]
    groups = group_related(records, extra_groups)
    # Include fixed-to-train groups in the allocation objective, not a later override.
    assignments = split_groups(groups, review_ids)
    manifest = {"version": VERSION, "audit_id": original_id, "seed": SEED,
                "target_ratios": RATIOS, "review_exclusions": sorted(exclude_ids),
                "extra_groups": extra_groups or {}, "reviewed_images_train_only": sorted(review_ids & known),
                "grouping": "dHash Hamming <=4, 100-number filename blocks, optional manual route groups; heuristic, not verified routes",
                "sources": report["sources"], "groups": [[r["id"] for r in g] for g in groups],
                "records": [{**r, "split": assignments[r["id"]]} for r in records]}
    manifest["dataset_id"] = identity(manifest)
    previous = backup / "preparation" / "manifest.json"
    if previous.exists() and read_json(previous)["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Saved attempt uses different data/review. Choose a new ATTEMPT name")
    dataset = root / "prepared" / manifest["dataset_id"][:12]
    for record in records:
        record["split"] = assignments[record["id"]]
        source = root / record["image"]
        if sha256(source) != record["sha256"] or sha256(root / record["xml"]) != record["xml_sha256"]:
            raise ValueError("Source changed after audit")
        stem = hashlib.sha256(record["id"].encode()).hexdigest()[:24]
        image = dataset / "images" / record["split"] / (stem + source.suffix.lower())
        label = dataset / "labels" / record["split"] / (stem + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if not image.exists():
            try:
                os.link(source, image)
            except OSError:
                shutil.copyfile(source, image)
        label.write_text(yolo_labels(record), encoding="utf-8")
        record["prepared_image"] = image.relative_to(root).as_posix()
    # Prepared paths are deterministic derived fields, excluded from identity.
    manifest["records"] = [{**r, "split": assignments[r["id"]]} for r in records]
    prep = root / "preparation"
    prep.mkdir(parents=True, exist_ok=True)
    write_json(prep / "manifest.json", manifest)
    summary = {}
    for split in RATIOS:
        subset = [r for r in manifest["records"] if r["split"] == split]
        summary[split] = {"images": len(subset), "positive_images": sum(bool(r["boxes"]) for r in subset),
                          "backgrounds": sum(not r["boxes"] for r in subset), "boxes": sum(len(r["boxes"]) for r in subset)}
        (prep / (split + ".txt")).write_text("\n".join(r["id"] for r in subset) + "\n", encoding="utf-8")
    write_json(prep / "summary.json", summary)
    # Training cannot discover test images through its dataset configuration.
    (prep / "train.yaml").write_text("path: " + json.dumps(str(dataset)) + "\ntrain: images/train\nval: images/val\nnames:\n  0: pothole\n", encoding="utf-8")
    (prep / "test.yaml").write_text((prep / "train.yaml").read_text() + "test: images/test\n", encoding="utf-8")
    backup_tree(prep, backup / "preparation")
    backup_tree(root / "inspection", backup / "inspection")
    print(json.dumps(summary, indent=2))
    print("Ratios approximate because related images remain together.")
    return manifest


def load_manifest(root):
    manifest = read_json(Path(root) / "preparation" / "manifest.json")
    bound = dict(manifest)
    dataset_id = bound.pop("dataset_id")
    bound["records"] = [{k: v for k, v in r.items() if k != "prepared_image"} for r in bound["records"]]
    if identity(bound) != dataset_id:
        raise ValueError("Prepared manifest changed")
    return manifest


def verify_prepared(root, manifest, splits):
    """Verify only requested splits; training never opens held-out test image bytes."""
    root = Path(root).resolve()
    for record in manifest["records"]:
        if record["split"] not in splits:
            continue
        image = (root / record["prepared_image"]).resolve()
        if not image.is_relative_to(root / "prepared"):
            raise ValueError("Prepared path escapes dataset")
        label = image.parent.parent.parent / "labels" / record["split"] / (image.stem + ".txt")
        if sha256(image) != record["sha256"] or label.read_text(encoding="utf-8") != yolo_labels(record):
            raise ValueError("Prepared data changed since review")


def require_ultralytics():
    if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
        raise RuntimeError("Run the notebook's pinned Colab installation cell")
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > GPU, then rerun setup")
    return torch, YOLO


def training_contract(manifest):
    return {"version": VERSION, "dataset_id": manifest["dataset_id"], "ultralytics": ULTRALYTICS_VERSION,
            "model": "yolo11n.pt", "imgsz": 640, "epochs": 30, "patience": 7,
            "seed": SEED, "optimizer": "AdamW", "lr0": .001, "lrf": .01,
            "mosaic": .25, "close_mosaic": 5, "mixup": 0., "copy_paste": 0.,
            "degrees": 5., "translate": .05, "scale": .2, "fliplr": .5, "flipud": 0.,
            "hsv_h": .015, "hsv_s": .4, "hsv_v": .3, "perspective": 0.}


def train_baseline(root, backup, start=False, resume=False):
    if not start:
        raise RuntimeError("Training is disabled. Review Phase 2 and set START_TRAINING=True")
    root, backup = Path(root), Path(backup)
    manifest = load_manifest(root)
    contract = training_contract(manifest)
    verify_prepared(root, manifest, {"train", "val"})
    contract_path = backup / "training_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("Attempt training configuration changed; use a new attempt")
    if (backup / "completed.json").exists():
        print("This attempt already completed; continue with Phase 4.")
        return backup / "run" / "weights" / "best.pt"
    if contract_path.exists() and not resume:
        raise RuntimeError("Existing attempt: set RESUME=True instead of overwriting")
    torch, YOLO = require_ultralytics()
    run = root / "runs" / "baseline"
    if resume:
        saved = backup / "run" / "weights" / "last.pt"
        if not saved.exists():
            raise FileNotFoundError("No backed-up resumable checkpoint yet")
        backup_tree(backup / "run", run)
        model = YOLO(str(run / "weights" / "last.pt"))
    else:
        if run.exists():
            raise RuntimeError("Local run exists; use a new attempt or resume its saved checkpoint")
        model = YOLO("yolo11n.pt")
    write_json(contract_path, contract)
    environment = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "packages": sorted(
                       str(d.metadata["Name"]) + "==" + d.version for d in importlib.metadata.distributions())}
    write_json(backup / "environment.json", environment)
    if not resume:
        atomic_copy(Path(model.ckpt_path), backup / "initial_yolo11n.pt")
        write_json(backup / "initial_weights.json", {"sha256": sha256(model.ckpt_path)})
    def checkpoint(trainer):
        for relative in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
            source = Path(trainer.save_dir) / relative
            if source.exists():
                atomic_copy(source, backup / "run" / relative)
        write_json(backup / "progress.json", {"completed_epochs": trainer.epoch + 1,
                                              "dataset_id": manifest["dataset_id"]})
    model.add_callback("on_model_save", checkpoint)
    settings = {k: v for k, v in contract.items() if k not in {"version", "dataset_id", "ultralytics", "model"}}
    common = {"data": str(root / "preparation" / "train.yaml"), "device": 0, "workers": 2,
              "project": str(root / "runs"), "name": "baseline", "exist_ok": True,
              "save": True, "plots": True, "cache": False, "amp": True}
    try:
        if resume:
            model.train(resume=True, **common)
        else:
            model.train(batch=.60, **settings, **common)
    except Exception:
        print("Training stopped. Completed epoch checkpoints remain in Drive; resume this attempt.")
        raise
    backup_tree(run, backup / "run")
    best = backup / "run" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("Training returned without best.pt")
    write_json(backup / "completed.json", {"dataset_id": manifest["dataset_id"], "best_sha256": sha256(best)})
    return best


def iou(left, right):
    width = max(0., min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0., min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.


def operating_counts(examples, threshold):
    true_positive = false_positive = false_negative = 0
    details = []
    for example in examples:
        matched = set()
        predictions = sorted((p for p in example["predictions"] if p["confidence"] >= threshold),
                             key=lambda p: p["confidence"], reverse=True)
        tp = 0
        for prediction in predictions:
            candidates = [(iou(prediction["box"], box), index) for index, box in enumerate(example["truth"]) if index not in matched]
            overlap, index = max(candidates, default=(0., -1))
            if overlap >= .5:
                matched.add(index)
                tp += 1
        fp, fn = len(predictions) - tp, len(example["truth"]) - tp
        true_positive += tp
        false_positive += fp
        false_negative += fn
        details.append({"id": example["id"], "tp": tp, "fp": fp, "fn": fn})
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {"threshold": threshold, "iou": .5, "tp": true_positive, "fp": false_positive,
            "fn": false_negative, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall), "images": details}


def predict_split(root, model, records):
    examples = []
    paths = [str(Path(root) / r["prepared_image"]) for r in records]
    results = model.predict(source=paths, stream=True, batch=8, imgsz=640, conf=.001,
                            iou=.7, max_det=300, device=0, verbose=False, save=False)
    for record, result in zip(records, results, strict=True):
        if Path(result.path).resolve() != Path(root, record["prepared_image"]).resolve():
            raise ValueError("Prediction order mismatch")
        predictions = [{"box": box, "confidence": float(score)} for box, score in
                       zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True)]
        examples.append({"id": record["id"], "truth": record["boxes"], "predictions": predictions})
    return examples


def select_threshold(root, backup):
    root, backup = Path(root), Path(backup)
    manifest = load_manifest(root)
    completed = read_json(backup / "completed.json")
    best = backup / "run" / "weights" / "best.pt"
    if completed["dataset_id"] != manifest["dataset_id"] or completed["best_sha256"] != sha256(best):
        raise ValueError("Completed model/dataset mismatch")
    saved = backup / "threshold.json"
    if saved.exists():
        result = read_json(saved)
        if result["best_sha256"] != sha256(best) or result["dataset_id"] != manifest["dataset_id"]:
            raise ValueError("Threshold belongs to another model/dataset")
        return result
    _, YOLO = require_ultralytics()
    verify_prepared(root, manifest, {"val"})
    model = YOLO(str(best))
    records = [r for r in manifest["records"] if r["split"] == "val"]
    examples = predict_split(root, model, records)
    sweep = [operating_counts(examples, number / 100) for number in range(5, 96, 5)]
    chosen = max(sweep, key=lambda m: (m["f1"], m["recall"], -m["threshold"]))
    result = {"dataset_id": manifest["dataset_id"], "best_sha256": sha256(best),
              "selection": "maximum validation F1 at IoU 0.50; ties favor recall then lower threshold",
              "threshold": chosen["threshold"], "validation": chosen,
              "sweep": [{k: v for k, v in item.items() if k != "images"} for item in sweep]}
    write_json(backup / "validation_predictions.json", examples)
    write_json(saved, result)
    print("Selected validation confidence threshold:", result["threshold"])
    return result


def evaluate_once(root, backup, approved=False):
    if not approved:
        raise RuntimeError("Test disabled: finish model/threshold selection, then approve one evaluation")
    root, backup = Path(root), Path(backup)
    manifest = load_manifest(root)
    selected = read_json(backup / "threshold.json")
    best = backup / "run" / "weights" / "best.pt"
    if selected["dataset_id"] != manifest["dataset_id"] or selected["best_sha256"] != sha256(best):
        raise ValueError("Selected model/dataset mismatch")
    evaluation = backup / "test_evaluation"
    if evaluation.exists():
        if (evaluation / "report.json").exists():
            print("Returning the saved test report; no test inference rerun.")
            return read_json(evaluation / "report.json")
        raise RuntimeError("A test attempt was interrupted. Preserve it and inspect before any new evaluation")
    _, YOLO = require_ultralytics()
    verify_prepared(root, manifest, {"test"})
    plot_directory = root / "evaluation" / "test"
    if plot_directory.exists():
        raise RuntimeError("Local test output already exists; inspect the previous attempt")
    evaluation.mkdir(parents=True)
    write_json(evaluation / "started.json", selected)
    model = YOLO(str(best))
    # mAP needs low-confidence predictions; operating P/R uses the selected threshold separately.
    metrics = model.val(data=str(root / "preparation" / "test.yaml"), split="test", imgsz=640,
                        conf=.001, iou=.7, device=0, batch=8, workers=2, plots=True,
                        project=str(root / "evaluation"), name="test", exist_ok=False)
    records = [r for r in manifest["records"] if r["split"] == "test"]
    examples = predict_split(root, model, records)
    operating = operating_counts(examples, selected["threshold"])
    report = {"dataset_id": manifest["dataset_id"], "best_sha256": sha256(best),
              "threshold": selected["threshold"], "mAP50": float(metrics.box.map50),
              "mAP50_95": float(metrics.box.map), "operating_point": operating,
              "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
              "limits": "Internal grouped RDD holdout, not official challenge results; route grouping is heuristic; no local camera validation yet."}
    write_json(evaluation / "predictions.json", examples)
    backup_tree(plot_directory, evaluation / "plots")
    write_json(evaluation / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "operating_point"}, indent=2))
    print("Operating precision/recall:", operating["precision"], operating["recall"])
    return report


def show_test_mistakes(root, backup, limit=6):
    from PIL import Image, ImageDraw, ImageOps
    from IPython.display import display
    backup = Path(backup)
    report = read_json(backup / "test_evaluation" / "report.json")
    predictions = {r["id"]: r for r in read_json(backup / "test_evaluation" / "predictions.json")}
    records = {r["id"]: r for r in load_manifest(root)["records"]}
    mistakes = sorted(report["operating_point"]["images"], key=lambda r: r["fp"] + r["fn"], reverse=True)
    for mistake in [m for m in mistakes if m["fp"] or m["fn"]][:limit]:
        record = records[mistake["id"]]
        with Image.open(Path(root) / record["prepared_image"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in record["boxes"]:
            draw.rectangle(box, outline="lime", width=3)
        for prediction in predictions[record["id"]]["predictions"]:
            if prediction["confidence"] >= report["threshold"]:
                draw.rectangle(prediction["box"], outline="red", width=3)
        print(mistake, "| green=annotation, red=prediction; labels may themselves be incomplete")
        display(ImageOps.contain(image, (900, 600)))


def package_model(root, backup):
    root, backup = Path(root), Path(backup)
    manifest = load_manifest(root)
    selected = read_json(backup / "threshold.json")
    report = read_json(backup / "test_evaluation" / "report.json")
    best = backup / "run" / "weights" / "best.pt"
    if not (selected["best_sha256"] == report["best_sha256"] == sha256(best)
            and selected["dataset_id"] == report["dataset_id"] == manifest["dataset_id"]):
        raise ValueError("Packaging inputs do not belong to the same evaluated run")
    output = root / "delivery"
    output.mkdir(parents=True, exist_ok=True)
    atomic_copy(best, output / "pothole_yolo11n.pt")
    write_json(output / "labels.json", {"0": "pothole"})
    write_json(output / "metadata.json", {**selected, "imgsz": 640, "ultralytics": ULTRALYTICS_VERSION,
               "input": "RGB image; Ultralytics handles letterbox resize and normalization",
               "output": "class 0 pothole; confidence score; xyxy boxes in original image pixels",
               "nms_iou": .7, "max_det": 300, "camera_validation": "not yet measured"})
    write_json(output / "evaluation.json", report)
    for relative in ("environment.json", "training_contract.json", "source_provenance.json"):
        source = backup / relative if relative != "source_provenance.json" else root / relative
        if source.exists():
            atomic_copy(source, output / relative)
    backup_tree(root / "preparation", output / "preparation")
    backup_tree(backup / "run", output / "training")
    (output / "INFERENCE.md").write_text(
        "# BANTAI pothole baseline\n\nInstall ultralytics==" + ULTRALYTICS_VERSION +
        ". Load pothole_yolo11n.pt with ultralytics.YOLO. Call predict(source=your_image, "
        "imgsz=640, conf=" + str(selected["threshold"]) + ", iou=0.7, max_det=300). "
        "Boxes are xyxy in original pixels. Scores are not measured accuracy or severity. "
        "Evaluate actual phone/camera images before claiming suitability.\n\n"
        "Dataset: https://github.com/sekilab/RoadDamageDetector ; cite RDD2022, "
        "Arya et al., Geoscience Data Journal, DOI 10.1002/gdj3.260. "
        "Retain dataset terms and Ultralytics AGPL-3.0 obligations.\n", encoding="utf-8")
    archive = root / "bantai_pothole_first_attempt.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    atomic_copy(archive, backup / archive.name)
    return archive


# NOTEBOOK_BUILDER: source above this marker is embedded verbatim in the notebook.
def build_notebook(destination=None):
    destination = Path(destination or Path(__file__).with_name("BANTAI_Pothole_First_Attempt.ipynb"))
    implementation = Path(__file__).read_text(encoding="utf-8").split("# NOTEBOOK_BUILDER:")[0]
    cells = []
    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": text.strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """# BANTAI — First pothole detector
Run one phase at a time. **Start with Phase 1 only.** No trained result is bundled.
Choose **Runtime → Change runtime type → GPU**. Downloads/extraction/training use
temporary `/content` storage. Only reports, manifests and model artifacts go to Drive.
The self-contained workflow below is generated from `colab/pothole_workflow.py`.
Dataset: [RDD2022 official downloads](https://github.com/sekilab/RoadDamageDetector).
We use only the annotated training portions of India, Japan and Czech; official
unlabeled challenge test images are excluded. Cite Arya et al., RDD2022,
DOI [10.1002/gdj3.260](https://doi.org/10.1002/gdj3.260).
""")
    add("markdown", """## Setup — configuration and Drive
Use the same `ATTEMPT` to recover an interrupted run. Use a new name when changing
data/review/training choices. On a new runtime rerun Setup and Phases 1–2 to rebuild
local files. Local data is temporary; no raw dataset is uploaded to Drive.
""")
    add("code", """from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
ATTEMPT = 'pothole_attempt_001'
assert ATTEMPT.replace('_', '').isalnum(), 'Use letters, numbers, underscores only'
ROOT = Path('/content') / ATTEMPT
BACKUP = Path('/content/drive/MyDrive/BANTAI/pothole') / ATTEMPT
ROOT.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)
print('Working files:', ROOT)
print('Persistent results:', BACKUP)
""")
    add("code", implementation)
    add("markdown", """## Phase 1 — download, audit and inspect
About 1.8 GB compressed plus extraction space. Rerunning verifies cached files;
an interrupted download retries the affected archive from the start. Download
hashes record provenance. If the old country link returns 403/404/410, the notebook
uses the official Figshare API and partial HTTP requests to retrieve just that
country ZIP from the large master archive. The full 13 GB archive is not downloaded.
Country-member CRC is checked; the full master-archive MD5 is not verified.
Missing XML is **not** assumed to mean a normal road. Invalid data is quarantined,
not silently repaired. Review both red boxes and candidate-background samples.
""")
    add("code", """download_sources(ROOT)
audit = audit_sources(ROOT)
show_samples(ROOT, audit)
backup_tree(ROOT / 'inspection', BACKUP / 'inspection')
atomic_copy(ROOT / 'source_provenance.json', BACKUP / 'source_provenance.json')
print('STOP HERE. Review counts, samples and inspection/audit.json before Phase 2.')
""")
    add("markdown", """## Phase 2 — approve review, group and prepare
**Do not use Run all.** Approval is off by default. Enter image IDs from the audit
for exclusions or manual route/capture groups. Small/difficult boxes are flagged
but retained unless excluded. All Phase 1 preview images and their related groups
are assigned to training, never validation/test.
Grouping uses near-duplicate dHash and conservative 100-number filename blocks.
These are heuristics, **not verified route identities**; supply manual groups when
known. 70/15/15 ratios are approximate; inspect actual counts. No augmentation or
resampling is applied to validation/test. No automatic oversampling in attempt 1.
""")
    add("code", """REVIEW_APPROVED = False  # Change only after inspecting Phase 1
EXCLUDE_IDS = []  # Example: 'India/India/train/images/India_000001.jpg'; use actual audit IDs
EXTRA_GROUPS = {}  # Optional: {'actual image ID': 'same-route-or-capture-session'}
manifest = prepare_dataset(ROOT, BACKUP, REVIEW_APPROVED, EXCLUDE_IDS, EXTRA_GROUPS)
""")
    add("markdown", """## Phase 3 — Colab-only dependency setup and training
This cell installs the exact Ultralytics release **in Colab only**, not your PC.
Colab supplies Torch, Pillow and GPU drivers; capture the resolved package inventory
in the result rather than replacing the GPU stack. Restart the session if Colab
requests it, then rerun setup/Phases 1–2. Use only official pretrained weights.
Training: YOLO11n, 640 px, max 30 epochs, patience 7, moderate augmentation, automatic
batch sizing at 60% GPU memory. Check actual class/background/country counts first.
Checkpoints are copied atomically to Drive after each completed epoch. An abrupt
disconnect can lose the current epoch; set RESUME=True to use the last saved one.
""")
    add("code", """import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.3.221'], check=True)
""")
    add("code", """START_TRAINING = False  # Change only after reviewing Phase 2
RESUME = False  # True after an interrupted run with saved last.pt
best_weights = train_baseline(ROOT, BACKUP, start=START_TRAINING, resume=RESUME)
print('Best validation checkpoint:', best_weights)
""")
    add("markdown", """## Phase 4a — choose confidence using validation only
Choose maximum F1 at IoU 0.50 on a fixed 0.05–0.95 grid. Keep test untouched.
The threshold report and validation predictions are saved to Drive.
""")
    add("code", """selection = select_threshold(ROOT, BACKUP)
print('Confidence threshold:', selection['threshold'])
print('Validation precision:', selection['validation']['precision'])
print('Validation recall:', selection['validation']['recall'])
""")
    add("markdown", """## Phase 4b — one held-out test evaluation
Enable only after freezing the model and threshold. This performs an mAP validation
pass and a separate prediction pass for P/R at your frozen threshold, as one recorded
evaluation. Rerunning returns the saved report. A failed attempt is retained and
blocks automatic retry. Do not tune against test results. This is an internal RDD
holdout, not an official challenge score or proof of local camera performance.
""")
    add("code", """RUN_TEST_ONCE = False
test_report = evaluate_once(ROOT, BACKUP, approved=RUN_TEST_ONCE)
show_test_mistakes(ROOT, BACKUP)
""")
    add("markdown", """## Phase 5 — package and try your own photos
Package weights, labels, preprocessing/threshold metadata, training curves/settings,
split manifests, evaluation and package inventory. Raw RDD images are not included.
Then upload **new** phone/camera photos: potholes, normal roads, shadows, puddles and
patches. Inspect misses/false alarms. Uploaded photos are used locally, not added
to training or copied to Drive. A confidence score is not accuracy or severity.
""")
    add("code", """package = package_model(ROOT, BACKUP)
print('Saved locally and to Drive:', package.name)
from google.colab import files
files.download(str(package))
""")
    add("code", """from PIL import Image, ImageOps
from IPython.display import display
from google.colab import files
import io
_, YOLO = require_ultralytics()
detector = YOLO(str(BACKUP / 'run/weights/best.pt'))
threshold = read_json(BACKUP / 'threshold.json')['threshold']
uploaded = files.upload()
for name, contents in uploaded.items():
    try:
        with Image.open(io.BytesIO(contents)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        result = detector.predict(image, imgsz=640, conf=threshold, iou=.7,
                                  max_det=300, device=0, verbose=False)[0]
        print(name, '| potholes:', len(result.boxes))
        display(Image.fromarray(result.plot()[:, :, ::-1]))
        print([{'class': 'pothole', 'confidence': round(float(score), 4), 'xyxy': box}
               for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())])
    except (OSError, ValueError):
        print(name, 'could not be read as an image; skipped')
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": destination.name}, "accelerator": "GPU",
        "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [{**cell, "id": f"bantai-{index:02d}"} for index, cell in enumerate(cells)]}
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-notebook"]:
        print(build_notebook())
    else:
        raise SystemExit("Use --build-notebook; execute training from Colab")
