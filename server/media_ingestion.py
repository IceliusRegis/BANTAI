"""Shared single-pass media ingestion service for BANTAI dual-model pipeline.

When a user uploads a video (.mp4, .mov, .avi, .mkv, .webm), RGB frames are extracted
ONCE at 1.0 FPS into a shared session extracted_frames/ directory.
When a user uploads a still photo (.jpg, .jpeg, .png, .bmp, .webp), it is staged
directly into extracted_frames/ as a single frame (id=10000).

Both Model A (Trash Detection) and Model B (Pothole Detection) consume the EXACT
same extracted_frames/ folder, eliminating redundant decoding.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageOps

# Connect to BANTAI-Modeling- root if present
DEFAULT_MODELING_ROOT = Path(os.environ.get(
    "BANTAI_MODELING_ROOT",
    r"c:\Users\Justine\Documents\Python\Project Image-object Detection"
))
if DEFAULT_MODELING_ROOT.is_dir() and str(DEFAULT_MODELING_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_MODELING_ROOT))


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def stage_single_image(
    image_path: Path,
    output_dir: Path,
    start_id: int = 10000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Stage a single still image with EXIF normalization into output_dir.

    Returns:
        (image_meta, [frame_entry])
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as raw_img:
        img_rgb = ImageOps.exif_transpose(raw_img).convert("RGB")
        width, height = img_rgb.size
        dest_filename = f"frame_0000_{image_path.name}"
        dest_path = output_dir / dest_filename
        img_rgb.save(dest_path, format="JPEG", quality=95)

    img_sha = sha256_file(dest_path)
    file_size = os.path.getsize(dest_path)

    media_meta = {
        "filename": image_path.name,
        "type": "image",
        "sha256": img_sha,
        "width": width,
        "height": height,
        "fps": 1.0,
        "duration_sec": 0.0,
        "total_frames": 1,
        "file_size_bytes": file_size,
    }

    frame_entry = {
        "id": start_id,
        "file_name": dest_filename,
        "file_path": str(dest_path),
        "width": width,
        "height": height,
        "frame_index": 0,
        "timestamp_sec": 0.0,
        "file_sha256": img_sha,
        "media_type": "still_image",
    }

    return media_meta, [frame_entry]


def extract_video_frames_safe(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 1.0,
    max_frames: int = 30,
    start_id: int = 10001,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract RGB JPEG frames from a video file at sample_fps.

    Uses cv2 if installed; gracefully falls back to a descriptive error if cv2 is absent.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError as err:
        raise RuntimeError(
            "OpenCV (cv2) is required for video frame extraction. "
            "Please install opencv-python or test with a still image."
        ) from err

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_sha = sha256_file(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    video_meta = {
        "filename": video_path.name,
        "type": "video",
        "sha256": video_sha,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "sample_fps": sample_fps,
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 3),
        "file_size_bytes": os.path.getsize(video_path),
    }

    frame_step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1
    extracted: List[Dict[str, Any]] = []
    current_frame = 0
    img_id = start_id

    while current_frame < total_frames and len(extracted) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp_sec = round(current_frame / fps, 3) if fps > 0 else 0.0
        file_name = f"frame_{len(extracted)+1:04d}_t{timestamp_sec:04.1f}s.jpg"
        file_path = output_dir / file_name

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img.save(file_path, format="JPEG", quality=95)

        frame_sha = sha256_file(file_path)
        extracted.append({
            "id": img_id,
            "file_name": file_name,
            "file_path": str(file_path),
            "width": width,
            "height": height,
            "frame_index": current_frame,
            "timestamp_sec": timestamp_sec,
            "file_sha256": frame_sha,
            "media_type": "video_frame",
        })

        img_id += 1
        current_frame += frame_step

    cap.release()
    return video_meta, extracted


def ingest_media(
    media_path: Path,
    extracted_frames_dir: Path,
    sample_fps: float = 1.0,
    max_frames: int = 30,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Single-pass ingestion entry point.

    Distinguishes video vs still image by extension and outputs to extracted_frames_dir.
    """
    media_path = Path(media_path)
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    is_video = media_path.suffix.lower() in video_exts

    if is_video:
        return extract_video_frames_safe(
            video_path=media_path,
            output_dir=extracted_frames_dir,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
    return stage_single_image(
        image_path=media_path,
        output_dir=extracted_frames_dir,
    )
