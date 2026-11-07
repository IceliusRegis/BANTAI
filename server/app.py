"""BANTAI Pothole & Road Defect Municipal Web API Server (Starlette + Uvicorn).

Serves:
- GET /api/v1/health: Environment and Pothole Model bundle readiness
- POST /api/v1/analyze: Single-pass upload, 1.0 FPS extraction, Pothole detection, annotated video preview
- GET /api/v1/media/{incident_id}/frames/{filename}: Serves extracted JPEG frames
- GET /api/v1/media/{incident_id}/preview.mp4: Streams annotated preview video
- POST /api/v1/generate-preview/{incident_id}: Triggers or returns preview video URL
"""
from __future__ import annotations

import datetime
from pathlib import Path
import shutil
import tempfile
import uuid

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from .media_ingestion import ingest_media
from .pothole_service import BantaiPotholeService
from .fusion import fuse_dual_model_incident

# Keep storage in system temp directory to prevent triggering Vite's file watcher in development
STORAGE_ROOT = Path(tempfile.gettempdir()) / "bantai_media_storage"
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

pothole_service = BantaiPotholeService()


async def health_check(request: Request) -> JSONResponse:
    """Returns system status, device, and readiness of the Pothole Model."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_b_pothole": {
            "name": pothole_service.bundle_name,
            "bundle_dir": str(pothole_service.bundle_dir),
            "weights_file": str(pothole_service.weights_path) if pothole_service.weights_path else None,
            "weights_present": pothole_service.is_weights_present(),
            "model_loaded": pothole_service.is_model_loaded(),
            "calibrated_threshold": pothole_service.calibrated_conf,
            "device": pothole_service.device,
        },
    })


async def analyze_media(request: Request) -> JSONResponse:
    """Single-pass media upload, 1.0 FPS extraction, and Pothole Model analysis."""
    form = await request.form()
    uploaded_file = form.get("file")
    if not uploaded_file or not hasattr(uploaded_file, "filename"):
        return JSONResponse({"error": "No media file uploaded"}, status_code=400)

    location = str(form.get("location", "J. P. Rizal St. near Market"))
    try:
        sample_fps = float(form.get("sample_fps", 1.0))
    except (ValueError, TypeError):
        sample_fps = 1.0

    today_str = datetime.datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    incident_id = f"INC-{today_str}-{short_uuid}"

    incident_dir = STORAGE_ROOT / incident_id
    frames_dir = incident_dir / "frames"
    incident_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file safely outside Vite workspace
    uploaded_path = incident_dir / uploaded_file.filename
    with open(uploaded_path, "wb") as f_out:
        shutil.copyfileobj(uploaded_file.file, f_out)

    try:
        # Step 1: Shared Single-Pass Ingestion
        media_meta, frame_entries = ingest_media(
            media_path=uploaded_path,
            extracted_frames_dir=frames_dir,
            sample_fps=sample_fps,
            max_frames=30,
        )

        is_video = media_meta["type"] == "video"

        # Step 2: Pothole Detection (Bundle v5 + Cell 8A Vision Engine)
        road_findings = pothole_service.run_on_frames(
            frame_entries=frame_entries,
            frames_dir=frames_dir,
            is_video=is_video,
        )

        # Empty trash findings (Trash model removed per user directive)
        empty_trash_findings = [
            {
                "model_version": "none",
                "total_items": 0,
                "regions": {"garbage_piles": 0, "groups_of_litter": 0, "individual_items": 0},
                "items_by_bantai_category": {},
                "region_list": [],
                "detections": [],
            }
            for _ in frame_entries
        ]

        # Step 3: Video Preview Generation if video
        preview_path = incident_dir / "preview.mp4"
        if is_video and not preview_path.is_file():
            try:
                pothole_service.render_annotated_video(
                    video_path=uploaded_path,
                    output_path=preview_path,
                    frame_entries=frame_entries,
                    pothole_findings_by_frame=road_findings,
                )
            except Exception as exc:
                print(f"[app.py] Could not render annotated video preview: {exc}", flush=True)

        # Step 4: Schema Assembly & Hotspot Prioritization
        unified_response = fuse_dual_model_incident(
            incident_id=incident_id,
            media_meta=media_meta,
            location_str=location,
            frame_entries=frame_entries,
            trash_findings_by_frame=empty_trash_findings,
            road_findings_by_frame=road_findings,
            media_url_prefix="/api/v1/media",
        )

        return JSONResponse(unified_response)

    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def serve_extracted_frame(request: Request) -> FileResponse | JSONResponse:
    """Serves extracted JPEG frames for display in the website."""
    incident_id = request.path_params.get("incident_id")
    filename = request.path_params.get("filename")
    frame_path = STORAGE_ROOT / incident_id / "frames" / filename
    if not frame_path.is_file():
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    return FileResponse(frame_path, media_type="image/jpeg")


async def serve_preview_video(request: Request) -> FileResponse | JSONResponse:
    """Streams the annotated MP4 video preview."""
    incident_id = request.path_params.get("incident_id")
    video_path = STORAGE_ROOT / incident_id / "preview.mp4"
    if not video_path.is_file():
        return JSONResponse({"error": "Preview video not found"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4")


async def generate_preview_endpoint(request: Request) -> JSONResponse:
    """Returns or checks the preview video URL for an incident."""
    incident_id = request.path_params.get("incident_id")
    video_path = STORAGE_ROOT / incident_id / "preview.mp4"
    if video_path.is_file():
        return JSONResponse({
            "status": "ready",
            "preview_video_url": f"/api/v1/media/{incident_id}/preview.mp4",
        })
    return JSONResponse({"error": "Preview video not available for this incident"}, status_code=404)


routes = [
    Route("/api/v1/health", health_check, methods=["GET"]),
    Route("/api/v1/analyze", analyze_media, methods=["POST"]),
    Route("/api/v1/media/{incident_id}/frames/{filename}", serve_extracted_frame, methods=["GET"]),
    Route("/api/v1/media/{incident_id}/preview.mp4", serve_preview_video, methods=["GET"]),
    Route("/api/v1/generate-preview/{incident_id}", generate_preview_endpoint, methods=["POST"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=True, routes=routes, middleware=middleware)
