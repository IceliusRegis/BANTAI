"""BANTAI Dual-Model Municipal Web API Server (Starlette + Uvicorn).

Serves:
- GET /api/v1/health: Environment, Model A (Trash 3-Model Consensus), and Model B (Pothole Bundle v5) readiness
- POST /api/v1/analyze: Single-pass upload, 1.0 FPS extraction, Dual-Model analysis, smart filtering, and annotated video preview
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
from .trash_service import BantaiTrashService
from .pothole_service import BantaiPotholeService
from .fusion import fuse_dual_model_incident

# Keep storage in system temp directory to prevent triggering Vite's file watcher in development
STORAGE_ROOT = Path(tempfile.gettempdir()) / "bantai_media_storage"
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

trash_service = BantaiTrashService()
pothole_service = BantaiPotholeService()


async def health_check(request: Request) -> JSONResponse:
    """Returns system status, device info, and readiness of both models."""
    trash_service.checkpoints_ready = trash_service._verify_checkpoints()
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_a_trash": {
            "name": "Grounding DINO-T 3-Model Consensus & D-112 Spatial Grouping",
            "checkpoints": trash_service.checkpoints_ready,
            "fully_ready": trash_service.is_fully_available(),
            "device": trash_service.device,
        },
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
    """Single-pass media upload, 1.0 FPS extraction, Model A + Model B analysis."""
    form = await request.form()
    uploaded_file = form.get("file")
    if not uploaded_file or not hasattr(uploaded_file, "filename"):
        return JSONResponse({"error": "No media file uploaded"}, status_code=400)

    location = str(form.get("location", "J. P. Rizal St. near Market"))
    try:
        sample_fps = float(form.get("sample_fps", 1.0))
    except (ValueError, TypeError):
        sample_fps = 1.0

    incident_id = str(uuid.uuid4())
    incident_dir = STORAGE_ROOT / incident_id
    frames_dir = incident_dir / "extracted_frames"
    incident_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file safely in system temp directory
    uploaded_path = incident_dir / uploaded_file.filename
    with open(uploaded_path, "wb") as f_out:
        shutil.copyfileobj(uploaded_file.file, f_out)

    try:
        # Step 1: Single-pass media ingestion
        media_meta, frame_entries = ingest_media(
            media_path=uploaded_path,
            extracted_frames_dir=frames_dir,
            sample_fps=sample_fps,
            max_frames=30,
        )

        is_video = media_meta["type"] == "video"

        # Step 2: Model A (Trash Detection & D-112 Spatial Grouping)
        trash_findings = trash_service.run_on_frames(
            frame_entries=frame_entries,
            frames_dir=frames_dir,
            score_threshold=0.15,
            top_k=25,
        )

        # Step 3: Model B (Pothole Detection - Bundle v5 + Cell 8A Vision Engine)
        road_findings = pothole_service.run_on_frames(
            frame_entries=frame_entries,
            frames_dir=frames_dir,
            is_video=is_video,
        )

        # Step 4: Video Preview Generation if video
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

        # Step 5: Fusion & Hotspot Prioritization (with Smart Frame Filtering)
        unified_response = fuse_dual_model_incident(
            incident_id=incident_id,
            media_meta=media_meta,
            location_str=location,
            frame_entries=frame_entries,
            trash_findings_by_frame=trash_findings,
            road_findings_by_frame=road_findings,
            media_url_prefix="/api/v1/media",
        )

        return JSONResponse(unified_response)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


async def serve_extracted_frame(request: Request) -> FileResponse | JSONResponse:
    """Streams an extracted JPEG frame from an incident session."""
    incident_id = request.path_params.get("incident_id")
    filename = request.path_params.get("filename")
    frame_path = STORAGE_ROOT / incident_id / "extracted_frames" / filename

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

