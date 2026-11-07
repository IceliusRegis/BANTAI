"""BANTAI Dual-Model Municipal Web API Server (Starlette + Uvicorn).

Serves:
- GET /api/v1/health: Environment, Model A, and Model B readiness
- POST /api/v1/analyze: Single-pass upload and dual-model analysis
- GET /api/v1/media/{incident_id}/frames/{filename}: Serves extracted frames
"""
from __future__ import annotations

import datetime
from pathlib import Path
import shutil
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

STORAGE_ROOT = Path(__file__).resolve().parent / "media_storage"
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

trash_service = BantaiTrashService()
pothole_service = BantaiPotholeService()


async def health_check(request: Request) -> JSONResponse:
    """Returns system status, CUDA device, and readiness of both models."""
    trash_service.checkpoints_ready = trash_service._verify_checkpoints()
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_a_trash": {
            "name": "Grounding DINO-T 3-Model Consensus",
            "checkpoints": trash_service.checkpoints_ready,
            "fully_ready": trash_service.is_fully_available(),
            "device": trash_service.device,
        },
        "model_b_pothole": {
            "name": "RF-DETR Nano Road Defect Detector (Disabled for Model A Focus)",
            "status": "disabled",
            "weights_present": False,
            "device": "none",
        },
    })


async def analyze_media(request: Request) -> JSONResponse:
    """Single-pass media upload, 1.0 FPS extraction, Model A analysis with Model B disabled."""
    form = await request.form()
    uploaded_file = form.get("file")
    if not uploaded_file or not hasattr(uploaded_file, "filename"):
        return JSONResponse({"error": "No media file uploaded"}, status_code=400)

    location = str(form.get("location", "J. P. Rizal St. near Market"))
    description = str(form.get("description", ""))
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

    # Save uploaded file
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

        # Step 2: Model A (Trash Detection & D-112 Spatial Grouping)
        trash_findings = trash_service.run_on_frames(
            frame_entries=frame_entries,
            frames_dir=frames_dir,
            score_threshold=0.15,
            top_k=25,
        )

        # Step 3: Model B (Pothole Detection) - Disabled for Model A Focus
        road_findings = [
            {"model_version": "disabled", "total_defects": 0, "defects": []}
            for _ in frame_entries
        ]

        # Step 4: Fusion & Hotspot Prioritization (with Smart Frame Filtering)
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
    """Serves extracted JPEG frames for display in the website."""
    incident_id = request.path_params.get("incident_id")
    filename = request.path_params.get("filename")
    frame_path = STORAGE_ROOT / incident_id / "frames" / filename
    if not frame_path.is_file():
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    return FileResponse(frame_path, media_type="image/jpeg")


routes = [
    Route("/api/v1/health", health_check, methods=["GET"]),
    Route("/api/v1/analyze", analyze_media, methods=["POST"]),
    Route("/api/v1/media/{incident_id}/frames/{filename}", serve_extracted_frame, methods=["GET"]),
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
