"""BANTAI Dual-Model Municipal Web API Server.

Serves:
- GET /api/v1/health: Environment, Model A, and Model B readiness
- POST /api/v1/analyze: Single-pass upload and dual-model analysis
- GET /api/v1/media/{incident_id}/frames/{filename}: Serves extracted frames
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .media_ingestion import ingest_media
from .trash_service import BantaiTrashService
from .pothole_service import BantaiPotholeService
from .fusion import fuse_dual_model_incident

app = FastAPI(
    title="BANTAI Street Issue AI Detection API",
    version="1.0.0",
    description="Dual-Model Street Issue & Spatial Grouping Municipal API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_ROOT = Path(__file__).resolve().parent / "media_storage"
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

trash_service = BantaiTrashService()
pothole_service = BantaiPotholeService()


@app.get("/api/v1/health")
def health_check():
    """Returns system status, CUDA device, and readiness of both models."""
    trash_service.checkpoints_ready = trash_service._verify_checkpoints()
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_a_trash": {
            "name": "Grounding DINO-T 3-Model Consensus",
            "checkpoints": trash_service.checkpoints_ready,
            "fully_ready": trash_service.is_fully_available(),
            "device": trash_service.device,
        },
        "model_b_pothole": {
            "name": "RF-DETR Nano Road Defect Detector",
            "weights_file": pothole_service.weights_path,
            "weights_present": pothole_service.is_weights_present(),
            "device": pothole_service.device,
        },
    }


@app.post("/api/v1/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    location: str = Form("J. P. Rizal St. near Market"),
    description: Optional[str] = Form(""),
    sample_fps: float = Form(1.0),
):
    """Single-pass media upload, 1.0 FPS extraction, Model A + Model B analysis."""
    # Generate unique incident ID
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    incident_id = f"INC-{today_str}-{short_uuid}"

    incident_dir = STORAGE_ROOT / incident_id
    frames_dir = incident_dir / "frames"
    incident_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    uploaded_path = incident_dir / file.filename
    with open(uploaded_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

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

        # Step 3: Model B (Pothole Detection)
        road_findings = pothole_service.run_on_frames(
            frame_entries=frame_entries,
            frames_dir=frames_dir,
            is_video=is_video,
        )

        # Step 4: Fusion & Hotspot Prioritization
        unified_response = fuse_dual_model_incident(
            incident_id=incident_id,
            media_meta=media_meta,
            location_str=location,
            frame_entries=frame_entries,
            trash_findings_by_frame=trash_findings,
            road_findings_by_frame=road_findings,
            media_url_prefix="/api/v1/media",
        )

        return JSONResponse(content=unified_response)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/media/{incident_id}/frames/{filename}")
def serve_extracted_frame(incident_id: str, filename: str):
    """Serves extracted JPEG frames for display in the website."""
    frame_path = STORAGE_ROOT / incident_id / "frames" / filename
    if not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path, media_type="image/jpeg")
