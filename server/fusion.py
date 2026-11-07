"""Incident Aggregator & Hotspot Prioritization Engine for BANTAI.

Implements:
- Section 6.2 Unified Dual-Model Response Schema (Model A Trash + Model B Road Defects)
- Section 6.3 Calibrated Municipal Hotspot Scoring Formula:
    H = min(100, W_pile * N_pile + W_group * N_group + W_litter * N_litter + W_pothole * N_pothole)
- Smart frame filtering:
    * Multi-frame video: only return frames containing valid target detections (surfacing confirmed
      piles and groups of litter, while filtering out clean/empty and transient frames).
    * Still image: return the staged single frame with complete regional grouping.
- Per-frame hotspot scoring and priority level.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

# Section 6.3 Calibrated Municipal Weights
WEIGHT_PILE = 35.0
WEIGHT_GROUP = 15.0
WEIGHT_LITTER = 3.0
WEIGHT_POTHOLE_BY_SEVERITY = {
    "low": 25.0,
    "moderate": 32.5,
    "high": 40.0,
    "critical": 40.0,
}


def calculate_hotspot_score(
    piles: int,
    groups: int,
    litter: int,
    potholes_by_severity: Dict[str, int],
) -> float:
    """Calculates municipal Hotspot Priority Index H bounded in [0.0, 100.0]."""
    score = (
        (WEIGHT_PILE * max(0, piles))
        + (WEIGHT_GROUP * max(0, groups))
        + (WEIGHT_LITTER * max(0, litter))
    )
    for sev, count in potholes_by_severity.items():
        w = WEIGHT_POTHOLE_BY_SEVERITY.get(sev.lower(), 30.0)
        score += w * max(0, count)

    return round(min(100.0, max(0.0, score)), 1)


def determine_priority_and_dispatch(hotspot_score: float, has_pile: bool, has_high_pothole: bool) -> Tuple[str, str]:
    """Assigns priority level and actionable municipal dispatch recommendation."""
    if hotspot_score >= 75.0 or (has_pile and has_high_pothole):
        priority = "CRITICAL"
        actions = []
        if has_pile:
            actions.append("immediate waste collection crew (garbage pile hazard)")
        if has_high_pothole:
            actions.append("rapid road-patch crew (severe pothole hazard)")
        if not actions:
            actions.append("immediate multi-hazard municipal dispatch")
        dispatch = " + ".join(actions)
    elif hotspot_score >= 40.0:
        priority = "ELEVATED"
        dispatch = "Schedule same-day barangay sanitation & maintenance patrol"
    else:
        priority = "ROUTINE"
        dispatch = "Include in routine scheduled street sweeping & road inspection"

    return priority, dispatch


def fuse_dual_model_incident(
    incident_id: str,
    media_meta: Dict[str, Any],
    location_str: str,
    frame_entries: List[Dict[str, Any]],
    trash_findings_by_frame: List[Dict[str, Any]],
    road_findings_by_frame: List[Dict[str, Any]],
    media_url_prefix: str = "/api/v1/media",
) -> Dict[str, Any]:
    """Constructs the unified dual-model response according to Guide Section 6.2 with smart frame filtering."""
    is_video = media_meta.get("type") == "video"

    # Determine which frames qualify for presentation
    # If the video contains multi-item regions (groups of litter or garbage piles),
    # surface only those confirmed hazard frames so clean frames and transient single-box
    # entries don't clutter the admin queue.
    has_multi_item_regions = any(
        (trash.get("regions", {}).get("garbage_piles", 0) > 0 or
         trash.get("regions", {}).get("groups_of_litter", 0) > 0)
        for trash in trash_findings_by_frame
    )

    retained_indices = []
    for idx, entry in enumerate(frame_entries):
        trash = trash_findings_by_frame[idx] if idx < len(trash_findings_by_frame) else {}
        total_items = trash.get("total_items", 0)
        piles = trash.get("regions", {}).get("garbage_piles", 0)
        groups = trash.get("regions", {}).get("groups_of_litter", 0)

        if is_video:
            if has_multi_item_regions:
                # Keep frames containing groups or piles (or multi-item accumulations)
                if piles > 0 or groups > 0 or total_items >= 2:
                    retained_indices.append(idx)
            else:
                # Keep frames with any valid detections
                if total_items > 0:
                    retained_indices.append(idx)
        else:
            # Still image: keep the frame if it has items or if single photo
            if total_items > 0 or len(frame_entries) == 1:
                retained_indices.append(idx)

    # Fallback: if all frames were filtered out (completely clean video), keep the first frame
    if not retained_indices and frame_entries:
        retained_indices = [0]

    total_piles = 0
    total_groups = 0
    total_litter = 0
    potholes_by_severity: Dict[str, int] = {"low": 0, "moderate": 0, "high": 0}

    frames_output = []
    for idx in retained_indices:
        entry = frame_entries[idx]
        trash = trash_findings_by_frame[idx] if idx < len(trash_findings_by_frame) else {}
        road = road_findings_by_frame[idx] if idx < len(road_findings_by_frame) else {}

        r_counts = trash.get("regions", {})
        f_piles = r_counts.get("garbage_piles", 0)
        f_groups = r_counts.get("groups_of_litter", 0)
        f_litter = r_counts.get("individual_items", 0)

        # Frame-level totals
        total_piles += f_piles
        total_groups += f_groups
        total_litter += f_litter

        # Frame-level Hotspot Score & Priority
        f_score = calculate_hotspot_score(f_piles, f_groups, f_litter, {})
        f_prio, _ = determine_priority_and_dispatch(f_score, f_piles > 0, False)

        frame_url = f"{media_url_prefix}/{incident_id}/frames/{entry['file_name']}"

        frames_output.append({
            "frame_index": entry.get("frame_index", idx),
            "timestamp_sec": entry.get("timestamp_sec", 0.0),
            "frame_url": frame_url,
            "width": entry.get("width", 1280),
            "height": entry.get("height", 720),
            "trash_findings": trash,
            "road_defect_findings": road,
            "frame_hotspot_score": f_score,
            "frame_priority_level": f_prio,
        })

    # Calculate overall incident Hotspot Score based on max frame hazard or combined totals
    # For a video of the same incident, use the peak frame severity
    if is_video and frames_output:
        hotspot_score = max(f["frame_hotspot_score"] for f in frames_output)
        incident_piles = max(f["trash_findings"].get("regions", {}).get("garbage_piles", 0) for f in frames_output)
        incident_groups = max(f["trash_findings"].get("regions", {}).get("groups_of_litter", 0) for f in frames_output)
        incident_litter = max(f["trash_findings"].get("regions", {}).get("individual_items", 0) for f in frames_output)
    else:
        hotspot_score = calculate_hotspot_score(
            piles=total_piles,
            groups=total_groups,
            litter=total_litter,
            potholes_by_severity=potholes_by_severity,
        )
        incident_piles = total_piles
        incident_groups = total_groups
        incident_litter = total_litter

    has_pile = incident_piles > 0
    has_high_pothole = potholes_by_severity.get("high", 0) > 0
    priority_level, dispatch_recommendation = determine_priority_and_dispatch(hotspot_score, has_pile, has_high_pothole)

    return {
        "incident_id": incident_id,
        "source_media": media_meta,
        "location": {
            "street_name": location_str,
            "latitude": 14.653889,   # Pilot Santa Elena, Marikina default
            "longitude": 121.103611,
        },
        "hotspot_score": hotspot_score,
        "priority_level": priority_level,
        "dispatch_recommendation": dispatch_recommendation,
        "summary": {
            "total_frames": len(frames_output),
            "total_extracted_frames": len(frame_entries),
            "total_garbage_piles": incident_piles,
            "total_litter_groups": incident_groups,
            "total_individual_litter": incident_litter,
            "total_potholes": sum(potholes_by_severity.values()),
            "potholes_by_severity": potholes_by_severity,
        },
        "frames": frames_output,
    }
