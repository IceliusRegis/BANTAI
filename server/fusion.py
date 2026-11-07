"""Incident Aggregator & Hotspot Prioritization Engine for BANTAI.

Implements:
- Section 6.2 Unified Dual-Model Response Schema (Model A Trash + Model B Road Defects)
- Universal 0-100% normalized box coordinates (box_pct / boxPct) for responsive UI rendering
- Section 6.3 Calibrated Municipal Hotspot Scoring Formula:
    H = min(100, W_pile * N_pile + W_group * N_group + W_litter * N_litter + W_pothole * N_pothole)
- Smart frame filtering:
    * Multi-frame video: surfaces confirmed hazard frames (potholes, garbage piles, litter groups)
      while filtering out redundant empty/transient frames.
    * Still image: retains staged frame with complete regional grouping.
- Per-frame hotspot scoring, priority level, and peak hazard incident scoring.
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

    # Smart frame filtering:
    # Determine which frames qualify for presentation.
    # If the video contains confirmed hazards (potholes, groups of litter, or garbage piles),
    # surface those confirmed frames so clean frames and transient single boxes don't clutter the queue.
    has_confirmed_hazards = any(
        (trash.get("regions", {}).get("garbage_piles", 0) > 0 or
         trash.get("regions", {}).get("groups_of_litter", 0) > 0 or
         len(road.get("defects", [])) > 0)
        for trash, road in zip(
            trash_findings_by_frame,
            road_findings_by_frame if road_findings_by_frame else [{}] * len(trash_findings_by_frame)
        )
    )

    retained_indices = []
    for idx, entry in enumerate(frame_entries):
        trash = trash_findings_by_frame[idx] if idx < len(trash_findings_by_frame) else {}
        road = road_findings_by_frame[idx] if idx < len(road_findings_by_frame) else {}

        total_trash_items = trash.get("total_items", 0)
        piles = trash.get("regions", {}).get("garbage_piles", 0)
        groups = trash.get("regions", {}).get("groups_of_litter", 0)
        potholes = len(road.get("defects", []))

        if is_video:
            if has_confirmed_hazards:
                if piles > 0 or groups > 0 or total_trash_items >= 2 or potholes > 0:
                    retained_indices.append(idx)
            else:
                if total_trash_items > 0 or potholes > 0:
                    retained_indices.append(idx)
        else:
            if total_trash_items > 0 or potholes > 0 or len(frame_entries) == 1:
                retained_indices.append(idx)

    # Fallback: if all frames were filtered out (completely clean video), keep first frame
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

        total_piles += f_piles
        total_groups += f_groups
        total_litter += f_litter

        combined_boxes = []

        # Process Road Potholes
        pothole_defects = road.get("defects", [])
        f_potholes_sev: Dict[str, int] = {"low": 0, "moderate": 0, "high": 0}
        for d in pothole_defects:
            sev = str(d.get("severity", "moderate")).lower()
            if sev in potholes_by_severity:
                potholes_by_severity[sev] += 1
                f_potholes_sev[sev] += 1
            else:
                potholes_by_severity["moderate"] += 1
                f_potholes_sev["moderate"] += 1

            b_box_pct = d.get("box_pct", {"x": 42.0, "y": 65.0, "w": 16.0, "h": 11.0})
            b_conf_pct = d.get("confidence_pct", int(d.get("confidence", 0.8) * 100))
            combined_boxes.append({
                "id": f"pothole-{d.get('defect_id', len(combined_boxes)+1)}",
                "kind": "pothole",
                "label": d.get("label", "Pothole"),
                "severity": d.get("severity", "moderate"),
                "confidence_pct": b_conf_pct,
                "confidencePct": b_conf_pct,
                "count": 1,
                "box_pct": b_box_pct,
                "boxPct": b_box_pct,
                "estimated_diameter_cm": d.get("estimated_diameter_cm", 35.0),
            })

        # Process Trash Regions
        trash_regions = trash.get("region_list", [])
        for r in trash_regions:
            t_box_pct = r.get("box_pct", {"x": 36.0, "y": 58.0, "w": 22.0, "h": 16.0})
            t_conf_pct = r.get("confidence_pct", 88)
            combined_boxes.append({
                "id": f"trash-{r.get('region_id', len(combined_boxes)+1)}",
                "kind": "trash",
                "label": r.get("display_label", "Trash"),
                "category": r.get("category", "individual_litter"),
                "confidence_pct": t_conf_pct,
                "confidencePct": t_conf_pct,
                "count": r.get("item_count", 1),
                "box_pct": t_box_pct,
                "boxPct": t_box_pct,
                "category_breakdown": r.get("category_breakdown", {}),
            })

        # Frame-level Hotspot Score & Priority
        f_score = calculate_hotspot_score(f_piles, f_groups, f_litter, f_potholes_sev)
        f_prio, _ = determine_priority_and_dispatch(f_score, f_piles > 0, f_potholes_sev.get("high", 0) > 0)

        frame_url = f"{media_url_prefix}/{incident_id}/frames/{entry['file_name']}"

        frames_output.append({
            "frame_index": entry.get("frame_index", idx),
            "timestamp_sec": entry.get("timestamp_sec", 0.0),
            "frame_url": frame_url,
            "width": entry.get("width", 1280),
            "height": entry.get("height", 720),
            "trash_findings": trash,
            "road_defect_findings": road,
            "boxes": combined_boxes,
            "pothole_count": len(pothole_defects),
            "trash_count": sum(r.get("item_count", 1) for r in trash_regions) or trash.get("total_items", 0),
            "frame_hotspot_score": f_score,
            "frame_priority_level": f_prio,
        })

    # Calculate overall incident Hotspot Score based on max frame hazard or combined totals
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

    preview_video_url = f"{media_url_prefix}/{incident_id}/preview.mp4" if is_video else None

    total_potholes = sum(potholes_by_severity.values())
    total_trash_items = (incident_piles * 8) + (incident_groups * 3) + incident_litter

    return {
        "incident_id": incident_id,
        "source_media": media_meta,
        "preview_video_url": preview_video_url,
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
            "total_potholes": total_potholes,
            "total_trash_items": total_trash_items,
            "total_garbage_piles": incident_piles,
            "total_litter_groups": incident_groups,
            "total_individual_litter": incident_litter,
            "potholes_by_severity": potholes_by_severity,
        },
        "frames": frames_output,
    }
