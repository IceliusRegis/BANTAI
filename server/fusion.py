"""Incident Aggregator & Hotspot Prioritization Engine for BANTAI.

Implements:
- Section 6.2 Unified Dual-Model Response Schema (Model A Trash + Model B Road Defects)
- Section 6.3 / AGENT_INTEGRATION_GUIDE.md Section 7 Calibrated Municipal Hotspot Scoring Formula:
    H = min(100, 35.0 * N_pile + 15.0 * N_group + 3.0 * N_litter + sum(W_s * N_pothole,s))
    where W_low = 25.0, W_moderate = 32.5, W_high = 40.0 (includes "Critical")
- Smart frame filtering:
    * Multi-frame video: only return frames containing valid target detections (confirmed
      piles/groups of litter or pothole defects, while filtering out clean/empty frames).
    * Still image: return the staged single frame with complete regional grouping and pothole findings.
- Per-frame hotspot scoring and priority level.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Calibrated Municipal Weights (Guide Section 6.3 & AGENT_INTEGRATION_GUIDE.md Section 7)
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

    has_multi_item_regions = any(
        (trash.get("regions", {}).get("garbage_piles", 0) > 0 or
         trash.get("regions", {}).get("groups_of_litter", 0) > 0)
        for trash in trash_findings_by_frame
    )
    has_any_trash = any(trash.get("total_items", 0) > 0 for trash in trash_findings_by_frame)

    retained_indices: List[int] = []
    for idx, _ in enumerate(frame_entries):
        trash = trash_findings_by_frame[idx] if idx < len(trash_findings_by_frame) else {}
        road = road_findings_by_frame[idx] if idx < len(road_findings_by_frame) else {}

        total_items = trash.get("total_items", 0)
        piles = trash.get("regions", {}).get("garbage_piles", 0)
        groups = trash.get("regions", {}).get("groups_of_litter", 0)
        defects = road.get("defects", [])
        num_defects = len(defects)

        if is_video:
            if has_multi_item_regions:
                # Primary multi-item trash scene: keep confirmed group/pile frames,
                # or frames with high-confidence pothole craters if no trash is present in the video
                if piles > 0 or groups > 0 or total_items >= 2:
                    retained_indices.append(idx)
            elif has_any_trash:
                if total_items > 0 or num_defects > 0:
                    retained_indices.append(idx)
            else:
                # Road/pothole video (0 trash items): keep frames with detected potholes
                if num_defects > 0:
                    retained_indices.append(idx)
        else:
            # Still image: always retain the single staged photo
            if total_items > 0 or num_defects > 0 or len(frame_entries) == 1:
                retained_indices.append(idx)

    # Fallback: if all frames were filtered out (completely clean video), keep the first frame
    if not retained_indices and frame_entries:
        retained_indices = [0]

    total_piles = 0
    total_groups = 0
    total_litter = 0
    potholes_by_severity: Dict[str, int] = {"low": 0, "moderate": 0, "high": 0}

    frames_output: List[Dict[str, Any]] = []
    for idx in retained_indices:
        entry = frame_entries[idx]
        trash = trash_findings_by_frame[idx] if idx < len(trash_findings_by_frame) else {}
        road = road_findings_by_frame[idx] if idx < len(road_findings_by_frame) else {}

        r_counts = trash.get("regions", {})
        f_piles = r_counts.get("garbage_piles", 0)
        f_groups = r_counts.get("groups_of_litter", 0)
        f_litter = r_counts.get("individual_items", 0)

        f_potholes_by_sev: Dict[str, int] = {"low": 0, "moderate": 0, "high": 0}
        for defect in road.get("defects", []):
            sev_raw = str(defect.get("severity", "moderate")).lower()
            sev_key = "high" if sev_raw in ("critical", "high") else ("low" if sev_raw == "low" else "moderate")
            f_potholes_by_sev[sev_key] = f_potholes_by_sev.get(sev_key, 0) + 1
            potholes_by_severity[sev_key] = potholes_by_severity.get(sev_key, 0) + 1

        # Accumulate totals
        total_piles += f_piles
        total_groups += f_groups
        total_litter += f_litter

        # Frame-level Hotspot Score & Priority
        f_has_high_pothole = f_potholes_by_sev.get("high", 0) > 0
        f_score = calculate_hotspot_score(f_piles, f_groups, f_litter, f_potholes_by_sev)
        f_prio, _ = determine_priority_and_dispatch(f_score, f_piles > 0, f_has_high_pothole)

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

    # Calculate overall incident Hotspot Score based on peak frame hazard (for video) or combined totals (for image)
    if is_video and frames_output:
        hotspot_score = max(f["frame_hotspot_score"] for f in frames_output)
        incident_piles = max(f["trash_findings"].get("regions", {}).get("garbage_piles", 0) for f in frames_output)
        incident_groups = max(f["trash_findings"].get("regions", {}).get("groups_of_litter", 0) for f in frames_output)
        incident_litter = max(f["trash_findings"].get("regions", {}).get("individual_items", 0) for f in frames_output)
        incident_potholes = max(len(f["road_defect_findings"].get("defects", [])) for f in frames_output)
        best_pothole_frame = max(frames_output, key=lambda f: len(f["road_defect_findings"].get("defects", [])))
        video_potholes_by_sev: Dict[str, int] = {"low": 0, "moderate": 0, "high": 0}
        for defect in best_pothole_frame["road_defect_findings"].get("defects", []):
            sev_raw = str(defect.get("severity", "moderate")).lower()
            sev_key = "high" if sev_raw in ("critical", "high") else ("low" if sev_raw == "low" else "moderate")
            video_potholes_by_sev[sev_key] = video_potholes_by_sev.get(sev_key, 0) + 1
        summary_potholes_by_severity = video_potholes_by_sev
        total_potholes = incident_potholes
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
        summary_potholes_by_severity = potholes_by_severity
        total_potholes = sum(potholes_by_severity.values())

    has_pile = incident_piles > 0
    has_high_pothole = summary_potholes_by_severity.get("high", 0) > 0
    priority_level, dispatch_recommendation = determine_priority_and_dispatch(hotspot_score, has_pile, has_high_pothole)

    source_media_out = dict(media_meta)
    if media_meta.get("filename"):
        source_media_out["media_url"] = f"{media_url_prefix}/{incident_id}/source/{media_meta['filename']}"

    return {
        "incident_id": incident_id,
        "source_media": source_media_out,
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
            "total_potholes": total_potholes,
            "potholes_by_severity": summary_potholes_by_severity,
        },
        "frames": frames_output,
    }
