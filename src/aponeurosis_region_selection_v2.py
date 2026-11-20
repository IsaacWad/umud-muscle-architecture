"""Geometry-only selection matching the reviewed notebook baseline.

Extraction is performed by the pipeline with probability threshold 0.20 and
minimum width fraction 0.25. Apo scores remain mean probability times width,
normalized relative to the strongest candidate in that image.

Selection uses vertical 12-model-pixel differences, exp(-median / 2 degrees),
and weights 0.50 boundary, 0.40 continuity, 0.15 internal clearance.
Scores and margins are ranking diagnostics, not probabilities.
No image-specific overrides or permissive detection fallback are applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


def _sample(array, x, y, outside=0.0):
    return map_coordinates(
        array, np.array([y, x]), order=1, mode="constant", cval=outside,
    )


def _sample_directions(field, mask, x, y):
    """Sample axial unit vectors with the notebook's full-mark mask check."""
    cosine = _sample(field[0], x, y, np.nan)
    sine = -_sample(field[1], x, y, np.nan)
    magnitude = np.hypot(cosine, sine)
    valid = (
        np.isfinite(cosine) & np.isfinite(sine) & (magnitude > 1e-8)
    )
    theta = 0.5 * np.arctan2(sine, cosine)
    for offset in np.linspace(-5, 5, 11):
        valid &= _sample(
            mask, x + offset * np.cos(theta), y + offset * np.sin(theta),
        ) > 0.999
    vectors = np.stack([cosine, sine], axis=-1)
    vectors /= np.maximum(magnitude, 1e-8)[..., None]
    vectors[~valid] = np.nan
    return vectors, valid


def _angular_difference(first, second):
    dot = np.sum(first * second, axis=-1)
    return 0.5 * np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def measure_fixed_vertical_continuity(
    geometry, gap_px=12, x_step=16, y_step=12,
):
    """Return fixed-gap continuity and the original depth-support eligibility.

    The five fractional depths are retained only for baseline eligibility.
    Their angular changes do not contribute to region_score. The full
    ten-pixel plotting-mark footprint is retained as a mask-clearance check
    because it was used throughout the reviewed notebook experiment.
    All distances are model pixels.
    """
    if gap_px <= 0 or x_step <= 0 or y_step <= 0:
        raise ValueError("Sampling distances must be positive")
    field = np.asarray(geometry["fascicle_v"], dtype=float)
    height, width = geometry["image_512"].shape
    grid_x, grid_y = np.meshgrid(
        np.arange(0, width, x_step, dtype=float),
        np.arange(0, height - gap_px, y_step, dtype=float),
    )
    x, y = grid_x.ravel(), grid_y.ravel()
    depth_x = np.arange(0, width, 16, dtype=float)
    rows = []
    for region in geometry["candidate_regions"]:
        mask = np.asarray(region["buffered_mask"], dtype=float)
        upper_vectors, upper_ok = _sample_directions(field, mask, x, y)
        lower_vectors, lower_ok = _sample_directions(field, mask, x, y + gap_px)
        valid = upper_ok & lower_ok
        for offset in np.linspace(0, gap_px, int(np.ceil(gap_px)) + 1):
            valid &= _sample(mask, x, y + offset) > 0.999
        changes = _angular_difference(upper_vectors, lower_vectors)[valid]

        upper, lower = region["upper_apo"], region["lower_apo"]
        upper_y = np.interp(depth_x, upper["extended_x"], upper["extended_y"])
        lower_y = np.interp(depth_x, lower["extended_x"], lower["extended_y"])
        depth_vectors, depth_validity = [], []
        for depth in (0.15, 0.325, 0.50, 0.675, 0.85):
            depth_y = upper_y + depth * (lower_y - upper_y)
            vectors, usable = _sample_directions(field, mask, depth_x, depth_y)
            depth_vectors.append(vectors)
            depth_validity.append(usable)
        depth_vectors = np.asarray(depth_vectors)
        depth_validity = np.asarray(depth_validity)
        adjacent_valid = depth_validity[:-1] & depth_validity[1:]
        adjacent_changes = _angular_difference(
            depth_vectors[:-1], depth_vectors[1:],
        )[adjacent_valid]

        rows.append({
            "image_id": geometry["image_id"],
            "pair_name": region["pair_name"],
            "gap_px": gap_px,
            "n_fixed_comparisons": len(changes),
            "fixed_vertical_change_deg": (
                float(np.median(changes)) if len(changes) else np.nan
            ),
            "fixed_vertical_iqr_deg": (
                float(np.percentile(changes, 75) - np.percentile(changes, 25))
                if len(changes) else np.nan
            ),
            "fixed_vertical_p90_deg": (
                float(np.percentile(changes, 90)) if len(changes) else np.nan
            ),
            "supported_depths": int(np.any(depth_validity, axis=1).sum()),
            "interior_coverage": float(depth_validity.mean()),
            "interior_adjacent_change_deg": (
                float(np.median(adjacent_changes))
                if len(adjacent_changes) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def select_region_from_pipeline_result(
    geometry, scale_deg=2.0, boundary_weight=0.50,
    continuity_weight=0.40, clearance_weight=0.15,
):
    """Select anatomy before measurement and return a scored candidate table.

    Result keys match the previous wrapper: image_id, status, candidate_table,
    selected, selected_region. selected_region is the original region object.
    Ties preserve candidate input order. Missing evidence returns an explicit
    insufficient_evidence status rather than selecting through a fallback.
    """
    weights = np.asarray(
        [boundary_weight, continuity_weight, clearance_weight], dtype=float,
    )
    if (
        not np.isfinite(scale_deg) or scale_deg <= 0
        or not np.isfinite(weights).all() or np.any(weights < 0)
    ):
        raise ValueError("Use a positive finite scale and nonnegative finite weights")
    settings = {
        "gap_px": 12, "scale_deg": scale_deg,
        "boundary_weight": boundary_weight,
        "continuity_weight": continuity_weight,
        "clearance_weight": clearance_weight,
    }
    result = {
        "image_id": geometry["image_id"],
        "status": "no_candidate_regions",
        "candidate_table": pd.DataFrame(),
        "selected": None, "selected_region": None,
        "settings": settings,
    }
    regions = geometry["candidate_regions"]
    if not regions:
        return result

    table = measure_fixed_vertical_continuity(geometry)
    candidates = sorted(
        geometry["apo_candidates"], key=lambda candidate: candidate["centroid_y"],
    )
    boundary_rows = []
    for region in regions:
        upper_index = region["upper_candidate_index"]
        lower_index = region["lower_candidate_index"]
        internal_score = max(
            (
                candidate["relative_score"]
                for candidate in candidates[upper_index + 1:lower_index]
            ),
            default=0.0,
        )
        boundary_rows.append({
            "pair_name": region["pair_name"],
            "pair_index": region["pair_index"],
            "upper_candidate_index": upper_index,
            "lower_candidate_index": lower_index,
            "geometry_valid": bool(region["valid"]),
            "upper_apo_score": region["upper_apo"]["relative_score"],
            "lower_apo_score": region["lower_apo"]["relative_score"],
            "boundary_score": min(
                region["upper_apo"]["relative_score"],
                region["lower_apo"]["relative_score"],
            ),
            "strongest_internal_apo_score": internal_score,
            "internal_clearance": 1.0 - np.clip(internal_score, 0.0, 1.0),
        })
    table = table.merge(
        pd.DataFrame(boundary_rows), on="pair_name", validate="one_to_one",
    )
    table["continuity_quality"] = np.exp(
        -table["fixed_vertical_change_deg"] / scale_deg,
    )
    table["evidence_available"] = (
        table["geometry_valid"] & table["supported_depths"].ge(2)
        & np.isfinite(table["interior_adjacent_change_deg"])
        & np.isfinite(table["boundary_score"])
        & np.isfinite(table["continuity_quality"])
    )
    table["boundary_contribution"] = boundary_weight * table["boundary_score"]
    table["continuity_contribution"] = continuity_weight * table["continuity_quality"]
    table["clearance_contribution"] = clearance_weight * table["internal_clearance"]
    table["region_score"] = (
        table["boundary_contribution"] + table["continuity_contribution"]
        + table["clearance_contribution"]
    )
    result["candidate_table"] = table
    ranked = table.loc[table["evidence_available"]].sort_values(
        "region_score", ascending=False, kind="stable",
    )
    if ranked.empty:
        result["status"] = "insufficient_evidence"
        return result

    selected = ranked.iloc[0].to_dict()
    selected.update({
        "n_candidate_regions": len(table),
        "n_eligible_regions": len(ranked),
        "score_margin": (
            float(ranked.iloc[0]["region_score"] - ranked.iloc[1]["region_score"])
            if len(ranked) >= 2 else np.nan
        ),
        "selection_reason": (
            "only_eligible_region" if len(ranked) == 1 else "highest_geometry_score"
        ),
        "selection_fallback": False,
    })
    result.update({
        "status": "selected", "selected": selected,
        "selected_region": next(
            region for region in regions
            if region["pair_name"] == selected["pair_name"]
        ),
    })
    return result
