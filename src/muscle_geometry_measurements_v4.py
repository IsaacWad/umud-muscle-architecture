"""Recompute fascicle lengths after fitting with scaled off-canvas apo slopes."""

from __future__ import annotations

import numpy as np


def _adjusted_hit_distance(seed, direction, apo, baseline_hit, slope_fraction):
    """Return the nearest forward hit after changing only the outward rays."""
    edge_x = np.asarray(apo["extended_x"], dtype=float)[[0, -1]]
    edge_y = np.asarray(apo["extended_y"], dtype=float)[[0, -1]]
    if edge_x[0] <= baseline_hit["point"][0] <= edge_x[1]:
        return baseline_hit["ray_distance_px"]

    observed_x = np.asarray(apo["center_x"], dtype=float)
    observed_y = np.asarray(apo["smooth_y"], dtype=float)
    window = min(25, len(observed_x))
    vx, vy = direction
    hits = []
    for edge, points, outward in (
        (0, slice(0, window), -1),
        (1, slice(-window, None), 1),
    ):
        slope = slope_fraction * np.polyfit(observed_x[points], observed_y[points], 1)[0]
        denominator = vy - slope * vx
        if abs(denominator) < 1e-10:
            continue
        distance = (edge_y[edge] + slope * (seed[0] - edge_x[edge]) - seed[1]) / denominator
        hit_x = seed[0] + distance * vx
        if np.isfinite(distance) and distance > 1e-8 and outward * (hit_x - edge_x[edge]) >= -1e-8:
            hits.append(float(distance))
    return min(hits) if hits else np.nan


def apply_fl_extension(measured, region, mm_per_model_px, *, slope_fraction=0.75):
    """Update the FL summary while preserving the original fit records."""
    if not np.isfinite(slope_fraction) or not 0 <= slope_fraction <= 1:
        raise ValueError("slope_fraction must be finite and between 0 and 1")

    summary = measured["summary"]
    adjustments = []
    measured["fl_extension_adjustments"] = adjustments
    for fit in measured["straight_fascicles"]:
        seed = np.array([fit["seed_x"], fit["seed_y"]], dtype=float)
        direction = np.array([fit["tx"], fit["ty"]], dtype=float)
        upper_direction = direction if direction[1] < 0 else -direction
        upper_distance = _adjusted_hit_distance(
            seed, upper_direction, region["upper_apo"], fit["upper_intersection"], slope_fraction,
        )
        lower_distance = _adjusted_hit_distance(
            seed, -upper_direction, region["lower_apo"], fit["lower_intersection"], slope_fraction,
        )
        adjusted_length = upper_distance + lower_distance
        missing_hit = not np.isfinite(adjusted_length)
        adjustments.append({
            "seed_x": fit["seed_x"],
            "seed_y": fit["seed_y"],
            "baseline_fl_px": fit["fl_px"],
            "fl_px": fit["fl_px"] if missing_hit or slope_fraction == 1 else adjusted_length,
            "upper_distance_px": upper_distance,
            "lower_distance_px": lower_distance,
            "retained_baseline": missing_hit,
        })

    if not adjustments:
        return

    summary["baseline_FL_straight_px"] = summary["FL_straight_px"]
    if "FL_straight_mm" in summary:
        summary["baseline_FL_straight_mm"] = summary["FL_straight_mm"]
    summary["fl_extension_slope_fraction"] = slope_fraction
    summary["n_fl_extension_fallbacks"] = sum(row["retained_baseline"] for row in adjustments)
    summary["FL_straight_px"] = float(np.median([row["fl_px"] for row in adjustments]))
    if "FL_straight_mm" in summary:
        summary["FL_straight_mm"] = summary["FL_straight_px"] * float(mm_per_model_px)
