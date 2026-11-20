"""Selected-region measurements with straight-line orientation fitting.

Existing PA and MT calculations are reused from v1. Straight FL fits one line
through each of five interior seeds by minimizing mean squared axial angular
error along the contiguous buffered segment containing that seed.

Uniform arc-length weights are the default. Optional field_weights must be
supplied explicitly. Raw vector magnitude is never treated as confidence.
Lengths are model pixels; optional calibrated summary lengths are millimetres.
A successful fit is a geometric result, not a physiological validity judgment.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar

from src.muscle_geometry_measurements_v1 import (
    estimate_muscle_thickness,
    estimate_pennation_angles,
    intersect_parametric_lines,
)


def _ray_apo_intersection(seed, direction, apo, endpoint_window=25):
    """Nearest forward intersection with reconstructed apo or end extension."""
    x = np.asarray(apo["extended_x"], dtype=float)
    y = np.asarray(apo["extended_y"], dtype=float)
    points = np.column_stack([x, y])
    segments = np.diff(points, axis=0)
    offset = points[:-1] - seed
    vx, vy = direction
    determinant = vx * segments[:, 1] - vy * segments[:, 0]
    valid = np.abs(determinant) > 1e-10
    distances = np.full(len(segments), np.nan)
    fractions = np.full(len(segments), np.nan)
    distances[valid] = (
        offset[valid, 0] * segments[valid, 1]
        - offset[valid, 1] * segments[valid, 0]
    ) / determinant[valid]
    fractions[valid] = (
        offset[valid, 0] * vy - offset[valid, 1] * vx
    ) / determinant[valid]
    valid &= (
        (distances > 1e-8) & (fractions >= -1e-8) & (fractions <= 1 + 1e-8)
    )
    hits = []
    if valid.any():
        distance = float(np.min(distances[valid]))
        hits.append((distance, seed + distance * direction))

    observed_x = np.asarray(apo["center_x"], dtype=float)
    observed_y = np.asarray(apo["smooth_y"], dtype=float)
    window = min(endpoint_window, len(observed_x))
    for side, edge, part, sign in (
        ("left", points[0], slice(0, window), -1.0),
        ("right", points[-1], slice(-window, None), 1.0),
    ):
        slope = float(np.polyfit(observed_x[part], observed_y[part], 1)[0])
        outward = np.array([sign, sign * slope])
        intersection = intersect_parametric_lines(seed, direction, edge, outward)
        if intersection is not None:
            distance = intersection["fascicle_distance_px"]
            extension = intersection["apo_extension_distance_px"]
            if distance > 1e-8 and extension >= -1e-8:
                hits.append((distance, np.asarray(intersection["point"], dtype=float)))

    if not hits:
        return None
    distance, point = min(hits, key=lambda hit: hit[0])
    if not np.isfinite(distance) or not np.isfinite(point).all():
        return None
    extension_side = None
    extension_distance = 0.0
    if point[0] < observed_x[0]:
        extension_side = "left"
        extension_distance = float(np.linalg.norm(
            point - np.array([observed_x[0], observed_y[0]]),
        ))
    elif point[0] > observed_x[-1]:
        extension_side = "right"
        extension_distance = float(np.linalg.norm(
            point - np.array([observed_x[-1], observed_y[-1]]),
        ))
    return {
        "point": (float(point[0]), float(point[1])),
        "ray_distance_px": float(distance),
        "intersection_type": "extended" if extension_side else "observed",
        "extension_side": extension_side,
        "apo_extension_distance_px": extension_distance,
    }


def fit_straight_fascicle_from_seed(
    seed_x, seed_y, fascicle_orientation, upper_apo, lower_apo,
    buffered_muscle_mask, field_weights=None, sample_step=1.0,
    angle_step_deg=1.0, min_field_samples=5,
):
    """Fit an axial line through a seed using a global angular grid and refinement.

    For each candidate angle, find the upper and lower apo intersections.
    Evaluate angular errors only on the contiguous, valid buffered segment
    containing the seed, bounded by those intersections. Uniformly sampled
    points have equal weights unless field_weights is supplied.

    The objective is weighted mean squared axial angle error in degrees squared.
    Search image-coordinate angles in [-90, 90), then locally refine around
    the best valid grid angle. Opposite axial directions are equivalent.
    Report missing fits as None. Endpoint extrapolation is reported explicitly;
    there is no physiological FL cap or guarantee that long extrapolation is
    accurate. Fits relying on very short observed segments need visual review.
    """
    if (
        not np.isfinite(sample_step) or sample_step <= 0
        or not np.isfinite(angle_step_deg) or not 0 < angle_step_deg <= 90
        or min_field_samples < 2
    ):
        raise ValueError("Use positive sampling steps and at least two field samples")
    mask = np.asarray(buffered_muscle_mask, dtype=float)
    field = np.asarray(fascicle_orientation, dtype=float)
    if mask.ndim != 2 or field.shape != (2, *mask.shape):
        raise ValueError("Expected a (2,H,W) field and matching (H,W) mask")
    if field_weights is None:
        weights = np.ones_like(mask)
    else:
        weights = np.asarray(field_weights, dtype=float)
        if (
            weights.shape != mask.shape or not np.isfinite(weights).all()
            or np.any(weights < 0)
        ):
            raise ValueError("field_weights must be finite, nonnegative, and match mask")

    seed = np.asarray([seed_x, seed_y], dtype=float)
    if not np.isfinite(seed).all():
        return None
    if map_coordinates(mask, [[seed_y], [seed_x]], order=1, mode="constant")[0] < 0.999:
        return None

    steps = int(np.ceil(np.hypot(*mask.shape) / sample_step))
    positions = np.arange(-steps, steps + 1) * sample_step
    center = steps

    def evaluate(angle_deg):
        theta = np.radians(angle_deg)
        direction = np.array([np.cos(theta), np.sin(theta)])
        upper_direction = direction if direction[1] < 0 else -direction
        lower_direction = -upper_direction
        upper = _ray_apo_intersection(seed, upper_direction, upper_apo)
        lower = _ray_apo_intersection(seed, lower_direction, lower_apo)
        if upper is None or lower is None:
            return None

        coordinates = seed[:, None] + lower_direction[:, None] * positions
        sample_coordinates = np.array([coordinates[1], coordinates[0]])
        inside = map_coordinates(
            mask, sample_coordinates, order=1, mode="constant", cval=0,
        ) > 0.999
        cosine = map_coordinates(
            field[0], sample_coordinates, order=1, mode="constant", cval=np.nan,
        )
        sine = -map_coordinates(
            field[1], sample_coordinates, order=1, mode="constant", cval=np.nan,
        )
        sampled_weights = map_coordinates(
            weights, sample_coordinates, order=1, mode="constant", cval=0,
        )
        valid = (
            inside & np.isfinite(cosine) & np.isfinite(sine)
            & (np.hypot(cosine, sine) > 1e-8)
            & (sampled_weights > 0)
            & (positions >= -upper["ray_distance_px"])
            & (positions <= lower["ray_distance_px"])
        )
        if not valid[center]:
            return None
        left_invalid = np.flatnonzero(~valid[:center])
        right_invalid = np.flatnonzero(~valid[center + 1:])
        start = int(left_invalid[-1] + 1) if len(left_invalid) else 0
        stop = int(center + 1 + right_invalid[0]) if len(right_invalid) else len(valid)
        if stop - start < min_field_samples:
            return None

        local_angle = 0.5 * np.arctan2(sine[start:stop], cosine[start:stop])
        difference = local_angle - theta
        error = 0.5 * np.degrees(
            np.arctan2(np.sin(2 * difference), np.cos(2 * difference)),
        )
        objective = float(np.average(
            error ** 2, weights=sampled_weights[start:stop],
        ))
        length = upper["ray_distance_px"] + lower["ray_distance_px"]
        supported_length = float(positions[stop - 1] - positions[start])
        supported_fraction = min(1.0, supported_length / length)
        return {
            "seed_x": float(seed_x), "seed_y": float(seed_y),
            "angle_image_deg": float((angle_deg + 90) % 180 - 90),
            "tx": float(direction[0]), "ty": float(direction[1]),
            "upper_intersection": upper, "lower_intersection": lower,
            "field_path_x": coordinates[0, start:stop].copy(),
            "field_path_y": coordinates[1, start:stop].copy(),
            "fit_sample_count": stop - start,
            "orientation_mse_deg2": objective,
            "orientation_rmse_deg": float(np.sqrt(objective)),
            "observed_length_px": supported_length,
            "extrapolated_length_px": float(max(0.0, length - supported_length)),
            "observed_fraction": float(supported_fraction),
            "fl_px": float(length),
            "weighting": "uniform" if field_weights is None else "provided_field_weights",
        }

    best = None
    best_grid_angle = None
    valid_angles = 0
    for angle in np.arange(-90.0, 90.0, angle_step_deg):
        fit = evaluate(float(angle))
        if fit is not None:
            valid_angles += 1
            if best is None or fit["orientation_mse_deg2"] < best["orientation_mse_deg2"]:
                best, best_grid_angle = fit, float(angle)
    if best is None:
        return None

    def objective(angle):
        fit = evaluate(float(angle))
        return fit["orientation_mse_deg2"] if fit is not None else 1e12

    refined = minimize_scalar(
        objective,
        bounds=(best_grid_angle - angle_step_deg, best_grid_angle + angle_step_deg),
        method="bounded", options={"xatol": 1e-4},
    )
    if refined.success:
        fit = evaluate(float(refined.x))
        if fit is not None and fit["orientation_mse_deg2"] < best["orientation_mse_deg2"]:
            best = fit
    best["valid_grid_angles"] = valid_angles
    best["fit_method"] = "straight_minimum_orientation_mse"
    return best


def _median(samples, key):
    values = np.asarray([sample[key] for sample in samples], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def measure_selected_region(geometry, region, mm_per_model_px=None):
    """Measure only the selected region and retain per-stage failure diagnostics.

    PA and MT reuse v1. Five straight FL fits use the same seed-column quantiles
    as the notebook. Choose a valid row nearest the column's median interior y.
    Stage errors and unsuccessful seed fits are recorded without changing
    the selected anatomy. No-region results are explicitly not_measured.
    """
    output = {
        "status": "not_measured", "pa_results": [], "mt_results": [],
        "straight_fascicles": [], "failed_fascicle_seeds": [],
        "errors": [], "summary": {},
    }
    if region is None:
        output["reason"] = "no_selected_region"
        return output

    common = {
        "upper_apo": region["upper_apo"],
        "lower_apo": region["lower_apo"],
    }
    try:
        output["pa_results"] = estimate_pennation_angles(
            lower_apo=region["lower_apo"],
            fascicle_orientation=geometry["fascicle_v"],
            buffered_muscle_mask=region["buffered_mask"],
        )
    except Exception as error:
        output["errors"].append({
            "stage": "pennation", "error": f"{type(error).__name__}: {error}",
        })
    try:
        output["mt_results"] = estimate_muscle_thickness(**common)
    except Exception as error:
        output["errors"].append({
            "stage": "thickness", "error": f"{type(error).__name__}: {error}",
        })

    mask = np.asarray(region["buffered_mask"]) > 0
    columns = np.flatnonzero(mask.any(axis=0))
    if len(columns):
        seed_columns = np.unique(
            np.quantile(columns, [0.15, 0.325, 0.50, 0.675, 0.85]).astype(int),
        )
        for seed_x in seed_columns:
            rows = np.flatnonzero(mask[:, seed_x])
            seed_y = float(rows[np.argmin(np.abs(rows - np.median(rows)))])
            failure = {"seed_x": float(seed_x), "seed_y": seed_y}
            try:
                fit = fit_straight_fascicle_from_seed(
                    seed_x=float(seed_x), seed_y=seed_y,
                    fascicle_orientation=geometry["fascicle_v"],
                    buffered_muscle_mask=region["buffered_mask"], **common,
                )
                if fit is None:
                    failure["reason"] = "no_supported_straight_fit"
                    output["failed_fascicle_seeds"].append(failure)
                else:
                    output["straight_fascicles"].append(fit)
            except Exception as error:
                failure["reason"] = f"{type(error).__name__}: {error}"
                output["failed_fascicle_seeds"].append(failure)

    summary = {
        "PA_deg": _median(output["pa_results"], "pa_deg"),
        "MT_px": _median(output["mt_results"], "mt_px"),
        "FL_straight_px": _median(output["straight_fascicles"], "fl_px"),
        "median_orientation_rmse_deg": _median(
            output["straight_fascicles"], "orientation_rmse_deg",
        ),
        "median_observed_fraction": _median(
            output["straight_fascicles"], "observed_fraction",
        ),
        "n_pa_samples": len(output["pa_results"]),
        "n_mt_samples": len(output["mt_results"]),
        "n_straight_fascicles": len(output["straight_fascicles"]),
        "n_failed_seeds": len(output["failed_fascicle_seeds"]),
    }
    if mm_per_model_px is not None:
        scale = float(mm_per_model_px)
        if np.isfinite(scale) and scale > 0:
            summary["MT_mm"] = summary["MT_px"] * scale
            summary["FL_straight_mm"] = summary["FL_straight_px"] * scale
    summary["majority_extrapolated"] = (
        bool(summary["median_observed_fraction"] < 0.5)
        if np.isfinite(summary["median_observed_fraction"]) else None
    )
    complete = (
        all(np.isfinite(summary[key]) for key in ("PA_deg", "MT_px", "FL_straight_px"))
        and not output["errors"] and not output["failed_fascicle_seeds"]
    )
    output["summary"] = summary
    output["status"] = "measured" if complete else "partial"
    return output
