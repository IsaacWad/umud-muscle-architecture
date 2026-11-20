"""Measure a selected region with denser, support-weighted straight fits."""

import numpy as np
from scipy.ndimage import map_coordinates

from src.muscle_geometry_measurements_v2 import fit_straight_fascicle_from_seed
from src.muscle_geometry_measurements_v3 import measure_selected_region as measure_five
from src.muscle_geometry_measurements_v4 import apply_fl_extension


def _distinct_fits(fits):
    """Keep the strongest representative of nearby, overlapping fitted lines."""
    kept = []
    for fit in sorted(fits, key=lambda item: item["length_weight"], reverse=True):
        for other in kept:
            angle_gap = abs((fit["angle_image_deg"] - other["angle_image_deg"] + 90) % 180 - 90)
            if angle_gap > 3:
                continue
            axis = "x" if abs(fit["tx"]) >= abs(fit["ty"]) else "y"
            path_key = f"field_path_{axis}"
            start = max(fit[path_key].min(), other[path_key].min())
            stop = min(fit[path_key].max(), other[path_key].max())
            if start > stop:
                continue
            distance = ((start + stop) / 2 - fit[f"seed_{axis}"]) / fit[f"t{axis}"]
            x = fit["seed_x"] + distance * fit["tx"]
            y = fit["seed_y"] + distance * fit["ty"]
            offset = abs(
                other["tx"] * (y - other["seed_y"])
                - other["ty"] * (x - other["seed_x"])
            )
            if offset <= 5:
                break
        else:
            kept.append(fit)
    return kept


def measure_selected_region(
    geometry, region, mm_per_model_px=None, *, field_weights=None,
    kappa_max=100.0, slope_fraction=0.75, dense_seeds=True,
):
    """Keep v6 PA and MT and aggregate distinct straight FL fits."""
    baseline = measure_five(geometry, region, mm_per_model_px, field_weights=field_weights)
    if region is None:
        return baseline
    original_summary = dict(baseline["summary"])
    apply_fl_extension(baseline, region, mm_per_model_px, slope_fraction=slope_fraction)
    if not dense_seeds:
        return baseline

    mask = np.asarray(region["buffered_mask"], dtype=bool)
    columns = np.flatnonzero(mask.any(axis=0))
    fits = [dict(fit) for fit in baseline["straight_fascicles"]]
    failures = list(baseline["failed_fascicle_seeds"])
    attempted = {(row["seed_x"], row["seed_y"]) for row in (*fits, *failures)}
    if columns.size:
        for x in np.unique(np.quantile(columns, np.linspace(0.20, 0.80, 15)).astype(int)):
            rows = np.flatnonzero(mask[:, x])
            for y in np.unique(np.quantile(rows, [0.25, 0.50, 0.75], method="nearest")):
                if (x, y) in attempted:
                    continue
                attempted.add((x, y))
                fit = fit_straight_fascicle_from_seed(
                    seed_x=float(x), seed_y=float(y),
                    fascicle_orientation=geometry["fascicle_v"],
                    upper_apo=region["upper_apo"], lower_apo=region["lower_apo"],
                    buffered_muscle_mask=region["buffered_mask"], field_weights=field_weights,
                )
                if fit is None:
                    failures.append({"seed_x": float(x), "seed_y": float(y),
                                     "reason": "no_supported_straight_fit"})
                else:
                    fits.append(fit)

    dense = dict(
        baseline, summary=original_summary, straight_fascicles=fits,
        failed_fascicle_seeds=failures,
    )
    apply_fl_extension(dense, region, mm_per_model_px, slope_fraction=slope_fraction)
    for fit, adjustment in zip(fits, dense["fl_extension_adjustments"]):
        length = adjustment["fl_px"]
        mean_joint = 1.0
        if field_weights is not None:
            coords = [fit["field_path_y"], fit["field_path_x"]]
            mean_joint = float(np.clip(map_coordinates(
                np.asarray(field_weights, dtype=float), coords, order=1,
            ).mean() / kappa_max, 0, 1))
        observed_fraction = min(1.0, fit["observed_length_px"] / length)
        fit.update(
            adjusted_fl_px=length, adjusted_observed_fraction=observed_fraction,
            mean_joint_score=mean_joint, length_weight=mean_joint * observed_fraction,
        )

    kept = _distinct_fits([fit for fit in fits if fit["length_weight"] > 0])
    diagnostics = {
        "n_attempted_seeds": len(attempted),
        "n_successful_seed_fits": len(fits),
        "n_distinct_seed_fits": len(kept),
        "five_seed_fl_mm": baseline["summary"].get("FL_straight_mm", np.nan),
    }
    if len(kept) < 5 and np.isfinite(baseline["summary"]["FL_straight_px"]):
        baseline["summary"].update(diagnostics, fl_seed_method="five_seed_fallback")
        return baseline
    if not kept:
        return baseline

    ordered = sorted(kept, key=lambda fit: fit["adjusted_fl_px"])
    cumulative = np.cumsum([fit["length_weight"] for fit in ordered])
    length = ordered[np.searchsorted(cumulative, cumulative[-1] / 2)]["adjusted_fl_px"]
    for fit in fits:
        fit["used_for_fl"] = any(fit is retained for retained in kept)
    summary = dense["summary"]
    observed_fraction = float(np.median([fit["adjusted_observed_fraction"] for fit in kept]))
    summary.update(
        diagnostics, FL_straight_px=length, fl_seed_method="dense_weighted_median",
        baseline_FL_straight_px=float(np.median([fit["fl_px"] for fit in kept])),
        n_straight_fascicles=len(fits), n_failed_seeds=len(failures),
        median_observed_fraction=observed_fraction,
        majority_extrapolated=observed_fraction < 0.5,
        median_orientation_rmse_deg=float(np.median([fit["orientation_rmse_deg"] for fit in kept])),
    )
    if "FL_straight_mm" in summary:
        summary["FL_straight_mm"] = length * float(mm_per_model_px)
        summary["baseline_FL_straight_mm"] = (
            summary["baseline_FL_straight_px"] * float(mm_per_model_px)
        )
    if not dense["errors"] and all(np.isfinite(summary[key]) for key in ("PA_deg", "MT_px", "FL_straight_px")):
        dense["status"] = "measured"
    return dense
