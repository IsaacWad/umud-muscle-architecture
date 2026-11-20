"""Selected-region PA, MT, and straight FL with optional FL weights."""

from __future__ import annotations

import numpy as np

from src.muscle_geometry_measurements_v1 import (
    estimate_muscle_thickness,
    estimate_pennation_angles,
)
from src.muscle_geometry_measurements_v2 import fit_straight_fascicle_from_seed


def _median(samples, key):
    values = np.asarray([sample[key] for sample in samples], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def measure_selected_region(geometry, region, mm_per_model_px=None, *, field_weights=None):
    """Measure only the selected region and retain per-stage failure diagnostics.

    field_weights applies to FL only. PA and MT reuse v1. Five straight FL fits use the same seed-column quantiles
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
    if field_weights is not None:
        field_weights = np.asarray(field_weights, dtype=float)
        if (
            field_weights.shape != mask.shape
            or not np.isfinite(field_weights).all()
            or np.any(field_weights < 0)
        ):
            raise ValueError("field_weights must be finite, nonnegative, and match buffered_mask")
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
                    buffered_muscle_mask=region["buffered_mask"],
                    field_weights=field_weights, **common,
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
