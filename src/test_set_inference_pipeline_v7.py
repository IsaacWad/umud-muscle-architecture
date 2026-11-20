"""Select the supported superficial region before measuring muscle geometry."""

import numpy as np

from src.muscle_geometry_measurements_v5 import measure_selected_region
from src.test_set_inference_pipeline_v5 import TARGET_SIZE, run_single_image as run_geometry


def _column_support(region, joint, valid):
    buffered = region.get("buffered_mask")
    if buffered is None:
        return np.nan
    y, x = np.where(np.asarray(buffered, dtype=bool) & valid)
    if not x.size:
        return np.nan
    counts = np.bincount(x)
    totals = np.bincount(x, weights=joint[y, x], minlength=counts.size)
    used = counts > 0
    return float(np.quantile(totals[used] / counts[used], 0.25))


def _prefer_supported_shallower_region(result, kappa_max):
    selection = result["region_selection"]
    winner = selection["selected_region"]
    weights = result.get("field_weights")
    if winner is None or weights is None or selection["selected"].get("selection_fallback"):
        return
    upper, lower = map(int, winner["pair_name"].split("-"))
    if upper == 0 or lower != upper + 1:
        return

    pair = f"{upper - 1}-{upper}"
    table = selection["candidate_table"]
    eligible = table.loc[
        table["pair_name"].eq(pair) & table["geometry_valid"] & table["evidence_available"]
    ]
    if eligible.empty:
        return
    candidate = next(region for region in result["candidate_regions"] if region["pair_name"] == pair)
    joint = np.clip(np.asarray(weights, dtype=float) / kappa_max, 0, 1)
    valid = np.asarray(result["valid_image_mask"], dtype=bool) & np.isfinite(joint)
    winner_support = _column_support(winner, joint, valid)
    candidate_support = _column_support(candidate, joint, valid)
    if not (candidate_support >= 0.65 and candidate_support >= winner_support - 0.10):
        return

    selected = dict(selection["selected"], **eligible.iloc[0].to_dict())
    selected.update(
        selection_reason="supported_shallower_region", score_margin=np.nan,
        previous_pair=winner["pair_name"], previous_joint_p25=winner_support,
        selected_joint_p25=candidate_support,
    )
    selection.update(selected=selected, selected_region=candidate)


def run_single_image(
    image_path, apo_model, fascicle_model, device, target_size=TARGET_SIZE,
    tick_interval_mm=10.0, *, measure=True, support_head=None, kappa_max=100.0,
    weighted_fl=True, coverage_weight=0.25, fl_extension_slope_fraction=0.75,
    dense_seeds=True,
):
    if not np.isfinite(kappa_max) or kappa_max <= 0:
        raise ValueError("kappa_max must be finite and positive")
    result = run_geometry(
        image_path, apo_model, fascicle_model, device, target_size, tick_interval_mm,
        measure=False, support_head=support_head, kappa_max=kappa_max,
        weighted_fl=weighted_fl, coverage_weight=coverage_weight,
    )
    _prefer_supported_shallower_region(result, kappa_max)
    region = result["region_selection"]["selected_region"]
    if measure:
        result["measurements"] = measure_selected_region(
            result, region, result["scale_info"].get("mm_per_model_px"),
            field_weights=result.get("field_weights") if weighted_fl else None,
            kappa_max=kappa_max, slope_fraction=fl_extension_slope_fraction,
            dense_seeds=dense_seeds,
        )
        if region is not None:
            region["measurements"] = result["measurements"]
    elif region is not None:
        region["measurements"] = result["measurements"]
    return result
