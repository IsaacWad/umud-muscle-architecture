"""V7 region selection and measurement with labeled ruler calibration."""

from pathlib import Path

import numpy as np
import torch

from src.test_set_inference_pipeline_v7 import (
    TARGET_SIZE,
    measure_selected_region,
    run_single_image as run_v7,
)
from src.ultrasound_calibration_v3 import analyze_vertical_scale


def run_single_image(
    image_path, apo_model, fascicle_model, device, target_size=TARGET_SIZE,
    tick_interval_mm=10.0, *, measure=True, support_head=None, kappa_max=100.0,
    weighted_fl=True, coverage_weight=0.25, fl_extension_slope_fraction=0.75,
    dense_seeds=True, read_text, measurement_support_head=None, depth_mm=None,
    labeled_span=None, horizontal_tick_mm=10.0,
):
    """Run selected-region geometry with an isotropic scale from the raw ruler."""
    result = run_v7(
        image_path, apo_model, fascicle_model, device, target_size, tick_interval_mm,
        measure=False, support_head=support_head, kappa_max=kappa_max,
        weighted_fl=weighted_fl, coverage_weight=coverage_weight,
    )
    calibration = analyze_vertical_scale(
        result["raw_image"], read_text, image_id=Path(image_path).name,
        rulers=result["calibration_result"], bounds=result["viewport"]["bounds"],
        depth_mm=depth_mm, labeled_span=labeled_span, horizontal_tick_mm=horizontal_tick_mm,
    )
    result["calibration_v3"] = calibration
    scale = dict(result["scale_info"])
    raw_scale = calibration["mm_per_raw_px"]
    scale.update(
        legacy_mm_per_model_px=scale.get("mm_per_model_px"),
        scale_found=raw_scale is not None,
        scale_source=calibration["source"],
        mm_per_px=raw_scale if raw_scale is not None else np.nan,
        sx_mm_per_px=raw_scale if raw_scale is not None else np.nan,
        sy_mm_per_px=raw_scale if raw_scale is not None else np.nan,
        mm_per_model_px=raw_scale / result["resize_info"]["scale"] if raw_scale is not None else np.nan,
    )
    result["scale_info"] = scale
    if not measure:
        return result

    region = result["region_selection"]["selected_region"]
    weights = result.get("field_weights") if weighted_fl else None
    if region is not None and weighted_fl and measurement_support_head is not None and measurement_support_head is not support_head:
        # The second head only supplies measurement weights after region selection.
        image = torch.from_numpy(np.asarray(result["image_512"], dtype=np.float32) / 255)
        features = []
        hook = fascicle_model.head.register_forward_pre_hook(
            lambda module, inputs: features.append(inputs[0])
        )
        measurement_support_head.eval()
        try:
            with torch.no_grad():
                prediction = fascicle_model(image[None, None].to(device))
                if prediction.shape[1] != 3 or len(features) != 1:
                    raise ValueError("Measurement support requires the three-channel uncertainty model")
                logits = measurement_support_head(features[0])
                if logits.shape != prediction[:, 2:3].shape:
                    raise ValueError("Measurement support must return one logit per pixel")
                weights = (kappa_max * prediction[:, 2:3].sigmoid() * logits.sigmoid())[0, 0].cpu().numpy()
                if not np.isfinite(weights).all():
                    raise ValueError("Measurement weights must be finite")
        finally:
            hook.remove()
    result["measurement_field_weights"] = weights
    measured = measure_selected_region(
        result, region, scale["mm_per_model_px"],
        field_weights=weights, kappa_max=kappa_max,
        slope_fraction=fl_extension_slope_fraction, dense_seeds=dense_seeds,
    )
    result["measurements"] = measured
    if region is not None:
        region["measurements"] = measured
    return result
