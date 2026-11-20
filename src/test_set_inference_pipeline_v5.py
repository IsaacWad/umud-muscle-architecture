"""Select a region, then measure it with optional support-weighted FL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.optimize import differential_evolution

import src.viewport_detection as vd
import src.muscle_geometry_integration_v2 as mg

import src.apo_fascicle_decoder as decoder
import src.aponeurosis_region_selection_v3 as region_selection
import src.muscle_geometry_measurements_v3 as measurements

import src.ultrasound_calibration_v2 as calibration_v2
import src.ultrasound_physical_calibration_v1 as physical_calibration_v1


# ============================================================
# SETTINGS
# ============================================================

TARGET_SIZE = 512
APO_PROBABILITY_THRESHOLD = 0.20
MIN_APO_WIDTH_FRACTION = 0.25


# ============================================================
# 1. MODEL INFERENCE
# ============================================================

def run_models(image_512, apo_model, fascicle_model, device, *, support_head=None, kappa_max=100.0):
    """Run both models and, when supplied, apply support_head to U-Net head inputs."""
    x = torch.from_numpy(np.asarray(image_512, dtype=np.float32) / 255.0)
    x = x[None, None].to(device)
    apo_model.eval()
    fascicle_model.eval()
    features = []
    hook = None
    if support_head is not None:
        if not np.isfinite(kappa_max) or kappa_max <= 0:
            raise ValueError("kappa_max must be finite and positive")
        support_head.eval()

        def capture_features(module, inputs):
            features.append(inputs[0])

        hook = fascicle_model.head.register_forward_pre_hook(capture_features)

    try:
        with torch.no_grad():
            apo_output = apo_model(x)
            raw = fascicle_model(x)
            if raw.ndim != 4 or raw.shape[1] not in (2, 3):
                raise ValueError("Expected two orientation channels and optional raw kappa")
            outputs = {"apo_output": apo_output, "fascicle_output": raw[:, :2]}
            if support_head is not None:
                if raw.shape[1] != 3 or len(features) != 1:
                    raise ValueError("Support weighting requires the three-channel uncertainty U-Net")
                logits = support_head(features[0])
                if logits.shape != raw[:, 2:3].shape:
                    raise ValueError("Support head must return one logit per model-space pixel")
                kappa = (kappa_max * raw[:, 2].sigmoid())[0].cpu().numpy()
                support = logits.sigmoid()[0, 0].cpu().numpy()
                joint = support * kappa
                if not np.isfinite(joint).all():
                    raise ValueError("Support times kappa must be finite")
                outputs.update(
                    fascicle_kappa=kappa, fascicle_support=support, field_weights=joint,
                )
            return outputs
    finally:
        if hook is not None:
            hook.remove()


# ============================================================
# 2. SINGLE-IMAGE PIPELINE
# ============================================================

def recover_lower_boundary(result, *, kappa_max):
    """Fit a smooth boundary at the drop in fascicle evidence below one apo."""
    upper = result["apo_candidates"][0]
    valid = np.asarray(result["valid_image_mask"], dtype=bool)
    joint = np.asarray(result["field_weights"], dtype=float) / kappa_max
    x = np.flatnonzero(valid.any(axis=0))
    if x.size < 4 or np.any(np.diff(x) != 1):
        return None

    smooth = uniform_filter(np.where(valid, joint, 0.0), size=(11, 13))
    smooth /= np.maximum(uniform_filter(valid.astype(float), size=(11, 13)), 1e-8)
    upper = mg.extend_apo_candidate_to_edges(upper, valid.shape[1], valid.shape[0])
    upper_y = upper["extended_y"][x]
    top = np.argmax(valid[:, x], axis=0)
    bottom = valid.shape[0] - 1 - np.argmax(valid[::-1, x], axis=0)
    min_y = np.maximum(upper_y + 25, top + 15)
    max_y = bottom - 15
    reference = float(np.median(upper_y))
    max_depth = float(max_y.max() - reference)
    if np.any(min_y > max_y) or max_depth <= 25:
        return None

    half_width = (x[-1] - x[0]) / 2
    t = (x - x.mean()) / half_width

    def curve(parameters):
        depth, tilt, bend = parameters
        return reference + depth + tilt * t + bend * (1 - t**2)

    def support_drop(y):
        above = map_coordinates(smooth, [y - 9, x], order=1)
        below = map_coordinates(smooth, [y + 9, x], order=1)
        return above - below

    def objective(parameters):
        y = curve(parameters)
        outside = np.maximum(min_y - y, 0) + np.maximum(y - max_y, 0)
        if np.any(outside):
            return 1 + outside.mean()
        return -support_drop(y).mean() + 0.002 * abs(parameters[2])

    bounds = [(25, max_depth), (-0.5 * half_width, 0.5 * half_width), (-25, 25)]
    fit = min(
        (differential_evolution(
            objective, bounds, seed=seed, tol=1e-7, popsize=30, maxiter=200,
        ) for seed in range(4)),
        key=lambda trial: trial.fun,
    )
    y = curve(fit.x)
    if np.any(y < min_y) or np.any(y > max_y):
        return None

    drop = support_drop(y)
    mean_drop = float(drop.mean())
    positive_fraction = float((drop > 0).mean())
    if mean_drop < 0.20 or positive_fraction < 0.75:
        return None

    tx, ty, _, _ = mg.compute_local_curve_tangents(x, y, window_radius=15)
    return {
        "center_x": x.astype(float),
        "center_y": y,
        "tx": tx,
        "ty": ty,
        "nx": -ty,
        "ny": tx,
        "smooth_y": y,
        "centroid_y": float(y.mean()),
        # The fallback has one possible pair. Use its joint-score drop as
        # boundary evidence and preserve its provenance for downstream review.
        "relative_score": mean_drop,
        "boundary_source": "fascicle_support_drop",
        "mean_support_drop": mean_drop,
        "positive_drop_fraction": positive_fraction,
    }


def run_single_image(
    image_path,
    apo_model,
    fascicle_model,
    device,
    target_size=TARGET_SIZE,
    tick_interval_mm=10.0,
    *,
    measure=True,
    support_head=None,
    kappa_max=100.0,
    weighted_fl=True,
    coverage_weight=0.25,
):
    """
    Run preprocessing, inference, geometry-only selection, then measurements.

    Supply the trained support head with the three-channel uncertainty U-Net.
    Weighted FL uses support times kappa. Set weighted_fl=False for uniform FL
    with the same models. Without a support head, FL uses uniform weights.

    Apo extraction uses probability threshold 0.20 and minimum model-width
    fraction 0.25 with unchanged linear apo scores. Region selection uses
    vertical 12-pixel continuity, a 2-degree exponential scale, and weights
    0.50 boundary, 0.40 continuity, 0.15 internal clearance. Joint
    coverage adds 0.25 times the mean normalized support times kappa
    inside the buffered region when a support head is supplied.
    Set coverage_weight=0 for the geometry-only control.

    measure=False stops after selection for notebook diagnostics. By default,
    PA, MT, and five straight-line orientation fits are calculated for the
    selected region. Measurement failure is reported separately from selection.

    Existing image, calibration, decoded-field, and candidate-region outputs
    are preserved. region_selection contains candidate_table, selected, and
    selected_region. measurements contains sample lists, fit diagnostics, and
    summary. Only the selected candidate has a measurements dictionary.

    If only one apo is detected and selection fails, try recovering the lower
    boundary from a sustained drop in the joint fascicle score.
    """

    image_path = Path(
        image_path
    )

    # ========================================================
    # 1. LOAD RAW IMAGE
    # ========================================================

    raw_image = (
        calibration_v2
        .load_raw_ultrasound(
            image_path
        )
    )

    # ========================================================
    # 2. RULER DETECTION
    #
    # Must operate in raw-image space before viewport cropping.
    # ========================================================

    calibration_result = (
        calibration_v2
        .analyze_ruler_candidates(
            raw_image
        )
    )

    # ========================================================
    # 3. VIEWPORT DETECTION
    # ========================================================

    viewport = (
        vd.detect_ultrasound_viewport(
            raw_image
        )
    )

    cropped_image = (
        viewport["crop"]
    )

    # ========================================================
    # 4. RESIZE + PAD INTO MODEL SPACE
    # ========================================================

    resize_info = (
        mg.resize_and_pad_to_square(
            cropped_image,
            target_size=target_size,
        )
    )

    image_512 = (
        resize_info["image"]
    )

    # ========================================================
    # 5. VALID MODEL-SPACE IMAGE MASK
    #
    # Excludes padding introduced during resize.
    # ========================================================

    valid_image_mask = (
        mg.build_valid_image_mask(
            original_shape=
                cropped_image.shape,

            resize_info=
                resize_info,

            target_size=
                target_size,
        )
    )

    # ========================================================
    # 6. PHYSICAL SCALE IN MODEL SPACE
    #
    # Raw ruler:
    #     mm_per_px
    #
    # After resizing:
    #     mm_per_model_px
    #         =
    #     mm_per_px / resize_scale
    # ========================================================

    scale_info = (
        physical_calibration_v1
        .derive_model_space_scale(
            calibration_result=
                calibration_result,

            resize_scale=
                resize_info["scale"],

            tick_interval_mm=
                tick_interval_mm,
        )
    )

    # ========================================================
    # 7. MODEL INFERENCE
    # ========================================================

    model_results = (
        run_models(
            image_512=
                image_512,

            apo_model=
                apo_model,

            fascicle_model=
                fascicle_model,

            device=
                device,
            support_head=support_head,
            kappa_max=kappa_max,
        )
    )


    # ========================================================
    # 8. DECODE MODEL OUTPUTS
    # ========================================================

    apo_decoded = (
        decoder.decode_apo_output(
            model_results["apo_output"]
        )
    )

    fascicle_decoded = (
        decoder.decode_fascicle_output(
            model_results["fascicle_output"]
        )
    )

    # ========================================================
    # 9. BUILD CANDIDATE MUSCLE REGIONS
    # ========================================================

    centerline_prob = apo_decoded["centerline_prob"]
    apo_candidates = mg.extract_apo_candidates_from_prediction(
        centerline_prob,
        probability_threshold=APO_PROBABILITY_THRESHOLD,
        min_width_fraction=MIN_APO_WIDTH_FRACTION,
    )
    candidate_regions = mg.build_candidate_regions(
        candidates=apo_candidates,
        image_shape=centerline_prob.shape,
        valid_image_mask=valid_image_mask,
        fascicle_tx=fascicle_decoded["fascicle_tx"],
        fascicle_ty=fascicle_decoded["fascicle_ty"],
    )

    # ========================================================
    # 10. ASSEMBLE GEOMETRY AND MODEL RESULTS
    # ========================================================

    result = {
        **{key: model_results[key] for key in (
            "fascicle_kappa", "fascicle_support", "field_weights",
        ) if key in model_results},
        "apo_candidates": apo_candidates,
        "candidate_regions": candidate_regions,

        "image_id":
            image_path.stem,

        "image_path":
            image_path,

        "raw_image":
            raw_image,

        "calibration_result":
            calibration_result,

        "scale_info":
            scale_info,

        "viewport":
            viewport,

        "cropped_image":
            cropped_image,

        "resize_info":
            resize_info,

        "image_512":
            image_512,

        "valid_image_mask":
            valid_image_mask,

        "apo_output":
            model_results[
                "apo_output"
            ],

        "fascicle_output":
            model_results[
                "fascicle_output"
            ],

        "centerline_prob":
            apo_decoded[
                "centerline_prob"
            ],

        "fascicle_raw_v":
            fascicle_decoded[
                "fascicle_raw_v"
            ],

        "fascicle_raw_magnitude":
            fascicle_decoded[
                "fascicle_raw_magnitude"
            ],

        "fascicle_v":
            fascicle_decoded[
                "fascicle_v"
            ],

        "fascicle_tx":
            fascicle_decoded[
                "fascicle_tx"
            ],

        "fascicle_ty":
            fascicle_decoded[
                "fascicle_ty"
            ],

        
    }

    # ========================================================
    # 11. SELECT ANATOMY BEFORE MEASUREMENT
    # ========================================================

    selection = region_selection.select_region_from_pipeline_result(
        result, coverage_weight=coverage_weight, kappa_max=kappa_max,
    )
    if (
        selection["selected_region"] is None
        and len(apo_candidates) == 1
        and result.get("field_weights") is not None
    ):
        lower = recover_lower_boundary(result, kappa_max=kappa_max)
        if lower is not None:
            recovered_apos = [*apo_candidates, lower]
            recovered_regions = mg.build_candidate_regions(
                candidates=recovered_apos,
                image_shape=centerline_prob.shape,
                valid_image_mask=valid_image_mask,
                fascicle_tx=result["fascicle_tx"],
                fascicle_ty=result["fascicle_ty"],
            )
            recovered_geometry = dict(
                result, apo_candidates=recovered_apos, candidate_regions=recovered_regions,
            )
            recovered_selection = region_selection.select_region_from_pipeline_result(
                recovered_geometry, coverage_weight=coverage_weight, kappa_max=kappa_max,
            )
            if recovered_selection["selected_region"] is not None:
                result = recovered_geometry
                selection = recovered_selection
                selection["selected"].update(
                    selection_fallback=True,
                    selection_reason="fascicle_support_drop",
                    mean_support_drop=lower["mean_support_drop"],
                    positive_drop_fraction=lower["positive_drop_fraction"],
                )

    result["region_selection"] = selection

    # ========================================================
    # 12. MEASURE THE SELECTED REGION ONLY
    # ========================================================

    if measure:
        measured = measurements.measure_selected_region(
            result,
            selection["selected_region"],
            mm_per_model_px=scale_info.get("mm_per_model_px"),
            field_weights=result.get("field_weights") if weighted_fl else None,
        )
    else:
        measured = {
            "status": "not_measured",
            "reason": "disabled",
            "pa_results": [],
            "mt_results": [],
            "straight_fascicles": [],
            "failed_fascicle_seeds": [],
            "errors": [],
            "summary": {},
        }

    result["measurements"] = measured
    if selection["selected_region"] is not None:
        selection["selected_region"]["measurements"] = measured

    return result

