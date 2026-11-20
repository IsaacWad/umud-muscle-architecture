from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import binary_erosion
import cv2

from src.aponeurosis_curve_utils import (
    compute_local_curve_tangents,
)


def extract_apo_candidates_from_prediction(
    probability_map,
    probability_threshold=0.20,
    min_width_fraction=0.30,
    min_area=20,
    smoothing=100,
    tangent_window_radius=15,
):
    """Extract permissive apo candidates from a probability map."""
    probability_map = np.asarray(probability_map, dtype=float)
    if probability_map.ndim != 2:
        raise ValueError("probability_map must be a 2D array")

    h, w = probability_map.shape
    binary_map = probability_map >= probability_threshold
    labeled_map, n_components = ndimage.label(binary_map)
    candidates = []

    for component_id in range(1, n_components + 1):
        ys, xs = np.where(labeled_map == component_id)
        if len(xs) == 0:
            continue

        area_px = len(xs)
        width_px = xs.max() - xs.min() + 1
        height_px = ys.max() - ys.min() + 1
        width_fraction = width_px / w

        if area_px < min_area or width_fraction < min_width_fraction:
            continue

        component_probabilities = probability_map[ys, xs]
        mean_probability = float(np.mean(component_probabilities))
        median_probability = float(np.median(component_probabilities))
        max_probability = float(np.max(component_probabilities))
        candidate_score = mean_probability * width_fraction

        unique_x = np.unique(xs)
        center_x = []
        center_y = []

        for x in unique_x:
            y_values = ys[xs == x]
            center_x.append(float(x))
            center_y.append(float(np.median(y_values)))

        center_x = np.asarray(center_x, dtype=float)
        center_y = np.asarray(center_y, dtype=float)
        if len(center_x) < 4:
            continue

        try:
            spline = UnivariateSpline(center_x, center_y, s=smoothing)
            smooth_y = spline(center_x)
        except Exception:
            continue

        tx, ty, tangent_rmse, tangent_confidence = compute_local_curve_tangents(
            center_x,
            smooth_y,
            window_radius=tangent_window_radius,
        )

        nx = -ty
        ny = tx
        mean_tangent_confidence = float(np.mean(tangent_confidence))

        candidates.append({
            "component_id": component_id,
            "area_px": area_px,
            "width_px": width_px,
            "height_px": height_px,
            "width_fraction": width_fraction,
            "mean_probability": mean_probability,
            "median_probability": median_probability,
            "max_probability": max_probability,
            "candidate_score": candidate_score,
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "center_x": center_x,
            "center_y": center_y,
            "smooth_y": smooth_y,
            "tx": tx,
            "ty": ty,
            "nx": nx,
            "ny": ny,
            "tangent_rmse": tangent_rmse,
            "tangent_confidence": tangent_confidence,
            "mean_tangent_confidence": mean_tangent_confidence,
        })

    if candidates:
        max_score = max(c["candidate_score"] for c in candidates)
        for candidate in candidates:
            candidate["relative_score"] = candidate["candidate_score"] / (max_score + 1e-8)

    return candidates


def sort_apo_candidates_top_to_bottom(candidates):
    """Sort candidate aponeuroses by centroid y."""
    return sorted(candidates, key=lambda c: c["centroid_y"])


def extend_apo_candidate_to_edges(
    candidate,
    image_width=512,
    image_height=512,
    endpoint_window=25,
):
    """Extend an apo to both image edges using terminal linear fits."""
    x = np.asarray(candidate["center_x"], dtype=float)
    y = np.asarray(candidate["smooth_y"], dtype=float)

    if len(x) < 2:
        raise ValueError("Apo candidate must contain at least two points.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    n_left = min(endpoint_window, len(x))
    n_right = min(endpoint_window, len(x))

    left_coef = np.polyfit(x[:n_left], y[:n_left], deg=1)
    right_coef = np.polyfit(x[-n_right:], y[-n_right:], deg=1)

    full_x = np.arange(0, image_width, dtype=float)
    full_y = np.empty_like(full_x)

    observed_min = x.min()
    observed_max = x.max()

    left_mask = full_x < observed_min
    observed_mask = (full_x >= observed_min) & (full_x <= observed_max)
    right_mask = full_x > observed_max

    full_y[left_mask] = left_coef[0] * full_x[left_mask] + left_coef[1]
    full_y[observed_mask] = np.interp(full_x[observed_mask], x, y)
    full_y[right_mask] = right_coef[0] * full_x[right_mask] + right_coef[1]
    full_y = np.clip(full_y, 0, image_height - 1)

    result = candidate.copy()
    result["extended_x"] = full_x
    result["extended_y"] = full_y
    result["observed_mask"] = observed_mask
    return result


def resize_and_pad_to_square(
    image,
    target_size=512,
):
    """
    Resize an image isotropically to fit inside a square canvas,
    then center-pad the remaining space.

    Returns
    -------
    dict containing:
        image
        scale
        pad_x
        pad_y
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            "image must be a 2D grayscale array"
        )

    h_raw, w_raw = image.shape

    scale = min(
        target_size / w_raw,
        target_size / h_raw,
    )

    new_w = int(
        round(
            w_raw * scale
        )
    )

    new_h = int(
        round(
            h_raw * scale
        )
    )

    new_w = min(
        new_w,
        target_size,
    )

    new_h = min(
        new_h,
        target_size,
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_x = (
        target_size - new_w
    ) // 2

    pad_y = (
        target_size - new_h
    ) // 2

    padded = np.zeros(
        (target_size, target_size),
        dtype=image.dtype,
    )

    padded[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w,
    ] = resized

    return {
        "image": padded,
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
    }



def build_valid_image_mask(original_shape, resize_info, target_size=512):
    """Mask the real resized image footprint, excluding square padding."""
    h_raw, w_raw = original_shape
    scale = float(resize_info["scale"])
    pad_x = int(resize_info["pad_x"])
    pad_y = int(resize_info["pad_y"])

    new_w = int(round(w_raw * scale))
    new_h = int(round(h_raw * scale))

    valid_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    x2 = min(target_size, pad_x + new_w)
    y2 = min(target_size, pad_y + new_h)
    valid_mask[pad_y:y2, pad_x:x2] = 1
    return valid_mask


def build_muscle_mask_from_extended_apos(
    upper_apo,
    lower_apo,
    image_shape=(512, 512),
):
    """Build a muscle-region mask between two extended apo curves."""
    h, w = image_shape

    upper_x = np.asarray(upper_apo["extended_x"], dtype=int)
    lower_x = np.asarray(lower_apo["extended_x"], dtype=int)
    upper_y = np.asarray(upper_apo["extended_y"], dtype=float)
    lower_y = np.asarray(lower_apo["extended_y"], dtype=float)
    upper_observed = np.asarray(upper_apo["observed_mask"], dtype=bool)
    lower_observed = np.asarray(lower_apo["observed_mask"], dtype=bool)

    common_x = np.intersect1d(upper_x, lower_x)
    if len(common_x) == 0:
        raise ValueError("No overlapping x-grid between extended apos.")

    upper_idx = np.searchsorted(upper_x, common_x)
    lower_idx = np.searchsorted(lower_x, common_x)

    upper_y_common = upper_y[upper_idx]
    lower_y_common = lower_y[lower_idx]
    upper_obs_common = upper_observed[upper_idx]
    lower_obs_common = lower_observed[lower_idx]

    observed_overlap_mask = upper_obs_common & lower_obs_common
    extrapolated_overlap_mask = ~observed_overlap_mask

    muscle_mask = np.zeros((h, w), dtype=np.uint8)

    for x, y_upper, y_lower in zip(common_x, upper_y_common, lower_y_common):
        if y_lower <= y_upper:
            continue

        y_start = max(0, int(np.ceil(y_upper)))
        y_end = min(h - 1, int(np.floor(y_lower)))

        if y_end >= y_start:
            muscle_mask[y_start:y_end + 1, x] = 1

    return {
        "muscle_mask": muscle_mask,
        "common_x": common_x,
        "upper_y": upper_y_common,
        "lower_y": lower_y_common,
        "observed_overlap_mask": observed_overlap_mask,
        "extrapolated_overlap_mask": extrapolated_overlap_mask,
    }


def build_buffered_muscle_mask(muscle_mask, buffer_pixels=8):
    """Shrink a muscle-region mask inward from apo boundaries."""
    muscle_mask = np.asarray(muscle_mask) > 0

    if buffer_pixels <= 0:
        return muscle_mask.astype(np.uint8)

    structure = np.ones(
        (2 * buffer_pixels + 1, 2 * buffer_pixels + 1),
        dtype=bool,
    )

    buffered_mask = binary_erosion(
        muscle_mask,
        structure=structure,
        border_value=0,
    )

    return buffered_mask.astype(np.uint8)


def summarize_candidate_region(
    upper_apo,
    lower_apo,
    buffered_mask,
    fascicle_tx,
    fascicle_ty,
):
    """Summarize angle, coherence, and separation for one candidate region."""
    upper_y = np.asarray(upper_apo["extended_y"], dtype=float)
    lower_y = np.asarray(lower_apo["extended_y"], dtype=float)
    separation = lower_y - upper_y

    valid_x = np.any(buffered_mask > 0, axis=0)
    separation_valid = separation[valid_x]

    if len(separation_valid) == 0:
        return {
            "valid": False,
            "reason": "no_valid_separation_after_buffer",
        }

    deep_dy_dx = np.gradient(lower_y)
    apo_tx = np.ones_like(deep_dy_dx)
    apo_ty = -deep_dy_dx

    apo_norm = np.sqrt(apo_tx ** 2 + apo_ty ** 2)
    apo_norm = np.maximum(apo_norm, 1e-8)
    apo_tx = apo_tx / apo_norm
    apo_ty = apo_ty / apo_norm

    yy, xx = np.where(buffered_mask > 0)
    if len(xx) == 0:
        return {
            "valid": False,
            "reason": "no_field_pixels_after_buffer",
        }

    f_tx = fascicle_tx[yy, xx]
    f_ty = fascicle_ty[yy, xx]
    a_tx = apo_tx[xx]
    a_ty = apo_ty[xx]

    dot = f_tx * a_tx + f_ty * a_ty
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    relative_angle_deg = np.degrees(np.arccos(dot))

    if len(relative_angle_deg) == 0:
        return {
            "valid": False,
            "reason": "no_relative_angles",
        }

    fascicle_theta = np.arctan2(f_ty, f_tx)
    mean_cos2 = np.mean(np.cos(2.0 * fascicle_theta))
    mean_sin2 = np.mean(np.sin(2.0 * fascicle_theta))
    coherence = np.sqrt(mean_cos2 ** 2 + mean_sin2 ** 2)

    angle_q25 = np.percentile(relative_angle_deg, 25)
    angle_q75 = np.percentile(relative_angle_deg, 75)
    separation_q25 = np.percentile(separation_valid, 25)
    separation_q75 = np.percentile(separation_valid, 75)

    return {
        "valid": True,
        "mean_angle_deg": float(np.mean(relative_angle_deg)),
        "median_angle_deg": float(np.median(relative_angle_deg)),
        "std_angle_deg": float(np.std(relative_angle_deg)),
        "angle_q25_deg": float(angle_q25),
        "angle_q75_deg": float(angle_q75),
        "angle_iqr_deg": float(angle_q75 - angle_q25),
        "coherence": float(coherence),
        "median_separation_px": float(np.median(separation_valid)),
        "mean_separation_px": float(np.mean(separation_valid)),
        "separation_q25_px": float(separation_q25),
        "separation_q75_px": float(separation_q75),
        "separation_iqr_px": float(separation_q75 - separation_q25),
        "n_field_pixels": int(len(relative_angle_deg)),
    }


def build_candidate_regions(
    candidates,
    image_shape,
    valid_image_mask,
    fascicle_tx,
    fascicle_ty,
    buffer_pixels=8,
    endpoint_window=25,
):
    """
    Build and summarize a muscle region for every upper/lower apo pair.

    After sorting candidates from top to bottom, pair each candidate
    with every candidate below it. Return the regions for downstream
    selection.
    """

    candidates_sorted = sort_apo_candidates_top_to_bottom(
        candidates
    )

    if len(candidates_sorted) < 2:
        return []

    h, w = image_shape

    extended_candidates = [
        extend_apo_candidate_to_edges(
            candidate,
            image_width=w,
            image_height=h,
            endpoint_window=endpoint_window,
        )
        for candidate in candidates_sorted
    ]

    valid_image_mask = (
        np.asarray(valid_image_mask) > 0
    ).astype(np.uint8)

    if valid_image_mask.shape != (h, w):
        raise ValueError(
            "valid_image_mask shape must match image_shape."
        )

    regions = []

    pair_index = 0

    # --------------------------------------------------
    # Pair each upper apo with every candidate below it.
    # --------------------------------------------------

    for upper_index in range(
        len(extended_candidates)
    ):

        for lower_index in range(
            upper_index + 1,
            len(extended_candidates)
        ):

            upper_apo = (
                extended_candidates[
                    upper_index
                ]
            )

            lower_apo = (
                extended_candidates[
                    lower_index
                ]
            )

            region_geometry = (
                build_muscle_mask_from_extended_apos(
                    upper_apo,
                    lower_apo,
                    image_shape=image_shape,
                )
            )

            valid_mask = (
                region_geometry[
                    "muscle_mask"
                ]
                *
                valid_image_mask
            )

            buffered_mask = (
                build_buffered_muscle_mask(
                    valid_mask,
                    buffer_pixels=buffer_pixels,
                )
            )

            summary = (
                summarize_candidate_region(
                    upper_apo=upper_apo,
                    lower_apo=lower_apo,
                    buffered_mask=buffered_mask,
                    fascicle_tx=fascicle_tx,
                    fascicle_ty=fascicle_ty,
                )
            )

            regions.append({
                "pair_index":
                    pair_index,

                "pair_name":
                    f"{upper_index}-{lower_index}",

                "upper_candidate_index":
                    upper_index,

                "lower_candidate_index":
                    lower_index,

                "upper_apo":
                    upper_apo,

                "lower_apo":
                    lower_apo,

                "region_geometry":
                    region_geometry,

                "valid_mask":
                    valid_mask,

                "buffered_mask":
                    buffered_mask,

                "summary":
                    summary,

                "valid":
                    bool(
                        summary.get(
                            "valid",
                            False,
                        )
                    ),
            })

            pair_index += 1

    return regions