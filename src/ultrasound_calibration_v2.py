# ============================================================
# ultrasound_calibration_v2.py
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

from scipy.signal import (
    find_peaks,
    peak_widths,
)


# ============================================================
# 1. Load raw ultrasound image
# ============================================================

def load_raw_ultrasound(path):
    path = Path(path)
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


# ============================================================
# 2. Extract border strips
# ============================================================

def get_border_strips(image, strip_fraction=0.10):
    image = np.asarray(image)
    height, width = image.shape

    strip_width = max(1, int(round(width * strip_fraction)))
    strip_height = max(1, int(round(height * strip_fraction)))

    return {
        "left": image[:, :strip_width],
        "right": image[:, width - strip_width:],
        "top": image[:strip_height, :],
        "bottom": image[height - strip_height:, :],
    }


# ============================================================
# 3. Build directional tick profile
# ============================================================

def build_tick_profile(strip, side):
    strip = np.asarray(strip, dtype=np.float32)

    if side in ("left", "right"):
        gradient = np.abs(np.diff(strip, axis=0))
        profile = np.mean(gradient, axis=1)

    elif side in ("top", "bottom"):
        gradient = np.abs(np.diff(strip, axis=1))
        profile = np.mean(gradient, axis=0)

    else:
        raise ValueError(f"Unknown side: {side}")

    profile_min = float(np.min(profile))
    profile_max = float(np.max(profile))

    if profile_max > profile_min:
        profile = (profile - profile_min) / (profile_max - profile_min)
    else:
        profile = np.zeros_like(profile)

    return profile.astype(np.float32)


# ============================================================
# 4. Build dynamic sub-band profiles
# ============================================================

def generate_subband_ranges(
    strip_length,
    band_fraction=0.20,
    overlap_fraction=0.50,
    min_band_width=8,
):
    strip_length = int(strip_length)

    if strip_length <= 0:
        return []

    band_width = max(
        min_band_width,
        int(round(strip_length * band_fraction)),
    )

    band_width = min(
        band_width,
        strip_length,
    )

    step = max(
        1,
        int(
            round(
                band_width
                *
                (1.0 - overlap_fraction)
            )
        ),
    )

    starts = list(
        range(
            0,
            strip_length - band_width + 1,
            step,
        )
    )

    final_start = strip_length - band_width

    if final_start not in starts:
        starts.append(final_start)

    starts = sorted(set(starts))

    return [
        (start, start + band_width)
        for start in starts
    ]


def build_subband_profiles(
    strip,
    side,
    band_fraction=0.20,
    overlap_fraction=0.50,
    min_band_width=8,
):
    strip = np.asarray(strip)

    if side in ("top", "bottom"):
        strip_length = strip.shape[0]

    elif side in ("left", "right"):
        strip_length = strip.shape[1]

    else:
        raise ValueError(f"Unknown side: {side}")

    ranges = generate_subband_ranges(
        strip_length=strip_length,
        band_fraction=band_fraction,
        overlap_fraction=overlap_fraction,
        min_band_width=min_band_width,
    )

    subbands = []

    for band_start, band_end in ranges:
        if side in ("top", "bottom"):
            band = strip[band_start:band_end, :]
        else:
            band = strip[:, band_start:band_end]

        profile = build_tick_profile(
            strip=band,
            side=side,
        )

        subbands.append(
            {
                "side": side,
                "band_start": band_start,
                "band_end": band_end,
                "band_width": band_end - band_start,
                "profile": profile,
            }
        )

    return subbands


# ============================================================
# 5. Detect candidate tick peaks
# ============================================================

def detect_tick_peaks(
    profile,
    min_distance=20,
    min_prominence=0.10,
    relative_prominence_threshold=0.30,
):
    profile = np.asarray(
        profile,
        dtype=np.float32,
    )

    peaks, properties = find_peaks(
        profile,
        distance=min_distance,
        prominence=min_prominence,
    )

    if len(peaks) == 0:
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=np.float32)

        return {
            "peaks": empty_int,
            "heights": empty_float,
            "prominences": empty_float,
            "widths": empty_float,
            "filtered_peaks": empty_int,
            "filtered_prominences": empty_float,
            "all_peaks": empty_int,
            "all_prominences": empty_float,
        }

    all_peaks = peaks.copy()

    all_prominences = (
        properties["prominences"]
        .astype(np.float32)
    )

    # Robust prominence reference:
    # use the median rather than the single strongest peak.
    prominence_reference = float(
        np.median(all_prominences)
    )

    relative_cutoff = (
        relative_prominence_threshold
        *
        prominence_reference
    )

    strong_mask = (
        all_prominences
        >=
        relative_cutoff
    )

    peaks = all_peaks[strong_mask]
    prominences = all_prominences[strong_mask]

    if len(peaks) > 0:
        heights = profile[peaks]

        widths = peak_widths(
            profile,
            peaks,
            rel_height=0.5,
        )[0]

    else:
        heights = np.array([], dtype=np.float32)
        widths = np.array([], dtype=np.float32)

    return {
        "peaks": peaks,
        "heights": heights,
        "prominences": prominences,
        "widths": widths,
        "filtered_peaks": peaks.copy(),
        "filtered_prominences": prominences.copy(),
        "all_peaks": all_peaks,
        "all_prominences": all_prominences,
    }


# ============================================================
# 6. Extract dominant regular peak sequence
# ============================================================

def extract_regular_peak_sequence(
    peaks,
    min_spacing_ratio=0.06,
    max_spacing_ratio=0.21,
    profile_length=None,
    spacing_tolerance=0.08,
    min_sequence_length=4,
):
    peaks = np.asarray(
        peaks,
        dtype=int,
    )

    if (
        profile_length is None
        or
        len(peaks) < min_sequence_length
    ):
        return {
            "sequence_peaks": peaks,
            "dominant_spacing": np.nan,
            "sequence_length": len(peaks),
            "sequence_span": 0.0,
            "spacing_support": 0,
        }

    min_spacing = (
        min_spacing_ratio
        *
        profile_length
    )

    max_spacing = (
        max_spacing_ratio
        *
        profile_length
    )

    pairwise_diffs = []

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            diff = peaks[j] - peaks[i]

            if min_spacing <= diff <= max_spacing:
                pairwise_diffs.append(diff)

    if len(pairwise_diffs) == 0:
        return {
            "sequence_peaks": peaks,
            "dominant_spacing": np.nan,
            "sequence_length": len(peaks),
            "sequence_span": 0.0,
            "spacing_support": 0,
        }

    pairwise_diffs = np.asarray(
        pairwise_diffs,
        dtype=np.float32,
    )

    best_spacing = None
    best_support = -1

    for spacing in pairwise_diffs:
        tolerance = (
            spacing_tolerance
            *
            spacing
        )

        support = int(
            np.sum(
                np.abs(
                    pairwise_diffs
                    -
                    spacing
                )
                <=
                tolerance
            )
        )

        if support > best_support:
            best_support = support
            best_spacing = float(spacing)

    tolerance = (
        spacing_tolerance
        *
        best_spacing
    )

    best_sequence = []

    for start_idx in range(len(peaks)):
        sequence = [
            int(peaks[start_idx])
        ]

        current = sequence[0]

        while True:
            target = (
                current
                +
                best_spacing
            )

            remaining = peaks[
                peaks > current
            ]

            if len(remaining) == 0:
                break

            errors = np.abs(
                remaining
                -
                target
            )

            best_match_idx = int(
                np.argmin(errors)
            )

            if errors[best_match_idx] <= tolerance:
                current = int(
                    remaining[
                        best_match_idx
                    ]
                )
                sequence.append(current)
            else:
                break

        if len(sequence) > len(best_sequence):
            best_sequence = sequence

    best_sequence = np.asarray(
        best_sequence,
        dtype=int,
    )

    if len(best_sequence) < min_sequence_length:
        best_sequence = peaks.copy()
        dominant_spacing = np.nan
        sequence_span = 0.0
    else:
        dominant_spacing = float(
            np.median(
                np.diff(best_sequence)
            )
        )
        sequence_span = float(
            best_sequence[-1]
            -
            best_sequence[0]
        )

    return {
        "sequence_peaks": best_sequence,
        "dominant_spacing": dominant_spacing,
        "sequence_length": len(best_sequence),
        "sequence_span": sequence_span,
        "spacing_support": int(best_support),
    }


def refine_peak_result_with_regular_sequence(
    peak_result,
    profile_length,
    axis,
    spacing_tolerance=0.08,
    min_sequence_length=4,
):
    peaks = np.asarray(
        peak_result["peaks"],
        dtype=int,
    )

    if axis == "horizontal":
        min_spacing_ratio = 0.06
        max_spacing_ratio = 0.19

    elif axis == "vertical":
        min_spacing_ratio = 0.07
        max_spacing_ratio = 0.21

    else:
        raise ValueError(
            f"Unknown axis: {axis}"
        )

    sequence = extract_regular_peak_sequence(
        peaks=peaks,
        min_spacing_ratio=min_spacing_ratio,
        max_spacing_ratio=max_spacing_ratio,
        profile_length=profile_length,
        spacing_tolerance=spacing_tolerance,
        min_sequence_length=min_sequence_length,
    )

    sequence_peaks = np.asarray(
        sequence["sequence_peaks"],
        dtype=int,
    )

    refined = dict(
        peak_result
    )

    refined[
        "dominant_spacing_px"
    ] = (
        sequence["dominant_spacing"]
    )

    refined[
        "regular_sequence_length"
    ] = (
        sequence["sequence_length"]
    )

    refined[
        "regular_sequence_span"
    ] = (
        sequence["sequence_span"]
    )

    refined[
        "spacing_support"
    ] = (
        sequence["spacing_support"]
    )

    if len(sequence_peaks) == 0:
        return refined

    peak_lookup = {
        int(peak): idx
        for idx, peak in enumerate(
            peak_result["peaks"]
        )
    }

    indices = [
        peak_lookup[int(peak)]
        for peak in sequence_peaks
        if int(peak) in peak_lookup
    ]

    if len(indices) == 0:
        return refined

    indices = np.asarray(
        indices,
        dtype=int,
    )

    refined["peaks"] = (
        peak_result["peaks"][indices]
    )

    refined["heights"] = (
        peak_result["heights"][indices]
    )

    refined["prominences"] = (
        peak_result["prominences"][indices]
    )

    refined["widths"] = (
        peak_result["widths"][indices]
    )

    return refined


# ============================================================
# 7. Evaluate peak sequence
# ============================================================

def evaluate_peak_sequence(
    peak_result,
    profile_length,
):
    peaks = np.asarray(
        peak_result["peaks"]
    )

    prominences = np.asarray(
        peak_result["prominences"]
    )

    widths = np.asarray(
        peak_result["widths"]
    )

    n_peaks = len(peaks)

    if n_peaks >= 2:
        gaps = np.diff(
            peaks
        ).astype(
            np.float32
        )

        median_gap = float(
            np.median(gaps)
        )

        gap_iqr = float(
            np.percentile(
                gaps,
                75,
            )
            -
            np.percentile(
                gaps,
                25,
            )
        )

        gap_mean = float(
            np.mean(gaps)
        )

        gap_std = float(
            np.std(gaps)
        )

        gap_cv = (
            gap_std
            /
            gap_mean
            if gap_mean > 0
            else np.nan
        )

        sequence_span = (
            float(
                peaks[-1]
                -
                peaks[0]
            )
            /
            max(
                profile_length - 1,
                1,
            )
        )

    else:
        gaps = np.array(
            [],
            dtype=np.float32,
        )

        median_gap = np.nan
        gap_iqr = np.nan
        gap_cv = np.nan
        sequence_span = 0.0

    if len(prominences) > 0:
        median_prominence = float(
            np.median(prominences)
        )

        prominence_mean = float(
            np.mean(prominences)
        )

        prominence_cv = (
            float(
                np.std(prominences)
            )
            /
            prominence_mean
            if prominence_mean > 0
            else np.nan
        )

    else:
        median_prominence = np.nan
        prominence_cv = np.nan

    if len(widths) > 0:
        median_width = float(
            np.median(widths)
        )

        width_mean = float(
            np.mean(widths)
        )

        width_cv = (
            float(
                np.std(widths)
            )
            /
            width_mean
            if width_mean > 0
            else np.nan
        )

    else:
        median_width = np.nan
        width_cv = np.nan

    return {
        "n_peaks": n_peaks,
        "median_gap_px": median_gap,
        "gap_iqr_px": gap_iqr,
        "gap_cv": gap_cv,
        "median_prominence": median_prominence,
        "prominence_cv": prominence_cv,
        "median_width_px": median_width,
        "width_cv": width_cv,
        "sequence_span_fraction": sequence_span,
        "gaps": gaps,
    }


# ============================================================
# 8. Evaluate whole-profile sparsity
# ============================================================

def evaluate_profile_sparsity(profile):
    profile = np.asarray(
        profile,
        dtype=np.float32,
    )

    if len(profile) == 0:
        return {
            "profile_median": np.nan,
            "profile_q75": np.nan,
            "profile_q90": np.nan,
            "active_fraction_010": np.nan,
            "active_fraction_020": np.nan,
            "active_fraction_030": np.nan,
        }

    return {
        "profile_median":
            float(
                np.median(profile)
            ),

        "profile_q75":
            float(
                np.percentile(
                    profile,
                    75,
                )
            ),

        "profile_q90":
            float(
                np.percentile(
                    profile,
                    90,
                )
            ),

        "active_fraction_010":
            float(
                np.mean(
                    profile > 0.10
                )
            ),

        "active_fraction_020":
            float(
                np.mean(
                    profile > 0.20
                )
            ),

        "active_fraction_030":
            float(
                np.mean(
                    profile > 0.30
                )
            ),
    }


# ============================================================
# 9. Evaluate local peak isolation
# ============================================================

def evaluate_peak_isolation(
    profile,
    peaks,
    local_radius=12,
    exclusion_radius=2,
):
    profile = np.asarray(
        profile,
        dtype=np.float32,
    )

    peaks = np.asarray(
        peaks,
        dtype=int,
    )

    if len(peaks) == 0:
        return {
            "median_local_contrast": np.nan,
            "peak_to_background_ratio": np.nan,
        }

    local_contrasts = []
    n = len(profile)

    for peak in peaks:
        start = max(
            0,
            peak - local_radius,
        )

        end = min(
            n,
            peak + local_radius + 1,
        )

        local_idx = np.arange(
            start,
            end,
        )

        mask = (
            np.abs(
                local_idx
                -
                peak
            )
            >
            exclusion_radius
        )

        background_idx = (
            local_idx[mask]
        )

        if len(background_idx) == 0:
            continue

        background = float(
            np.median(
                profile[
                    background_idx
                ]
            )
        )

        peak_value = float(
            profile[peak]
        )

        contrast = (
            peak_value
            -
            background
        ) / (
            peak_value
            +
            1e-8
        )

        local_contrasts.append(
            contrast
        )

    if len(local_contrasts) == 0:
        median_local_contrast = np.nan
    else:
        median_local_contrast = float(
            np.median(
                local_contrasts
            )
        )

    keep_mask = np.ones(
        len(profile),
        dtype=bool,
    )

    for peak in peaks:
        start = max(
            0,
            peak - exclusion_radius,
        )

        end = min(
            len(profile),
            peak + exclusion_radius + 1,
        )

        keep_mask[start:end] = False

    background_values = profile[
        keep_mask
    ]

    if len(background_values) == 0:
        peak_to_background_ratio = np.nan

    else:
        background_q75 = float(
            np.percentile(
                background_values,
                75,
            )
        )

        median_peak = float(
            np.median(
                profile[peaks]
            )
        )

        peak_to_background_ratio = (
            median_peak
            /
            (
                background_q75
                +
                1e-8
            )
        )

    return {
        "median_local_contrast":
            median_local_contrast,

        "peak_to_background_ratio":
            float(
                peak_to_background_ratio
            )
            if np.isfinite(
                peak_to_background_ratio
            )
            else np.nan,
    }


# ============================================================
# 10. Broad physical tick-spacing plausibility
# ============================================================

def is_plausible_tick_spacing(
    gap_ratio,
    axis,
):
    if not np.isfinite(
        gap_ratio
    ):
        return False

    if axis == "horizontal":
        return (
            0.06
            <=
            gap_ratio
            <=
            0.19
        )

    if axis == "vertical":
        return (
            0.07
            <=
            gap_ratio
            <=
            0.21
        )

    return False


# ============================================================
# 11. Tick-likelihood score
# ============================================================

def score_tick_likelihood(
    sequence_result,
    isolation_result,
    sparsity_result,
    min_peaks=4,
    full_confidence_peaks=8,
):
    n_peaks = sequence_result[
        "n_peaks"
    ]

    gap_cv = sequence_result[
        "gap_cv"
    ]

    prominence = sequence_result[
        "median_prominence"
    ]

    width_cv = sequence_result[
        "width_cv"
    ]

    span = sequence_result[
        "sequence_span_fraction"
    ]

    local_contrast = isolation_result[
        "median_local_contrast"
    ]

    peak_background_ratio = isolation_result[
        "peak_to_background_ratio"
    ]

    profile_q75 = sparsity_result[
        "profile_q75"
    ]

    active_fraction_020 = sparsity_result[
        "active_fraction_020"
    ]

    if n_peaks < min_peaks:
        return {
            "tick_score": 0.0,
            "sequence_confidence": 0.0,
            "spacing_score": 0.0,
            "prominence_score": 0.0,
            "width_score": 0.0,
            "span_score": 0.0,
            "isolation_score": 0.0,
            "background_ratio_score": 0.0,
            "sparsity_score": 0.0,
        }

    sequence_confidence = float(
        np.clip(
            (
                n_peaks
                -
                min_peaks
                +
                1
            )
            /
            (
                full_confidence_peaks
                -
                min_peaks
                +
                1
            ),
            0.0,
            1.0,
        )
    )

    if np.isfinite(gap_cv):
        spacing_score = float(
            np.clip(
                1.0
                -
                gap_cv
                /
                0.40,
                0.0,
                1.0,
            )
        )
    else:
        spacing_score = 0.0

    if np.isfinite(prominence):
        prominence_score = float(
            np.clip(
                prominence,
                0.0,
                1.0,
            )
        )
    else:
        prominence_score = 0.0

    if np.isfinite(width_cv):
        width_score = float(
            np.clip(
                1.0
                -
                width_cv
                /
                0.60,
                0.0,
                1.0,
            )
        )
    else:
        width_score = 0.0

    if np.isfinite(span):
        span_score = float(
            np.clip(
                span,
                0.0,
                1.0,
            )
        )
    else:
        span_score = 0.0

    if np.isfinite(local_contrast):
        isolation_score = float(
            np.clip(
                local_contrast,
                0.0,
                1.0,
            )
        )
    else:
        isolation_score = 0.0

    if np.isfinite(peak_background_ratio):
        background_ratio_score = float(
            np.clip(
                (
                    peak_background_ratio
                    -
                    1.0
                )
                /
                4.0,
                0.0,
                1.0,
            )
        )
    else:
        background_ratio_score = 0.0

    if (
        np.isfinite(profile_q75)
        and
        np.isfinite(active_fraction_020)
    ):
        q75_score = float(
            np.clip(
                1.0
                -
                profile_q75
                /
                0.30,
                0.0,
                1.0,
            )
        )

        active_score = float(
            np.clip(
                1.0
                -
                active_fraction_020
                /
                0.35,
                0.0,
                1.0,
            )
        )

        sparsity_score = (
            0.50
            *
            q75_score
            +
            0.50
            *
            active_score
        )

    else:
        sparsity_score = 0.0

    base_score = (
        0.20 * spacing_score
        +
        0.15 * prominence_score
        +
        0.10 * width_score
        +
        0.10 * span_score
        +
        0.15 * isolation_score
        +
        0.10 * background_ratio_score
        +
        0.20 * sparsity_score
    )

    tick_score = (
        base_score
        *
        sequence_confidence
    )

    return {
        "tick_score":
            float(tick_score),

        "sequence_confidence":
            sequence_confidence,

        "spacing_score":
            spacing_score,

        "prominence_score":
            prominence_score,

        "width_score":
            width_score,

        "span_score":
            span_score,

        "isolation_score":
            isolation_score,

        "background_ratio_score":
            background_ratio_score,

        "sparsity_score":
            sparsity_score,
    }


# ============================================================
# 12. Edge-location selection prior
# ============================================================

def add_edge_selection_score(
    candidates,
    image_shape,
    strip_fraction=0.10,
    horizontal_tick_weight=0.70,
    horizontal_edge_weight=0.30,
    vertical_tick_weight=0.85,
    vertical_edge_weight=0.15,
):
    """
    Add an edge-location prior to each ruler candidate.

    Horizontal ruler candidates receive a stronger edge prior because,
    in this dataset, horizontal calibration rulers occur at the bottom
    edge.

    Vertical candidates keep the weaker edge prior because valid rulers
    may appear on either the left or right side.
    """

    height, width = image_shape

    strip_width = int(
        round(
            width * strip_fraction
        )
    )

    strip_height = int(
        round(
            height * strip_fraction
        )
    )

    df = candidates.copy()

    edge_distances = []
    edge_relative = []
    edge_scores = []
    selection_scores = []

    for _, row in df.iterrows():

        side = row["side"]
        axis = row["axis"]

        band_center = (
            row["band_start"]
            +
            row["band_end"]
        ) / 2.0

        if side == "left":

            distance_px = band_center
            strip_size = strip_width

        elif side == "right":

            distance_px = (
                strip_width
                -
                band_center
            )

            strip_size = strip_width

        elif side == "top":

            distance_px = band_center
            strip_size = strip_height

        elif side == "bottom":

            distance_px = (
                strip_height
                -
                band_center
            )

            strip_size = strip_height

        else:

            raise ValueError(
                f"Unknown side: {side}"
            )

        distance_relative = (
            distance_px
            /
            max(
                strip_size,
                1,
            )
        )

        distance_relative = float(
            np.clip(
                distance_relative,
                0.0,
                1.0,
            )
        )

        edge_score = (
            1.0
            -
            distance_relative
        )

        if axis == "horizontal":

            selection_score = (
                horizontal_tick_weight
                *
                row["tick_score"]
                +
                horizontal_edge_weight
                *
                edge_score
            )

        elif axis == "vertical":

            selection_score = (
                vertical_tick_weight
                *
                row["tick_score"]
                +
                vertical_edge_weight
                *
                edge_score
            )

        else:

            raise ValueError(
                f"Unknown axis: {axis}"
            )

        edge_distances.append(
            distance_px
        )

        edge_relative.append(
            distance_relative
        )

        edge_scores.append(
            edge_score
        )

        selection_scores.append(
            float(
                selection_score
            )
        )

    df[
        "edge_distance_px"
    ] = edge_distances

    df[
        "edge_distance_relative"
    ] = edge_relative

    df[
        "edge_score"
    ] = edge_scores

    df[
        "selection_score"
    ] = selection_scores

    return (
        df
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def accept_horizontal_ruler(
    candidate,
    min_selection_score=0.45,
):
    """
    Accept or reject the selected horizontal ruler candidate.
    """

    if candidate is None:
        return False

    score = candidate.get(
        "selection_score",
        np.nan,
    )

    return bool(
        np.isfinite(score)
        and
        score >= min_selection_score
    )


def accept_vertical_ruler(
    candidate,
    min_sparsity_score=0.90,
    min_isolation_score=0.90,
):
    """
    Accept or reject the selected vertical ruler candidate.

    Vertical ultrasound texture can imitate regular tick spacing,
    so acceptance uses the sparse and isolated response structure
    that separated real rulers from false positives during review.
    """

    if candidate is None:
        return False

    sparsity_score = candidate.get(
        "sparsity_score",
        np.nan,
    )

    isolation_score = candidate.get(
        "isolation_score",
        np.nan,
    )

    return bool(
        np.isfinite(
            sparsity_score
        )
        and
        np.isfinite(
            isolation_score
        )
        and
        sparsity_score >= min_sparsity_score
        and
        isolation_score >= min_isolation_score
    )

# ============================================================
# 13. Full candidate analysis
# ============================================================

def analyze_ruler_candidates(
    image,
    strip_fraction=0.10,
    band_fraction=0.20,
    overlap_fraction=0.50,
    min_band_width=8,
    min_distance=20,
    min_prominence=0.10,
    relative_prominence_threshold=0.30,
    spacing_tolerance=0.08,
    min_sequence_length=4,
    apply_spacing_filter=True,
):
    image = np.asarray(image)

    height, width = image.shape

    strips = get_border_strips(
        image=image,
        strip_fraction=strip_fraction,
    )

    rows = []

    search_sides = [
        "left",
        "right",
        "bottom",

    ]

    for side in search_sides:

        strip = strips[side]

        bands = build_subband_profiles(
            strip=strip,
            side=side,
            band_fraction=band_fraction,
            overlap_fraction=overlap_fraction,
            min_band_width=min_band_width,
        )

        if side in ("top", "bottom"):
            axis = "horizontal"
            dimension_px = width
        else:
            axis = "vertical"
            dimension_px = height

        for band in bands:
            profile = band["profile"]

            peak_result = detect_tick_peaks(
                profile=profile,
                min_distance=min_distance,
                min_prominence=min_prominence,
                relative_prominence_threshold=
                    relative_prominence_threshold,
            )

            peak_result = (
                refine_peak_result_with_regular_sequence(
                    peak_result=peak_result,
                    profile_length=len(profile),
                    axis=axis,
                    spacing_tolerance=spacing_tolerance,
                    min_sequence_length=min_sequence_length,
                )
            )

            sequence_result = (
                evaluate_peak_sequence(
                    peak_result=peak_result,
                    profile_length=len(profile),
                )
            )

            isolation_result = (
                evaluate_peak_isolation(
                    profile=profile,
                    peaks=peak_result["peaks"],
                )
            )

            sparsity_result = (
                evaluate_profile_sparsity(
                    profile
                )
            )

            median_gap = (
                sequence_result[
                    "median_gap_px"
                ]
            )

            if np.isfinite(median_gap):
                gap_ratio = (
                    median_gap
                    /
                    dimension_px
                )
            else:
                gap_ratio = np.nan

            spacing_plausible = (
                is_plausible_tick_spacing(
                    gap_ratio=gap_ratio,
                    axis=axis,
                )
            )

            score_result = (
                score_tick_likelihood(
                    sequence_result=
                        sequence_result,

                    isolation_result=
                        isolation_result,

                    sparsity_result=
                        sparsity_result,
                )
            )

            rows.append(
                {
                    "side":
                        side,

                    "axis":
                        axis,

                    "band_start":
                        band[
                            "band_start"
                        ],

                    "band_end":
                        band[
                            "band_end"
                        ],

                    "band_width":
                        band[
                            "band_width"
                        ],

                    "tick_score":
                        score_result[
                            "tick_score"
                        ],

                    "spacing_plausible":
                        spacing_plausible,

                    "gap_ratio":
                        gap_ratio,

                    "n_peaks":
                        sequence_result[
                            "n_peaks"
                        ],

                    "median_gap_px":
                        median_gap,

                    "gap_iqr_px":
                        sequence_result[
                            "gap_iqr_px"
                        ],

                    "gap_cv":
                        sequence_result[
                            "gap_cv"
                        ],

                    "median_prominence":
                        sequence_result[
                            "median_prominence"
                        ],

                    "median_width_px":
                        sequence_result[
                            "median_width_px"
                        ],

                    "width_cv":
                        sequence_result[
                            "width_cv"
                        ],

                    "sequence_span_fraction":
                        sequence_result[
                            "sequence_span_fraction"
                        ],

                    "dominant_spacing_px":
                        peak_result.get(
                            "dominant_spacing_px",
                            np.nan,
                        ),

                    "regular_sequence_length":
                        peak_result.get(
                            "regular_sequence_length",
                            0,
                        ),

                    "regular_sequence_span":
                        peak_result.get(
                            "regular_sequence_span",
                            0.0,
                        ),

                    "spacing_support":
                        peak_result.get(
                            "spacing_support",
                            0,
                        ),

                    "median_local_contrast":
                        isolation_result[
                            "median_local_contrast"
                        ],

                    "peak_to_background_ratio":
                        isolation_result[
                            "peak_to_background_ratio"
                        ],

                    "profile_median":
                        sparsity_result[
                            "profile_median"
                        ],

                    "profile_q75":
                        sparsity_result[
                            "profile_q75"
                        ],

                    "profile_q90":
                        sparsity_result[
                            "profile_q90"
                        ],

                    "active_fraction_010":
                        sparsity_result[
                            "active_fraction_010"
                        ],

                    "active_fraction_020":
                        sparsity_result[
                            "active_fraction_020"
                        ],

                    "active_fraction_030":
                        sparsity_result[
                            "active_fraction_030"
                        ],

                    "spacing_score":
                        score_result[
                            "spacing_score"
                        ],

                    "prominence_score":
                        score_result[
                            "prominence_score"
                        ],

                    "width_score":
                        score_result[
                            "width_score"
                        ],

                    "span_score":
                        score_result[
                            "span_score"
                        ],

                    "isolation_score":
                        score_result[
                            "isolation_score"
                        ],

                    "background_ratio_score":
                        score_result[
                            "background_ratio_score"
                        ],

                    "sparsity_score":
                        score_result[
                            "sparsity_score"
                        ],

                    "sequence_confidence":
                        score_result[
                            "sequence_confidence"
                        ],

                    "peaks":
                        peak_result[
                            "peaks"
                        ],

                    "filtered_peaks":
                        peak_result.get(
                            "filtered_peaks",
                            peak_result["peaks"],
                        ),

                    "all_peaks":
                        peak_result.get(
                            "all_peaks",
                            peak_result["peaks"],
                        ),

                    "all_prominences":
                        peak_result.get(
                            "all_prominences",
                            np.array(
                                [],
                                dtype=np.float32,
                            ),
                        ),

                    "profile":
                        profile,
                }
            )

    ranked = pd.DataFrame(
        rows
    )

    ranked = add_edge_selection_score(
        candidates=ranked,
        image_shape=image.shape,
        strip_fraction=strip_fraction,
        horizontal_tick_weight=0.70,
        horizontal_edge_weight=0.30,
        vertical_tick_weight=0.85,
        vertical_edge_weight=0.15,
        )

    if apply_spacing_filter:
        ranking_pool = (
            ranked[
                ranked[
                    "spacing_plausible"
                ]
            ]
            .copy()
        )
    else:
        ranking_pool = ranked.copy()

    ranking_pool = (
        ranking_pool
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    horizontal = (
        ranking_pool[
            ranking_pool["axis"]
            ==
            "horizontal"
        ]
    )

    if len(horizontal) > 0:
        best_horizontal = (
            horizontal
            .iloc[0]
            .to_dict()
        )
    else:
        best_horizontal = None

    vertical = (
        ranking_pool[
            ranking_pool["axis"]
            ==
            "vertical"
        ]
    )

    if len(vertical) > 0:
        best_vertical = (
            vertical
            .iloc[0]
            .to_dict()
        )
    else:
        best_vertical = None



    horizontal_accepted = (
    accept_horizontal_ruler(
        best_horizontal
    )
)

    vertical_accepted = (
        accept_vertical_ruler(
            best_vertical
        )
    )

    accepted_horizontal = (
        best_horizontal
        if horizontal_accepted
        else None
    )

    accepted_vertical = (
        best_vertical
        if vertical_accepted
        else None
    )

    return {
        "ranked_candidates":
            ranking_pool,

        "all_candidates":
            ranked,

        "best_horizontal_candidate":
            best_horizontal,

        "best_vertical_candidate":
            best_vertical,

        "strips":
            strips,

        "horizontal_accepted":
        horizontal_accepted,

        "vertical_accepted":
            vertical_accepted,

        "accepted_horizontal_candidate":
            accepted_horizontal,

        "accepted_vertical_candidate":
            accepted_vertical,
    
    }
