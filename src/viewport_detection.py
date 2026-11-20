from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


# ============================================================
# TEXTURE MAP
# ============================================================

def compute_local_texture_map(
    image: np.ndarray,
    kernel_size: int = 11,
) -> np.ndarray:
    """
    Compute local intensity standard deviation.

    High values indicate locally varied / textured regions.
    Low values indicate locally uniform regions.
    """

    if image.ndim != 2:
        raise ValueError(
            "image must be a 2D grayscale array"
        )

    image_f = image.astype(
        np.float32
    )

    mean = cv2.blur(
        image_f,
        (
            kernel_size,
            kernel_size,
        )
    )

    mean_sq = cv2.blur(
        image_f ** 2,
        (
            kernel_size,
            kernel_size,
        )
    )

    variance = (
        mean_sq
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        0.0
    )

    return np.sqrt(
        variance
    )


# ============================================================
# HORIZONTAL EXPANSION
# ============================================================

def _expand_horizontal_side(
    texture_map: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    side: str,
    step: int = 5,
) -> pd.DataFrame:
    """
    Expand left or right from an initial box.

    At each step, score only the newly added strip.
    """

    _, w = texture_map.shape

    current_x1 = int(x1)
    current_x2 = int(x2)

    records = []
    step_number = 0

    while True:

        if side == "left":

            new_x1 = max(
                0,
                current_x1 - step
            )

            strip = texture_map[
                y1:y2,
                new_x1:current_x1
            ]

            if strip.size == 0:
                break

            current_x1 = new_x1

            distance = (
                x1 - current_x1
            )

        elif side == "right":

            new_x2 = min(
                w,
                current_x2 + step
            )

            strip = texture_map[
                y1:y2,
                current_x2:new_x2
            ]

            if strip.size == 0:
                break

            current_x2 = new_x2

            distance = (
                current_x2 - x2
            )

        else:
            raise ValueError(
                "side must be 'left' or 'right'"
            )

        step_number += 1

        records.append({
            "step":
                step_number,

            "distance":
                int(distance),

            "texture_score":
                float(
                    strip.mean()
                ),
        })

        if (
            current_x1 == 0
            or current_x2 == w
        ):
            break

    return pd.DataFrame(
        records
    )


# ============================================================
# VERTICAL EXPANSION
# ============================================================

def _expand_vertical_side(
    texture_map: np.ndarray,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    side: str,
    step: int = 5,
) -> pd.DataFrame:
    """
    Expand top or bottom using the detected horizontal
    viewport width.

    Scoring a full-width strip makes the detector less
    sensitive to local anatomical texture changes.
    """

    h, _ = texture_map.shape

    current_y1 = int(y1)
    current_y2 = int(y2)

    records = []
    step_number = 0

    while True:

        if side == "top":

            new_y1 = max(
                0,
                current_y1 - step
            )

            strip = texture_map[
                new_y1:current_y1,
                x1:x2
            ]

            if strip.size == 0:
                break

            current_y1 = new_y1

            distance = (
                y1 - current_y1
            )

        elif side == "bottom":

            new_y2 = min(
                h,
                current_y2 + step
            )

            strip = texture_map[
                current_y2:new_y2,
                x1:x2
            ]

            if strip.size == 0:
                break

            current_y2 = new_y2

            distance = (
                current_y2 - y2
            )

        else:
            raise ValueError(
                "side must be 'top' or 'bottom'"
            )

        step_number += 1

        records.append({
            "step":
                step_number,

            "distance":
                int(distance),

            "texture_score":
                float(
                    strip.mean()
                ),
        })

        if (
            current_y1 == 0
            or current_y2 == h
        ):
            break

    return pd.DataFrame(
        records
    )


# ============================================================
# PRIMARY BOUNDARY DETECTOR
# ============================================================

def _detect_abrupt_sustained_boundary(
    strip_df: pd.DataFrame,
    drop_ratio: float = 0.35,
    sustain_ratio: float = 0.50,
    sustain_steps: int = 4,
    history_steps: int = 5,
) -> dict:
    """
    Primary boundary detector.

    Requires:

    1. An abrupt local texture drop.

    2. A sustained low-texture regime afterward.

    This is intentionally conservative so that internal
    anatomical texture changes are less likely to be
    mistaken for viewport boundaries.
    """

    if len(strip_df) == 0:
        return {
            "boundary_found": False,
            "boundary_distance": None,
            "method": None,
        }

    scores = strip_df[
        "texture_score"
    ].to_numpy(
        dtype=float
    )

    distances = strip_df[
        "distance"
    ].to_numpy(
        dtype=float
    )

    start_index = max(
        1,
        history_steps
    )

    stop_index = (
        len(scores)
        - sustain_steps
        + 1
    )

    for i in range(
        start_index,
        stop_index
    ):

        previous_score = scores[
            i - 1
        ]

        if previous_score <= 0:
            continue

        local_drop_ratio = (
            scores[i]
            / previous_score
        )

        # Require an abrupt transition.
        if (
            local_drop_ratio
            >= drop_ratio
        ):
            continue

        reference = np.median(
            scores[
                i - history_steps:i
            ]
        )

        if reference <= 0:
            continue

        future_scores = scores[
            i:i + sustain_steps
        ]

        # Require a sustained lower regime.
        if np.all(
            future_scores
            <
            reference * sustain_ratio
        ):

            return {
                "boundary_found":
                    True,

                "boundary_index":
                    int(i),

                "boundary_distance":
                    float(
                        distances[i]
                    ),

                "previous_score":
                    float(
                        previous_score
                    ),

                "current_score":
                    float(
                        scores[i]
                    ),

                "local_drop_ratio":
                    float(
                        local_drop_ratio
                    ),

                "reference_texture":
                    float(
                        reference
                    ),

                "method":
                    "abrupt_sustained",
            }

    return {
        "boundary_found": False,
        "boundary_distance": None,
        "method": None,
    }


# ============================================================
# HORIZONTAL FALLBACK DETECTOR
# ============================================================

def _detect_regime_shift_boundary(
    strip_df: pd.DataFrame,
    history_steps: int = 5,
    future_steps: int = 5,
    regime_ratio: float = 0.50,
) -> dict:
    """
    Fallback boundary detector for LEFT / RIGHT only.

    Compares a short texture regime before a candidate
    transition with a short regime afterward.

    This handles cases where the UI transition occurs over
    several strips rather than through one abrupt step.

    Do NOT use this as the primary vertical detector because
    top/bottom anatomical changes can create similar sustained
    texture shifts inside the ultrasound image.
    """

    if len(strip_df) == 0:
        return {
            "boundary_found": False,
            "boundary_distance": None,
            "method": None,
        }

    scores = strip_df[
        "texture_score"
    ].to_numpy(
        dtype=float
    )

    distances = strip_df[
        "distance"
    ].to_numpy(
        dtype=float
    )

    minimum_length = (
        history_steps
        + future_steps
    )

    if len(scores) < minimum_length:
        return {
            "boundary_found": False,
            "boundary_distance": None,
            "method": None,
        }

    for i in range(
        history_steps,
        len(scores) - future_steps + 1
    ):

        previous_region = scores[
            i - history_steps:i
        ]

        future_region = scores[
            i:i + future_steps
        ]

        previous_level = float(
            np.median(
                previous_region
            )
        )

        future_level = float(
            np.median(
                future_region
            )
        )

        if previous_level <= 0:
            continue

        ratio = (
            future_level
            / previous_level
        )

        if ratio < regime_ratio:

            return {
                "boundary_found":
                    True,

                "boundary_index":
                    int(i),

                "boundary_distance":
                    float(
                        distances[i]
                    ),

                "previous_level":
                    previous_level,

                "future_level":
                    future_level,

                "regime_ratio":
                    float(
                        ratio
                    ),

                "method":
                    "regime_shift_fallback",
            }

    return {
        "boundary_found": False,
        "boundary_distance": None,
        "method": None,
    }


# ============================================================
# MAIN VIEWPORT DETECTOR
# ============================================================

def detect_ultrasound_viewport(
    image: np.ndarray,
    start_half_width: int = 100,
    start_half_height: int = 100,
    step: int = 5,
    kernel_size: int = 11,
    drop_ratio: float = 0.35,
    sustain_ratio: float = 0.50,
    sustain_steps: int = 4,
    history_steps: int = 5,
    fallback_future_steps: int = 5,
    fallback_regime_ratio: float = 0.50,
) -> dict:
    """
    Detect the ultrasound viewport from local texture.

    Strategy
    --------
    1. Start from a centered seed box.

    2. Expand left and right independently.

    3. Use abrupt + sustained detection first.

    4. If LEFT or RIGHT fails, use the more permissive
       regime-shift detector as a horizontal fallback.

    5. Once horizontal bounds are known, evaluate full-width
       top/bottom strips.

    6. TOP and BOTTOM use ONLY abrupt + sustained detection.

    7. If no valid boundary is found on a side, use the image
       edge itself.

    This intentionally favors preserving scan content over
    aggressively trimming UI.
    """

    if image.ndim != 2:
        raise ValueError(
            "image must be a 2D grayscale array"
        )

    h, w = image.shape

    texture_map = (
        compute_local_texture_map(
            image=image,
            kernel_size=kernel_size,
        )
    )

    center_x = w // 2
    center_y = h // 2

    # Prevent the seed from exceeding smaller images.
    half_width = min(
        start_half_width,
        max(
            1,
            center_x - 1
        ),
        max(
            1,
            w - center_x - 1
        ),
    )

    half_height = min(
        start_half_height,
        max(
            1,
            center_y - 1
        ),
        max(
            1,
            h - center_y - 1
        ),
    )

    x1_start = (
        center_x
        - half_width
    )

    x2_start = (
        center_x
        + half_width
    )

    y1_start = (
        center_y
        - half_height
    )

    y2_start = (
        center_y
        + half_height
    )

    # ========================================================
    # HORIZONTAL STRIPS
    # ========================================================

    left_df = (
        _expand_horizontal_side(
            texture_map=texture_map,
            x1=x1_start,
            y1=y1_start,
            x2=x2_start,
            y2=y2_start,
            side="left",
            step=step,
        )
    )

    right_df = (
        _expand_horizontal_side(
            texture_map=texture_map,
            x1=x1_start,
            y1=y1_start,
            x2=x2_start,
            y2=y2_start,
            side="right",
            step=step,
        )
    )

    # ========================================================
    # LEFT — PRIMARY
    # ========================================================

    left_result = (
        _detect_abrupt_sustained_boundary(
            strip_df=left_df,
            drop_ratio=drop_ratio,
            sustain_ratio=sustain_ratio,
            sustain_steps=sustain_steps,
            history_steps=history_steps,
        )
    )

    # Horizontal fallback only.
    if not left_result[
        "boundary_found"
    ]:

        left_result = (
            _detect_regime_shift_boundary(
                strip_df=left_df,
                history_steps=history_steps,
                future_steps=fallback_future_steps,
                regime_ratio=fallback_regime_ratio,
            )
        )

    # ========================================================
    # RIGHT — PRIMARY
    # ========================================================

    right_result = (
        _detect_abrupt_sustained_boundary(
            strip_df=right_df,
            drop_ratio=drop_ratio,
            sustain_ratio=sustain_ratio,
            sustain_steps=sustain_steps,
            history_steps=history_steps,
        )
    )

    # Horizontal fallback only.
    if not right_result[
        "boundary_found"
    ]:

        right_result = (
            _detect_regime_shift_boundary(
                strip_df=right_df,
                history_steps=history_steps,
                future_steps=fallback_future_steps,
                regime_ratio=fallback_regime_ratio,
            )
        )

    # ========================================================
    # HORIZONTAL BOUNDS
    # ========================================================

    if left_result[
        "boundary_found"
    ]:

        x1 = max(
            0,
            x1_start
            - int(
                left_result[
                    "boundary_distance"
                ]
            )
        )

        left_source = (
            left_result[
                "method"
            ]
        )

    else:

        x1 = 0
        left_source = "image_edge"

    if right_result[
        "boundary_found"
    ]:

        x2 = min(
            w,
            x2_start
            + int(
                right_result[
                    "boundary_distance"
                ]
            )
        )

        right_source = (
            right_result[
                "method"
            ]
        )

    else:

        x2 = w
        right_source = "image_edge"

    # ========================================================
    # VERTICAL STRIPS
    #
    # Use the detected horizontal viewport width.
    # ========================================================

    top_df = (
        _expand_vertical_side(
            texture_map=texture_map,
            x1=x1,
            x2=x2,
            y1=y1_start,
            y2=y2_start,
            side="top",
            step=step,
        )
    )

    bottom_df = (
        _expand_vertical_side(
            texture_map=texture_map,
            x1=x1,
            x2=x2,
            y1=y1_start,
            y2=y2_start,
            side="bottom",
            step=step,
        )
    )

    # ========================================================
    # VERTICAL DETECTION
    #
    # IMPORTANT:
    # No regime-shift fallback here.
    # ========================================================

    top_result = (
        _detect_abrupt_sustained_boundary(
            strip_df=top_df,
            drop_ratio=drop_ratio,
            sustain_ratio=sustain_ratio,
            sustain_steps=sustain_steps,
            history_steps=history_steps,
        )
    )

    bottom_result = (
        _detect_abrupt_sustained_boundary(
            strip_df=bottom_df,
            drop_ratio=drop_ratio,
            sustain_ratio=sustain_ratio,
            sustain_steps=sustain_steps,
            history_steps=history_steps,
        )
    )

    # ========================================================
    # VERTICAL BOUNDS
    # ========================================================

    if top_result[
        "boundary_found"
    ]:

        y1 = max(
            0,
            y1_start
            - int(
                top_result[
                    "boundary_distance"
                ]
            )
        )

        top_source = (
            top_result[
                "method"
            ]
        )

    else:

        y1 = 0
        top_source = "image_edge"

    if bottom_result[
        "boundary_found"
    ]:

        y2 = min(
            h,
            y2_start
            + int(
                bottom_result[
                    "boundary_distance"
                ]
            )
        )

        bottom_source = (
            bottom_result[
                "method"
            ]
        )

    else:

        y2 = h
        bottom_source = "image_edge"

    # ========================================================
    # FINAL CROP
    # ========================================================

    crop = image[
        y1:y2,
        x1:x2
    ].copy()

    return {
        "crop":
            crop,

        "bounds": {
            "x1":
                int(x1),

            "x2":
                int(x2),

            "y1":
                int(y1),

            "y2":
                int(y2),
        },

        "seed_bounds": {
            "x1":
                int(x1_start),

            "x2":
                int(x2_start),

            "y1":
                int(y1_start),

            "y2":
                int(y2_start),
        },

        "boundary_sources": {
            "left":
                left_source,

            "right":
                right_source,

            "top":
                top_source,

            "bottom":
                bottom_source,
        },

        "diagnostics": {
            "left":
                left_result,

            "right":
                right_result,

            "top":
                top_result,

            "bottom":
                bottom_result,

            "left_strips":
                left_df,

            "right_strips":
                right_df,

            "top_strips":
                top_df,

            "bottom_strips":
                bottom_df,
        },

        "texture_map":
            texture_map,
    }