import numpy as np


def derive_mm_per_px(
    calibration_result,
    tick_interval_mm=10.0,
):
    """
    Derive one isotropic physical scale in mm/pixel.

    Priority:
      1. accepted vertical ruler
      2. accepted horizontal ruler fallback
    """
    vertical = calibration_result.get("accepted_vertical_candidate")
    horizontal = calibration_result.get("accepted_horizontal_candidate")

    if vertical is not None:
        gap_px = float(vertical["median_gap_px"])
        source = "vertical"
    elif horizontal is not None:
        gap_px = float(horizontal["median_gap_px"])
        source = "horizontal"
    else:
        return {
            "scale_found": False,
            "scale_source": None,
            "gap_px": np.nan,
            "mm_per_px": np.nan,
            "sx_mm_per_px": np.nan,
            "sy_mm_per_px": np.nan,
        }

    if (not np.isfinite(gap_px)) or gap_px <= 0:
        return {
            "scale_found": False,
            "scale_source": source,
            "gap_px": gap_px,
            "mm_per_px": np.nan,
            "sx_mm_per_px": np.nan,
            "sy_mm_per_px": np.nan,
        }

    mm_per_px = float(tick_interval_mm) / gap_px

    return {
        "scale_found": True,
        "scale_source": source,
        "gap_px": gap_px,
        "mm_per_px": mm_per_px,
        "sx_mm_per_px": mm_per_px,
        "sy_mm_per_px": mm_per_px,
    }


def point_px_to_mm(x_px, y_px, mm_per_px):
    scale = float(mm_per_px)
    return float(x_px) * scale, float(y_px) * scale


def points_px_to_mm(points_px, mm_per_px):
    points = np.asarray(points_px, dtype=np.float64)

    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError("points_px must have shape (..., 2)")

    return points * float(mm_per_px)


def length_px_to_mm(length_px, mm_per_px):
    if not np.isfinite(length_px):
        return np.nan

    return float(length_px) * float(mm_per_px)


def model_points_to_crop_px(
    points_model,
    resize_scale,
    pad_left=0.0,
    pad_top=0.0,
):
    """
    Reverse:
      crop -> resize by resize_scale -> pad
    """

    points = np.asarray(points_model, dtype=np.float64)

    if points.shape[-1] != 2:
        raise ValueError("points_model must have shape (..., 2)")

    resize_scale = float(resize_scale)

    if (not np.isfinite(resize_scale)) or resize_scale <= 0:
        raise ValueError("resize_scale must be positive and finite")

    output = points.copy()
    output[..., 0] = (output[..., 0] - float(pad_left)) / resize_scale
    output[..., 1] = (output[..., 1] - float(pad_top)) / resize_scale

    return output


def crop_points_to_raw_px(
    points_crop,
    crop_left=0.0,
    crop_top=0.0,
):
    points = np.asarray(points_crop, dtype=np.float64)

    if points.shape[-1] != 2:
        raise ValueError("points_crop must have shape (..., 2)")

    output = points.copy()
    output[..., 0] += float(crop_left)
    output[..., 1] += float(crop_top)

    return output


def model_points_to_raw_px(
    points_model,
    resize_scale,
    pad_left=0.0,
    pad_top=0.0,
    crop_left=0.0,
    crop_top=0.0,
):
    crop_points = model_points_to_crop_px(
        points_model=points_model,
        resize_scale=resize_scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )

    return crop_points_to_raw_px(
        points_crop=crop_points,
        crop_left=crop_left,
        crop_top=crop_top,
    )


def model_points_to_mm(
    points_model,
    resize_scale,
    mm_per_px,
    pad_left=0.0,
    pad_top=0.0,
):
    """
    Convert model-space coordinates directly to mm relative to crop origin.
    Translation offsets do not affect lengths or angles.
    """

    crop_points = model_points_to_crop_px(
        points_model=points_model,
        resize_scale=resize_scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )

    return points_px_to_mm(
        points_px=crop_points,
        mm_per_px=mm_per_px,
    )


def euclidean_distance_mm(
    point_a_px,
    point_b_px,
    mm_per_px,
):
    a = np.asarray(point_a_px, dtype=np.float64)
    b = np.asarray(point_b_px, dtype=np.float64)

    distance_px = float(np.linalg.norm(b - a))

    return distance_px * float(mm_per_px)


def polyline_length_mm(
    points_px,
    mm_per_px,
):
    points = np.asarray(points_px, dtype=np.float64)

    if len(points) < 2:
        return np.nan

    segment_lengths_px = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1,
    )

    return float(
        np.sum(segment_lengths_px)
        * float(mm_per_px)
    )


def acute_angle_deg(
    vector_a,
    vector_b,
):
    """
    Acute angle between two 2D axial directions.
    Isotropic physical scaling does not change the angle.
    """

    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a <= 0 or norm_b <= 0:
        return np.nan

    cosine = np.dot(a, b) / (norm_a * norm_b)
    cosine = abs(float(cosine))
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))




def derive_model_space_scale(
    calibration_result,
    resize_scale,
    tick_interval_mm=10.0,
):
    """
    Convert raw-image ruler calibration into the physical scale
    of the resized model coordinate system.
    """

    scale_info = derive_mm_per_px(
        calibration_result=calibration_result,
        tick_interval_mm=tick_interval_mm,
    )

    if not scale_info["scale_found"]:

        return {
            **scale_info,
            "resize_scale": resize_scale,
            "mm_per_model_px": np.nan,
        }

    resize_scale = float(
        resize_scale
    )

    if (
        not np.isfinite(resize_scale)
        or
        resize_scale <= 0
    ):

        raise ValueError(
            "resize_scale must be positive and finite"
        )

    mm_per_model_px = (
        scale_info["mm_per_px"]
        /
        resize_scale
    )

    return {
        **scale_info,

        "resize_scale":
            resize_scale,

        "mm_per_model_px":
            float(mm_per_model_px),
    }



def convert_measurements_to_physical(
    measurements,
    mm_per_model_px,
):
    """
    Convert pixel-space geometry measurements to physical units.

    PA is unchanged because scaling is isotropic.
    """

    scale = float(
        mm_per_model_px
    )

    output = dict(
        measurements
    )

    # Angle is invariant under isotropic scaling.
    if "PA_deg" in measurements:

        output[
            "PA_deg_physical"
        ] = measurements[
            "PA_deg"
        ]

    # Convert every length field we currently care about.
    length_fields = [
        "MT_px",
        "FL_curved_px",
        "FL_straight_px",
        "FL_combined_px",
        "FL_geom_px",
    ]

    for field in length_fields:

        if field not in measurements:
            continue

        value = measurements[
            field
        ]

        output[
            field.replace(
                "_px",
                "_mm",
            )
        ] = (
            np.nan
            if not np.isfinite(value)
            else
            float(value)
            *
            scale
        )

    return output