# src/muscle_geometry_measurements_v1.py

import numpy as np
from scipy.optimize import brentq

def estimate_pennation_angles(
    lower_apo,
    fascicle_orientation,
    buffered_muscle_mask,
    sample_step=20,
    inward_offset=12,
    neighborhood_radius=6,
):
    """
    Estimate local pennation angles along the deep/lower aponeurosis.

    Parameters
    ----------
    lower_apo : dict
        Reconstructed lower aponeurosis candidate.
        Expected keys:
        x, smooth_y, tx, ty, nx, ny

    fascicle_orientation : np.ndarray
        Doubled-angle fascicle orientation field with shape (2, H, W).

    buffered_muscle_mask : np.ndarray
        Binary mask defining valid interior muscle region.

    sample_step : int
        Step along lower apo x-coordinates.

    inward_offset : float
        Distance in pixels moved inward from lower apo.

    neighborhood_radius : int
        Radius used to average fascicle orientation around each sample point.

    Returns
    -------
    list of dict
    """

    H, W = buffered_muscle_mask.shape

    results = []

    x_vals = lower_apo["center_x"]
    y_vals = lower_apo["center_y"]

    tx_vals = lower_apo["tx"]
    ty_vals = lower_apo["ty"]

    nx_vals = lower_apo["nx"]
    ny_vals = lower_apo["ny"]

    for i in range(
        0,
        len(x_vals),
        sample_step
    ):

        x0 = x_vals[i]
        y0 = y_vals[i]

        apo_tx = tx_vals[i]
        apo_ty = ty_vals[i]

        # Lower apo stored normal generally points downward.
        # For anatomy, inward direction points upward into muscle.
        inward_nx = -nx_vals[i]
        inward_ny = -ny_vals[i]

        sample_x = (
            x0
            + inward_offset * inward_nx
        )

        sample_y = (
            y0
            + inward_offset * inward_ny
        )

        cx = int(round(sample_x))
        cy = int(round(sample_y))

        if not (
            0 <= cx < W
            and 0 <= cy < H
        ):
            continue

        if buffered_muscle_mask[
            cy,
            cx
        ] == 0:
            continue

        x_min = max(
            0,
            cx - neighborhood_radius
        )

        x_max = min(
            W,
            cx + neighborhood_radius + 1
        )

        y_min = max(
            0,
            cy - neighborhood_radius
        )

        y_max = min(
            H,
            cy + neighborhood_radius + 1
        )

        local_mask = (
            buffered_muscle_mask[
                y_min:y_max,
                x_min:x_max
            ]
            > 0
        )

        if not np.any(local_mask):
            continue

        local_v1 = fascicle_orientation[
            0,
            y_min:y_max,
            x_min:x_max
        ][local_mask]

        local_v2 = fascicle_orientation[
            1,
            y_min:y_max,
            x_min:x_max
        ][local_mask]

        mean_v1 = np.mean(
            local_v1
        )

        mean_v2 = np.mean(
            local_v2
        )

        norm = np.hypot(
            mean_v1,
            mean_v2
        )

        if norm < 1e-8:
            continue

        mean_v1 /= norm
        mean_v2 /= norm

        theta = (
            0.5
            * np.arctan2(
                mean_v2,
                mean_v1
            )
        )

        fasc_tx = np.cos(
            theta
        )

        # Convert from Cartesian orientation
        # to image coordinates.
        fasc_ty = -np.sin(
            theta
        )

        dot = (
            apo_tx * fasc_tx
            + apo_ty * fasc_ty
        )

        dot = np.clip(
            np.abs(dot),
            0.0,
            1.0
        )

        pa_deg = np.degrees(
            np.arccos(dot)
        )

        results.append(
            {
                "apo_x": float(x0),
                "apo_y": float(y0),
                "sample_x": float(sample_x),
                "sample_y": float(sample_y),
                "apo_tx": float(apo_tx),
                "apo_ty": float(apo_ty),
                "fasc_tx": float(fasc_tx),
                "fasc_ty": float(fasc_ty),
                "pa_deg": float(pa_deg),
            }
        )

    return results


def estimate_muscle_thickness(
    upper_apo,
    lower_apo,
    sample_step=20,
    max_distance=600,
):
    """
    Estimate muscle thickness in pixels by intersecting
    inward normals from the superficial/upper aponeurosis
    with the deep/lower aponeurosis.

    Returns
    -------
    list of dict
        One MT measurement per successful normal intersection.
    """

    upper_x = np.asarray(
        upper_apo["center_x"],
        dtype=float
    )

    upper_y = np.asarray(
        upper_apo["center_y"],
        dtype=float
    )

    upper_nx = np.asarray(
        upper_apo["nx"],
        dtype=float
    )

    upper_ny = np.asarray(
        upper_apo["ny"],
        dtype=float
    )

    lower_x = np.asarray(
        lower_apo["center_x"],
        dtype=float
    )

    lower_y = np.asarray(
        lower_apo["center_y"],
        dtype=float
    )

    lower_x_min = lower_x.min()
    lower_x_max = lower_x.max()

    results = []

    for i in range(
        0,
        len(upper_x),
        sample_step
    ):

        x0 = upper_x[i]
        y0 = upper_y[i]

        # For the upper apo, the stored normal
        # generally points downward into the muscle.
        nx = upper_nx[i]
        ny = upper_ny[i]

        # Ensure normal points toward the lower apo.
        if ny < 0:
            nx = -nx
            ny = -ny

        def intersection_function(t):

            x_ray = (
                x0
                + t * nx
            )

            y_ray = (
                y0
                + t * ny
            )

            if (
                x_ray < lower_x_min
                or x_ray > lower_x_max
            ):
                return np.nan

            y_lower = np.interp(
                x_ray,
                lower_x,
                lower_y
            )

            return (
                y_ray
                - y_lower
            )

        # Search outward along the normal
        # until the sign changes.
        previous_t = 0.0
        previous_value = intersection_function(
            previous_t
        )

        intersection_t = None

        for t in np.arange(
            1.0,
            max_distance + 1.0,
            1.0
        ):

            value = intersection_function(
                t
            )

            if (
                not np.isfinite(value)
                or not np.isfinite(previous_value)
            ):
                previous_t = t
                previous_value = value
                continue

            if (
                previous_value == 0
                or value == 0
                or np.sign(previous_value)
                != np.sign(value)
            ):

                try:
                    intersection_t = brentq(
                        intersection_function,
                        previous_t,
                        t
                    )
                except ValueError:
                    intersection_t = None

                break

            previous_t = t
            previous_value = value

        if intersection_t is None:
            continue

        x1 = (
            x0
            + intersection_t * nx
        )

        y1 = (
            y0
            + intersection_t * ny
        )

        results.append(
            {
                "upper_x": float(x0),
                "upper_y": float(y0),

                "lower_x": float(x1),
                "lower_y": float(y1),

                "nx": float(nx),
                "ny": float(ny),

                # normal is unit length,
                # therefore t is distance in pixels
                "mt_px": float(
                    intersection_t
                ),
            }
        )

    return results



import numpy as np


def bilinear_sample_field(
    field_x,
    field_y,
    x,
    y,
):
    """
    Bilinearly sample a 2D vector field at floating-point coordinates.
    """

    H, W = field_x.shape

    if not (
        0 <= x < W - 1
        and 0 <= y < H - 1
    ):
        return None

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    vx = (
        (1 - dx) * (1 - dy) * field_x[y0, x0]
        + dx * (1 - dy) * field_x[y0, x1]
        + (1 - dx) * dy * field_x[y1, x0]
        + dx * dy * field_x[y1, x1]
    )

    vy = (
        (1 - dx) * (1 - dy) * field_y[y0, x0]
        + dx * (1 - dy) * field_y[y0, x1]
        + (1 - dx) * dy * field_y[y1, x0]
        + dx * dy * field_y[y1, x1]
    )

    norm = np.hypot(
        vx,
        vy
    )

    if norm < 1e-8:
        return None

    return (
        vx / norm,
        vy / norm
    )




def trace_fascicle_between_apos(
    seed_x,
    seed_y,
    fascicle_tx,
    fascicle_ty,
    start_apo,
    target_apo,
    buffered_muscle_mask,
    step_size=1.0,
    max_steps=1000,
    initial_window=10,
    terminal_window=10,
    apo_tangent_window=10,
):
    """
    Trace a fascicle from one apo toward the other.

    Works for either:
        lower -> upper
    or:
        upper -> lower

    The orientation field is axial, so vector signs are chosen
    automatically according to the target direction and then
    preserved continuously during tracing.
    """

    H, W = (
        buffered_muscle_mask.shape
    )

    start_x = np.asarray(
        start_apo["center_x"],
        dtype=float,
    )

    start_y = np.asarray(
        start_apo["center_y"],
        dtype=float,
    )

    target_x = np.asarray(
        target_apo["center_x"],
        dtype=float,
    )

    target_y = np.asarray(
        target_apo["center_y"],
        dtype=float,
    )

    # --------------------------------------------------
    # Determine whether target is above or below start
    # in image coordinates.
    # --------------------------------------------------

    target_is_above = (
        np.median(target_y)
        <
        np.median(start_y)
    )

    x = float(seed_x)
    y = float(seed_y)

    path_x = [x]
    path_y = [y]

    directions = []

    previous_direction = None

    reached_target_directly = False

    # --------------------------------------------------
    # 1. Field-driven trace
    # --------------------------------------------------

    for _ in range(max_steps):

        sampled = bilinear_sample_field(
            fascicle_tx,
            fascicle_ty,
            x,
            y,
        )

        if sampled is None:
            break

        vx, vy = sampled

        # ----------------------------------------------
        # First vector:
        # choose sign toward target apo.
        # ----------------------------------------------

        if previous_direction is None:

            if target_is_above:

                if vy > 0:
                    vx = -vx
                    vy = -vy

            else:

                if vy < 0:
                    vx = -vx
                    vy = -vy

        # ----------------------------------------------
        # Remaining vectors:
        # preserve axial continuity.
        # ----------------------------------------------

        else:

            prev_x, prev_y = (
                previous_direction
            )

            dot = (
                vx * prev_x
                + vy * prev_y
            )

            if dot < 0:
                vx = -vx
                vy = -vy

        x_new = (
            x
            + step_size * vx
        )

        y_new = (
            y
            + step_size * vy
        )

        if not (
            0 <= x_new < W
            and 0 <= y_new < H
        ):
            break

        # ----------------------------------------------
        # Check direct target apo crossing
        # ----------------------------------------------

        if (
            target_x.min()
            <= x_new
            <= target_x.max()
        ):

            target_y_here = np.interp(
                x_new,
                target_x,
                target_y,
            )

            if target_is_above:

                crossed_target = (
                    y_new <= target_y_here
                )

            else:

                crossed_target = (
                    y_new >= target_y_here
                )

            if crossed_target:

                path_x.append(
                    x_new
                )

                path_y.append(
                    target_y_here
                )

                directions.append(
                    (vx, vy)
                )

                reached_target_directly = True

                break

        # ----------------------------------------------
        # Stop if trusted field ends
        # ----------------------------------------------

        ix = int(
            round(x_new)
        )

        iy = int(
            round(y_new)
        )

        if not (
            0 <= ix < W
            and 0 <= iy < H
        ):
            break

        if buffered_muscle_mask[
            iy,
            ix
        ] == 0:
            break

        path_x.append(
            x_new
        )

        path_y.append(
            y_new
        )

        directions.append(
            (vx, vy)
        )

        x = x_new
        y = y_new

        previous_direction = (
            vx,
            vy
        )

    path_x = np.asarray(
        path_x,
        dtype=float,
    )

    path_y = np.asarray(
        path_y,
        dtype=float,
    )

    # --------------------------------------------------
    # 2. Field-traced length
    # --------------------------------------------------

    if len(path_x) > 1:

        observed_fl_px = np.sum(
            np.hypot(
                np.diff(path_x),
                np.diff(path_y),
            )
        )

    else:

        observed_fl_px = 0.0

    # --------------------------------------------------
    # 3. Starting apo correction
    # --------------------------------------------------

    start_intersection = None
    start_correction_px = 0.0
    initial_direction = None

    if len(directions) > 0:

        initial_dirs = np.asarray(
            directions[
                :initial_window
            ],
            dtype=float,
        )

        initial_v = (
            initial_dirs.mean(
                axis=0
            )
        )

        initial_norm = np.hypot(
            initial_v[0],
            initial_v[1]
        )

        if initial_norm > 1e-8:

            initial_v /= initial_norm

            initial_direction = (
                float(initial_v[0]),
                float(initial_v[1]),
            )

            # Go backward from seed toward start apo.
            backward_v = (
                -initial_direction[0],
                -initial_direction[1],
            )

            start_intersection = (
                intersect_ray_with_apo(
                    p0=(
                        float(path_x[0]),
                        float(path_y[0]),
                    ),
                    v=backward_v,
                    apo=start_apo,
                    apo_tangent_window=
                        apo_tangent_window,
                )
            )

            if start_intersection is not None:

                start_correction_px = (
                    start_intersection[
                        "ray_distance_px"
                    ]
                )

    # --------------------------------------------------
    # 4. Target apo intersection / extrapolation
    # --------------------------------------------------

    target_intersection = None
    target_extrapolation_px = 0.0

    terminal_direction = None

    if reached_target_directly:

        target_intersection = {
            "point": (
                float(path_x[-1]),
                float(path_y[-1]),
            ),

            "ray_distance_px":
                0.0,

            "intersection_type":
                "observed",

            "apo_extension_distance_px":
                0.0,

            "extension_side":
                None,
        }

    elif len(directions) > 0:

        terminal_dirs = np.asarray(
            directions[
                -terminal_window:
            ],
            dtype=float,
        )

        terminal_v = (
            terminal_dirs.mean(
                axis=0
            )
        )

        terminal_norm = np.hypot(
            terminal_v[0],
            terminal_v[1]
        )

        if terminal_norm > 1e-8:

            terminal_v /= terminal_norm

            terminal_direction = (
                float(terminal_v[0]),
                float(terminal_v[1]),
            )

            target_intersection = (
                intersect_ray_with_apo(
                    p0=(
                        float(path_x[-1]),
                        float(path_y[-1]),
                    ),
                    v=terminal_direction,
                    apo=target_apo,
                    apo_tangent_window=
                        apo_tangent_window,
                )
            )

            if target_intersection is not None:

                target_extrapolation_px = (
                    target_intersection[
                        "ray_distance_px"
                    ]
                )

    # --------------------------------------------------
    # 5. Total FL
    # --------------------------------------------------

    fl_px = (
        start_correction_px
        + observed_fl_px
        + target_extrapolation_px
    )

    # Start correction is supported geometry:
    # it exists only because we intentionally buffered
    # away from the starting apo.

    if fl_px > 0:

        supported_px = (
            start_correction_px
            + observed_fl_px
        )

        supported_fraction = (
            supported_px
            / fl_px
        )

    else:

        supported_fraction = 0.0

    return {
        "path_x": path_x,
        "path_y": path_y,

        "start_intersection":
            start_intersection,

        "target_intersection":
            target_intersection,

        "initial_direction":
            initial_direction,

        "terminal_direction":
            terminal_direction,

        "target_is_above":
            target_is_above,

        "reached_target_directly":
            reached_target_directly,

        "start_correction_px":
            float(
                start_correction_px
            ),

        "observed_fl_px":
            float(
                observed_fl_px
            ),

        "target_extrapolation_px":
            float(
                target_extrapolation_px
            ),

        "supported_fraction":
            float(
                supported_fraction
            ),

        "fl_px":
            float(
                fl_px
            ),
    }



def intersect_parametric_lines(
    p0,
    v,
    q0,
    w,
):
    """
    Solve:

        p0 + t*v = q0 + s*w

    where:
        p0 = fascicle endpoint
        v  = fascicle continuation direction
        q0 = apo edge point
        w  = apo tangent direction

    Returns
    -------
    dict or None
        Intersection point plus distances along
        the fascicle and apo extensions.
    """

    A = np.array(
        [
            [v[0], -w[0]],
            [v[1], -w[1]],
        ],
        dtype=float
    )

    b = np.array(
        [
            q0[0] - p0[0],
            q0[1] - p0[1],
        ],
        dtype=float
    )

    det = np.linalg.det(
        A
    )

    if abs(det) < 1e-8:
        return None

    t, s = np.linalg.solve(
        A,
        b
    )

    point = (
        p0[0] + t * v[0],
        p0[1] + t * v[1],
    )

    return {
        "point": point,
        "fascicle_distance_px": float(t),
        "apo_extension_distance_px": float(s),
    }



def intersect_ray_with_apo(
    p0,
    v,
    apo,
    max_distance=1000.0,
    step_size=1.0,
    apo_tangent_window=10,
):
    """
    Intersect a forward ray with an aponeurosis.

    First attempts intersection with the observed apo curve.
    If that fails, locally extends the appropriate apo edge
    using an averaged edge tangent.

    Parameters
    ----------
    p0 : tuple
        Ray starting point (x, y).

    v : tuple
        Unit ray direction (vx, vy).

    apo : dict
        Apo candidate containing:
        center_x, center_y, tx, ty.

    Returns
    -------
    dict or None
        point
        ray_distance_px
        intersection_type  -> "observed" or "extended"
        apo_extension_distance_px
        extension_side
    """

    apo_x = np.asarray(
        apo["center_x"],
        dtype=float,
    )

    apo_y = np.asarray(
        apo["center_y"],
        dtype=float,
    )

    apo_tx = np.asarray(
        apo["tx"],
        dtype=float,
    )

    apo_ty = np.asarray(
        apo["ty"],
        dtype=float,
    )

    vx, vy = v

    norm = np.hypot(
        vx,
        vy
    )

    if norm < 1e-8:
        return None

    vx /= norm
    vy /= norm

    # --------------------------------------------------
    # 1. Try observed apo first
    # --------------------------------------------------

    previous_t = None
    previous_value = None

    for t in np.arange(
        0.0,
        max_distance + step_size,
        step_size,
    ):

        x_ray = (
            p0[0]
            + t * vx
        )

        y_ray = (
            p0[1]
            + t * vy
        )

        if (
            apo_x.min()
            <= x_ray
            <= apo_x.max()
        ):

            y_apo = np.interp(
                x_ray,
                apo_x,
                apo_y,
            )

            value = (
                y_ray
                - y_apo
            )

            if (
                previous_value is not None
                and np.sign(value)
                != np.sign(previous_value)
            ):

                alpha = (
                    abs(previous_value)
                    / (
                        abs(previous_value)
                        + abs(value)
                    )
                )

                t_intersection = (
                    previous_t
                    + alpha
                    * (
                        t
                        - previous_t
                    )
                )

                x_int = (
                    p0[0]
                    + t_intersection * vx
                )

                y_int = (
                    p0[1]
                    + t_intersection * vy
                )

                return {
                    "point": (
                        float(x_int),
                        float(y_int),
                    ),

                    "ray_distance_px":
                        float(
                            t_intersection
                        ),

                    "intersection_type":
                        "observed",

                    "apo_extension_distance_px":
                        0.0,

                    "extension_side":
                        None,
                }

            previous_t = t
            previous_value = value

        else:

            previous_t = None
            previous_value = None

    # --------------------------------------------------
    # 2. No observed intersection:
    #    extend relevant apo edge
    # --------------------------------------------------

    if vx >= 0:

        extension_side = "right"

        apo_edge_point = (
            float(apo_x[-1]),
            float(apo_y[-1]),
        )

        tangent_x = (
            apo_tx[
                -apo_tangent_window:
            ]
        )

        tangent_y = (
            apo_ty[
                -apo_tangent_window:
            ]
        )

    else:

        extension_side = "left"

        apo_edge_point = (
            float(apo_x[0]),
            float(apo_y[0]),
        )

        tangent_x = (
            apo_tx[
                :apo_tangent_window
            ]
        )

        tangent_y = (
            apo_ty[
                :apo_tangent_window
            ]
        )

    mean_tx = np.mean(
        tangent_x
    )

    mean_ty = np.mean(
        tangent_y
    )

    tangent_norm = np.hypot(
        mean_tx,
        mean_ty
    )

    if tangent_norm < 1e-8:
        return None

    mean_tx /= tangent_norm
    mean_ty /= tangent_norm

    intersection = (
        intersect_parametric_lines(
            p0=p0,
            v=(vx, vy),
            q0=apo_edge_point,
            w=(mean_tx, mean_ty),
        )
    )

    if intersection is None:
        return None

    ray_distance = (
        intersection[
            "fascicle_distance_px"
        ]
    )

    apo_distance = (
        intersection[
            "apo_extension_distance_px"
        ]
    )

    # Must lie forward along fascicle ray.
    if ray_distance <= 0:
        return None

    # Must lie on correct apo extension.
    if extension_side == "right":

        if apo_distance < 0:
            return None

    else:

        if apo_distance > 0:
            return None

    return {
        "point":
            intersection["point"],

        "ray_distance_px":
            float(
                ray_distance
            ),

        "intersection_type":
            "extended",

        "apo_extension_distance_px":
            float(
                abs(apo_distance)
            ),

        "extension_side":
            extension_side,
    }




def trace_fascicle_from_seed_to_apo(
    seed_x,
    seed_y,
    initial_direction,
    fascicle_tx,
    fascicle_ty,
    target_apo,
    buffered_muscle_mask,
    step_size=1.0,
    max_steps=1000,
    terminal_window=10,
    apo_tangent_window=10,
):
    """
    Trace from an arbitrary interior seed toward one target apo.

    Uses the learned fascicle field while inside the trusted muscle
    region. If the trace leaves the trusted region before reaching
    the target apo, extrapolates using the averaged terminal direction.

    Returns
    -------
    dict or None
    """

    H, W = buffered_muscle_mask.shape

    target_x = np.asarray(
        target_apo["center_x"],
        dtype=float,
    )

    target_y = np.asarray(
        target_apo["center_y"],
        dtype=float,
    )

    # Normalize requested initial direction
    initial_direction = np.asarray(
        initial_direction,
        dtype=float,
    )

    initial_norm = np.linalg.norm(
        initial_direction
    )

    if initial_norm < 1e-8:
        return None

    initial_direction /= initial_norm

    x = float(seed_x)
    y = float(seed_y)

    path_x = [x]
    path_y = [y]

    directions = []

    previous_direction = (
        float(initial_direction[0]),
        float(initial_direction[1]),
    )

    reached_target_directly = False

    # Determine whether target apo is above or below seed.
    target_y_at_seed = None

    if (
        target_x.min()
        <= seed_x
        <= target_x.max()
    ):
        target_y_at_seed = np.interp(
            seed_x,
            target_x,
            target_y,
        )

    if target_y_at_seed is not None:

        target_is_above = (
            target_y_at_seed < seed_y
        )

    else:

        target_is_above = (
            np.median(target_y) < seed_y
        )

    # --------------------------------------------------
    # Field-driven trace
    # --------------------------------------------------

    for _ in range(max_steps):

        sampled = bilinear_sample_field(
            fascicle_tx,
            fascicle_ty,
            x,
            y,
        )

        if sampled is None:
            break

        vx, vy = sampled

        # Axial orientation:
        # choose sign agreeing with current direction.
        dot = (
            vx * previous_direction[0]
            + vy * previous_direction[1]
        )

        if dot < 0:
            vx = -vx
            vy = -vy

        x_new = (
            x
            + step_size * vx
        )

        y_new = (
            y
            + step_size * vy
        )

        if not (
            0 <= x_new < W
            and 0 <= y_new < H
        ):
            break

        # --------------------------------------------------
        # Check direct target-apo intersection
        # --------------------------------------------------

        if (
            target_x.min()
            <= x_new
            <= target_x.max()
        ):

            target_y_here = np.interp(
                x_new,
                target_x,
                target_y,
            )

            if target_is_above:

                crossed_target = (
                    y_new <= target_y_here
                )

            else:

                crossed_target = (
                    y_new >= target_y_here
                )

            if crossed_target:

                path_x.append(
                    x_new
                )

                path_y.append(
                    target_y_here
                )

                directions.append(
                    (vx, vy)
                )

                reached_target_directly = True
                break

        # --------------------------------------------------
        # Stop once we leave trusted field
        # --------------------------------------------------

        ix = int(
            round(x_new)
        )

        iy = int(
            round(y_new)
        )

        if not (
            0 <= ix < W
            and 0 <= iy < H
        ):
            break

        if buffered_muscle_mask[
            iy,
            ix
        ] == 0:
            break

        path_x.append(
            x_new
        )

        path_y.append(
            y_new
        )

        directions.append(
            (vx, vy)
        )

        x = x_new
        y = y_new

        previous_direction = (
            vx,
            vy
        )

    path_x = np.asarray(
        path_x,
        dtype=float,
    )

    path_y = np.asarray(
        path_y,
        dtype=float,
    )

    # --------------------------------------------------
    # Observed path length
    # --------------------------------------------------

    if len(path_x) > 1:

        observed_length_px = np.sum(
            np.hypot(
                np.diff(path_x),
                np.diff(path_y),
            )
        )

    else:

        observed_length_px = 0.0

    target_intersection = None
    extrapolated_length_px = 0.0
    terminal_direction = None

    # --------------------------------------------------
    # Direct target hit
    # --------------------------------------------------

    if reached_target_directly:

        target_intersection = {
            "point": (
                float(path_x[-1]),
                float(path_y[-1]),
            ),
            "ray_distance_px": 0.0,
            "intersection_type": "observed",
            "apo_extension_distance_px": 0.0,
            "extension_side": None,
        }

    # --------------------------------------------------
    # Otherwise extrapolate from terminal direction
    # --------------------------------------------------

    elif len(directions) > 0:

        terminal_dirs = np.asarray(
            directions[-terminal_window:],
            dtype=float,
        )

        terminal_v = terminal_dirs.mean(
            axis=0
        )

        terminal_norm = np.linalg.norm(
            terminal_v
        )

        if terminal_norm > 1e-8:

            terminal_v /= terminal_norm

            terminal_direction = (
                float(terminal_v[0]),
                float(terminal_v[1]),
            )

            target_intersection = (
                intersect_ray_with_apo(
                    p0=(
                        float(path_x[-1]),
                        float(path_y[-1]),
                    ),
                    v=terminal_direction,
                    apo=target_apo,
                    apo_tangent_window=
                        apo_tangent_window,
                )
            )

            if target_intersection is not None:

                extrapolated_length_px = (
                    target_intersection[
                        "ray_distance_px"
                    ]
                )

    total_length_px = (
        observed_length_px
        + extrapolated_length_px
    )

    if total_length_px > 0:

        observed_fraction = (
            observed_length_px
            / total_length_px
        )

    else:

        observed_fraction = 0.0

    return {
        "path_x": path_x,
        "path_y": path_y,

        "target_intersection":
            target_intersection,

        "terminal_direction":
            terminal_direction,

        "reached_target_directly":
            reached_target_directly,

        "observed_length_px":
            float(
                observed_length_px
            ),

        "extrapolated_length_px":
            float(
                extrapolated_length_px
            ),

        "observed_fraction":
            float(
                observed_fraction
            ),

        "length_px":
            float(
                total_length_px
            ),
    }


def trace_fascicle_from_interior_seed(
    seed_x,
    seed_y,
    fascicle_tx,
    fascicle_ty,
    upper_apo,
    lower_apo,
    buffered_muscle_mask,
    step_size=1.0,
    max_steps=1000,
    terminal_window=10,
    apo_tangent_window=10,
):
    """
    Trace a complete fascicle from one interior seed.

    One branch travels toward the upper apo.
    The opposite branch travels toward the lower apo.

    Returns
    -------
    dict or None
    """

    H, W = buffered_muscle_mask.shape

    ix = int(
        round(seed_x)
    )

    iy = int(
        round(seed_y)
    )

    if not (
        0 <= ix < W
        and 0 <= iy < H
    ):
        return None

    if buffered_muscle_mask[
        iy,
        ix
    ] == 0:
        return None

    # --------------------------------------------------
    # Local axial orientation at seed
    # --------------------------------------------------

    sampled = bilinear_sample_field(
        fascicle_tx,
        fascicle_ty,
        seed_x,
        seed_y,
    )

    if sampled is None:
        return None

    vx, vy = sampled

    # Image coordinates:
    # negative y = upward
    # positive y = downward

    if vy < 0:

        upper_direction = (
            vx,
            vy,
        )

        lower_direction = (
            -vx,
            -vy,
        )

    else:

        upper_direction = (
            -vx,
            -vy,
        )

        lower_direction = (
            vx,
            vy,
        )

    # --------------------------------------------------
    # Seed -> upper apo
    # --------------------------------------------------

    upper_trace = (
        trace_fascicle_from_seed_to_apo(
            seed_x=seed_x,
            seed_y=seed_y,
            initial_direction=
                upper_direction,
            fascicle_tx=fascicle_tx,
            fascicle_ty=fascicle_ty,
            target_apo=upper_apo,
            buffered_muscle_mask=
                buffered_muscle_mask,
            step_size=step_size,
            max_steps=max_steps,
            terminal_window=
                terminal_window,
            apo_tangent_window=
                apo_tangent_window,
        )
    )

    # --------------------------------------------------
    # Seed -> lower apo
    # --------------------------------------------------

    lower_trace = (
        trace_fascicle_from_seed_to_apo(
            seed_x=seed_x,
            seed_y=seed_y,
            initial_direction=
                lower_direction,
            fascicle_tx=fascicle_tx,
            fascicle_ty=fascicle_ty,
            target_apo=lower_apo,
            buffered_muscle_mask=
                buffered_muscle_mask,
            step_size=step_size,
            max_steps=max_steps,
            terminal_window=
                terminal_window,
            apo_tangent_window=
                apo_tangent_window,
        )
    )

    if (
        upper_trace is None
        or lower_trace is None
    ):
        return None

    if (
        upper_trace[
            "target_intersection"
        ] is None
        or lower_trace[
            "target_intersection"
        ] is None
    ):
        return None

    # --------------------------------------------------
    # Complete fascicle length
    # --------------------------------------------------

    fl_px = (
        upper_trace["length_px"]
        + lower_trace["length_px"]
    )

    observed_px = (
        upper_trace[
            "observed_length_px"
        ]
        + lower_trace[
            "observed_length_px"
        ]
    )

    extrapolated_px = (
        upper_trace[
            "extrapolated_length_px"
        ]
        + lower_trace[
            "extrapolated_length_px"
        ]
    )

    if fl_px > 0:

        observed_fraction = (
            observed_px
            / fl_px
        )

    else:

        observed_fraction = 0.0

    return {
        "seed_x": float(seed_x),
        "seed_y": float(seed_y),

        "upper_trace":
            upper_trace,

        "lower_trace":
            lower_trace,

        "upper_length_px":
            float(
                upper_trace["length_px"]
            ),

        "lower_length_px":
            float(
                lower_trace["length_px"]
            ),

        "observed_length_px":
            float(
                observed_px
            ),

        "extrapolated_length_px":
            float(
                extrapolated_px
            ),

        "observed_fraction":
            float(
                observed_fraction
            ),

        "fl_px":
            float(
                fl_px
            ),
    }



def estimate_local_straight_fascicle_direction(
    seed_x,
    seed_y,
    fascicle_orientation,
    buffered_muscle_mask,
    neighborhood_radius=8,
):
    """
    Estimate one representative straight fascicle direction
    around an interior seed by averaging the doubled-angle
    orientation field locally.

    Returns
    -------
    dict or None
    """

    H, W = buffered_muscle_mask.shape

    cx = int(round(seed_x))
    cy = int(round(seed_y))

    x_min = max(
        0,
        cx - neighborhood_radius
    )

    x_max = min(
        W,
        cx + neighborhood_radius + 1
    )

    y_min = max(
        0,
        cy - neighborhood_radius
    )

    y_max = min(
        H,
        cy + neighborhood_radius + 1
    )

    local_mask = (
        buffered_muscle_mask[
            y_min:y_max,
            x_min:x_max
        ]
        > 0
    )

    if not np.any(local_mask):
        return None

    local_v1 = fascicle_orientation[
        0,
        y_min:y_max,
        x_min:x_max
    ][local_mask]

    local_v2 = fascicle_orientation[
        1,
        y_min:y_max,
        x_min:x_max
    ][local_mask]

    mean_v1 = np.mean(
        local_v1
    )

    mean_v2 = np.mean(
        local_v2
    )

    coherence = np.hypot(
        mean_v1,
        mean_v2
    )

    if coherence < 1e-8:
        return None

    mean_v1 /= coherence
    mean_v2 /= coherence

    theta = (
        0.5
        * np.arctan2(
            mean_v2,
            mean_v1
        )
    )

    tx = np.cos(
        theta
    )

    # Cartesian -> image coordinates
    ty = -np.sin(
        theta
    )

    return {
        "tx": float(tx),
        "ty": float(ty),
        "coherence": float(coherence),
    }




def estimate_straight_fl_from_seed(
    seed_x,
    seed_y,
    fascicle_orientation,
    upper_apo,
    lower_apo,
    buffered_muscle_mask,
    neighborhood_radius=8,
    apo_tangent_window=10,
):
    """
    Estimate straight-line fascicle length from one interior seed.

    A locally averaged fascicle direction is extended in both
    axial directions until it intersects the upper and lower apos.
    """

    direction = (
        estimate_local_straight_fascicle_direction(
            seed_x=seed_x,
            seed_y=seed_y,
            fascicle_orientation=
                fascicle_orientation,
            buffered_muscle_mask=
                buffered_muscle_mask,
            neighborhood_radius=
                neighborhood_radius,
        )
    )

    if direction is None:
        return None

    tx = direction["tx"]
    ty = direction["ty"]

    # Choose sign convention:
    # one direction upward, one downward.
    if ty < 0:

        upper_direction = (
            tx,
            ty,
        )

        lower_direction = (
            -tx,
            -ty,
        )

    else:

        upper_direction = (
            -tx,
            -ty,
        )

        lower_direction = (
            tx,
            ty,
        )

    upper_intersection = (
        intersect_ray_with_apo(
            p0=(
                seed_x,
                seed_y,
            ),
            v=upper_direction,
            apo=upper_apo,
            apo_tangent_window=
                apo_tangent_window,
        )
    )

    lower_intersection = (
        intersect_ray_with_apo(
            p0=(
                seed_x,
                seed_y,
            ),
            v=lower_direction,
            apo=lower_apo,
            apo_tangent_window=
                apo_tangent_window,
        )
    )

    if (
        upper_intersection is None
        or lower_intersection is None
    ):
        return None

    upper_length = (
        upper_intersection[
            "ray_distance_px"
        ]
    )

    lower_length = (
        lower_intersection[
            "ray_distance_px"
        ]
    )

    fl_px = (
        upper_length
        + lower_length
    )

    return {
        "seed_x": float(seed_x),
        "seed_y": float(seed_y),

        "tx": float(tx),
        "ty": float(ty),

        "local_coherence":
            direction["coherence"],

        "upper_intersection":
            upper_intersection,

        "lower_intersection":
            lower_intersection,

        "upper_length_px":
            float(upper_length),

        "lower_length_px":
            float(lower_length),

        "fl_px":
            float(fl_px),
    }