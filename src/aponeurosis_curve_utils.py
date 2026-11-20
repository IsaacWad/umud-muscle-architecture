from __future__ import annotations

import numpy as np

from scipy import ndimage
from scipy.interpolate import UnivariateSpline






def extract_component_centerline(
    labeled_mask,
    component_id
):
    """
    Extract one centerline point per x-column
    using the median y-value of component pixels.
    """

    ys, xs = np.where(
        labeled_mask == component_id
    )

    unique_x = np.unique(xs)

    centerline_x = []
    centerline_y = []

    for x in unique_x:

        y_at_x = ys[
            xs == x
        ]

        center_y = np.median(
            y_at_x
        )

        centerline_x.append(
            x
        )

        centerline_y.append(
            center_y
        )

    return (
        np.asarray(
            centerline_x,
            dtype=float
        ),
        np.asarray(
            centerline_y,
            dtype=float
        )
    )




def compute_local_curve_tangents(
    x,
    y_smooth,
    window_radius=15
):
    """
    Estimate a local tangent direction at each curve point
    by fitting a straight line to nearby smooth-curve points.

    Parameters
    ----------
    x:
        1D array of x-coordinates.
    y_smooth:
        1D array of smoothed curve y-coordinates.
    window_radius:
        Number of neighboring curve samples to use on each side.

    Returns
    -------
    tx:
        Local unit tangent x-component.
    ty:
        Local unit tangent y-component.
    fit_rmse:
        Local line-fit RMSE in pixels.
    confidence:
        Local tangent confidence based on fit RMSE.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y_smooth = np.asarray(
        y_smooth,
        dtype=float
    )

    n = len(x)

    tx = np.zeros(
        n,
        dtype=float
    )

    ty = np.zeros(
        n,
        dtype=float
    )

    fit_rmse = np.zeros(
        n,
        dtype=float
    )

    confidence = np.zeros(
        n,
        dtype=float
    )

    for i in range(n):

        lo = max(
            0,
            i - window_radius
        )

        hi = min(
            n,
            i + window_radius + 1
        )

        x_local = x[
            lo:hi
        ]

        y_local = y_smooth[
            lo:hi
        ]

        # ------------------------------------------
        # Fit y = m*x + b locally
        # ------------------------------------------

        m, b = np.polyfit(
            x_local,
            y_local,
            deg=1
        )

        y_hat = (
            m * x_local
            + b
        )

        residuals = (
            y_local
            - y_hat
        )

        rmse = np.sqrt(
            np.mean(
                residuals**2
            )
        )

        # ------------------------------------------
        # Unit tangent vector
        # ------------------------------------------

        norm = np.sqrt(
            1.0
            + m**2
        )

        tx[i] = (
            1.0
            / norm
        )

        ty[i] = (
            m
            / norm
        )

        fit_rmse[i] = rmse


    # ----------------------------------------------
    # Convert local fit quality to confidence
    # ----------------------------------------------

    # Start simple:
    # ~1 px error -> high confidence
    # larger local deviation -> lower confidence

    confidence = np.exp(
        -fit_rmse / 2.0
    )

    return (
        tx,
        ty,
        fit_rmse,
        confidence
    )




def build_apo_curve_candidates(
    mask_512,
    smoothing=100,
    min_area=20,
    min_width=25,
    min_unique_x=10,
    tangent_window_radius=15,
    confidence_scale=2.0
):
    """
    Convert connected apo mask components into smooth
    directional curve candidates.

    Each candidate stores:
        - centerline
        - smoothed curve
        - local-fit tangent vectors
        - local tangent RMSE
        - local tangent confidence

    No final upper/lower assignment is made here.
    """

    mask_binary = (
        mask_512 > 0
    ).astype(np.uint8)

    labeled_mask, n_components = ndimage.label(
        mask_binary
    )

    candidates = []

    for component_id in range(
        1,
        n_components + 1
    ):

        ys, xs = np.where(
            labeled_mask == component_id
        )

        if len(xs) == 0:
            continue


        # ------------------------------------------
        # Basic geometry
        # ------------------------------------------

        area_px = len(xs)

        width_px = (
            xs.max()
            - xs.min()
            + 1
        )

        height_px = (
            ys.max()
            - ys.min()
            + 1
        )


        # ------------------------------------------
        # Permissive filtering
        # ------------------------------------------

        if area_px < min_area:
            continue

        if width_px < min_width:
            continue


        # ------------------------------------------
        # Centerline:
        # median y for each observed x
        # ------------------------------------------

        unique_x = np.unique(
            xs
        )

        if len(unique_x) < min_unique_x:
            continue


        center_x = []
        center_y = []

        for x in unique_x:

            y_values = ys[
                xs == x
            ]

            center_x.append(
                float(x)
            )

            center_y.append(
                float(
                    np.median(
                        y_values
                    )
                )
            )


        center_x = np.asarray(
            center_x,
            dtype=float
        )

        center_y = np.asarray(
            center_y,
            dtype=float
        )


        # ------------------------------------------
        # Smooth curve
        # ------------------------------------------

        try:

            spline = UnivariateSpline(
                center_x,
                center_y,
                s=smoothing
            )

            smooth_y = spline(
                center_x
            )

        except Exception:

            continue


        # ------------------------------------------
        # Local-fit tangent vectors
        # ------------------------------------------

        (
            tangent_tx,
            tangent_ty,
            tangent_rmse,
            _
        ) = compute_local_curve_tangents(
            center_x,
            smooth_y,
            window_radius=tangent_window_radius
        )


        # ------------------------------------------
        # Tangent confidence
        # ------------------------------------------

        tangent_confidence = np.exp(
            -tangent_rmse
            / confidence_scale
        )


        # ------------------------------------------
        # Candidate summaries
        # ------------------------------------------

        mean_tangent_confidence = float(
            np.mean(
                tangent_confidence
            )
        )

        median_tangent_confidence = float(
            np.median(
                tangent_confidence
            )
        )

        p10_tangent_confidence = float(
            np.percentile(
                tangent_confidence,
                10
            )
        )

        mean_tangent_rmse = float(
            np.mean(
                tangent_rmse
            )
        )


        candidates.append({
            "component_id":
                component_id,

            "area_px":
                area_px,

            "width_px":
                width_px,

            "height_px":
                height_px,

            "centroid_x":
                float(
                    xs.mean()
                ),

            "centroid_y":
                float(
                    ys.mean()
                ),

            "x_min":
                int(
                    xs.min()
                ),

            "x_max":
                int(
                    xs.max()
                ),

            "center_x":
                center_x,

            "center_y":
                center_y,

            "smooth_y":
                smooth_y,

            "tx":
                tangent_tx,

            "ty":
                tangent_ty,

            "tangent_rmse":
                tangent_rmse,

            "tangent_confidence":
                tangent_confidence,

            "mean_tangent_confidence":
                mean_tangent_confidence,

            "median_tangent_confidence":
                median_tangent_confidence,

            "p10_tangent_confidence":
                p10_tangent_confidence,

            "mean_tangent_rmse":
                mean_tangent_rmse
        })


    return candidates




def build_apo_training_targets(
    candidates,
    scored_candidates,
    image_shape=(512, 512),
    sigma=2.0,
    radius=6,
    quality_weights=None
):
    """
    Build aponeurosis training targets from reconstructed curve candidates.

    Parameters
    ----------
    candidates:
        List of candidate dictionaries returned by
        build_apo_curve_candidates().

    scored_candidates:
        DataFrame containing rows for the same image with:
            candidate_rank
            quality_class

    image_shape:
        Output target shape.

    sigma:
        Gaussian spread around the centerline.

    radius:
        Maximum distance from the centerline used for
        target/supervision generation.

    quality_weights:
        Optional mapping from quality class to training weight.

        Default:
            good -> 1.0
            weak -> 0.5
            bad -> 0.0

    Returns
    -------
    centerline_target:
        Shape [H, W].
        Gaussian apo evidence target.

    orientation_map:
        Shape [2, H, W].
        Axial orientation represented as:
            cos(2 theta)
            sin(2 theta)

    supervision_mask:
        Shape [H, W].
        Indicates pixels where apo orientation supervision exists.

    confidence_map:
        Shape [H, W].
        Quality-weighted supervision strength.
    """

    if quality_weights is None:

        quality_weights = {
            "good": 1.0,
            "weak": 0.5,
            "bad": 0.0
        }


    h, w = image_shape


    # --------------------------------------------------
    # Initialize targets
    # --------------------------------------------------

    centerline_target = np.zeros(
        (h, w),
        dtype=np.float32
    )

    orientation_cos = np.zeros(
        (h, w),
        dtype=np.float32
    )

    orientation_sin = np.zeros(
        (h, w),
        dtype=np.float32
    )

    orientation_weight = np.zeros(
        (h, w),
        dtype=np.float32
    )

    supervision_mask = np.zeros(
        (h, w),
        dtype=np.float32
    )

    confidence_map = np.zeros(
        (h, w),
        dtype=np.float32
    )


    # --------------------------------------------------
    # Precompute local Gaussian neighborhood
    # --------------------------------------------------

    offsets = np.arange(
        -radius,
        radius + 1
    )

    yy_offset, xx_offset = np.meshgrid(
        offsets,
        offsets,
        indexing="ij"
    )

    dist_sq = (
        xx_offset**2
        + yy_offset**2
    )

    gaussian_kernel = np.exp(
        -dist_sq
        / (2.0 * sigma**2)
    )

    gaussian_kernel[
        dist_sq > radius**2
    ] = 0.0


    # --------------------------------------------------
    # Process each candidate
    # --------------------------------------------------

    for candidate_rank, candidate in enumerate(
        candidates,
        start=1
    ):

        # ----------------------------------------------
        # Find quality classification
        # ----------------------------------------------

        row = scored_candidates[
            scored_candidates["candidate_rank"]
            == candidate_rank
        ]

        if len(row) == 0:
            continue

        quality_class = row.iloc[0][
            "quality_class"
        ]

        training_weight = quality_weights.get(
            quality_class,
            0.0
        )


        # ----------------------------------------------
        # Ignore bad candidates
        # ----------------------------------------------

        if training_weight <= 0:
            continue


        # ----------------------------------------------
        # Curve geometry
        # ----------------------------------------------

        x = np.asarray(
            candidate["center_x"],
            dtype=float
        )

        y = np.asarray(
            candidate["smooth_y"],
            dtype=float
        )

        tx = np.asarray(
            candidate["tx"],
            dtype=float
        )

        ty = np.asarray(
            candidate["ty"],
            dtype=float
        )


        # ----------------------------------------------
        # Axial orientation
        #
        # theta and theta + pi represent the same
        # anatomical line orientation.
        # ----------------------------------------------

        theta = np.arctan2(
            ty,
            tx
        )

        cos2 = np.cos(
            2.0 * theta
        )

        sin2 = np.sin(
            2.0 * theta
        )


        # ----------------------------------------------
        # Rasterize each curve point
        # ----------------------------------------------

        for (
            x0,
            y0,
            c2,
            s2
        ) in zip(
            x,
            y,
            cos2,
            sin2
        ):

            cx = int(
                round(
                    x0
                )
            )

            cy = int(
                round(
                    y0
                )
            )

            if (
                cx < 0
                or cx >= w
                or cy < 0
                or cy >= h
            ):
                continue


            # ------------------------------------------
            # Kernel bounds in image coordinates
            # ------------------------------------------

            x_start = max(
                0,
                cx - radius
            )

            x_end = min(
                w,
                cx + radius + 1
            )

            y_start = max(
                0,
                cy - radius
            )

            y_end = min(
                h,
                cy + radius + 1
            )


            # ------------------------------------------
            # Corresponding Gaussian kernel bounds
            # ------------------------------------------

            kernel_x_start = (
                x_start
                - (cx - radius)
            )

            kernel_x_end = (
                kernel_x_start
                + (x_end - x_start)
            )

            kernel_y_start = (
                y_start
                - (cy - radius)
            )

            kernel_y_end = (
                kernel_y_start
                + (y_end - y_start)
            )


            local_gaussian = gaussian_kernel[
                kernel_y_start:
                kernel_y_end,

                kernel_x_start:
                kernel_x_end
            ]


            # ------------------------------------------
            # Centerline heatmap
            #
            # Keep the strongest Gaussian response if
            # multiple curve points overlap.
            # ------------------------------------------

            current_target = centerline_target[
                y_start:y_end,
                x_start:x_end
            ]

            centerline_target[
                y_start:y_end,
                x_start:x_end
            ] = np.maximum(
                current_target,
                local_gaussian
            )


            # ------------------------------------------
            # Orientation supervision
            #
            # Weight local contribution by:
            # Gaussian proximity × annotation quality
            # ------------------------------------------

            local_weight = (
                local_gaussian
                * training_weight
            )


            orientation_cos[
                y_start:y_end,
                x_start:x_end
            ] += (
                local_weight
                * c2
            )

            orientation_sin[
                y_start:y_end,
                x_start:x_end
            ] += (
                local_weight
                * s2
            )

            orientation_weight[
                y_start:y_end,
                x_start:x_end
            ] += local_weight


            # ------------------------------------------
            # Supervision / confidence maps
            # ------------------------------------------

            supervision_mask[
                y_start:y_end,
                x_start:x_end
            ] = np.maximum(
                supervision_mask[
                    y_start:y_end,
                    x_start:x_end
                ],
                (
                    local_gaussian > 0
                ).astype(
                    np.float32
                )
            )

            confidence_map[
                y_start:y_end,
                x_start:x_end
            ] = np.maximum(
                confidence_map[
                    y_start:y_end,
                    x_start:x_end
                ],
                local_weight
            )


    # --------------------------------------------------
    # Normalize orientation vectors
    # --------------------------------------------------

    valid = (
        orientation_weight > 0
    )

    orientation_cos[
        valid
    ] /= orientation_weight[
        valid
    ]

    orientation_sin[
        valid
    ] /= orientation_weight[
        valid
    ]


    # Normalize averaged doubled-angle vectors
    norm = np.sqrt(
        orientation_cos**2
        + orientation_sin**2
    )

    valid_norm = (
        norm > 1e-8
    )

    orientation_cos[
        valid_norm
    ] /= norm[
        valid_norm
    ]

    orientation_sin[
        valid_norm
    ] /= norm[
        valid_norm
    ]


    orientation_map = np.stack(
        [
            orientation_cos,
            orientation_sin
        ],
        axis=0
    ).astype(
        np.float32
    )


    return {
        "centerline_target":
            centerline_target,

        "orientation_map":
            orientation_map,

        "supervision_mask":
            supervision_mask,

        "confidence_map":
            confidence_map
    }