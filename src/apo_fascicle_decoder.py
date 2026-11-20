# src/apo_fascicle_decoder.py

from __future__ import annotations

import numpy as np
import torch


# ============================================================
# 1. APONEUROSIS DECODER
# ============================================================

def decode_apo_output(
    apo_output,
):
    """
    Convert raw aponeurosis model output into a 2D
    centerline probability map.

    Parameters
    ----------
    apo_output : dict
        Raw output from the aponeurosis model.

        Expected key:
            "centerline_logits"

        Expected tensor shape:
            (1, 1, H, W)

    Returns
    -------
    dict
        centerline_prob : np.ndarray
            2D aponeurosis centerline probability map
            with shape (H, W).
    """

    centerline_logits = (
        apo_output[
            "centerline_logits"
        ]
    )

    centerline_prob = (
        torch.sigmoid(
            centerline_logits
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if centerline_prob.ndim != 2:
        raise ValueError(
            "Decoded aponeurosis probability map "
            "must be 2D."
        )

    return {
        "centerline_prob":
            centerline_prob,
    }


# ============================================================
# 2. FASCICLE ORIENTATION DECODER
# ============================================================

def decode_fascicle_output(
    fascicle_output,
):
    """
    Convert raw fascicle model output into a normalized
    doubled-angle field and image-coordinate direction field.

    Parameters
    ----------
    fascicle_output : torch.Tensor
        Raw fascicle model output.

        Expected shape:
            (1, 2, H, W)

    Returns
    -------
    dict
        fascicle_raw_v : np.ndarray
            Raw two-channel model prediction.

        fascicle_raw_magnitude : np.ndarray
            Magnitude of the raw two-channel prediction.

        fascicle_v : np.ndarray
            Normalized doubled-angle representation.

        fascicle_tx : np.ndarray
            Horizontal fascicle tangent component.

        fascicle_ty : np.ndarray
            Vertical fascicle tangent component in image
            coordinates, where positive y points downward.
    """

    fascicle_raw_v = (
        fascicle_output
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if (
        fascicle_raw_v.ndim != 3
        or
        fascicle_raw_v.shape[0] != 2
    ):
        raise ValueError(
            "Fascicle model output must decode to "
            "shape (2, H, W)."
        )

    # --------------------------------------------------------
    # Magnitude of raw doubled-angle prediction
    # --------------------------------------------------------

    fascicle_raw_magnitude = np.sqrt(
        fascicle_raw_v[0] ** 2
        +
        fascicle_raw_v[1] ** 2
    )

    safe_magnitude = np.maximum(
        fascicle_raw_magnitude,
        1e-8,
    )

    # --------------------------------------------------------
    # Normalize doubled-angle representation
    # --------------------------------------------------------

    fascicle_v = (
        fascicle_raw_v
        /
        safe_magnitude[None, ...]
    )

    # --------------------------------------------------------
    # Decode axial orientation
    #
    # Network representation:
    #
    #     [cos(2 theta), sin(2 theta)]
    #
    # therefore:
    #
    #     theta = 0.5 * atan2(v2, v1)
    # --------------------------------------------------------

    theta = (
        0.5
        *
        np.arctan2(
            fascicle_v[1],
            fascicle_v[0],
        )
    )

    # --------------------------------------------------------
    # Cartesian -> image coordinates
    #
    # Cartesian:
    #     +y = upward
    #
    # Image:
    #     +y = downward
    #
    # Therefore the vertical component must be reflected.
    # --------------------------------------------------------

    fascicle_tx = np.cos(
        theta
    ).astype(
        np.float32
    )

    fascicle_ty = (
        -np.sin(
            theta
        )
    ).astype(
        np.float32
    )

    return {
        "fascicle_raw_v":
            fascicle_raw_v,

        "fascicle_raw_magnitude":
            fascicle_raw_magnitude,

        "fascicle_v":
            fascicle_v.astype(
                np.float32
            ),

        "fascicle_tx":
            fascicle_tx,

        "fascicle_ty":
            fascicle_ty,
    }