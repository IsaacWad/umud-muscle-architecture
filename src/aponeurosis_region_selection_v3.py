"""Add mean joint fascicle coverage to the existing geometry selector."""

from __future__ import annotations

import numpy as np

from src.aponeurosis_region_selection_v2 import (
    select_region_from_pipeline_result as select_geometry_region,
)


def select_region_from_pipeline_result(
    geometry, scale_deg=2.0, boundary_weight=0.50,
    continuity_weight=0.40, clearance_weight=0.15, *,
    coverage_weight=0.25, kappa_max=100.0,
):
    """Rank eligible pairs by geometry score plus joint coverage.

    Coverage is the mean of support times kappa / kappa_max inside each
    buffered mask. A zero weight or absent joint map uses geometry alone.
    """
    if not np.isfinite(coverage_weight) or coverage_weight < 0:
        raise ValueError("coverage_weight must be finite and nonnegative")

    result = select_geometry_region(
        geometry, scale_deg, boundary_weight, continuity_weight, clearance_weight,
    )
    weights = geometry.get("field_weights")
    effective_weight = coverage_weight if weights is not None else 0.0
    result["settings"].update(
        coverage_weight=coverage_weight,
        effective_coverage_weight=effective_weight,
    )
    table = result["candidate_table"]
    if table.empty:
        return result

    table["base_region_score"] = table["region_score"]
    table["joint_coverage"] = np.nan
    table["coverage_contribution"] = 0.0
    if effective_weight == 0:
        if result["selected"] is not None:
            result["selected"].update(
                base_region_score=result["selected"]["region_score"],
                joint_coverage=np.nan,
                coverage_contribution=0.0,
            )
        return result

    if not np.isfinite(kappa_max) or kappa_max <= 0:
        raise ValueError("kappa_max must be finite and positive")
    joint = np.asarray(weights, dtype=float) / kappa_max
    if (
        joint.shape != np.shape(geometry["image_512"])
        or not np.isfinite(joint).all()
        or np.any((joint < 0) | (joint > 1 + 1e-6))
    ):
        raise ValueError("Expected a finite model-space joint map in [0, 1]")

    regions = {region["pair_name"]: region for region in geometry["candidate_regions"]}
    coverage = {}
    for pair, region in regions.items():
        mask = np.asarray(region["buffered_mask"], dtype=bool)
        coverage[pair] = float(joint[mask].mean()) if mask.any() else np.nan

    table["joint_coverage"] = table["pair_name"].map(coverage)
    eligible = table["evidence_available"]
    if not np.isfinite(table.loc[eligible, "joint_coverage"]).all():
        raise ValueError("Eligible candidate has no buffered pixels")
    table["coverage_contribution"] = effective_weight * table["joint_coverage"]
    table["region_score"] = table["base_region_score"] + table["coverage_contribution"]
    ranked = table.loc[eligible].sort_values(
        "region_score", ascending=False, kind="stable",
    )
    if ranked.empty:
        return result

    selected = ranked.iloc[0].to_dict()
    selected.update(
        n_candidate_regions=len(table),
        n_eligible_regions=len(ranked),
        score_margin=(
            float(ranked.iloc[0]["region_score"] - ranked.iloc[1]["region_score"])
            if len(ranked) > 1 else np.nan
        ),
        selection_reason=(
            "highest_score_with_joint_coverage" if len(ranked) > 1 else "only_eligible_region"
        ),
        selection_fallback=False,
    )
    result.update(selected=selected, selected_region=regions[selected["pair_name"]])
    return result
