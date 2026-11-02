"""Population Stability Index without third-party dependencies."""

from __future__ import annotations

import math
from typing import Any


EPSILON = 1e-6


def _validate(values: list[float], name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} needs at least two numeric observations")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError(f"{name} contains a non-finite value")


def population_stability_index(reference: list[float], current: list[float], bins: int = 10) -> dict[str, Any]:
    """Compare numeric distributions using equal-width bins derived from reference.

    Values outside reference's observed range are included in the nearest edge
    bin; this makes a shift visible rather than dropping the observation.
    """
    _validate(reference, "reference")
    _validate(current, "current")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    low, high = min(reference), max(reference)
    if low == high:
        raise ValueError("reference must contain more than one distinct value")
    width = (high - low) / bins
    ref_counts, cur_counts = [0] * bins, [0] * bins
    def bucket(value: float) -> int:
        return min(bins - 1, max(0, int((value - low) / width)))
    for value in reference:
        ref_counts[bucket(value)] += 1
    for value in current:
        cur_counts[bucket(value)] += 1
    components = []
    for index, (ref_count, cur_count) in enumerate(zip(ref_counts, cur_counts)):
        ref_share = max(ref_count / len(reference), EPSILON)
        cur_share = max(cur_count / len(current), EPSILON)
        component = (cur_share - ref_share) * math.log(cur_share / ref_share)
        components.append({
            "lower": low + index * width,
            "upper": low + (index + 1) * width,
            "reference_share": round(ref_share, 6),
            "current_share": round(cur_share, 6),
            "psi_component": round(component, 6),
        })
    psi = sum(component["psi_component"] for component in components)
    status = "review" if psi >= 0.25 else "watch" if psi >= 0.1 else "stable"
    return {"psi": round(psi, 6), "status": status, "bins": bins, "reference_size": len(reference), "current_size": len(current), "components": components}
