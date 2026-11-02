from __future__ import annotations

import argparse
import json
from pathlib import Path

from .psi import population_stability_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate local numeric feature drift with PSI.")
    parser.add_argument("reference", type=Path, help="JSON numeric array representing baseline data")
    parser.add_argument("current", type=Path, help="JSON numeric array representing current data")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("drift_report.json"))
    args = parser.parse_args()
    report = population_stability_index(json.loads(args.reference.read_text()), json.loads(args.current.read_text()), args.bins)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("psi", "status", "reference_size", "current_size")}))
