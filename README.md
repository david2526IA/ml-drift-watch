# ML Drift Watch

An explainable, dependency-free Population Stability Index (PSI) checker for numeric ML features. Compare a baseline distribution with current data before retraining, deployment review or investigation.

## Run

```bash
pip install .
ml-drift-watch baseline.json current.json --bins 10 --output drift_report.json
```

Both input files are JSON arrays of finite numbers. The report shows PSI, a transparent per-bin breakdown and a conservative status:

- `stable`: PSI under 0.10
- `watch`: PSI from 0.10 to under 0.25
- `review`: PSI 0.25 or above

## What this is—and is not

It is a local feature-distribution signal. It does not prove model degradation, diagnose the cause of a shift, evaluate fairness or replace production monitoring. Thresholds should be calibrated to the model, feature and operational context.

## Development

```bash
pip install ".[dev]"
pytest -q
```

## License

MIT
