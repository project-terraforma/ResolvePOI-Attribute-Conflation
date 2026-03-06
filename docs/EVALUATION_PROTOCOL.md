# Evaluation Protocol

This protocol avoids overfitting to repeated ad hoc evaluation and keeps experiment comparisons fair.

## Baseline

Use `data/results/start_report.txt` as the frozen starting baseline.

## Reproducible Run Command

Run experiments through the helper script instead of manual one-off commands.

```bash
.venv/bin/python scripts/run_repro_eval.py --tag exp_baseline_recheck --mode full
```

Modes:
- `--mode full`: regenerates synthetic data, features, retrains ML, evaluates 200 real records. Skips 2k final inference for speed.
- `--mode eval-only`: evaluates existing model artifacts without retraining.

Optional attribute subset:

```bash
.venv/bin/python scripts/run_repro_eval.py --tag exp_name_v1 --mode full --attributes name
```

## Outputs

Each run writes three files under `data/results/experiment_reports/`:
- `<tag>_pipeline.log`: full pipeline command output.
- `<tag>_report.txt`: output of `scripts/analyze_results.py`.
- `<tag>_meta.json`: environment metadata (Python, package versions, git commit, command used).

## Keep/Reject Criteria

Keep a change only if all apply:
- Target attribute F1 improves versus baseline.
- No severe regression on non-target attributes.
- Coverage stays at or near 1.0.
- Inference speed remains acceptable.

## Notes

- 200-record metrics are the scored benchmark in this repo.
- 2k inference outputs are useful for qualitative checks, not direct accuracy scoring unless labels exist.
