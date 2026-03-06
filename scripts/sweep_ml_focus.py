"""
Targeted ML sweep for high-value attributes.

Sweeps synthetic labeling and feature metadata options, evaluates on 200 real records,
and writes a ranked report plus best config per attribute.
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

REAL_GOLDEN_PATH = "data/golden_dataset_200.json"


def run_step(cmd: list[str]) -> bool:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Command failed:", " ".join(cmd))
        print(proc.stdout)
        print(proc.stderr)
        return False
    return True


def run_single(attribute: str, label_mode: str, include_metadata: str, use_record_confidence: str, seed: int) -> float | None:
    py = sys.executable

    if not run_step([
        py,
        "scripts/generate_synthetic_dataset.py",
        "--limit",
        "2000",
        "--seed",
        str(seed),
        "--label-mode",
        label_mode,
    ]):
        return None

    if not run_step([
        py,
        "-m",
        "scripts.process_synthetic_data",
        "--attribute",
        attribute,
        "--include-metadata",
        include_metadata,
        "--use-record-confidence",
        use_record_confidence,
    ]):
        return None

    model_dir = Path(f"models/ml/{attribute}")
    if not run_step([
        py,
        "scripts/train_models.py",
        "--features",
        f"data/processed/features_{attribute}_synthetic.parquet",
        "--output-dir",
        str(model_dir),
    ]):
        return None

    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_name = summary["best_model"]
    model_path = model_dir / f"best_model_{model_name}.joblib"

    pred_path = Path(f"data/results/ml_predictions_200_real_{attribute}.json")
    if not run_step([
        py,
        "-m",
        "scripts.run_inference",
        "--attribute",
        attribute,
        "--data",
        REAL_GOLDEN_PATH,
        "--model",
        str(model_path),
        "--output",
        str(pred_path),
    ]):
        return None

    eval_path = Path(f"data/results/ml_evaluation_200_real_{attribute}.json")
    if not run_step([
        py,
        "scripts/evaluate_models.py",
        "--predictions",
        str(pred_path),
        "--golden",
        REAL_GOLDEN_PATH,
        "--attribute",
        attribute,
        "--algorithm-name",
        f"ML Model ({attribute})",
        "--output",
        str(eval_path),
    ]):
        return None

    eval_json = json.loads(eval_path.read_text(encoding="utf-8"))
    return float(eval_json["metrics"]["f1"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep ML configs for selected attributes")
    parser.add_argument("--attributes", nargs="*", default=["name", "address"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/results/experiment_reports/exp_step6_ml_focus_sweep.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {a: [] for a in args.attributes}
    best = {}

    for attribute in args.attributes:
        print(f"\n=== Sweeping {attribute} ===")
        for label_mode, include_metadata, use_record_confidence in itertools.product(
            ["quality", "confidence"],
            ["true", "false"],
            ["true", "false"],
        ):
            cfg = {
                "label_mode": label_mode,
                "include_metadata": include_metadata,
                "use_record_confidence": use_record_confidence,
            }
            print(f"Trying: {cfg}")
            f1 = run_single(
                attribute=attribute,
                label_mode=label_mode,
                include_metadata=include_metadata,
                use_record_confidence=use_record_confidence,
                seed=args.seed,
            )
            if f1 is None:
                continue
            row = {"config": cfg, "f1": f1}
            results[attribute].append(row)

        results[attribute].sort(key=lambda x: x["f1"], reverse=True)
        best[attribute] = results[attribute][0] if results[attribute] else None
        print(f"Best {attribute}: {best[attribute]}")

    report = {
        "attributes": args.attributes,
        "seed": args.seed,
        "best": best,
        "results": results,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved sweep report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
