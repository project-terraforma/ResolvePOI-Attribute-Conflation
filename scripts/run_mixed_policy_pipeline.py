"""
Run a mixed-policy ML pipeline where each attribute uses its own synthetic policy.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ALL_ATTRIBUTES = ["name", "phone", "website", "address", "category"]
REAL_GOLDEN_PATH = "data/golden_dataset_200.json"

# Mixed policy v1:
# - name/address: Step 1.1 style (quality labels, no metadata)
# - phone/website/category: Step 2-ish style (confidence labels, metadata on, fixed confidence)
MIXED_POLICY_V1 = {
    "name": {
        "label_mode": "quality",
        "include_metadata": "false",
        "use_record_confidence": "true",
    },
    "address": {
        "label_mode": "quality",
        "include_metadata": "false",
        "use_record_confidence": "true",
    },
    "phone": {
        "label_mode": "confidence",
        "include_metadata": "true",
        "use_record_confidence": "false",
    },
    "website": {
        "label_mode": "confidence",
        "include_metadata": "true",
        "use_record_confidence": "false",
    },
    "category": {
        "label_mode": "confidence",
        "include_metadata": "true",
        "use_record_confidence": "false",
    },
}


def run_step(name: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("=" * 80)
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr)
    if proc.returncode != 0:
        print(f"ERROR: {name} failed with exit code {proc.returncode}")
        return False
    return True


def run_attribute(attribute: str, synthetic_limit: int, seed: int) -> bool:
    policy = MIXED_POLICY_V1[attribute]

    print("\n" + "#" * 80)
    print(f"PROCESSING ATTRIBUTE: {attribute.upper()}")
    print(f"POLICY: {policy}")
    print("#" * 80)

    if not run_step(
        f"Generate synthetic ({attribute})",
        [
            sys.executable,
            "scripts/generate_synthetic_dataset.py",
            "--limit",
            str(synthetic_limit),
            "--seed",
            str(seed),
            "--label-mode",
            policy["label_mode"],
        ],
    ):
        return False

    if not run_step(
        f"Process synthetic features ({attribute})",
        [
            sys.executable,
            "-m",
            "scripts.process_synthetic_data",
            "--attribute",
            attribute,
            "--include-metadata",
            policy["include_metadata"],
            "--use-record-confidence",
            policy["use_record_confidence"],
        ],
    ):
        return False

    model_dir = Path(f"models/ml/{attribute}")
    if not run_step(
        f"Train models ({attribute})",
        [
            sys.executable,
            "scripts/train_models.py",
            "--features",
            f"data/processed/features_{attribute}_synthetic.parquet",
            "--output-dir",
            str(model_dir),
        ],
    ):
        return False

    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        print(f"ERROR: Missing training summary at {summary_path}")
        return False

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    best_model_name = summary["best_model"]
    model_path = model_dir / f"best_model_{best_model_name}.joblib"

    if not run_step(
        f"Inference on 200 real ({attribute})",
        [
            sys.executable,
            "-m",
            "scripts.run_inference",
            "--attribute",
            attribute,
            "--data",
            REAL_GOLDEN_PATH,
            "--model",
            str(model_path),
            "--output",
            f"data/results/ml_predictions_200_real_{attribute}.json",
        ],
    ):
        return False

    if not run_step(
        f"Evaluate ML on 200 real ({attribute})",
        [
            sys.executable,
            "scripts/evaluate_models.py",
            "--predictions",
            f"data/results/ml_predictions_200_real_{attribute}.json",
            "--golden",
            REAL_GOLDEN_PATH,
            "--attribute",
            attribute,
            "--algorithm-name",
            f"ML Model ({attribute})",
            "--output",
            f"data/results/ml_evaluation_200_real_{attribute}.json",
        ],
    ):
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mixed-policy ML pipeline")
    parser.add_argument("--attributes", nargs="*", default=ALL_ATTRIBUTES)
    parser.add_argument("--synthetic-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MIXED POLICY PIPELINE (V1)")
    print("=" * 80)

    for attribute in args.attributes:
        if attribute not in MIXED_POLICY_V1:
            print(f"ERROR: No policy for attribute '{attribute}'")
            return 1
        ok = run_attribute(attribute, args.synthetic_limit, args.seed)
        if not ok:
            return 1

    if not args.skip_analysis:
        if not run_step("Analyze final results", [sys.executable, "scripts/analyze_results.py"]):
            return 1

    print("\nMixed-policy run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
