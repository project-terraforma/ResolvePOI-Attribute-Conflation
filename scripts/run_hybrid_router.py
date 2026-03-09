"""
Route final decisions between ML and a baseline fallback per attribute.

This script is intentionally conservative: use ML where it is trusted,
fall back to baseline elsewhere.
"""

import argparse
import json
from pathlib import Path

import subprocess
import sys

ALL_ATTRIBUTES = ["name", "phone", "website", "address", "category"]
REAL_GOLDEN_PATH = "data/golden_dataset_200.json"

# Hybrid router v1: safe defaults after mixed-policy findings.
HYBRID_ROUTER_V1 = {
    "name": {"mode": "confidence_gate", "threshold": 0.80},
    "address": {"mode": "confidence_gate", "threshold": 0.70},
    "phone": {"mode": "baseline_only", "threshold": 1.00},
    "website": {"mode": "baseline_only", "threshold": 1.00},
    "category": {"mode": "baseline_only", "threshold": 1.00},
}


def load_policy(policy_json: str | None) -> dict:
    """Load router policy from JSON file or fall back to built-in v1 policy."""
    if not policy_json:
        return HYBRID_ROUTER_V1

    policy_path = Path(policy_json)
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)
    return policy


def load_ml_predictions(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}
    for item in raw:
        out[item["id"]] = {
            "selected_source": item.get("selected_source", "unclear"),
            "model_confidence": float(item.get("model_confidence", 0.0)),
            "model_confidence_raw": float(item.get("model_confidence_raw", item.get("model_confidence", 0.0))),
            "current_probability_calibrated": float(item.get("current_probability_calibrated", 0.5)),
            "is_calibrated": bool(item.get("is_calibrated", False)),
        }
    return out


def load_baseline_predictions(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def route_prediction(
    ml_pred: dict,
    baseline_pred: str,
    mode: str,
    threshold: float,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    prefer_ml_on_agreement_midband: bool = True,
) -> tuple[str, str]:
    ml_choice = ml_pred.get("selected_source", "unclear")
    ml_conf = float(ml_pred.get("model_confidence", 0.0))

    if mode == "ml_only":
        return ml_choice, "ml"
    if mode == "baseline_only":
        return baseline_pred, "baseline"
    if mode == "dual_threshold_gate":
        low = float(low_threshold if low_threshold is not None else 0.70)
        high = float(high_threshold if high_threshold is not None else 0.90)

        if ml_conf >= high:
            return ml_choice, "ml_high"
        if ml_conf <= low:
            return baseline_pred, "baseline_low"

        # Mid-band: use disagreement signal conservatively.
        if ml_choice == baseline_pred and prefer_ml_on_agreement_midband:
            return ml_choice, "ml_mid_agree"
        return baseline_pred, "baseline_mid_disagree"

    # confidence_gate
    if ml_conf >= threshold:
        return ml_choice, "ml"
    return baseline_pred, "baseline"


def run_step(name: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("=" * 80)
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr)
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hybrid router predictions and evaluation")
    parser.add_argument("--attributes", nargs="*", default=ALL_ATTRIBUTES)
    parser.add_argument(
        "--fallback-baseline",
        choices=["hybrid", "most_recent", "confidence", "completeness"],
        default="hybrid",
        help="Baseline source used as fallback.",
    )
    parser.add_argument(
        "--router-tag",
        default="hybrid_router_v1",
        help="Tag for output files.",
    )
    parser.add_argument(
        "--policy-json",
        default=None,
        help="Optional path to policy JSON overriding built-in HYBRID_ROUTER_V1.",
    )
    args = parser.parse_args()

    policy_map = load_policy(args.policy_json)

    out_dir = Path("data/results")

    summary = {}

    for attribute in args.attributes:
        if attribute not in policy_map:
            print(f"No router policy for attribute: {attribute}")
            return 1

        policy = policy_map[attribute]
        baseline_name = policy.get("baseline", args.fallback_baseline)
        ml_path = out_dir / f"ml_predictions_200_real_{attribute}.json"
        baseline_path = out_dir / f"predictions_baseline_{baseline_name}_200_real_{attribute}.json"

        if not ml_path.exists() or not baseline_path.exists():
            print(f"Missing input files for {attribute}: {ml_path} / {baseline_path}")
            return 1

        ml_predictions = load_ml_predictions(ml_path)
        baseline_predictions = load_baseline_predictions(baseline_path)

        combined = {}
        routed_to_ml = 0
        routed_to_baseline = 0

        all_ids = set(ml_predictions.keys()) | set(baseline_predictions.keys())
        for rid in all_ids:
            ml_pred = ml_predictions.get(rid, {"selected_source": "unclear", "model_confidence": 0.0})
            baseline_pred = baseline_predictions.get(rid, "unclear")
            final_pred, routed_source = route_prediction(
                ml_pred=ml_pred,
                baseline_pred=baseline_pred,
                mode=policy["mode"],
                threshold=float(policy.get("threshold", 0.0)),
                low_threshold=policy.get("low_threshold"),
                high_threshold=policy.get("high_threshold"),
                prefer_ml_on_agreement_midband=bool(policy.get("prefer_ml_on_agreement_midband", True)),
            )
            combined[rid] = final_pred

            if routed_source.startswith("ml"):
                routed_to_ml += 1
            else:
                routed_to_baseline += 1

        out_pred = out_dir / f"predictions_{args.router_tag}_{attribute}.json"
        with open(out_pred, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

        out_eval = out_dir / f"evaluation_{args.router_tag}_{attribute}.json"
        ok = run_step(
            f"Evaluate routed predictions ({attribute})",
            [
                sys.executable,
                "scripts/evaluate_models.py",
                "--predictions",
                str(out_pred),
                "--golden",
                REAL_GOLDEN_PATH,
                "--attribute",
                attribute,
                "--algorithm-name",
                f"Hybrid Router ({attribute})",
                "--output",
                str(out_eval),
            ],
        )
        if not ok:
            return 1

        summary[attribute] = {
            "policy": policy,
            "baseline": baseline_name,
            "routed_to_ml": routed_to_ml,
            "routed_to_baseline": routed_to_baseline,
            "prediction_file": str(out_pred),
            "evaluation_file": str(out_eval),
        }

    summary_path = out_dir / f"{args.router_tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved router summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
