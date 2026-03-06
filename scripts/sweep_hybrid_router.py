"""
Sweep hybrid-router policies and report top-performing configs.
"""

import argparse
import itertools
import json
from pathlib import Path

try:
    from scripts.evaluate_models import evaluate_algorithm, load_golden_labels
except ModuleNotFoundError:
    from evaluate_models import evaluate_algorithm, load_golden_labels

ALL_ATTRIBUTES = ["name", "phone", "website", "address", "category"]
BASELINES = ["hybrid", "most_recent", "confidence", "completeness"]
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_ml_predictions(attribute: str) -> dict:
    path = Path(f"data/results/ml_predictions_200_real_{attribute}.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for item in raw:
        out[item["id"]] = {
            "selected_source": item.get("selected_source", "unclear"),
            "model_confidence": float(item.get("model_confidence", 0.0)),
        }
    return out


def load_baseline_predictions(attribute: str, baseline: str) -> dict:
    path = Path(f"data/results/predictions_baseline_{baseline}_200_real_{attribute}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def route(ml_pred: dict, baseline_pred: str, mode: str, threshold: float) -> str:
    ml_choice = ml_pred.get("selected_source", "unclear")
    ml_conf = float(ml_pred.get("model_confidence", 0.0))

    if mode == "ml_only":
        return ml_choice
    if mode == "baseline_only":
        return baseline_pred
    if ml_conf >= threshold:
        return ml_choice
    return baseline_pred


def build_predictions(
    attribute: str,
    ml_preds: dict,
    baseline_preds: dict,
    mode: str,
    threshold: float,
) -> dict:
    ids = set(ml_preds.keys()) | set(baseline_preds.keys())
    out = {}
    for rid in ids:
        out[rid] = route(
            ml_pred=ml_preds.get(rid, {"selected_source": "unclear", "model_confidence": 0.0}),
            baseline_pred=baseline_preds.get(rid, "unclear"),
            mode=mode,
            threshold=threshold,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep hybrid router policies")
    parser.add_argument("--golden", default="data/golden_dataset_200.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--weights-json",
        default="",
        help="Optional JSON object mapping attribute->weight for weighted score.",
    )
    parser.add_argument(
        "--output-policy",
        default="data/results/experiment_reports/exp_step5_hybrid_router_best_policy.json",
    )
    parser.add_argument(
        "--output-report",
        default="data/results/experiment_reports/exp_step5_hybrid_router_sweep.json",
    )
    args = parser.parse_args()

    weights = {a: 1.0 for a in ALL_ATTRIBUTES}
    if args.weights_json:
        custom = json.loads(args.weights_json)
        for k, v in custom.items():
            if k in weights:
                weights[k] = float(v)

    golden_by_attr = {a: load_golden_labels(args.golden, a) for a in ALL_ATTRIBUTES}
    ml_by_attr = {a: load_ml_predictions(a) for a in ALL_ATTRIBUTES}
    baseline_by_attr = {
        a: {b: load_baseline_predictions(a, b) for b in BASELINES}
        for a in ALL_ATTRIBUTES
    }

    results = []

    # Keep phone/website/category as baseline-only; sweep which baseline to use.
    # Sweep gate threshold + fallback baseline for name/address.
    for phone_b, web_b, cat_b, name_fallback, addr_fallback, name_th, addr_th in itertools.product(
        BASELINES,
        BASELINES,
        BASELINES,
        BASELINES,
        BASELINES,
        THRESHOLDS,
        THRESHOLDS,
    ):
        policy = {
            "name": {"mode": "confidence_gate", "threshold": name_th, "baseline": name_fallback},
            "address": {"mode": "confidence_gate", "threshold": addr_th, "baseline": addr_fallback},
            "phone": {"mode": "baseline_only", "threshold": 1.0, "baseline": phone_b},
            "website": {"mode": "baseline_only", "threshold": 1.0, "baseline": web_b},
            "category": {"mode": "baseline_only", "threshold": 1.0, "baseline": cat_b},
        }

        per_attr_f1 = {}
        for attribute in ALL_ATTRIBUTES:
            p = policy[attribute]
            preds = build_predictions(
                attribute=attribute,
                ml_preds=ml_by_attr[attribute],
                baseline_preds=baseline_by_attr[attribute][p["baseline"]],
                mode=p["mode"],
                threshold=float(p["threshold"]),
            )
            eval_result = evaluate_algorithm(
                predictions=preds,
                golden_labels=golden_by_attr[attribute],
                algorithm_name=f"sweep:{attribute}",
            )
            per_attr_f1[attribute] = float(eval_result["metrics"]["f1"])

        macro_f1 = sum(per_attr_f1.values()) / len(ALL_ATTRIBUTES)
        weighted_total = sum(per_attr_f1[a] * weights[a] for a in ALL_ATTRIBUTES)
        weighted_norm = sum(weights.values()) if weights else 1.0
        weighted_f1 = weighted_total / weighted_norm

        results.append(
            {
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "per_attr_f1": per_attr_f1,
                "policy": policy,
            }
        )

    results.sort(key=lambda r: (r["weighted_f1"], r["macro_f1"]), reverse=True)
    top_k = results[: args.top_k]
    best = results[0]

    output_policy_path = Path(args.output_policy)
    output_policy_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep full per-attribute policy including fallback baseline.
    best_runner_policy = best["policy"]

    with open(output_policy_path, "w", encoding="utf-8") as f:
        json.dump(best_runner_policy, f, indent=2)

    report = {
        "weights": weights,
        "searched_configs": len(results),
        "best": best,
        "top_k": top_k,
        "runner_policy_path": str(output_policy_path),
    }

    output_report_path = Path(args.output_report)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Sweep complete")
    print("searched_configs:", len(results))
    print("best_macro_f1:", round(best["macro_f1"], 6))
    print("best_weighted_f1:", round(best["weighted_f1"], 6))
    print("best_per_attr_f1:", best["per_attr_f1"])
    print("policy file:", output_policy_path)
    print("report file:", output_report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
