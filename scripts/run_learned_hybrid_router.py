"""
Train and evaluate a learned hybrid router.

The router learns when to trust ML vs a selected baseline for each attribute.
It uses existing 200-record prediction artifacts and golden labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from scripts.evaluate_models import evaluate_algorithm, load_golden_labels
except ModuleNotFoundError:
    from evaluate_models import evaluate_algorithm, load_golden_labels

ALL_ATTRIBUTES = ["name", "phone", "website", "address", "category"]


def load_ml_predictions(attribute: str) -> dict:
    path = Path(f"data/results/ml_predictions_200_real_{attribute}.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}
    for item in raw:
        out[item["id"]] = {
            "selected_source": item.get("selected_source", "unclear"),
            "model_confidence": float(item.get("model_confidence", 0.0)),
            "model_confidence_raw": float(item.get("model_confidence_raw", item.get("model_confidence", 0.0))),
            "current_probability_calibrated": float(item.get("current_probability_calibrated", 0.5)),
            "is_calibrated": 1.0 if bool(item.get("is_calibrated", False)) else 0.0,
        }
    return out


def load_baseline_predictions(attribute: str, baseline: str) -> dict:
    path = Path(f"data/results/predictions_baseline_{baseline}_200_real_{attribute}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_training_frame(attribute: str, baseline: str, golden: dict, ml_preds: dict, base_preds: dict) -> pd.DataFrame:
    rows = []
    ids = set(golden.keys()) & set(ml_preds.keys()) & set(base_preds.keys())

    for rid in ids:
        gold = golden[rid]
        ml = ml_preds[rid]["selected_source"]
        base = base_preds[rid]

        if gold == "unclear" or ml == "unclear" or base == "unclear":
            continue

        ml_correct = 1 if ml == gold else 0
        base_correct = 1 if base == gold else 0

        # Learn only from informative rows where one choice is better.
        if ml_correct == base_correct:
            continue

        choose_ml = 1 if (ml_correct == 1 and base_correct == 0) else 0

        conf = float(ml_preds[rid]["model_confidence"])
        conf_raw = float(ml_preds[rid]["model_confidence_raw"])
        curr_prob_cal = float(ml_preds[rid]["current_probability_calibrated"])

        rows.append(
            {
                "id": rid,
                "choose_ml": choose_ml,
                "ml_conf": conf,
                "ml_conf_raw": conf_raw,
                "ml_margin": abs(conf - 0.5),
                "curr_prob_cal": curr_prob_cal,
                "curr_prob_margin": abs(curr_prob_cal - 0.5),
                "agree": 1.0 if ml == base else 0.0,
                "is_calibrated": float(ml_preds[rid]["is_calibrated"]),
            }
        )

    return pd.DataFrame(rows)


def choose_best_threshold(y_true: np.ndarray, p_ml: np.ndarray, thresholds: list[float]) -> tuple[float, float]:
    best_th = 0.5
    best_score = -1.0
    for th in thresholds:
        pred = (p_ml >= th).astype(int)
        score = float((pred == y_true).mean())
        if score > best_score:
            best_score = score
            best_th = th
    return best_th, best_score


def tune_router_with_cv(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Tune logistic C and decision threshold with stratified CV.

    Returns:
      best_c, best_threshold, best_mean_accuracy
    """
    threshold_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    c_grid = [0.2, 0.5, 1.0, 2.0, 5.0]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        return 1.0, 0.5, -1.0

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_cfg = (1.0, 0.5, -1.0)
    for c_val in c_grid:
        fold_scores = {th: [] for th in threshold_grid}

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            clf = LogisticRegression(max_iter=1000, random_state=42, C=c_val)
            clf.fit(X_train_s, y_train)
            p_val = clf.predict_proba(X_val_s)[:, 1]

            for th in threshold_grid:
                pred = (p_val >= th).astype(int)
                fold_scores[th].append(float((pred == y_val).mean()))

        for th in threshold_grid:
            if not fold_scores[th]:
                continue
            mean_acc = float(np.mean(fold_scores[th]))
            if mean_acc > best_cfg[2]:
                best_cfg = (c_val, th, mean_acc)

    return best_cfg


def run_attribute(attribute: str, baseline: str, golden_path: str) -> dict:
    golden = load_golden_labels(golden_path, attribute)
    ml_preds = load_ml_predictions(attribute)
    base_preds = load_baseline_predictions(attribute, baseline)

    train_df = build_training_frame(attribute, baseline, golden, ml_preds, base_preds)

    # Conservative fallback when there is little signal: keep baseline.
    if train_df.empty or train_df["choose_ml"].nunique() < 2:
        final_preds = {rid: base_preds[rid] for rid in golden.keys() if rid in base_preds}
        eval_result = evaluate_algorithm(final_preds, golden, f"Learned Hybrid Router ({attribute})")
        eval_result["router_meta"] = {
            "baseline": baseline,
            "mode": "fallback_baseline_only",
            "n_train_rows": int(len(train_df)),
        }
        return {
            "predictions": final_preds,
            "evaluation": eval_result,
            "threshold": None,
            "best_c": None,
            "cv_accuracy": None,
            "n_train_rows": int(len(train_df)),
        }

    feature_cols = [
        "ml_conf",
        "ml_conf_raw",
        "ml_margin",
        "curr_prob_cal",
        "curr_prob_margin",
        "agree",
        "is_calibrated",
    ]

    X = train_df[feature_cols].values
    y = train_df["choose_ml"].values

    best_c, threshold, cv_accuracy = tune_router_with_cv(X, y)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42, C=best_c)
    clf.fit(X_s, y)

    # Optional in-sample calibration of threshold tie-breaking if CV fails to tune.
    p_full = clf.predict_proba(X_s)[:, 1]
    if cv_accuracy < 0:
        threshold, _ = choose_best_threshold(y, p_full, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])

    # Apply router model to all evaluable records.
    final_preds = {}
    for rid in golden.keys():
        if rid not in ml_preds or rid not in base_preds:
            continue

        ml = ml_preds[rid]["selected_source"]
        base = base_preds[rid]

        if ml == "unclear":
            final_preds[rid] = base
            continue
        if base == "unclear":
            final_preds[rid] = ml
            continue

        row = np.array(
            [
                float(ml_preds[rid]["model_confidence"]),
                float(ml_preds[rid]["model_confidence_raw"]),
                abs(float(ml_preds[rid]["model_confidence"]) - 0.5),
                float(ml_preds[rid]["current_probability_calibrated"]),
                abs(float(ml_preds[rid]["current_probability_calibrated"]) - 0.5),
                1.0 if ml == base else 0.0,
                float(ml_preds[rid]["is_calibrated"]),
            ]
        ).reshape(1, -1)

        p_choose_ml = float(clf.predict_proba(scaler.transform(row))[:, 1][0])
        final_preds[rid] = ml if p_choose_ml >= threshold else base

    eval_result = evaluate_algorithm(final_preds, golden, f"Learned Hybrid Router ({attribute})")
    eval_result["router_meta"] = {
        "baseline": baseline,
        "mode": "learned_logreg_router_cv",
        "threshold": threshold,
        "best_c": best_c,
        "cv_accuracy": cv_accuracy,
        "n_train_rows": int(len(train_df)),
    }

    return {
        "predictions": final_preds,
        "evaluation": eval_result,
        "threshold": float(threshold),
        "best_c": float(best_c),
        "cv_accuracy": float(cv_accuracy),
        "n_train_rows": int(len(train_df)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate learned hybrid router")
    parser.add_argument("--golden", default="data/golden_dataset_200.json")
    parser.add_argument("--attributes", nargs="*", default=ALL_ATTRIBUTES)
    parser.add_argument("--policy-json", default="data/results/experiment_reports/exp_step5_hybrid_router_best_policy.json")
    parser.add_argument("--router-tag", default="learned_hybrid_router_v2")
    args = parser.parse_args()

    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)

    out_dir = Path("data/results")
    summary = {}

    for attribute in args.attributes:
        baseline = policy.get(attribute, {}).get("baseline", "hybrid")
        result = run_attribute(attribute, baseline, args.golden)

        pred_path = out_dir / f"predictions_{args.router_tag}_{attribute}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(result["predictions"], f, indent=2)

        eval_path = out_dir / f"evaluation_{args.router_tag}_{attribute}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(result["evaluation"], f, indent=2)

        summary[attribute] = {
            "baseline": baseline,
            "threshold": result["threshold"],
            "best_c": result["best_c"],
            "cv_accuracy": result["cv_accuracy"],
            "n_train_rows": result["n_train_rows"],
            "f1": result["evaluation"]["metrics"]["f1"],
            "prediction_file": str(pred_path),
            "evaluation_file": str(eval_path),
        }

    summary_path = out_dir / f"{args.router_tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved learned router summary:", summary_path)
    print("Per-attribute F1:", {k: round(v["f1"], 4) for k, v in summary.items()})
    macro = sum(v["f1"] for v in summary.values()) / max(len(summary), 1)
    print("Macro F1:", round(macro, 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
