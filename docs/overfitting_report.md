# Overfitting Report

Date: March 14, 2026

## Objective

This report checks whether the current attribute-selection ML models show evidence of overfitting.

For this project, there are two different failure modes worth separating:

- Classic overfitting:
  - the model memorizes the synthetic training split
  - typical sign: `train_f1` much higher than `val_f1`
- Weak generalization / shortcut behavior:
  - the model learns biased or synthetic-specific decision rules
  - typical signs:
    - collapse toward one class such as always predicting `current`
    - unstable thresholds or high fold variance
    - poor transfer from synthetic validation to real labeled data

## Checks Performed

The following diagnostics were run:

1. Reconstructed train/validation evaluation on the synthetic feature sets for the saved best model type per attribute
2. Compared synthetic `train_f1` to synthetic `val_f1`
3. Compared synthetic `val_f1` to real-label benchmark `f1`
4. Ran threshold-tuned 5-fold cross-validation on the synthetic feature sets
5. Computed real-data confusion-matrix counts and false-positive / false-negative patterns

## Train vs Validation vs Real

These numbers use the best saved model type for each attribute and reconstruct the training logic with the same split, threshold tuning, and calibration behavior.

| Attribute | Best Model | Train F1 | Val F1 | Train-Val Gap | Real F1 | Val-Real Gap |
|---|---|---:|---:|---:|---:|---:|
| Address | logistic_regression | 0.7105 | 0.7391 | -0.0286 | 0.8315 | -0.0924 |
| Category | logistic_regression | 0.6360 | 0.6348 | 0.0012 | 0.8563 | -0.2215 |
| Name | logistic_regression | 0.6438 | 0.6489 | -0.0051 | 0.8079 | -0.1590 |
| Phone | random_forest | 0.8234 | 0.8377 | -0.0143 | 0.8094 | 0.0283 |
| Website | random_forest | 0.6239 | 0.6378 | -0.0139 | 0.8563 | -0.2185 |

### Interpretation

- There is no strong classic overfitting pattern here.
- The `train_f1` to `val_f1` gaps are very small.
- In several attributes, validation F1 is slightly higher than training F1.
- That means the saved best model types do not appear to be memorizing the synthetic training split in an obvious way.

At the same time:

- The synthetic validation split does not fully explain real-data behavior.
- For most attributes, real F1 is actually higher than synthetic validation F1.
- That suggests the issue is not simply "synthetic validation looks great but real data collapses."
- Instead, the synthetic and real distributions appear different in a more complicated way.

## Threshold-Tuned Cross-Validation

To better match the actual pipeline, cross-validation was run with per-fold threshold tuning rather than default 0.5 predictions.

| Attribute | Best Model | Tuned CV Mean F1 | CV Std | Fold Thresholds |
|---|---|---:|---:|---|
| Address | logistic_regression | 0.7171 | 0.0133 | [0.40, 0.40, 0.42, 0.20, 0.20] |
| Category | logistic_regression | 0.6357 | 0.0011 | [0.20, 0.20, 0.20, 0.20, 0.20] |
| Name | logistic_regression | 0.6468 | 0.0038 | [0.22, 0.30, 0.40, 0.38, 0.34] |
| Phone | random_forest | 0.8062 | 0.0215 | [0.42, 0.40, 0.42, 0.40, 0.46] |
| Website | random_forest | 0.6374 | 0.0033 | [0.20, 0.20, 0.20, 0.38, 0.34] |

### Interpretation

- Fold-to-fold variance is fairly low overall.
- This argues against strong instability on the synthetic side.
- `phone` has the highest variance, but it is still modest.
- `category` is extremely stable, but the threshold is fixed at `0.20` in every fold.
- That is a warning sign of biased decision behavior rather than a sign of healthy generalization.

## Real-Data Class Balance

For the 400-label rerun used in the current evaluation artifacts, the usable records are imbalanced:

- `current` or `same`: 295
- `base`: 99
- usable records per attribute: 394

This matters because a method that predicts `current` most of the time can still achieve strong recall and competitive F1.

## Confusion-Matrix Counts on Real Labels

### Name

| Method | TP | TN | FP | FN | N |
|---|---:|---:|---:|---:|---:|
| ML | 244 | 34 | 65 | 51 | 394 |
| Most Recent | 295 | 0 | 99 | 0 | 394 |
| Hybrid | 240 | 25 | 74 | 55 | 394 |
| Router Best | 295 | 0 | 99 | 0 | 394 |

### Phone

| Method | TP | TN | FP | FN | N |
|---|---:|---:|---:|---:|---:|
| ML | 242 | 38 | 61 | 53 | 394 |
| Most Recent | 291 | 18 | 81 | 4 | 394 |
| Hybrid | 291 | 18 | 81 | 4 | 394 |
| Router Best | 291 | 18 | 81 | 4 | 394 |

### Website

| Method | TP | TN | FP | FN | N |
|---|---:|---:|---:|---:|---:|
| ML | 295 | 0 | 99 | 0 | 394 |
| Most Recent | 267 | 29 | 70 | 28 | 394 |
| Hybrid | 267 | 29 | 70 | 28 | 394 |
| Router Best | 267 | 29 | 70 | 28 | 394 |

### Address

| Method | TP | TN | FP | FN | N |
|---|---:|---:|---:|---:|---:|
| ML | 269 | 16 | 83 | 26 | 394 |
| Most Recent | 295 | 0 | 99 | 0 | 394 |
| Hybrid | 295 | 0 | 99 | 0 | 394 |
| Router Best | 295 | 0 | 99 | 0 | 394 |

### Category

| Method | TP | TN | FP | FN | N |
|---|---:|---:|---:|---:|---:|
| ML | 295 | 0 | 99 | 0 | 394 |
| Most Recent | 294 | 1 | 98 | 0 | 393 |
| Hybrid | 252 | 39 | 60 | 42 | 393 |
| Router Best | 294 | 1 | 98 | 0 | 393 |

### Interpretation

These counts show the most important warning sign in the current system:

- several methods are close to one-class behavior on some attributes
- specifically, they predict `current` almost all the time

Examples:

- `ML` on `website`: `TN = 0`, `FP = 99`
- `ML` on `category`: `TN = 0`, `FP = 99`
- `Most Recent` on `name` and `address`: also near-zero true negatives

This does not prove classic train-set overfitting by itself.

However, it does show:

- shortcut-like behavior
- class-bias toward `current`
- weak negative-class detection for some attributes

For this project, that is a more important practical concern than simple memorization.

## Compact FP/FN Summary

| Attribute | Method | FP | FN |
|---|---|---:|---:|
| Name | ML | 65 | 51 |
| Name | Most Recent | 99 | 0 |
| Name | Router Best | 99 | 0 |
| Phone | ML | 61 | 53 |
| Phone | Most Recent | 81 | 4 |
| Phone | Router Best | 81 | 4 |
| Website | ML | 99 | 0 |
| Website | Most Recent | 70 | 28 |
| Website | Router Best | 70 | 28 |
| Address | ML | 83 | 26 |
| Address | Most Recent | 99 | 0 |
| Address | Router Best | 99 | 0 |
| Category | ML | 99 | 0 |
| Category | Most Recent | 98 | 0 |
| Category | Router Best | 98 | 0 |

## Overall Conclusion

The current evidence does not show strong classic overfitting.

Why:

- `train_f1` is not much higher than `val_f1`
- threshold-tuned cross-validation is reasonably stable

But the evidence does show a different risk:

- several methods behave in a shortcut-like way
- some attributes collapse toward predicting `current`
- negative-class (`base`) detection is weak for certain methods

So the main concern is not:

- "the model memorized the synthetic training split"

It is more likely:

- class-bias
- synthetic-to-real mismatch
- shortcut learning around current-side patterns

## Practical Takeaway

The best summary is:

- No strong evidence of classical overfitting
- Clear evidence of biased or shortcut-like behavior on some attributes
- More real labeled data would still help:
  - reduce uncertainty in evaluation
  - improve calibration and threshold selection
  - give the learned router a better chance to generalize

## Suggested Next Steps

1. Expand the real labeled benchmark further
2. Manually inspect false positives and false negatives for `website`, `category`, and `address`
3. Report per-class metrics explicitly in future experiment summaries
4. Save train-vs-validation metrics directly in `training_summary.json`
5. Consider stronger imbalance-aware diagnostics for attributes with near-zero true negatives
