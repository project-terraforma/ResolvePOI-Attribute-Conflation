---
title: "Mayhem Attribute Conflation"
subtitle: "Kate Mikhailova"
author: ""
date: "March 11, 2026"
theme: "Madrid"
colortheme: "default"
fontsize: 10pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{array}
  - \usepackage{tabularx}
  - \usepackage{longtable}
---

# What We Started With

- Goal: choose the better value between `current` and `base` for `name`, `phone`, `website`, `address`, and `category`
- Starting system had:
  - rule-based baselines: `Most Recent`, `Completeness`, `Confidence`, heuristic `Hybrid`
  - per-attribute ML trained on synthetic data
- Strongest starting performance came from simple heuristics, not ML
- I did not want production logic to rely mainly on recency, even if it scored well offline

\small

| Attribute | Most Recent F1 | Heuristic Hybrid F1 | Starting ML F1 |
|---|---:|---:|---:|
| Address | 0.8338 | 0.8338 | 0.2515 |
| Category | 0.8338 | 0.8094 | 0.6417 |
| Name | 0.8338 | 0.7667 | 0.3656 |
| Phone | 0.8554 | 0.8554 | 0.6929 |
| Website | 0.8323 | 0.8323 | 0.3483 |
| Macro Avg | 0.8370 | 0.8195 | 0.4600 |

\normalsize

# My Approach

- Main goal: make ML more trustworthy, not just more complex
- I improved:
  - synthetic data realism using Yelp
  - size of the labeled benchmark: `200 -> 400`
  - ML confidence quality and decision thresholds
  - the interaction between ML and rule-based methods
- Shift in strategy:
  - from `ML vs rules`
  - to `ML selectively combined with rules`

## Why Yelp

- Large-scale business records with the same fields as the task
- Real labels were too limited for ML-only training
- Yelp made it possible to generate many realistic training pairs cheaply

## Why More Labels

- The task has ambiguous edge cases
- More labels improve evaluation stability and confidence in conclusions

# How I Improved ML

- Per-attribute models instead of one global model
- Improved synthetic generation:
  - attribute-specific corruption
  - near-equal cases
  - both-noisy cases
  - labels based on quality, not confidence shortcuts
- Added calibration-aware confidence with Platt scaling
- Tuned decision thresholds for F1 instead of fixed `0.5`

## Why calibration and threshold tuning mattered

- Raw probabilities are often overconfident
- Calibration makes confidence better match actual correctness
- Different attributes need different cutoffs
- I optimized directly for validation F1

\small

| Milestone | Address | Category | Name | Phone | Website | ML Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Starting ML | 0.2515 | 0.6417 | 0.3656 | 0.6929 | 0.3483 | 0.4600 |
| Early cleanup | 0.7921 | 0.8338 | 0.2410 | 0.8000 | 0.8338 | 0.7001 |
| Gated ML phase | 0.7921 | 0.4623 | 0.8225 | 0.8519 | 0.7491 | 0.7356 |
| Final ML refresh | 0.8315 | 0.8563 | 0.8079 | 0.8094 | 0.8563 | 0.8323 |

\normalsize

# Hybrid Strategy

- Original Mayhem hybrid:
  - heuristic combination of rule signals
  - still fully rule-based
  - no learned routing between ML and rules
- My hybrid idea:
  - baseline is the safe default
  - ML only overrides when confidence is strong enough
  - routing policy differs by attribute
- I searched `541,696` policy configurations

## Example decision

- Baseline says `base`
- ML says `current`
- calibrated ML confidence = `0.91`
- name policy threshold = `0.85`
- final choice = `ML`

If confidence were `0.68`, the router would fall back to the baseline.

## Best final policy

- `name`: confidence-gated ML over `Most Recent`
- `address`: confidence-gated ML over `Hybrid`
- `phone`, `website`, `category`: baseline-only

\small

| Method | Macro F1 |
|---|---:|
| Starting ML | 0.4600 |
| Final ML | 0.8323 |
| Best baseline (`Most Recent`) | 0.8574 |
| Best swept hybrid | 0.8491 |
| Learned router v2 | 0.8040 |

\normalsize

# Findings, Limitations, and Company Relevance

## Findings

- Strong baselines can hide fragile logic
- ML improved a lot with better realism, calibration, and thresholds
- Best system design was selective override, not full ML replacement
- Confidence quality mattered almost as much as classifier quality

## Limitations

- Synthetic-to-real gap still exists
- Real labeled benchmark is still relatively small
- Learned routing is data-hungry
- Some report artifacts still use legacy wording

## Why this matters to a company

- Do not over-trust a shortcut baseline just because it wins one benchmark
- Safer deployment pattern:
  - keep strong heuristics
  - use calibrated ML selectively
  - preserve interpretable fallback behavior

The goal is not to replace logic with ML. The goal is to use ML where it adds real signal, while keeping reliable fallback behavior where it does not.
