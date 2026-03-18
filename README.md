# Places Attributes Conflation
**CRWN102 Project A Submission**  
**Kate Mikhailova**  
Based on prior work in `Mayhem_Attribute_Conflation` by **Jaskaran Singh** and **Varnit Balivada**

## Project Overview

This project studies **place attribute conflation**: given two candidate versions of the same place record, choose whether the better value comes from the `current` side or the `base` side.

The attributes evaluated are:

- `name`
- `phone`
- `website`
- `address`
- `category`

I built on top of the existing Mayhem pipeline, which already included:

- rule-based baselines:
  - `Most Recent`
  - `Completeness`
  - `Confidence`
  - heuristic `Hybrid`
- per-attribute ML trained on synthetic data

My work focused on:

- improving synthetic Yelp-based data realism
- expanding the labeled benchmark from 200 to 400 records
- improving ML confidence quality with calibration
- tuning decision thresholds for F1 instead of using fixed `0.5`
- building and sweeping a **hybrid router** that selectively chooses between ML and rule-based methods
- exploring a **learned router** as a future direction

The main result is that **pure ML improved substantially**, but the best final system was still a **selective hybrid**, not full ML replacement.

## What I Changed

Compared with the original Mayhem repo, I added or extended the following:

- **Synthetic data realism improvements**
  - more realistic Yelp-derived corruptions
  - attribute-specific perturbations
  - near-equal and both-noisy cases
  - labels based on quality rather than confidence shortcuts
- **Expanded benchmark**
  - added `data/golden_dataset_400.json`
  - added `data/golden_dataset_next_200_template.json`
- **ML improvements**
  - per-attribute threshold tuning for F1
  - calibration-aware confidence using Platt scaling
  - refreshed training summaries and predictions
- **Hybrid routing**
  - swept a large policy space to find the best per-attribute routing strategy
  - saved best policy and summary artifacts
- **Learned router**
  - implemented a meta-model that learns when to trust ML vs baseline
  - evaluated it as an exploratory future direction
- **Reporting**
  - unified experiment reports and combined summaries
  - added overfitting/generalization analysis

## Repository Structure

```markdown
Mayhem_Attribute_Conflation/
├── data/
│   ├── golden_dataset_200.json                    # Original labeled benchmark
│   ├── golden_dataset_400.json                    # Expanded benchmark used in later reruns
│   ├── golden_dataset_next_200_template.json      # Template for collecting additional labels
│   ├── synthetic_golden_dataset_2k.json           # Yelp-derived synthetic training set
│   ├── project_b_samples_2k.parquet               # Original Overture sample records
│   ├── processed/
│   │   ├── features_*_synthetic.parquet           # Per-attribute synthetic feature sets
│   │   └── golden_dataset_*.json                  # Train/validation/test splits and processed sets
│   └── results/
│       ├── experiment_reports/                    # Main reports, logs, sweeps, and summaries
│       ├── ml_evaluation_200_real_*.json          # ML evaluation artifacts
│       ├── ml_predictions_200_real_*.json         # ML predictions on labeled real data
│       ├── predictions_baseline_*.json            # Rule-based baseline predictions
│       ├── predictions_exp_step5_hybrid_router_*.json
│       ├── predictions_learned_hybrid_router_*.json
│       └── *_summary.json                         # Summary files for hybrid and learned router runs
│
├── docs/
│   ├── README.md                                  # Documentation index
│   ├── AI_ANNOTATION_REPORT.md                    # Earlier annotation comparison report
│   ├── EVALUATION_PROTOCOL.md                     # Reproducible experiment protocol
│   ├── overfitting_report.md                      # Overfitting/generalization analysis
│   ├── kate_mayhem_presentation.md                # Presentation draft used for final slides
│   └── ...                                        # Guidelines, notes, reports, and project docs
│
├── models/
│   ├── ml/
│   │   ├── name/                                  # Best models and training summary for name
│   │   ├── phone/                                 # Best models and training summary for phone
│   │   ├── website/                               # Best models and training summary for website
│   │   ├── address/                               # Best models and training summary for address
│   │   └── category/                              # Best models and training summary for category
│   ├── rule_based/                                # Baseline evaluation outputs
│   └── hybrid/                                    # Hybrid evaluation outputs
│
├── notebooks/
│   └── colab_pipeline_setup.ipynb                 # Notebook workflow for Colab
│
├── scripts/
│   ├── run_algorithm_pipeline.py                  # Main pipeline orchestrator
│   ├── run_repro_eval.py                          # Reproducible experiment runner
│   ├── generate_synthetic_dataset.py              # Yelp-based synthetic data generator
│   ├── extract_features.py                        # Feature engineering for attribute comparison
│   ├── train_models.py                            # Trains logistic regression / RF / GB per attribute
│   ├── run_inference.py                           # Runs the best ML model on target records
│   ├── baseline_heuristics.py                     # Most Recent / Completeness / Confidence / Hybrid rules
│   ├── run_hybrid_router.py                       # Applies a fixed hybrid router policy
│   ├── sweep_hybrid_router.py                     # Searches many hybrid routing policies
│   ├── run_learned_hybrid_router.py               # Learned router meta-model
│   ├── evaluate_models.py                         # Shared evaluation utilities
│   ├── analyze_results.py                         # Builds comparison tables and reports
│   └── calculate_agreement.py                     # Agreement analysis for annotation workflows
│
└── yelp/
    ├── yelp_academic_dataset_business.json        # Yelp business data used for synthetic generation
    └── Dataset_User_Agreement.pdf                 # Yelp dataset license / terms
```

## Methodology

### 1. Rule-Based Baselines

The original pipeline compared several heuristic strategies:

- **Most Recent**
  - prefer the newer/current-side value
- **Completeness**
  - prefer the side with more complete information
- **Confidence**
  - prefer the side with the higher source confidence
- **Hybrid**
  - heuristic combination of rule signals

These baselines were already strong and served as the reference point for all later work.

### 2. Machine Learning Pipeline

The ML system is a **per-attribute tabular classification pipeline**, not a deep neural network.

For each attribute:

1. extract comparison features from `current` and `base`
2. train three candidate models:
   - logistic regression
   - random forest
   - gradient boosting
3. select the best model by validation F1
4. calibrate the probability output with **Platt scaling**
5. tune the decision threshold for best validation F1

This produces a binary decision:

- `current` is better
- or `base` is better

### 3. Hybrid Router

The hybrid router is the core system-level contribution.

Instead of trusting ML everywhere, it:

- uses a selected baseline as the **safe default**
- lets ML override only when the policy says confidence is strong enough

The best hybrid policy was found by sweeping a large policy space over:

- fallback baseline choice
- routing mode
- threshold values

This repository searched **541,696 policy configurations**.

### 4. Learned Router

The learned router is a **meta-model**.

It does not directly predict `current` vs `base`.  
Instead, it learns:

- should we trust ML here?
- or should we trust the baseline here?

This is an exploratory direction. In the current project it did not outperform the swept hybrid, but it remains a promising future improvement if more real labeled disagreement data is collected.

## Results Summary

### Starting Point

The original repo’s starting results were dominated by rule-based baselines:

| Method | Macro F1 |
|---|---:|
| Most Recent | 0.8370 |
| Heuristic Hybrid | 0.8195 |
| Starting ML | 0.4600 |

This is why my work focused on improving ML quality and then combining ML with baselines more intelligently.

### Final ML Improvement

Pure ML improved substantially over the course of the project:

| Stage | ML Macro F1 |
|---|---:|
| Starting ML | 0.4600 |
| Early cleanup | 0.7001 |
| Gated ML phase | 0.7356 |
| Final ML refresh | 0.8323 |

### Final Method Comparison

Current final comparisons from the combined report:

| Method | Macro F1 |
|---|---:|
| Starting ML | 0.4600 |
| Final ML | 0.8323 |
| Best baseline (`Most Recent`) | 0.8574 |
| Best swept hybrid | 0.8491 |
| Learned router v2 | 0.8040 |

### Best Final Hybrid Policy

The best swept hybrid policy was:

- `name`
  - confidence-gated ML over `most_recent`
- `address`
  - confidence-gated ML over `hybrid`
- `phone`
  - baseline-only using `hybrid`
- `website`
  - baseline-only using `hybrid`
- `category`
  - baseline-only using `most_recent`

This is the main project conclusion:

> ML was useful, but only in specific places. The winning design was not to trust ML everywhere, but to let it help only where it had strong evidence.

## Overfitting / Generalization Notes

I also analyzed whether the final ML setup showed signs of overfitting.

Main conclusion:

- there was **no strong evidence of classic train-set overfitting**
- but there **was evidence of shortcut-like behavior** on some attributes, especially a tendency for some methods to over-predict `current`

See:

- [`docs/overfitting_report.md`](docs/overfitting_report.md)

This report includes:

- train vs validation vs real F1 comparisons
- tuned cross-validation checks
- confusion-matrix counts
- FP/FN summaries

## Key Files for Submission

If a grader or reviewer only opens a few files, these are the most useful:

- [`README.md`](README.md)
  - project overview and submission summary
- [`data/results/experiment_reports/current_combined_report.txt`](data/results/experiment_reports/current_combined_report.txt)
  - combined comparison table across methods
- [`data/results/experiment_reports/exp_step5_hybrid_router_sweep.json`](data/results/experiment_reports/exp_step5_hybrid_router_sweep.json)
  - best hybrid policy and search summary
- [`data/results/exp_step5_hybrid_router_best_summary.json`](data/results/exp_step5_hybrid_router_best_summary.json)
  - final best hybrid configuration
- [`data/results/learned_hybrid_router_v2_summary.json`](data/results/learned_hybrid_router_v2_summary.json)
  - learned router summary
- [`docs/overfitting_report.md`](docs/overfitting_report.md)
  - overfitting and generalization analysis
- [`scripts/generate_synthetic_dataset.py`](scripts/generate_synthetic_dataset.py)
  - synthetic data generation logic
- [`scripts/train_models.py`](scripts/train_models.py)
  - ML training, calibration, threshold tuning
- [`scripts/sweep_hybrid_router.py`](scripts/sweep_hybrid_router.py)
  - hybrid policy search
- [`scripts/run_learned_hybrid_router.py`](scripts/run_learned_hybrid_router.py)
  - learned router implementation

## Installation

```bash
git clone https://github.com/project-terraforma/Mayhem_Attribute_Conflation.git
cd Mayhem_Attribute_Conflation
pip install -r requirements.txt
```

If using Git LFS for large files:

```bash
git lfs pull
```

## Running the Pipeline

### Full pipeline

```bash
python scripts/run_algorithm_pipeline.py
```

### Reproducible experiment run

```bash
python scripts/run_repro_eval.py --tag exp_name_v1 --mode full --attributes name
```

### Analyze results

```bash
python scripts/analyze_results.py
```

### Sweep hybrid policies

```bash
python scripts/sweep_hybrid_router.py
```

### Run learned router

```bash
python scripts/run_learned_hybrid_router.py
```

## Future Improvements

The clearest next steps are:

- collect more real labeled data
- improve hard edge-case coverage
- strengthen per-class evaluation and false-positive / false-negative analysis
- revisit learned routing once more real disagreement data is available

## Acknowledgements

- Original repository work by **Jaskaran Singh** and **Varnit Balivada**
- Data sources:
  - Overture Maps Foundation
  - Yelp Academic Dataset
- Course context:
  - **CRWN102**

See [`LICENSE`](LICENSE) for terms.
