# Results

9,750 dog stays, intake 2017-01-01 to 2024-12-31.

## Where dogs are N days after intake (Aalen-Johansen cumulative incidence)

|   day |   Adoption |   Euthanasia / died |   Rescue / transfer |   Returned to owner |   In shelter |
|------:|-----------:|--------------------:|--------------------:|--------------------:|-------------:|
|     7 |        5.5 |                 4.5 |                21.1 |                25.7 |         43.1 |
|    30 |       18.6 |                 6.1 |                31.4 |                29   |         14.9 |
|    90 |       24.6 |                 7   |                34.1 |                29.3 |          5   |
|   180 |       26.6 |                 7.4 |                35   |                29.3 |          1.7 |

Naive Kaplan-Meier for adoption at day 180: 85.5% (vs 26.6%).

## Adoption model, test on 2023-24 intakes

Train n=6,848 (recent window n=2,304), test n=2,767. Adoption base rate: train 21.6%, test 41.6%.

| model                              |   roc_auc |   pr_auc |   brier |
|:-----------------------------------|----------:|---------:|--------:|
| base_rate                          |     0.5   |    0.416 |  0.2829 |
| logistic_intake_only               |     0.647 |    0.532 |  0.2571 |
| lightgbm_intake_only               |     0.673 |    0.558 |  0.2503 |
| logistic_with_leaky_fields         |     0.869 |    0.795 |  0.1552 |
| lightgbm_with_leaky_fields         |     0.874 |    0.806 |  0.1478 |
| logistic_intake_only_recent_window |     0.692 |    0.583 |  0.2164 |
| lightgbm_intake_only_recent_window |     0.699 |    0.59  |  0.2178 |

Permutation importance (ROC-AUC drop, intake-only LightGBM):

|                  |   auc_drop |
|:-----------------|-----------:|
| age_years        |     0.1183 |
| intake_condition |     0.0285 |
| intake_type      |     0.0198 |
| intake_subtype   |     0.0182 |
| month            |     0.0071 |
| primary_color    |     0.0036 |
| sex_base         |     0.0015 |
| jurisdiction     |     0.0015 |
