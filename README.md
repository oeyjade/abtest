# Data

Place `cookie_cats.csv` here.

Source: Kaggle — "Mobile Games A/B Testing with Cookie Cats"
(https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing)

Expected columns:

| column | type | meaning |
|---|---|---|
| `userid` | int | unique player |
| `version` | str | `gate_30` (control) or `gate_40` (treatment) |
| `sum_gamerounds` | int | rounds played in the first week |
| `retention_1` | bool | returned 1 day after install |
| `retention_7` | bool | returned 7 days after install |

Without this file `analysis.py` runs on clearly-labelled synthetic data so the
pipeline stays testable. Synthetic results are watermarked and must not be cited.
