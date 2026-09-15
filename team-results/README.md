# Sample submissions (for testing the evaluation code)

Ten mock submissions in the format Section 9 of the hands-on notebook produces:
the top 200 compounds, two columns, `SMILES` and `Prediction_Score`, ranked best first.

They were built by training on `sample_data/sample-train.parquet` and scoring
`sample_data/sample-test.parquet`, which holds 9 actives among 5000 compounds (0.18%).

Each team uses a different model and fingerprint, so the files span a range of
quality. `team10` is a random ranking with no model at all - it is there so the
evaluation code gets tested against a submission that should score badly.

| File | Method | Fingerprint | Actives in top 200 |
|---|---|---|---|
| `team1.csv` | LightGBM (tuned) | ECFP4 | 9 of 9 |
| `team2.csv` | LightGBM | ECFP6 | 9 of 9 |
| `team3.csv` | LightGBM | MACCS | 4 of 9 |
| `team4.csv` | Random forest | ECFP4 | 9 of 9 |
| `team5.csv` | Random forest | MACCS | 5 of 9 |
| `team6.csv` | Logistic regression | ECFP4 | 8 of 9 |
| `team7.csv` | LightGBM | RDK | 9 of 9 |
| `team8.csv` | LightGBM | AVALON | 9 of 9 |
| `team9.csv` | Random forest | FCFP4 | 9 of 9 |
| `team10.csv` | Random ranking (no model) | - | 0 of 9 |

Participants will submit to a different folder; this one exists only so the
evaluation script has realistic input to develop against.
