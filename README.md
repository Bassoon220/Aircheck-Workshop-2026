# Aircheck Workshop 2026

Materials for the Aircheck Workshop 2026: teaching notebooks, sample data, and the hackathon
leaderboard.

## Repository layout

| Folder | What is in it |
|---|---|
| `Introduction-Notebooks/` | the two introductions and the data-exploration hands-on |
| `Hands-On-MachineLearning/` | the machine-learning hands-on, and the hackathon notebook |
| `Evaluation/` | the organisers' leaderboard |
| `Hands-On-DataProcessing/` | the DEL label and feature notebooks, with their `aircheck` helper |
| `sample_data/` | the sample datasets the notebooks read |
| `src/` | helper modules the notebooks import: `metrics`, `plots`, `features` |
| `team-results/` | team submissions, plus ten samples for testing the scorer |
| `results/` | generated output — trained models and leaderboard files. Created when a notebook runs; contents are gitignored |
| `requirements.txt` | Python dependencies |

## Workshop sample data

Small samples, committed to the repository so every participant starts from identical data,
in Colab and locally alike.

| File | Compounds | Label | SMILES | What it is |
|---|---|---|---|---|
| `sample-train.parquet` | 4,000 | yes, balanced 50/50 | no | DEL screen against WDR91. Training set. |
| `sample-test.parquet` | 5,000 | yes, 9 actives (0.18%) | yes | Test set, and the library that gets screened. |
| `sample-screen.parquet` | 5,000 | no | yes | Unlabelled compounds to screen and nominate. |

Every fingerprint column (`ECFP4`, `ECFP6`, `FCFP4`, `FCFP6`, `MACCS`, `RDK`, `AVALON`,
`ATOMPAIR`, `TOPTOR`) is stored as an **array of counts**, so reading a column gives NumPy
arrays directly — there are no comma-separated strings to parse.

## Main workshop data

*To be added.*

<!-- The full dataset used on the day. Fill in: file names and sizes, how many compounds and
     actives, where they live, and what the notebooks should point at instead of the samples
     above. -->

The full AIRCHECK datasets are available from https://www.aircheck.ai/datasets

## Open in Colab

Click a link and Colab opens the notebook straight from this repository. No GitHub account
needed — the repository is public.

| Notebook | Open |
|---|---|
| Introduction to Python | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/Introduction-Notebooks/Introduction-Python-AircheckWorkshop2026.ipynb) |
| Introduction to Machine Learning | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/Introduction-Notebooks/Introduction-MachineLearning-AircheckWorkshop2026.ipynb) |
| Hands-On: Data Exploration | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/Introduction-Notebooks/HandsOn-DataExploration-AircheckWorkshop2026.ipynb) |
| Hands-On: Machine Learning | [Open in Colab](https://colab.research.google.com/github/ShagReza/Aircheck-Workshop-2026/blob/main/Hands-On-MachineLearning/HandsOn-MachineLearning-AircheckWorkshop2026.ipynb) |

`Evaluation/` is for organisers: it reads the gold labels, so it is not listed
above.

**Notes for participants**

- Colab opens a **copy**. Use `File → Save a copy in Drive` before editing to keep your work.
- If a notebook is updated during the workshop, re-open the link to get the new version; your
  Drive copy will not update on its own.
- The hands-on notebooks install what Colab is missing (`rdkit`, `lightgbm`, `umap-learn`) in
  their first cell.

## Local setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Then open the notebooks in VS Code with the Jupyter extension.

## How the notebooks find their data

Each hands-on notebook opens with a bootstrap cell that works in Colab, on Databricks and
locally. It looks for the repository on disk first — a local clone or a Databricks Git folder
is found there — and clones it only if it is missing, which is what happens in Colab. The
next cell installs `requirements.txt` in Colab and Databricks, and assumes a prepared virtual
environment locally.

Either way you end up with `REPO_ROOT`, `DATA_DIR` and `RESULTS_DIR`, and every later cell
uses those rather than absolute paths.
