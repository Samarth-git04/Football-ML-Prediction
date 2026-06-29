# Football ML Prediction

A machine learning project for predicting football match outcomes using historical match results, team form, Pi-style ratings, match statistics, and FIFAIndex / FC player ratings.

The project focuses on **3-way football match prediction**:

- `H` = home win
- `D` = draw
- `A` = away win

The main notebook builds a player and position-aware prediction pipeline using FIFAIndex data from **2005 to 2025**, historical match data, rolling team form, and model comparison across several machine learning algorithms.

---

## Project Overview

Football match prediction is difficult because outcomes depend on both team-level history and player-level quality. This project combines:

- historical match results
- rolling team form
- home and away Pi-style ratings
- head-to-head history
- FIFAIndex / FC player ratings
- position-group strength by goalkeeper, defence, midfield and attack
- custom player-point aggregates
- probability-based model evaluation

The goal is not only to predict the most likely outcome, but to produce reliable class probabilities for home win, draw and away win.

---

## Main Notebook

The main modelling notebook is:

```text
player_position_match_prediction_time_cv_2026.ipynb
```

It performs the full workflow:

1. Load match data
2. Load FIFAIndex / FC player data
3. Clean and standardise team, nation and player names
4. Create match outcome labels
5. Build rolling team-form features
6. Build Pi-style dynamic rating features
7. Aggregate FIFA player ratings by team/nation and position group
8. Merge match-level and player-level features
9. Train multiple models
10. Apply chronological cross-validation
11. Apply probability calibration
12. Evaluate using probability and classification metrics
13. Export model artefacts and prediction outputs

---

## Data Sources

This project is designed to work with local datasets, including:

### Match Data

Supported match data formats include:

```text
data/raw_data/results.csv
data/raw_data/international/results.csv
raw_data/database.sqlite
```

The notebook also supports common football-data style columns such as:

```text
Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HS, AS, HST, AST
```

### FIFAIndex / FC Player Data

Expected FIFAIndex data structure:

```text
data/raw_data/fifaindex/csv/*.csv
data/raw_data/fifaindex/parquet/*.parquet
```

The player data is expected to include fields such as:

```text
player
team
nation
position
ovr
pot
game_year
```

The notebook can aggregate ratings by either club/team name or nation name.

---

## Feature Engineering

The model uses only pre-match features to avoid data leakage.

### Match History Features

Rolling team-form features are created using previous matches only:

- points average
- goals for
- goals against
- goal difference
- win rate
- draw rate
- loss rate
- match availability count
- days since previous match

Rolling windows used:

```text
last 3 matches
last 5 matches
last 10 matches
```

### Pi-Style Rating Features

The project includes a simplified Pi-style dynamic rating system with separate home and away ratings.

Example features:

```text
home_pi_home_pre
home_pi_away_pre
away_pi_home_pre
away_pi_away_pre
pi_expected_goal_diff
pi_home_advantage_rating_diff
pi_away_travel_rating_diff
pi_rating_diff
```

The key idea is:

```text
expected goal difference = home team's home Pi rating - away team's away Pi rating
```

Ratings are updated after each match based on the prediction error between expected and actual goal difference.

### FIFAIndex / Player Features

FIFAIndex player data is aggregated by team or nation and by position group:

```text
GK  = goalkeeper
DEF = defender
MID = midfielder
ATT = attacker
```

Example feature types:

- average overall rating
- average potential rating
- top 3 player overall rating
- average player points
- top 3 player points
- player counts by position group
- home minus away FIFA strength differences

---

## Models Used

The notebook compares the following models:

- Logistic Regression
- Random Forest
- Neural Network / MLP
- XGBoost
- CatBoost

Each model is evaluated using both raw and calibrated probabilities where applicable.

---

## Validation Strategy

The project uses chronological validation instead of random splitting.

This is important because football prediction is a time-based problem. Future matches must not leak into the training data.

Current split:

```text
Training:    matches up to 2023
Validation:  2024-2025
Prediction:  2026
```

Cross-validation uses expanding yearly windows inside the training period.

Example:

```text
Fold 1: train up to 2017, validate 2018
Fold 2: train up to 2018, validate 2019
Fold 3: train up to 2019, validate 2020
...
Fold 6: train up to 2022, validate 2023
```

The 2026 data is held out from training, validation, cross-validation and calibration.

---

## Evaluation Metrics

The project evaluates both classification quality and probability quality.

Metrics include:

- Accuracy
- Balanced accuracy
- Macro F1-score
- ROC-AUC one-vs-rest
- Log loss
- Brier score
- Multiclass Brier score
- Ranked Probability Score, also known as RPS
- Expected Calibration Error, also known as ECE

Probability metrics are especially important because the model is intended to output probabilities, not just hard class labels.

---

## Current Results

In the latest validation run on the 2024-2025 period, the strongest model was CatBoost.

| Model | Calibrated | Log Loss | RPS | Accuracy | ROC-AUC OVR |
|---|---:|---:|---:|---:|---:|
| CatBoost | No | 0.8550 | 0.1648 | 0.6082 | 0.7517 |
| XGBoost | No | 0.8560 | 0.1652 | 0.6028 | 0.7507 |
| CatBoost | Yes | 0.8685 | 0.1666 | 0.6064 | 0.7462 |
| XGBoost | Yes | 0.8700 | 0.1666 | 0.6005 | 0.7465 |

Lower is better for:

```text
log loss
Brier score
RPS
ECE
```

Higher is better for:

```text
accuracy
balanced accuracy
macro F1
ROC-AUC
```

---

## Project Structure

Expected local structure:

```text
Football-ML-Prediction/
│
├── player_position_match_prediction_time_cv_2026.ipynb
│
├── data/
│   └── raw_data/
│       ├── results.csv
│       ├── international/
│       │   └── results.csv
│       ├── database.sqlite
│       └── fifaindex/
│           ├── csv/
│           └── parquet/
│
├── models/
│   └── player_position_match_model/
│
├── .gitignore
└── README.md
```

Large data files and generated model outputs should usually stay local and should not be committed to GitHub.

---

## Installation

Create and activate a Python environment:

```bash
conda create -n football_ml python=3.10
conda activate football_ml
```

Install the main dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib jupyter
pip install xgboost catboost
pip install pyarrow fastparquet
```

Optional dependencies for data collection and scraping:

```bash
pip install requests beautifulsoup4 tqdm
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Samarth-git04/Football-ML-Prediction.git
cd Football-ML-Prediction
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
player_position_match_prediction_time_cv_2026.ipynb
```

Check the configuration cell near the top of the notebook.

Important path settings:

```python
MATCH_FILE_OVERRIDE = None
FIFAINDEX_DIR_OVERRIDE = None
FIFA_JOIN_MODE = "auto"
FIFA_YEAR_MODE = "season_release"
```

Use `MATCH_FILE_OVERRIDE` if the notebook does not automatically find the correct match file.

Example:

```python
MATCH_FILE_OVERRIDE = Path("data/raw_data/results.csv")
```

Use `FIFAINDEX_DIR_OVERRIDE` if the FIFAIndex folder is in a different location.

Example:

```python
FIFAINDEX_DIR_OVERRIDE = Path("data/raw_data/fifaindex")
```

---

## Outputs

The notebook saves model comparison and prediction artefacts under:

```text
models/player_position_match_model/
```

Typical outputs include:

```text
cv_results_player_position_models.csv
cv_summary_player_position_models.csv
cv_leaderboard_player_position_models.csv
oof_predictions_player_position_models.csv
```

Depending on the notebook run, it may also export holdout predictions, final model artefacts and 2026 prediction files.

---

## Why Time-Based Cross-Validation?

Random cross-validation is not suitable for football prediction because it can leak future information into earlier training folds.

For example, a random split could train on 2024 matches and validate on 2018 matches. That would not reflect real forecasting.

This project therefore uses chronological validation, where every validation fold occurs after its training data.

---

## Limitations

This project is still experimental and has several limitations:

- FIFAIndex ratings are yearly, not match-specific.
- Actual lineups and injuries are not fully modelled.
- International team-name matching can reduce FIFA join coverage.
- Some older matches have limited match-stat detail.
- Draw prediction remains difficult because draws are less frequent and harder to separate.
- Model performance can vary depending on the match dataset used.
- Predictions should not be treated as betting advice.

---

## Future Improvements

Planned improvements include:

- Better country and club alias mapping
- Separate models for club and international matches
- Elo and Glicko rating features
- Player availability and starting XI features
- Injury and suspension data
- Tournament-stage and match-importance features
- Hyperparameter tuning with Optuna
- Ensemble model using CatBoost, XGBoost and calibrated logistic regression
- Streamlit dashboard for upcoming fixture predictions

---

## Tech Stack

Main tools used:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- CatBoost
- matplotlib
- joblib
- Jupyter Notebook

---

## Author

Built by Samarth Gohel as part of a football match prediction machine learning project.
