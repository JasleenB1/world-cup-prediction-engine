# 2026 FIFA World Cup Prediction Engine

A data-driven World Cup forecasting system built using historical international football results, Elo ratings, recent form analysis, squad-strength adjustments, and Monte Carlo tournament simulation.

## Overview

This project predicts the 2026 FIFA World Cup by combining:

* Dynamic Elo ratings trained on 49,000+ international matches
* Recent form weighting
* Squad valuation adjustments
* Injury and availability impacts
* Host/travel effects
* 100,000 Monte Carlo tournament simulations

The model produces:

* Team strength ratings
* Group-stage forecasts
* Knockout-stage probabilities
* Champion probabilities
* Full 48-team rankings

---

## Methodology

### 1. Elo Rating System

Historical international matches dating back to 1872 are used to build team Elo ratings.

Features include:

* Goal-difference adjustments
* Tournament weighting
* Home-field advantage
* Rating updates after every match

### 2. Recent Form

Recent matches are weighted more heavily to capture current team strength while preserving long-term historical performance.

Final team rating:

Rating = 80% Career Elo + 20% Recent Form Elo + External Adjustments

### 3. External Adjustments

The model incorporates:

* Squad market values
* Injury impacts
* Host advantages
* Travel effects

### 4. Tournament Simulation

The complete World Cup is simulated 100,000 times.

For each simulation:

1. Group stage is played
2. Knockout bracket is generated
3. Round of 32 through Final are simulated
4. Tournament results are recorded

Probabilities are calculated from aggregate simulation outcomes.

---

## 2026 Predictions

### Top 10 Teams

| Rank | Team        |
| ---- | ----------- |
| 1    | Argentina   |
| 2    | Spain       |
| 3    | France      |
| 4    | Brazil      |
| 5    | England     |
| 6    | Portugal    |
| 7    | Germany     |
| 8    | Netherlands |
| 9    | Colombia    |
| 10   | Belgium     |

### Championship Favorites

| Team      | Champion Probability |
| --------- | -------------------- |
| Spain     | 11.8%                |
| Argentina | 10.4%                |
| France    | 9.8%                 |
| Brazil    | 8.4%                 |
| England   | 6.2%                 |

---

## Backtesting

The model was tested using the 2022 FIFA World Cup.

Metrics:

* Champion prediction probability
* Final ranking accuracy
* Top-4 overlap
* Mean Absolute Rank Error (MAE)

Backtesting was used to evaluate model modifications and remove features that did not improve predictive performance.

---

## Project Structure

```text
WorldCup/
├── predict_worldcup.py
├── backtest_2022.py
├── model_enhancements.py
├── external_factors.py
├── data/
├── output/
└── README.md
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run predictions:

```bash
python predict_worldcup.py
```

Run the 2022 backtest:

```bash
python backtest_2022.py
```

---

## Technologies

* Python
* Pandas
* NumPy
* Monte Carlo Simulation
* Elo Rating Systems
* Sports Analytics

---

## Disclaimer

This project is intended for analytical and educational purposes. Football matches contain significant randomness, and tournament outcomes are inherently uncertain.
