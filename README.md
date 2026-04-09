# Football Player Performance Prediction

A production-quality machine learning project that predicts daily football player performance using training, recovery, and lifestyle data.

**Authors:** Ritam Rabha, Deepanshi, Ravinder Kaur

---

## Problem Statement

Football players' daily performance depends on a complex interplay of training load, physical recovery, sleep quality, and lifestyle habits. This project builds a **supervised binary classification** system that predicts whether a player will have a **good (1)** or **bad (0)** performance day, using measurable daily metrics.

The dataset presents realistic challenges: **noisy features**, **weak correlations**, **temporal dependencies**, and **class imbalance** — making it a genuine test of ML methodology.

## Dataset

| Property | Value |
|----------|-------|
| Records | 420 |
| Players | 7 |
| Time span | Feb 1 – Apr 1, 2026 (59 days) |
| Features | 8 original + 4 engineered |
| Target | `performance_today` (0/1) |
| Class split | 55% bad / 45% good |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `training_minutes` | Numeric | Daily training session duration |
| `distance` | Numeric | Distance covered (km) |
| `sprint_count` | Numeric | Number of high-intensity sprints |
| `sleep_hours` | Numeric | Hours of sleep previous night |
| `screen_time` | Numeric | Daily screen time (hours) |
| `soreness` | Numeric | Self-reported muscle soreness (0–10) |
| `prev_performance` | Numeric | Previous day's performance score (0–10) |
| `fatigue_index` | Engineered | Composite of training load, soreness, sleep |
| `recovery_score` | Engineered | Composite of sleep, soreness, screen time |
| `training_intensity` | Engineered | Sprint-adjusted training effort per km |
| `day_of_week` | Engineered | Day of week (0=Mon, 6=Sun) |

## Approach

### 1. Preprocessing
- Data validation and range-checking
- Feature engineering (4 composite features)
- StandardScaler normalization
- Stratified 80/20 train-test split

### 2. Exploratory Data Analysis
- Class distribution analysis (slight imbalance confirmed)
- Feature distributions segmented by performance class
- Correlation heatmap revealing weak-to-moderate feature relationships
- Per-player time-series performance trends with 7-day rolling averages
- Box plot analysis for feature discriminative power

### 3. Model Training
Five classifiers trained with **GridSearchCV** (5-fold stratified CV, F1-optimized):

| Model | Best CV F1 |
|-------|-----------|
| Logistic Regression | 0.615 |
| Random Forest | 0.607 |
| Gradient Boosting | 0.604 |
| HistGradientBoosting | 0.593 |
| Decision Tree | 0.555 |

### 4. Clustering Analysis (Bonus)
K-Means clustering on aggregated player profiles identified distinct archetypes:
- **High Performers:** Balanced training and recovery metrics
- **Fatigue-Prone:** Lower performance despite similar training volume

## Results

### Best Model: Logistic Regression

| Metric | Score |
|--------|-------|
| **Accuracy** | 77.4% |
| **Precision** | 78.8% |
| **Recall** | 68.4% |
| **F1-Score** | 73.2% |
| **ROC-AUC** | 82.2% |

### Why Logistic Regression Wins
1. **Small dataset** (420 rows) — regularized linear models generalize better than deep ensembles
2. **Noisy features** — L2 regularization (C=0.1) effectively dampens noise
3. **Engineered features** created linearly separable signals
4. Tree-based models overfit on limited data despite hyperparameter tuning

## Key Insights

1. **Soreness is the #1 predictor** — high soreness strongly signals bad performance
2. **Engineered composite features** (`recovery_score`, `fatigue_index`) outperform raw metrics
3. **Previous performance** carries temporal momentum but isn't dominant
4. **Simple models beat complex ones** on small, noisy datasets
5. **Screen time and distance** are weak individual predictors (noise-like)
6. **Player clustering** reveals actionable recovery-intervention opportunities

## Project Structure

```
football_ml_project/
├── data/
│   └── football_dataset_refined3.csv
├── notebooks/
│   └── Football_Performance_Prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data loading, cleaning, feature engineering
│   ├── eda.py                 # Exploratory data analysis & plots
│   ├── modelling.py           # Model training, evaluation, visualization
│   └── clustering.py          # K-Means player clustering
├── models/
│   └── best_model.pkl         # Saved best model (Logistic Regression)
├── outputs/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── player_performance_trends.png
│   ├── boxplots_by_class.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── model_comparison_table.png
│   ├── model_comparison.csv
│   ├── optimal_k.png
│   ├── cluster_profiles.png
│   └── cluster_scatter.png
├── main.py                    # Full pipeline runner
├── requirements.txt
└── README.md
```

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py
```
This executes the entire workflow — preprocessing, EDA, model training, evaluation, clustering, and model saving — in under 2 minutes. All outputs are saved to `outputs/`.

### Jupyter Notebook
```bash
cd notebooks/
jupyter notebook Football_Performance_Prediction.ipynb
```
The notebook walks through every step interactively with explanations and visualizations.

### Load Saved Model
```python
import joblib
model = joblib.load('models/best_model.pkl')
# model.predict(X_new)  # pass scaled feature array
```

## Technologies

- **Python 3.10+**
- **scikit-learn 1.8** — preprocessing, models, evaluation
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **joblib** — model serialization

## Future Improvements

- Larger dataset with more players and longer time horizons
- Time-series models (LSTM, temporal cross-validation)
- Additional features: nutrition, match-day pressure, weather conditions
- Deployment as REST API for real-time prediction
- SHAP values for granular feature interpretation

---

*Built as a portfolio-grade ML project demonstrating end-to-end classification methodology.*
