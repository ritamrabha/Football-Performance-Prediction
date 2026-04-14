# Football Player Performance Prediction

A production-quality machine learning project that predicts daily football player performance using training, recovery, and lifestyle data.

**Authors:** Ritam Rabha, Deepanshi, Ravinder Kaur

---

## Problem Statement

Football players' daily performance depends on a mix of training load, recovery, and lifestyle factors. This project builds a **binary classification model** to predict whether a player will have a **good (1)** or **bad (0)** performance day.

The dataset is intentionally realistic: **noisy features, weak correlations, and small size**, making model selection non-trivial.

---

## Dataset

| Property | Value |
|----------|-------|
| Records | 420 |
| Players | 7 |
| Time span | Feb 1 – Apr 1, 2026 |
| Features | 8 original + 4 engineered |
| Target | `performance_today` (0/1) |
| Class split | ~55% bad / ~45% good |

---

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `training_minutes` | Numeric | Daily training duration |
| `distance` | Numeric | Distance covered (km) |
| `sprint_count` | Numeric | Number of sprints |
| `sleep_hours` | Numeric | Sleep duration |
| `screen_time` | Numeric | Screen usage (hours) |
| `soreness` | Numeric | Muscle soreness (0–10) |
| `prev_performance` | Numeric | Previous day performance |
| `fatigue_index` | Engineered | Combines load + soreness + sleep |
| `recovery_score` | Engineered | Combines sleep + soreness + screen |
| `training_intensity` | Engineered | Effort per distance |
| `day_of_week` | Engineered | Temporal feature |

---

## Approach

### 1. Preprocessing
- Data cleaning and validation  
- Feature engineering (4 new features)  
- Standard scaling  
- Stratified train-test split (80/20)  

---

### 2. Exploratory Data Analysis
- Class imbalance visualization  
- Feature distributions across classes  
- Correlation heatmap (weak correlations observed)  
- Player-wise performance trends  
- Boxplots for feature separation  

---

### 3. Model Training

Five models trained using **GridSearchCV (5-fold CV, F1-score optimized):**

| Model | Performance |
|-------|-----------|
| Logistic Regression | Moderate |
| Decision Tree | Low |
| Random Forest | Moderate |
| Gradient Boosting | Good |
| HistGradientBoosting | **Best** |

---

### 4. Clustering Analysis (Bonus)

K-Means clustering on player-level aggregates:
- **Cluster 0:** Average performers (majority)
- **Cluster 1:** Slightly better performers  
- Separation is weak → dataset not strongly clusterable  

---

## Results

### Best Model: HistGradientBoosting

| Metric | Score |
|--------|-------|
| **Accuracy** | ~73.8% |
| **Precision** | ~0.70 |
| **Recall** | ~0.74 |
| **F1-Score** | ~0.72 |
| **ROC-AUC** | ~0.64 |

---

## Key Insights

1. **Recovery score is the strongest feature**  
2. **Soreness strongly affects performance**  
3. **Engineered features outperform raw ones**  
4. **Boosting models perform best**  
5. **Clustering shows weak separation**  

---

## Project Structure

```
football_ml_project/
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py
```

---

### Jupyter Notebook
```bash
cd notebooks/
jupyter notebook Football_Performance_Prediction.ipynb
```

---

### Load Saved Model
```python
import joblib
model = joblib.load('models/best_model.pkl')
```

---

## Technologies

- Python  
- scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  
- joblib  

---

## Future Improvements

- More data (players + time)  
- Time-series models (LSTM)  
- Add features (nutrition, stress)  
- Better clustering  
- Deploy as API  

---

*Built as an end-to-end ML project demonstrating real-world noisy data handling.*
