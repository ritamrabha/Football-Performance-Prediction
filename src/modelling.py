"""
Model Training & Evaluation Module
====================================
Trains, tunes, and evaluates multiple classifiers.

Authors: Ritam Rabha, Deepanshi, Ravinder Kaur
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Model Definitions with Hyperparameter Grids
# ---------------------------------------------------------------------------

def get_models_and_params():
    """Return dict of {name: (estimator, param_grid)}."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [3, 5, 8, 12, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 3, 5],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 3],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        ),
        "XGBoost (HistGB)": (
            HistGradientBoostingClassifier(
                random_state=42, max_iter=200,
            ),
            {
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_leaf": [5, 10, 20],
                "max_leaf_nodes": [15, 31, None],
            },
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(X_train, y_train, cv_folds: int = 5):
    """
    Train all models with GridSearchCV.
    Returns dict of {name: best_estimator}.
    """
    models_params = get_models_and_params()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trained = {}

    for name, (estimator, params) in models_params.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        grid = GridSearchCV(
            estimator, params, cv=cv, scoring="f1",
            n_jobs=-1, verbose=0, refit=True,
        )
        grid.fit(X_train, y_train)
        trained[name] = grid.best_estimator_

        print(f"  Best params : {grid.best_params_}")
        print(f"  Best CV F1  : {grid.best_score_:.4f}")

    return trained


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Evaluate a single model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    return metrics, y_pred, y_proba


def evaluate_all(trained_models: dict, X_test, y_test):
    """Evaluate all models and return comparison DataFrame + predictions."""
    results = []
    predictions = {}
    probas = {}

    for name, model in trained_models.items():
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        predictions[name] = y_pred
        probas[name] = y_proba
        print(f"\n[{name}]")
        print(classification_report(y_test, y_pred, target_names=["Bad (0)", "Good (1)"]))

    comparison_df = pd.DataFrame(results).set_index("Model")
    comparison_df = comparison_df.sort_values("F1-Score", ascending=False)
    return comparison_df, predictions, probas


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrices(trained_models, predictions, y_test, save_dir):
    """Plot confusion matrices for all models in a grid."""
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, (name, _) in enumerate(trained_models.items()):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"],
            annot_kws={"size": 14}, linewidths=0.5,
        )
        axes[i].set_title(name, fontweight="bold", fontsize=11)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_roc_curves(trained_models, probas, y_test, save_dir):
    """Plot ROC curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(trained_models)))

    for i, (name, _) in enumerate(trained_models.items()):
        if probas[name] is not None:
            fpr, tpr, _ = roc_curve(y_test, probas[name])
            auc = roc_auc_score(y_test, probas[name])
            ax.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_feature_importance(trained_models, feature_names, save_dir):
    """Plot feature importance for tree-based models."""
    tree_models = {
        k: v for k, v in trained_models.items()
        if hasattr(v, "feature_importances_")
    }
    if not tree_models:
        print("[WARN] No tree-based models found for feature importance.")
        return None

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for i, (name, model) in enumerate(tree_models.items()):
        importances = model.feature_importances_
        idx = np.argsort(importances)

        axes[i].barh(
            range(len(idx)), importances[idx],
            color=plt.cm.viridis(importances[idx] / importances.max()),
            edgecolor="white", linewidth=0.5,
        )
        axes[i].set_yticks(range(len(idx)))
        axes[i].set_yticklabels([feature_names[j] for j in idx], fontsize=9)
        axes[i].set_title(name, fontweight="bold", fontsize=11)
        axes[i].set_xlabel("Importance")

    plt.suptitle("Feature Importance (Tree-Based Models)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_comparison_table(comparison_df, save_dir):
    """Save the comparison table as an image."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    rounded = comparison_df.round(4)
    table = ax.table(
        cellText=rounded.values,
        colLabels=rounded.columns,
        rowLabels=rounded.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Highlight best model row
    best_idx = 0
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[0] == best_idx + 1:
            cell.set_facecolor("#d5f4e6")
        if key[1] == -1:
            cell.set_text_props(fontweight="bold")

    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


# ---------------------------------------------------------------------------
# Model Saving
# ---------------------------------------------------------------------------

def save_best_model(trained_models, comparison_df, model_dir="models"):
    """Save the best model (by F1-Score) using joblib."""
    os.makedirs(model_dir, exist_ok=True)
    best_name = comparison_df.index[0]
    best_model = trained_models[best_name]
    path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, path)
    print(f"\n[SAVED] Best model ({best_name}) -> {path}")
    return path, best_name
