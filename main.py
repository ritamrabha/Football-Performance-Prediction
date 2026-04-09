"""
main.py — Full ML Pipeline Runner
===================================
Runs the entire project pipeline end-to-end:
  1. Data loading & preprocessing
  2. Exploratory data analysis
  3. Model training & evaluation
  4. Clustering analysis
  5. Model saving & report generation

Usage:
    python main.py

Authors: Ritam Rabha, Deepanshi, Ravinder Kaur
"""

import os
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure src is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_data, inspect_data, clean_data, engineer_features, prepare_splits
from src.eda import run_full_eda
from src.modelling import (
    train_models, evaluate_all,
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_comparison_table,
    save_best_model,
)
from src.clustering import run_full_clustering


def main():
    print("=" * 70)
    print("  FOOTBALL PLAYER PERFORMANCE PREDICTION PIPELINE")
    print("  Authors: Ritam Rabha, Deepanshi, Ravinder Kaur")
    print("=" * 70)

    DATA_PATH = os.path.join("data", "football_dataset_refined3.csv")
    OUTPUT_DIR = "outputs"
    MODEL_DIR = "models"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. DATA LOADING & PREPROCESSING
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 70)

    df = load_data(DATA_PATH)
    summary = inspect_data(df)
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_splits(df)

    print(f"\nFeatures used ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # 2. EXPLORATORY DATA ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    eda_results = run_full_eda(df, save_dir=OUTPUT_DIR)

    print("\n--- KEY INSIGHTS ---")
    for i, insight in enumerate(eda_results["insights"], 1):
        print(f"  {i}. {insight}")

    # ------------------------------------------------------------------
    # 3. MODEL TRAINING
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 3: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 70)

    trained_models = train_models(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. MODEL EVALUATION & COMPARISON
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 4: MODEL EVALUATION & COMPARISON")
    print("=" * 70)

    comparison_df, predictions, probas = evaluate_all(trained_models, X_test, y_test)

    print("\n--- COMPARISON TABLE ---")
    print(comparison_df.round(4).to_string())

    best_name = comparison_df.index[0]
    print(f"\n>>> BEST MODEL: {best_name} (F1={comparison_df.loc[best_name, 'F1-Score']:.4f})")

    # ------------------------------------------------------------------
    # 5. VISUALIZATIONS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_confusion_matrices(trained_models, predictions, y_test, OUTPUT_DIR)
    plot_roc_curves(trained_models, probas, y_test, OUTPUT_DIR)
    plot_feature_importance(trained_models, feature_names, OUTPUT_DIR)
    plot_comparison_table(comparison_df, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 6. CLUSTERING ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 6: CLUSTERING ANALYSIS (BONUS)")
    print("=" * 70)

    cluster_results = run_full_clustering(df, save_dir=OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 7. SAVE BEST MODEL
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 7: SAVING BEST MODEL")
    print("=" * 70)

    model_path, best_name = save_best_model(trained_models, comparison_df, model_dir=MODEL_DIR)

    # ------------------------------------------------------------------
    # 8. SAVE COMPARISON CSV
    # ------------------------------------------------------------------
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
    print(f"[SAVED] {os.path.join(OUTPUT_DIR, 'model_comparison.csv')}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Best Model     : {best_name}")
    print(f"  Saved to       : {model_path}")
    print(f"  Outputs in     : {OUTPUT_DIR}/")
    print(f"  Total plots    : {len(os.listdir(OUTPUT_DIR))} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
