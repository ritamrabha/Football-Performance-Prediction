"""
Clustering Analysis Module
============================
K-Means clustering of players based on behavioural patterns.

Authors: Ritam Rabha, Deepanshi, Ravinder Kaur
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


sns.set_theme(style="whitegrid", font_scale=1.1)


def aggregate_player_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily records into per-player behavioural profiles."""
    agg = df.groupby("player_id").agg(
        avg_training=("training_minutes", "mean"),
        avg_distance=("distance", "mean"),
        avg_sprints=("sprint_count", "mean"),
        avg_sleep=("sleep_hours", "mean"),
        avg_screen=("screen_time", "mean"),
        avg_soreness=("soreness", "mean"),
        avg_prev_perf=("prev_performance", "mean"),
        perf_rate=("performance_today", "mean"),       # proportion of good days
        perf_std=("performance_today", "std"),          # consistency measure
        total_records=("performance_today", "count"),
    ).reset_index()

    agg["perf_std"] = agg["perf_std"].fillna(0)
    print(f"[INFO] Aggregated profiles for {len(agg)} players")
    return agg


def find_optimal_k(profiles: pd.DataFrame, features: list, max_k: int = 6, save_dir: str = "outputs"):
    """Elbow method + silhouette analysis to pick k."""
    scaler = StandardScaler()
    X = scaler.fit_transform(profiles[features])

    inertias = []
    sil_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(list(K_range), inertias, "o-", color="#3498db", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method", fontweight="bold")

    axes[1].plot(list(K_range), sil_scores, "s-", color="#e74c3c", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis", fontweight="bold")

    plt.suptitle("Optimal k Selection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "optimal_k.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")

    best_k = list(K_range)[np.argmax(sil_scores)]
    print(f"[INFO] Best k by silhouette: {best_k} (score={max(sil_scores):.3f})")
    return best_k, X, scaler


def run_kmeans(profiles: pd.DataFrame, X_scaled, k: int):
    """Fit final K-Means and attach labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    profiles = profiles.copy()
    profiles["cluster"] = km.fit_predict(X_scaled)
    return profiles, km


def interpret_clusters(profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate cluster summary statistics and assign labels."""
    summary = profiles.groupby("cluster").agg(
        n_players=("player_id", "count"),
        avg_perf_rate=("perf_rate", "mean"),
        avg_soreness=("avg_soreness", "mean"),
        avg_sleep=("avg_sleep", "mean"),
        avg_training=("avg_training", "mean"),
        perf_consistency=("perf_std", "mean"),
    ).round(3)

    # Assign interpretive labels
    labels = []
    for _, row in summary.iterrows():
        if row["avg_perf_rate"] >= summary["avg_perf_rate"].max() * 0.9:
            labels.append("High Performers")
        elif row["avg_soreness"] >= summary["avg_soreness"].max() * 0.9:
            labels.append("Fatigue-Prone")
        elif row["perf_consistency"] >= summary["perf_consistency"].max() * 0.85:
            labels.append("Inconsistent")
        else:
            labels.append("Average / Balanced")
    summary["label"] = labels
    print("\n[CLUSTER SUMMARY]")
    print(summary.to_string())
    return summary


def plot_cluster_profiles(profiles: pd.DataFrame, save_dir: str):
    """Radar / bar chart of cluster centroids."""
    cluster_summary = profiles.groupby("cluster")[
        ["avg_training", "avg_distance", "avg_sprints", "avg_sleep",
         "avg_screen", "avg_soreness", "perf_rate"]
    ].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_summary.T.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", linewidth=0.8)
    ax.set_title("Cluster Profiles — Average Feature Values", fontweight="bold", fontsize=13)
    ax.set_ylabel("Value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Cluster", loc="upper right")

    plt.tight_layout()
    path = os.path.join(save_dir, "cluster_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_cluster_scatter(profiles: pd.DataFrame, save_dir: str):
    """Scatter plot: performance rate vs soreness, coloured by cluster."""
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        profiles["avg_soreness"], profiles["perf_rate"],
        c=profiles["cluster"], cmap="Set1", s=120, edgecolors="black", linewidths=0.8,
        alpha=0.85,
    )
    for _, row in profiles.iterrows():
        ax.annotate(
            f"P{int(row['player_id'])}", (row["avg_soreness"], row["perf_rate"]),
            fontsize=9, fontweight="bold", ha="center", va="bottom",
        )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("Avg Soreness", fontsize=12)
    ax.set_ylabel("Performance Rate (% Good Days)", fontsize=12)
    ax.set_title("Player Clusters: Soreness vs Performance", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "cluster_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def run_full_clustering(df: pd.DataFrame, save_dir: str = "outputs") -> dict:
    """Execute the complete clustering pipeline."""
    os.makedirs(save_dir, exist_ok=True)

    profiles = aggregate_player_profiles(df)

    cluster_features = [
        "avg_training", "avg_distance", "avg_sprints", "avg_sleep",
        "avg_screen", "avg_soreness", "avg_prev_perf", "perf_rate", "perf_std",
    ]

    best_k, X_scaled, scaler = find_optimal_k(profiles, cluster_features, save_dir=save_dir)
    profiles, km_model = run_kmeans(profiles, X_scaled, best_k)
    summary = interpret_clusters(profiles)

    paths = {
        "cluster_profiles": plot_cluster_profiles(profiles, save_dir),
        "cluster_scatter": plot_cluster_scatter(profiles, save_dir),
    }

    return {
        "profiles": profiles,
        "summary": summary,
        "model": km_model,
        "plots": paths,
    }
