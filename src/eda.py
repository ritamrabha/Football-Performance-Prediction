"""
Exploratory Data Analysis Module
=================================
Generates all EDA visualizations and returns textual insights.

Authors: Ritam Rabha, Deepanshi, Ravinder Kaur
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#e74c3c", "#2ecc71"]  # bad=red, good=green


def plot_class_distribution(df: pd.DataFrame, save_dir: str):
    """Bar chart of target variable class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    counts = df["performance_today"].value_counts().sort_index()
    axes[0].bar(["Bad (0)", "Good (1)"], counts.values, color=PALETTE, edgecolor="black", linewidth=0.8)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold", fontsize=13)
    axes[0].set_title("Class Distribution (Count)", fontweight="bold")
    axes[0].set_ylabel("Number of Records")

    # Percentage pie
    axes[1].pie(
        counts.values,
        labels=["Bad (0)", "Good (1)"],
        colors=PALETTE,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class Distribution (%)", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_feature_distributions(df: pd.DataFrame, save_dir: str):
    """Histograms of all numeric features, coloured by target."""
    numeric_cols = [
        "training_minutes", "distance", "sprint_count", "sleep_hours",
        "screen_time", "soreness", "prev_performance",
    ]
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        for label, color in zip([0, 1], PALETTE):
            subset = df[df["performance_today"] == label][col]
            axes[i].hist(subset, bins=20, alpha=0.6, color=color,
                         label=f"{'Bad' if label == 0 else 'Good'}", edgecolor="white")
        axes[i].set_title(col, fontweight="bold")
        axes[i].legend(fontsize=9)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Performance Class", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str):
    """Correlation heatmap of all numeric features + target."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_player_performance_trends(df: pd.DataFrame, save_dir: str):
    """Time-series of daily performance per player using rolling average."""
    players = sorted(df["player_id"].unique())
    n_players = len(players)
    fig, axes = plt.subplots(n_players, 1, figsize=(14, 3 * n_players), sharex=True)
    if n_players == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_players))

    for i, pid in enumerate(players):
        pdf = df[df["player_id"] == pid].sort_values("date")
        axes[i].plot(pdf["date"], pdf["performance_today"], "o", alpha=0.35, color=colors[i], markersize=4)
        # Rolling 7-day mean
        rolling = pdf.set_index("date")["performance_today"].rolling("7D").mean()
        axes[i].plot(rolling.index, rolling.values, "-", color=colors[i], linewidth=2, label="7-day rolling avg")
        axes[i].set_ylabel("Perf.")
        axes[i].set_title(f"Player {pid}", fontweight="bold", fontsize=11, loc="left")
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].legend(loc="upper right", fontsize=8)

    plt.suptitle("Player Performance Trends Over Time", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "player_performance_trends.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_boxplots_by_class(df: pd.DataFrame, save_dir: str):
    """Box plots of key features grouped by target class."""
    features = ["training_minutes", "distance", "sprint_count", "sleep_hours",
                 "screen_time", "soreness", "prev_performance"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(data=df, x="performance_today", y=col, ax=axes[i],
                    palette=PALETTE, width=0.5, fliersize=3)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_xticklabels(["Bad (0)", "Good (1)"])

    axes[-1].set_visible(False)
    plt.suptitle("Feature Distributions by Performance Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "boxplots_by_class.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def generate_insights(df: pd.DataFrame) -> list:
    """Return a list of textual EDA insights."""
    insights = []
    corr_with_target = df.select_dtypes(include=[np.number]).corr()["performance_today"].drop("performance_today")
    top_pos = corr_with_target.nlargest(3)
    top_neg = corr_with_target.nsmallest(3)

    insights.append(
        f"Dataset contains {df.shape[0]} records across {df['player_id'].nunique()} players "
        f"over {(df['date'].max() - df['date'].min()).days} days."
    )

    class_pct = df["performance_today"].value_counts(normalize=True)
    insights.append(
        f"Class distribution is slightly imbalanced: {class_pct.get(1, 0)*100:.1f}% good days vs "
        f"{class_pct.get(0, 0)*100:.1f}% bad days."
    )

    insights.append(
        f"Top positively correlated features with performance: "
        + ", ".join([f"{k} ({v:+.3f})" for k, v in top_pos.items()])
    )
    insights.append(
        f"Top negatively correlated features with performance: "
        + ", ".join([f"{k} ({v:+.3f})" for k, v in top_neg.items()])
    )

    # Per-player performance variation
    player_perf = df.groupby("player_id")["performance_today"].mean()
    best = player_perf.idxmax()
    worst = player_perf.idxmin()
    insights.append(
        f"Player {best} has the highest avg performance ({player_perf[best]:.2f}), "
        f"while Player {worst} has the lowest ({player_perf[worst]:.2f})."
    )

    insights.append(
        "Most features show weak-to-moderate correlations with the target, "
        "confirming that the dataset contains noise and no single dominant predictor."
    )

    return insights


def run_full_eda(df: pd.DataFrame, save_dir: str = "outputs") -> dict:
    """Run all EDA analyses and return paths + insights."""
    os.makedirs(save_dir, exist_ok=True)

    paths = {
        "class_dist": plot_class_distribution(df, save_dir),
        "feat_dist": plot_feature_distributions(df, save_dir),
        "corr_heatmap": plot_correlation_heatmap(df, save_dir),
        "player_trends": plot_player_performance_trends(df, save_dir),
        "boxplots": plot_boxplots_by_class(df, save_dir),
    }
    insights = generate_insights(df)
    return {"plots": paths, "insights": insights}
