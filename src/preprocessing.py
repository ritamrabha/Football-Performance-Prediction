"""
Data Preprocessing Module
=========================
Handles loading, cleaning, validating, and splitting the football performance dataset.

Authors: Ritam Rabha, Deepanshi, Ravinder Kaur
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset and parse dates."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Run basic data quality checks and return a summary dict."""
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "class_distribution": df["performance_today"].value_counts().to_dict(),
        "class_ratio": df["performance_today"].value_counts(normalize=True).to_dict(),
        "n_players": df["player_id"].nunique(),
        "date_range": (df["date"].min(), df["date"].max()),
    }
    print(f"[INFO] Players: {summary['n_players']}")
    print(f"[INFO] Date range: {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")
    print(f"[INFO] Missing values: {sum(summary['missing_values'].values())}")
    print(f"[INFO] Duplicates: {summary['duplicates']}")
    print(f"[INFO] Class distribution: {summary['class_distribution']}")
    return summary


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle inconsistencies: drop duplicates, handle missing values, validate ranges."""
    df = df.copy()

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[INFO] Dropped {n_dropped} duplicate rows")

    # Fill missing numerics with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Filled {col} NaNs with median={median_val:.2f}")

    # Validate reasonable ranges and clip outliers
    range_constraints = {
        "training_minutes": (0, 200),
        "distance": (0, 15),
        "sprint_count": (0, 30),
        "sleep_hours": (0, 14),
        "screen_time": (0, 16),
        "soreness": (0, 10),
        "prev_performance": (0, 10),
    }
    for col, (lo, hi) in range_constraints.items():
        if col in df.columns:
            n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_clipped > 0:
                df[col] = df[col].clip(lo, hi)
                print(f"[INFO] Clipped {n_clipped} values in {col} to [{lo}, {hi}]")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for richer modelling."""
    df = df.copy()

    # Fatigue index: high training + high soreness + low sleep
    df["fatigue_index"] = (
        (df["training_minutes"] / df["training_minutes"].max())
        + (df["soreness"] / 10)
        - (df["sleep_hours"] / df["sleep_hours"].max())
    )

    # Recovery score: good sleep + low soreness + low screen time
    df["recovery_score"] = (
        (df["sleep_hours"] / df["sleep_hours"].max())
        + (1 - df["soreness"] / 10)
        + (1 - df["screen_time"] / df["screen_time"].max())
    )

    # Training intensity: minutes * sprints / distance (effort density)
    df["training_intensity"] = (
        df["training_minutes"] * df["sprint_count"]
    ) / (df["distance"] + 0.1)

    # Day of week from date (cyclical patterns)
    df["day_of_week"] = df["date"].dt.dayofweek

    print(f"[INFO] Engineered 4 new features: fatigue_index, recovery_score, training_intensity, day_of_week")
    return df


def prepare_splits(
    df: pd.DataFrame,
    target: str = "performance_today",
    test_size: float = 0.20,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Split into train/test and optionally scale features.
    Returns X_train, X_test, y_train, y_test, feature_names, scaler.
    """
    drop_cols = [target, "date", "player_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("[INFO] Applied StandardScaler to features")

    return X_train, X_test, y_train, y_test, feature_cols, scaler
