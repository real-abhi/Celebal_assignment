"""End-to-end ML pipeline for Tesla delivery regression.

The pipeline uses the provided 2015-2025 Tesla delivery dataset to predict
Estimated_Deliveries from pricing, production, vehicle, region, time, and
charging-infrastructure signals.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("tesla_deliveries_dataset_2015_2025.csv")
OUTPUT_DIR = Path("outputs")
TARGET = "Estimated_Deliveries"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and lightly validate the source dataset."""
    df = pd.read_csv(path)
    required = {
        "Year",
        "Month",
        "Region",
        "Model",
        TARGET,
        "Production_Units",
        "Avg_Price_USD",
        "Battery_Capacity_kWh",
        "Range_km",
        "CO2_Saved_tons",
        "Source_Type",
        "Charging_Stations",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready time and business features."""
    featured = df.copy()
    featured["Date"] = pd.to_datetime(
        dict(year=featured["Year"], month=featured["Month"], day=1)
    )
    featured["Quarter"] = featured["Date"].dt.quarter
    featured["Month_Sin"] = np.sin(2 * np.pi * featured["Month"] / 12)
    featured["Month_Cos"] = np.cos(2 * np.pi * featured["Month"] / 12)
    featured["Price_Per_kWh"] = (
        featured["Avg_Price_USD"] / featured["Battery_Capacity_kWh"]
    )
    featured["Range_Per_kWh"] = featured["Range_km"] / featured["Battery_Capacity_kWh"]
    featured["Stations_Per_1000_Production"] = (
        featured["Charging_Stations"] / featured["Production_Units"].clip(lower=1)
    ) * 1000
    return featured.sort_values("Date").reset_index(drop=True)


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical predictors."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return common regression metrics for a fitted model."""
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    return {
        "model": name,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": rmse,
        "r2": r2_score(y_test, predictions),
    }


def save_eda(df: pd.DataFrame) -> None:
    """Save compact EDA outputs used in the notebook/report."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    df.describe(include="all").transpose().to_csv(OUTPUT_DIR / "eda_summary.csv")

    monthly = (
        df.groupby("Date", as_index=False)[[TARGET, "Avg_Price_USD"]]
        .mean()
        .sort_values("Date")
    )
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(monthly["Date"], monthly[TARGET], color="#2563eb", label="Deliveries")
    ax1.set_ylabel("Average deliveries", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(monthly["Date"], monthly["Avg_Price_USD"], color="#dc2626", label="Price")
    ax2.set_ylabel("Average price (USD)", color="#dc2626")
    ax1.set_title("Tesla Deliveries and Average Price Over Time")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deliveries_price_trend.png", dpi=160)
    plt.close(fig)

    corr_cols = [
        TARGET,
        "Production_Units",
        "Avg_Price_USD",
        "Battery_Capacity_kWh",
        "Range_km",
        "CO2_Saved_tons",
        "Charging_Stations",
        "Price_Per_kWh",
        "Range_Per_kWh",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[corr_cols].corr(), cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Numeric Feature Correlations")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    by_model_region = (
        df.groupby(["Model", "Region"], as_index=False)[TARGET]
        .mean()
        .sort_values(TARGET, ascending=False)
    )
    by_model_region.to_csv(OUTPUT_DIR / "deliveries_by_model_region.csv", index=False)


def train_pipeline() -> tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    """Train, tune, evaluate, and persist the best delivery model."""
    raw = load_data()
    df = engineer_features(raw)
    save_eda(df)

    numeric_features = [
        "Year",
        "Month",
        "Quarter",
        "Month_Sin",
        "Month_Cos",
        "Production_Units",
        "Avg_Price_USD",
        "Battery_Capacity_kWh",
        "Range_km",
        "CO2_Saved_tons",
        "Charging_Stations",
        "Price_Per_kWh",
        "Range_Per_kWh",
        "Stations_Per_1000_Production",
    ]
    categorical_features = ["Region", "Model", "Source_Type"]
    feature_columns = numeric_features + categorical_features

    split_date = df["Date"].quantile(0.8)
    train_mask = df["Date"] <= split_date
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    candidates = {
        "Ridge": Ridge(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    fitted_models: dict[str, Pipeline] = {}
    metrics = []
    for name, estimator in candidates.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        pipe.fit(x_train, y_train)
        fitted_models[name] = pipe
        metrics.append(evaluate_model(name, pipe, x_test, y_test))

    metrics_df = pd.DataFrame(metrics).sort_values("rmse")
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    tuned_rf_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    rf_param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [8, 14, None],
        "model__min_samples_leaf": [1, 3],
    }
    rf_search = GridSearchCV(
        tuned_rf_pipe,
        param_grid=rf_param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rf_search.fit(x_train, y_train)

    tuned_ridge_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge()),
        ]
    )
    ridge_search = GridSearchCV(
        tuned_ridge_pipe,
        param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ridge_search.fit(x_train, y_train)

    tuned_rows = []
    for tuned_name, search in {
        "Tuned Random Forest": rf_search,
        "Tuned Ridge": ridge_search,
    }.items():
        tuned_metrics = evaluate_model(tuned_name, search.best_estimator_, x_test, y_test)
        tuned_metrics["best_params"] = search.best_params_
        tuned_rows.append(tuned_metrics)

    final_metrics = pd.concat(
        [metrics_df, pd.DataFrame(tuned_rows)], ignore_index=True
    ).sort_values("rmse")
    final_metrics.to_csv(OUTPUT_DIR / "final_model_metrics.csv", index=False)

    best_model_name = final_metrics.iloc[0]["model"]
    tuned_lookup = {
        "Tuned Random Forest": rf_search.best_estimator_,
        "Tuned Ridge": ridge_search.best_estimator_,
    }
    if best_model_name in tuned_lookup:
        tuned_model = tuned_lookup[best_model_name]
    else:
        tuned_model = fitted_models[best_model_name]

    predictions = test_df[
        ["Date", "Region", "Model", "Avg_Price_USD", "Production_Units", TARGET]
    ].copy()
    predictions["Predicted_Deliveries"] = tuned_model.predict(x_test)
    predictions["Residual"] = predictions[TARGET] - predictions["Predicted_Deliveries"]
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predictions[TARGET], predictions["Predicted_Deliveries"], alpha=0.7)
    low = min(predictions[TARGET].min(), predictions["Predicted_Deliveries"].min())
    high = max(predictions[TARGET].max(), predictions["Predicted_Deliveries"].max())
    ax.plot([low, high], [low, high], color="#dc2626", linewidth=2)
    ax.set_xlabel("Actual deliveries")
    ax.set_ylabel("Predicted deliveries")
    ax.set_title("Actual vs Predicted Deliveries")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    joblib.dump(tuned_model, OUTPUT_DIR / "tesla_delivery_model.joblib")
    return final_metrics, tuned_model, predictions


if __name__ == "__main__":
    results, _, preds = train_pipeline()
    print("Final model metrics:")
    print(results.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
    print("\nSample predictions:")
    print(preds.head().to_string(index=False))
