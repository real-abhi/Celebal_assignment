from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Country-data.csv"
OUTPUT_DIR = ROOT / "outputs" / "customer_intelligence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "income",
    "inflation",
    "life_expec",
    "total_fer",
    "gdpp",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df[FEATURES].isna().any().any():
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    return df


def add_actual_trade_values(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["exports_value"] = enriched["exports"] * enriched["gdpp"] / 100
    enriched["imports_value"] = enriched["imports"] * enriched["gdpp"] / 100
    enriched["health_value"] = enriched["health"] * enriched["gdpp"] / 100
    return enriched


def save_eda(df: pd.DataFrame) -> None:
    df[FEATURES].describe().T.to_csv(OUTPUT_DIR / "eda_summary.csv")
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Country Indicator Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()


def scaled_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[FEATURES])
    return x_scaled, scaler


def evaluate_kmeans(x_scaled: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(x_scaled, labels),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "kmeans_k_selection.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=results, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("K-Means Elbow Curve")
    sns.lineplot(data=results, x="k", y="silhouette_score", marker="o", ax=axes[1])
    axes[1].set_title("K-Means Silhouette Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_selection.png", dpi=180)
    plt.close()
    return results


def fit_kmeans(df: pd.DataFrame, x_scaled: np.ndarray, n_clusters: int = 3) -> tuple[pd.DataFrame, KMeans]:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
    clustered = df.copy()
    clustered["kmeans_cluster"] = model.fit_predict(x_scaled)
    profiles = clustered.groupby("kmeans_cluster")[FEATURES].mean()

    risk_score = (
        profiles["child_mort"].rank(ascending=False)
        + profiles["total_fer"].rank(ascending=False)
        + profiles["income"].rank(ascending=True)
        + profiles["gdpp"].rank(ascending=True)
        + profiles["life_expec"].rank(ascending=True)
    )
    ordered_clusters = risk_score.sort_values(ascending=True).index.tolist()
    label_map = {
        ordered_clusters[0]: "High Priority",
        ordered_clusters[1]: "Developing",
        ordered_clusters[2]: "Stable / Developed",
    }
    clustered["segment"] = clustered["kmeans_cluster"].map(label_map)
    clustered["priority_rank"] = clustered["segment"].map(
        {"High Priority": 1, "Developing": 2, "Stable / Developed": 3}
    )

    clustered.sort_values(["priority_rank", "child_mort", "gdpp"], ascending=[True, False, True]).to_csv(
        OUTPUT_DIR / "country_segments.csv", index=False
    )
    profiles.assign(segment=profiles.index.map(label_map)).to_csv(OUTPUT_DIR / "segment_profiles.csv")
    return clustered, model


def tune_dbscan(x_scaled: np.ndarray, kmeans_labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for eps in np.arange(1.1, 3.1, 0.2):
        for min_samples in [3, 4, 5, 6, 8]:
            labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(x_scaled)
            non_noise = labels != -1
            clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if clusters >= 2 and non_noise.sum() > clusters:
                score = silhouette_score(x_scaled[non_noise], labels[non_noise])
            else:
                score = np.nan
            rows.append(
                {
                    "eps": round(float(eps), 2),
                    "min_samples": min_samples,
                    "clusters": clusters,
                    "noise_points": int((labels == -1).sum()),
                    "silhouette_score": score,
                    "ari_vs_kmeans": adjusted_rand_score(kmeans_labels, labels),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "dbscan_tuning.csv", index=False)

    neighbors = NearestNeighbors(n_neighbors=5)
    distances, _ = neighbors.fit(x_scaled).kneighbors(x_scaled)
    kth_distances = np.sort(distances[:, -1])
    plt.figure(figsize=(8, 4))
    plt.plot(kth_distances)
    plt.title("DBSCAN 5-Nearest Neighbor Distance")
    plt.xlabel("Sorted countries")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dbscan_k_distance.png", dpi=180)
    plt.close()
    return results


def plot_segments(clustered: pd.DataFrame, x_scaled: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "segment": clustered["segment"],
            "country": clustered["country"],
        }
    )
    pca_df.to_csv(OUTPUT_DIR / "pca_coordinates.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="segment", s=80)
    plt.title("Country Segments Visualized with PCA")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_pca_scatter.png", dpi=180)
    plt.close()


def get_boosting_model():
    try:
        from xgboost import XGBClassifier

        return (
            "XGBoost",
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
            ),
        )
    except Exception:
        return (
            "Gradient Boosting (XGBoost fallback)",
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=42),
        )


def train_classifiers(clustered: pd.DataFrame) -> None:
    x = clustered[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(clustered["segment"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    boosting_name, boosting_model = get_boosting_model()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        ),
        boosting_name: boosting_model,
    }

    rows = []
    reports = []
    best_name, best_pipeline, best_accuracy = None, None, -1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="accuracy")
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, preds)
        rows.append(
            {
                "model": name,
                "test_accuracy": accuracy,
                "cv_accuracy_mean": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
            }
        )
        reports.append(f"\n{name}\n" + classification_report(y_test, preds, target_names=encoder.classes_))

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{name.lower().replace(' ', '_').replace('/', '')}_confusion_matrix.png", dpi=180)
        plt.close()

        if accuracy > best_accuracy:
            best_name, best_pipeline, best_accuracy = name, pipeline, accuracy

    metrics = pd.DataFrame(rows).sort_values(["test_accuracy", "cv_accuracy_mean"], ascending=False)
    metrics.to_csv(OUTPUT_DIR / "classification_model_comparison.csv", index=False)
    (OUTPUT_DIR / "classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")

    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        importances = best_pipeline.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        importance_df.to_csv(OUTPUT_DIR / "best_model_feature_importance.csv", index=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=importance_df, x="importance", y="feature", color="#4C78A8")
        plt.title(f"Feature Importance: {best_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "best_model_feature_importance.png", dpi=180)
        plt.close()

    joblib.dump({"model": best_pipeline, "label_encoder": encoder, "features": FEATURES}, OUTPUT_DIR / "best_classifier.joblib")


def plain_table(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string() + "\n```"


def write_business_report(clustered: pd.DataFrame, kmeans_results: pd.DataFrame, dbscan_results: pd.DataFrame) -> None:
    profile = clustered.groupby("segment")[FEATURES + ["priority_rank"]].mean().sort_values("priority_rank")
    top_priority = clustered[clustered["segment"] == "High Priority"].sort_values(
        ["child_mort", "gdpp"], ascending=[False, True]
    )[["country", "child_mort", "income", "life_expec", "total_fer", "gdpp"]]
    best_k = kmeans_results.sort_values("silhouette_score", ascending=False).iloc[0]
    best_dbscan = dbscan_results.dropna(subset=["silhouette_score"]).sort_values(
        ["silhouette_score", "ari_vs_kmeans"], ascending=False
    ).head(1)

    dbscan_line = "DBSCAN did not produce a stronger stable segmentation than K-Means."
    if not best_dbscan.empty:
        row = best_dbscan.iloc[0]
        dbscan_line = (
            f"Best DBSCAN trial used eps={row['eps']} and min_samples={int(row['min_samples'])}, "
            f"creating {int(row['clusters'])} clusters with {int(row['noise_points'])} noise points."
        )

    text = f"""# Customer Intelligence System - Country Segmentation

## Objective
Build an end-to-end intelligence pipeline using clustering and classification to identify priority country segments from socio-economic indicators.

## Data
- Records: {len(clustered)} countries
- Numeric indicators: {len(FEATURES)}
- Missing values: 0
- Source files: `Country-data.csv` and `data-dictionary.csv`

## Methodology
1. Loaded and validated the country data.
2. Standardized all numeric indicators before clustering.
3. Compared K-Means cluster counts using inertia and silhouette score.
4. Built final K-Means segmentation with 3 interpretable groups.
5. Compared DBSCAN as a density-based clustering alternative.
6. Trained classification models to predict the final customer/country segment:
   Logistic Regression, Random Forest, and XGBoost when installed. This machine used the scikit-learn gradient boosting fallback if XGBoost was unavailable.

## Clustering Findings
- Best silhouette from K-Means scan: k={int(best_k['k'])}, score={best_k['silhouette_score']:.3f}
- Final business segmentation used k=3 for interpretability.
- {dbscan_line}

## Segment Profiles
{plain_table(profile.round(2))}

## Highest Priority Countries
{plain_table(top_priority.head(15).set_index("country"))}

## Business Interpretation
- High Priority countries show the weakest development indicators: high child mortality, low income, low GDP per capita, lower life expectancy, and higher fertility.
- Developing countries sit between the other groups and are suitable for targeted growth, health, education, and infrastructure programs.
- Stable / Developed countries show stronger income, GDP, and life expectancy, so they are lower urgency for aid-style intervention.

## Deliverables Generated
- `country_segments.csv`: final country-level segment assignments
- `segment_profiles.csv`: average indicator profile by segment
- `classification_model_comparison.csv`: classification performance summary
- `classification_reports.txt`: precision, recall, and F1-score by model
- `best_classifier.joblib`: saved best predictive model
- PNG charts for EDA, clustering selection, PCA segmentation, DBSCAN, and classification confusion matrices
"""
    (OUTPUT_DIR / "assignment_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    df = load_data()
    df = add_actual_trade_values(df)
    save_eda(df)
    x_scaled, scaler = scaled_features(df)
    kmeans_results = evaluate_kmeans(x_scaled)
    clustered, kmeans_model = fit_kmeans(df, x_scaled)
    dbscan_results = tune_dbscan(x_scaled, clustered["kmeans_cluster"].to_numpy())
    plot_segments(clustered, x_scaled)
    train_classifiers(clustered)
    write_business_report(clustered, kmeans_results, dbscan_results)
    joblib.dump({"scaler": scaler, "kmeans": kmeans_model, "features": FEATURES}, OUTPUT_DIR / "kmeans_segmenter.joblib")
    print(f"Done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
