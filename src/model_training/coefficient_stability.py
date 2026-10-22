
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    train_test_split,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42


def run_coefficient_stability(X, y, feature_names=None):
    X = np.asarray(X)
    y = np.asarray(y)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if len(feature_names) != X.shape[1]:
        raise ValueError("Feature names must match X columns")

    mlflow.set_experiment("DM01_Coefficient_Stability")

    methods = {
        "5_Fold": list(
            KFold(n_splits=5, shuffle=True, random_state=42)
            .split(X)
        ),
        "Repeated_5_Fold": list(
            RepeatedKFold(
                n_splits=5, n_repeats=5, random_state=42
            ).split(X)
        ),
    }

    results = []

    for method, splits in methods.items():
        coefficients = []
        metrics = []

        with mlflow.start_run(run_name=method):
            mlflow.log_param("method", method)
            mlflow.log_param("number_of_splits", len(splits))
            mlflow.log_param("random_state", RANDOM_STATE)

            for train_idx, test_idx in splits:
                model = LinearRegression()
                model.fit(X[train_idx], y[train_idx])

                predictions = model.predict(X[test_idx])

                coefficients.append(model.coef_)

                metrics.append({
                    "mae": mean_absolute_error(
                        y[test_idx], predictions
                    ),
                    "rmse": np.sqrt(mean_squared_error(
                        y[test_idx], predictions
                    )),
                    "r2": r2_score(y[test_idx], predictions),
                })

            coef_df = pd.DataFrame(
                coefficients, columns=feature_names
            )
            metrics_df = pd.DataFrame(metrics)

            for feature in feature_names:
                safe_name = str(feature).replace(" ", "_").replace("/", "_")
                mlflow.log_metric(
                    f"coef_std_{safe_name}",
                    float(coef_df[feature].std())
                )

            for metric in ["mae", "rmse", "r2"]:
                mlflow.log_metric(
                    f"mean_{metric}",
                    float(metrics_df[metric].mean())
                )

            mlflow.log_metric(
                "mean_coefficient_std",
                float(coef_df.std().mean())
            )

            output_dir = Path("stability_outputs")
            output_dir.mkdir(exist_ok=True)

            coef_df.to_csv(
                output_dir / f"{method}_coefficients.csv",
                index=False
            )
            metrics_df.to_csv(
                output_dir / f"{method}_metrics.csv",
                index=False
            )

            mlflow.log_artifacts(str(output_dir))

            results.append({
                "method": method,
                "mean_mae": metrics_df["mae"].mean(),
                "mean_rmse": metrics_df["rmse"].mean(),
                "mean_r2": metrics_df["r2"].mean(),
                "mean_coefficient_std": coef_df.std().mean(),
            })

    return pd.DataFrame(results)
