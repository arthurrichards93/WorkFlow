
from pathlib import Path
import argparse

import mlflow
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42


def run_coefficient_stability(X, y, feature_names=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y must have compatible dimensions")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if len(feature_names) != X.shape[1]:
        raise ValueError("Feature names must match X columns")

    if len(X) < 6:
        raise ValueError("At least 6 observations are required")

    mlflow.set_experiment("DM01_Coefficient_Stability")

    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(len(X))

    methods = {
        "5_Fold": list(
            KFold(
                n_splits=5, shuffle=True,
                random_state=RANDOM_STATE
            ).split(X)
        ),
        "Repeated_5_Fold": list(
            RepeatedKFold(
                n_splits=5, n_repeats=5,
                random_state=RANDOM_STATE
            ).split(X)
        ),
    }

    # Bootstrap with out-of-bag evaluation
    bootstrap_splits = []

    for _ in range(100):
        train_idx = rng.choice(
            indices, size=len(X), replace=True
        )

        test_idx = np.setdiff1d(
            indices, np.unique(train_idx)
        )

        if len(test_idx) >= 2:
            bootstrap_splits.append((train_idx, test_idx))

    methods["Bootstrap_100"] = bootstrap_splits

    output_dir = Path("stability_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for method, splits in methods.items():
        coefficients = []
        metrics = []

        with mlflow.start_run(run_name=method):
            mlflow.log_params({
                "method": method,
                "number_of_splits": len(splits),
                "random_state": RANDOM_STATE,
                "model": "LinearRegression",
            })

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
                    "r2": r2_score(
                        y[test_idx], predictions
                    ),
                })

            coef_df = pd.DataFrame(
                coefficients, columns=feature_names
            )
            metrics_df = pd.DataFrame(metrics)

            summary = coef_df.agg(
                ["mean", "std", "min", "max"]
            ).T

            summary.index.name = "feature"

            for i, feature in enumerate(feature_names):
                mlflow.log_metric(
                    f"coefficient_{i}_std",
                    float(summary.loc[feature, "std"])
                )

            for metric in ["mae", "rmse", "r2"]:
                mlflow.log_metric(
                    f"mean_{metric}",
                    float(metrics_df[metric].mean())
                )

            mean_std = float(
                coef_df.std().mean()
            )

            mlflow.log_metric(
                "mean_coefficient_std", mean_std
            )

            coef_df.to_csv(
                output_dir / f"{method}_coefficients.csv",
                index=False
            )

            metrics_df.to_csv(
                output_dir / f"{method}_metrics.csv",
                index=False
            )

            summary.to_csv(
                output_dir / f"{method}_summary.csv"
            )

            mlflow.log_artifacts(str(output_dir))

            results.append({
                "method": method,
                "mean_mae": metrics_df["mae"].mean(),
                "mean_rmse": metrics_df["rmse"].mean(),
                "mean_r2": metrics_df["r2"].mean(),
                "mean_coefficient_std": mean_std,
            })

            print(
                f"{method}: "
                f"MAE={metrics_df['mae'].mean():.4f}, "
                f"Coefficient SD={mean_std:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_dir / "comparison_summary.csv",
        index=False
    )

    print("\nCOEFFICIENT STABILITY COMPARISON")
    print(results_df.to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to DM01 CSV dataset"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(
            f"Target '{args.target}' not found"
        )

    feature_names = [
        col for col in df.select_dtypes(
            include="number"
        ).columns
        if col != args.target
    ]

    X = df[feature_names]
    y = df[args.target]

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Dataset contains missing numeric values")

    run_coefficient_stability(
        X, y, feature_names
    )


if __name__ == "__main__":
    main()
