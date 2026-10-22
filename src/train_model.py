
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("raw")
OUTPUT_DIR = Path("artifacts")

BANDS = [
    "06 < 07 Weeks",
    "07 < 08 Weeks",
    "08 < 09 Weeks",
    "09 < 10 Weeks",
    "10 < 11 Weeks",
    "11 < 12 Weeks",
    "12 < 13 Weeks",
    "13+ Weeks",
]

KEYS = ["Month", "Provider Org Code", "Diagnostic Tests"]


def load_data():
    files = sorted(DATA_DIR.glob("DM01-*.csv"))

    if not files:
        raise FileNotFoundError("No DM01 files found")

    frames = []

    for file in files:
        print(f"Loading: {file.name}")

        df = pd.read_csv(
            file,
            usecols=["Period", "Provider Org Code",
                     "Diagnostic Tests", "Total WL"] + BANDS,
            low_memory=False,
        )

        period = df["Period"].astype(str).str.extract(
            r"DM01-([A-Za-z]+)-(20\d{2})"
        )

        df["Month"] = pd.to_datetime(
            period[0] + " " + period[1],
            format="%B %Y",
            errors="coerce",
        )

        for col in ["Total WL"] + BANDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=KEYS + ["Total WL"] + BANDS
        )

        df["Over_6"] = df[BANDS].sum(axis=1)

        grouped = df.groupby(
            KEYS, as_index=False
        ).agg(
            Total_WL=("Total WL", "sum"),
            Over_6=("Over_6", "sum"),
        )

        frames.append(grouped)

    data = pd.concat(frames, ignore_index=True)

    # Combine any duplicate month/provider/test records
    data = data.groupby(
        KEYS, as_index=False
    )[["Total_WL", "Over_6"]].sum()

    data = data[
        (data["Total_WL"] > 0)
        & (data["Over_6"] >= 0)
        & (data["Over_6"] <= data["Total_WL"])
    ].copy()

    data["Pct_Over_6"] = (
        100 * data["Over_6"] / data["Total_WL"]
    )

    return data


def prepare_features(data):
    data = data.sort_values(KEYS).copy()

    group = data.groupby(
        ["Provider Org Code", "Diagnostic Tests"]
    )

    previous_month = group["Month"].shift(1)
    previous_value = group["Pct_Over_6"].shift(1)

    # Only use genuinely consecutive months
    consecutive = (
        previous_month.dt.to_period("M") + 1
        == data["Month"].dt.to_period("M")
    )

    data["Previous_Month_Pct"] = (
        previous_value.where(consecutive)
    )

    return data.dropna(
        subset=["Previous_Month_Pct"]
    ).copy()


def main():
    print("NHS DM01 MODEL TRAINING")

    data = load_data()
    data = prepare_features(data)

    months = sorted(data["Month"].unique())

    if len(months) < 2:
        raise ValueError(
            "At least two target months are required"
        )

    # Time-based split: final month is held out
    test_month = months[-1]

    train = data[data["Month"] < test_month]
    test = data[data["Month"] == test_month]

    if train.empty or test.empty:
        raise ValueError("Training or test data is empty")

    features = ["Previous_Month_Pct"]

    X_train = train[features]
    y_train = train["Pct_Over_6"]

    X_test = test[features]
    y_test = test["Pct_Over_6"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    r2 = r2_score(y_test, predictions)

    # Compare against previous-month baseline
    baseline = test["Previous_Month_Pct"]
    baseline_mae = mean_absolute_error(
        y_test, baseline
    )

    print("\nMODEL RESULTS")
    print(f"Training records: {len(train):,}")
    print(f"Testing records: {len(test):,}")
    print(f"Test month: {pd.Timestamp(test_month):%B %Y}")
    print(f"Linear Regression MAE: {mae:.4f}")
    print(f"Linear Regression RMSE: {rmse:.4f}")
    print(f"Linear Regression R2: {r2:.4f}")
    print(f"Naive Baseline MAE: {baseline_mae:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": features,
        },
        OUTPUT_DIR / "dm01_model.joblib",
    )

    results = pd.DataFrame([
        {
            "Model": "Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        },
        {
            "Model": "Naive Baseline",
            "MAE": baseline_mae,
            "RMSE": np.sqrt(
                mean_squared_error(y_test, baseline)
            ),
            "R2": r2_score(y_test, baseline),
        },
    ])

    results.to_csv(
        OUTPUT_DIR / "model_results.csv",
        index=False,
    )

    print("\nModel and results saved successfully")


if __name__ == "__main__":
    main()
