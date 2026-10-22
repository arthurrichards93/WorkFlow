
from pathlib import Path
import pandas as pd

# Folder containing the DM01 CSV files
DATA_DIR = Path("raw")

# Required columns from the DM01 dataset
REQUIRED_COLUMNS = [
    "Period",
    "Provider Org Code",
    "Diagnostic Tests",
    "Total WL",
    "06 < 07 Weeks",
    "07 < 08 Weeks",
    "08 < 09 Weeks",
    "09 < 10 Weeks",
    "10 < 11 Weeks",
    "11 < 12 Weeks",
    "12 < 13 Weeks",
    "13+ Weeks",
]

WAITING_BANDS = REQUIRED_COLUMNS[4:]


def validate_file(file_path):
    print(f"\nValidating: {file_path.name}")

    df = pd.read_csv(
        file_path,
        usecols=lambda column: column in REQUIRED_COLUMNS,
        low_memory=False,
    )

    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Dataset is empty")

    # Validate numeric waiting-list columns
    numeric_columns = ["Total WL"] + WAITING_BANDS

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing or invalid required values")

    if (df[numeric_columns] < 0).any().any():
        raise ValueError("Negative waiting-list values found")

    # Patients waiting over six weeks cannot exceed total WL
    over_six_weeks = df[WAITING_BANDS].sum(axis=1)

    if (over_six_weeks > df["Total WL"]).any():
        raise ValueError(
            "Patients waiting over six weeks exceed total WL"
        )

    print(f"PASS: {len(df):,} rows validated")

    return len(df)


def main():
    files = sorted(DATA_DIR.glob("DM01-*.csv"))

    if not files:
        raise FileNotFoundError("No DM01 CSV files found")

    print("=" * 50)
    print("NHS DM01 DATA VALIDATION")
    print("=" * 50)

    total_rows = 0

    for file in files:
        total_rows += validate_file(file)

    print("\n" + "=" * 50)
    print("VALIDATION SUCCESSFUL")
    print(f"Files validated: {len(files)}")
    print(f"Total rows: {total_rows:,}")
    print("=" * 50)


if __name__ == "__main__":
    main()
