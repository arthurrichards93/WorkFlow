
import pandas as pd
import pytest

from src.validate_data import validate_file, REQUIRED_COLUMNS


def create_test_data():
    """Create a small, valid DM01 dataset."""
    data = {column: [0] for column in REQUIRED_COLUMNS}

    data["Period"] = ["DM01-MAY-2026"]
    data["Provider Org Code"] = ["TEST001"]
    data["Diagnostic Tests"] = ["MRI"]
    data["Total WL"] = [100]
    data["06 < 07 Weeks"] = [10]
    data["07 < 08 Weeks"] = [5]

    return pd.DataFrame(data)


def test_valid_data(tmp_path):
    """Valid DM01 data should pass."""
    file = tmp_path / "valid.csv"
    create_test_data().to_csv(file, index=False)

    assert validate_file(file) == 1


def test_missing_column(tmp_path):
    """Missing required columns should fail."""
    file = tmp_path / "missing.csv"

    df = create_test_data().drop(columns=["Total WL"])
    df.to_csv(file, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        validate_file(file)


def test_negative_waiting_list(tmp_path):
    """Negative waiting-list values should fail."""
    file = tmp_path / "negative.csv"

    df = create_test_data()
    df["Total WL"] = -10
    df.to_csv(file, index=False)

    with pytest.raises(ValueError, match="Negative"):
        validate_file(file)


def test_waiting_list_exceeded(tmp_path):
    """Patients waiting over six weeks cannot exceed total WL."""
    file = tmp_path / "exceeded.csv"

    df = create_test_data()
    df["Total WL"] = 5
    df.to_csv(file, index=False)

    with pytest.raises(ValueError, match="exceed"):
        validate_file(file)


def test_empty_dataset(tmp_path):
    """Empty datasets should fail."""
    file = tmp_path / "empty.csv"

    create_test_data().iloc[0:0].to_csv(file, index=False)

    with pytest.raises(ValueError, match="empty"):
        validate_file(file)
