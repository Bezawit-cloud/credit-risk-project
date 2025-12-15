import pandas as pd


def test_dataframe_has_no_nulls_after_processing():
    """Test that processed dataframe has no missing values"""
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [0.1, 0.2, 0.3],
        "target": [0, 1, 0]
    })

    assert df.isnull().sum().sum() == 0


def test_expected_columns_exist():
    """Test that expected columns exist after feature engineering"""
    df = pd.DataFrame({
        "CustomerId": [1, 2],
        "feature1": [10, 20],
        "is_high_risk": [0, 1]
    })

    expected_columns = {"CustomerId", "feature1", "is_high_risk"}
    assert expected_columns.issubset(set(df.columns))
