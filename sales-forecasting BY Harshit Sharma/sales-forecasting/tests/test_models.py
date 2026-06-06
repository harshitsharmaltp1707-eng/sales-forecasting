"""
test_models.py
--------------
Simple tests to make sure our code doesn't break.
Run with:  python -m pytest tests/
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd
from data_generator import generate_sales_data, load_data
from evaluator import compare_models


# ── Data Generator Tests ──────────────────────────────────────────────────────

class TestDataGenerator:

    def test_generates_correct_number_of_rows(self):
        df = generate_sales_data(periods=52, save_path="data/test_sales.csv")
        assert len(df) == 52

    def test_has_required_columns(self):
        df = generate_sales_data(periods=52, save_path="data/test_sales.csv")
        for col in ["ds", "y", "trend", "seasonal", "promo"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_negative_sales(self):
        df = generate_sales_data(periods=52, save_path="data/test_sales.csv")
        assert (df["y"] > 0).all(), "Sales should always be positive"

    def test_dates_are_weekly(self):
        df = generate_sales_data(periods=10, save_path="data/test_sales.csv")
        diffs = pd.to_datetime(df["ds"]).diff().dropna()
        # All differences should be 7 days
        assert all(diffs == pd.Timedelta("7 days"))

    def test_december_higher_than_february(self):
        """December should have higher seasonal factor than February."""
        df = generate_sales_data(periods=104, save_path="data/test_sales.csv")
        dec_avg = df[pd.to_datetime(df["ds"]).dt.month == 12]["seasonal"].mean()
        feb_avg = df[pd.to_datetime(df["ds"]).dt.month == 2]["seasonal"].mean()
        assert dec_avg > feb_avg, "December should be busier than February"

    def test_load_creates_file_if_missing(self, tmp_path):
        path = str(tmp_path / "missing.csv")
        df = load_data(path)
        assert len(df) > 0


# ── Metric Calculation Tests ──────────────────────────────────────────────────

class TestMetrics:

    def test_perfect_prediction_gives_zero_mae(self):
        actual    = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        mae = np.mean(np.abs(actual - predicted))
        assert mae == 0.0

    def test_mape_on_known_values(self):
        actual    = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        assert abs(mape - 7.5) < 0.01   # (10% + 5%) / 2 = 7.5%

    def test_larger_errors_inflate_rmse_more_than_mae(self):
        """RMSE penalizes outliers harder than MAE — good property."""
        actual       = np.array([100, 100, 100])
        pred_outlier = np.array([100, 100, 200])   # one big error
        pred_smooth  = np.array([133, 133, 134])   # same total error, spread out

        rmse_out = np.sqrt(np.mean((actual - pred_outlier)**2))
        rmse_smo = np.sqrt(np.mean((actual - pred_smooth)**2))
        assert rmse_out > rmse_smo


# ── Sequence Building Tests (LSTM) ───────────────────────────────────────────

class TestLSTMSequences:

    def test_sequence_shapes(self):
        from lstm_model import make_sequences
        data = np.arange(20).reshape(-1, 1).astype(float)
        X, y = make_sequences(data, seq_len=5)
        assert X.shape == (15, 5)   # 20 - 5 = 15 samples
        assert y.shape == (15,)

    def test_first_sequence_correct(self):
        from lstm_model import make_sequences
        data = np.arange(10).reshape(-1, 1).astype(float)
        X, y = make_sequences(data, seq_len=3)
        np.testing.assert_array_equal(X[0].flatten(), [0, 1, 2])
        assert y[0] == 3

    def test_last_sequence_correct(self):
        from lstm_model import make_sequences
        data = np.arange(10).reshape(-1, 1).astype(float)
        X, y = make_sequences(data, seq_len=3)
        np.testing.assert_array_equal(X[-1].flatten(), [6, 7, 8])
        assert y[-1] == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
