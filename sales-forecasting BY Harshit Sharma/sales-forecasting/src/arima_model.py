"""
arima_model.py
--------------
ARIMA = AutoRegressive Integrated Moving Average

Think of ARIMA like this:
  AR (AutoRegressive)  → "Future sales depend on PAST sales"
  I  (Integrated)      → "We remove trends to make data stable"
  MA (Moving Average)  → "We also learn from past prediction ERRORS"

The three magic numbers ARIMA(p, d, q):
  p = how many past sales to look at (lag)
  d = how many times to subtract to remove trend (usually 1)
  q = how many past errors to learn from
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

import sys, os
sys.path.append(os.path.dirname(__file__))
from data_generator import load_data

# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80          # 80% train, 20% test
FORECAST_WEEKS = 12         # how many weeks ahead to predict
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def check_stationarity(series: pd.Series) -> bool:
    """
    Stationarity test — ARIMA needs data without a drifting mean.
    We use the Augmented Dickey-Fuller test:
      p-value < 0.05  → stationary (good!)
      p-value >= 0.05 → needs differencing
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"   ADF p-value: {p_value:.4f}  →  "
          f"{'[OK] Stationary' if p_value < 0.05 else '[WARN]  Not stationary — will difference'}")
    return p_value < 0.05


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_arima(df: pd.DataFrame) -> dict:
    print("\n" + "="*55)
    print("  ARIMA MODEL")
    print("="*55)

    series = df.set_index("ds")["y"]

    # ── Train / test split ────────────────────────────────────────────────────
    split = int(len(series) * TRAIN_RATIO)
    train, test = series.iloc[:split], series.iloc[split:]
    print(f"\n Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} weeks)")
    print(f"   Test : {test.index[0].date()}  → {test.index[-1].date()} ({len(test)} weeks)")

    # ── Stationarity ──────────────────────────────────────────────────────────
    print("\n Checking stationarity...")
    check_stationarity(train)

    # ── Fit SARIMA(1,1,1)(1,1,0,52) ──────────────────────────────────────────
    # SARIMA adds Seasonal terms: (P,D,Q,m) where m=52 for weekly data
    print("\n  Fitting SARIMA(1,1,1)(1,1,0,52)  [this may take ~30 sec]...")
    model = SARIMAX(
        train,
        order=(1, 1, 1),            # (p, d, q)
        seasonal_order=(1, 1, 0, 52),  # (P, D, Q, season_length)
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    print("   [OK] Model fitted!")

    # ── In-sample predictions (on test set) ──────────────────────────────────
    pred_test = fitted.forecast(steps=len(test))
    pred_test.index = test.index

    # ── Future forecast ───────────────────────────────────────────────────────
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=FORECAST_WEEKS,
        freq="W",
    )
    future_pred = fitted.forecast(steps=len(test) + FORECAST_WEEKS)
    future_pred = future_pred.iloc[len(test):]
    future_pred.index = future_dates

    # Confidence intervals
    forecast_ci = fitted.get_forecast(steps=len(test) + FORECAST_WEEKS).conf_int()
    future_ci = forecast_ci.iloc[len(test):]
    future_ci.index = future_dates

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(test.values, pred_test.values)
    print(f"\n Test-set metrics:")
    print(f"   MAE  : ${metrics['MAE']:>10,.0f}")
    print(f"   RMSE : ${metrics['RMSE']:>10,.0f}")
    print(f"   MAPE : {metrics['MAPE']:>9.2f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("ARIMA Sales Forecast", fontsize=16, fontweight="bold", y=0.98)

    # -- Top: full history + test pred + future
    ax = axes[0]
    ax.plot(train.index, train.values, color="#378ADD", linewidth=1.5, label="Training data")
    ax.plot(test.index,  test.values,  color="#378ADD", linewidth=1.5, linestyle="--", alpha=0.6, label="Actual (test)")
    ax.plot(pred_test.index, pred_test.values, color="#E24B4A", linewidth=2, label="ARIMA prediction (test)")
    ax.plot(future_pred.index, future_pred.values, color="#1D9E75", linewidth=2, linestyle="--", label=f"Forecast ({FORECAST_WEEKS} weeks)")
    ax.fill_between(future_ci.index,
                    future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                    color="#EF9F27", alpha=0.25, label="95% confidence band")
    ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(test.index[0], ax.get_ylim()[0], " test start", fontsize=9, color="gray")
    ax.set_title("Forecast vs Actual Sales")
    ax.set_ylabel("Weekly Sales ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # -- Bottom: residuals (prediction errors)
    ax2 = axes[1]
    residuals = test.values - pred_test.values
    ax2.bar(test.index, residuals, color=["#E24B4A" if r < 0 else "#1D9E75" for r in residuals],
            width=5, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Prediction Errors (Residuals)  — closer to 0 = better")
    ax2.set_ylabel("Error ($)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "arima_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n Plot saved → {path}")

    return {
        "model": "ARIMA",
        "metrics": metrics,
        "forecast": future_pred,
        "confidence": future_ci,
    }


if __name__ == "__main__":
    df = load_data()
    result = run_arima(df)
