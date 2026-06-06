"""
prophet_model.py
----------------
Facebook Prophet — the friendliest forecasting tool!

Prophet breaks your sales into 3 simple parts:
   Trend     — is the business growing or shrinking overall?
   Seasonal  — which months/weeks are always busy or slow?
   Holidays  — special events that cause unexpected spikes

You just feed it a table with two columns:
  ds = date
  y  = sales number
...and Prophet does the rest! No PhD required.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# Prophet needs to be imported carefully (it can be verbose)
try:
    from prophet import Prophet
    from prophet.plot import plot_components_plotly
except ImportError:
    raise ImportError("Run:  pip install prophet")

import sys
sys.path.append(os.path.dirname(__file__))
from data_generator import load_data

# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO    = 0.80
FORECAST_WEEKS = 12
PLOT_DIR       = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def build_holidays() -> pd.DataFrame:
    """
    Tell Prophet about special shopping events.
    It will automatically learn that sales spike around these dates.
    """
    years = [2021, 2022, 2023, 2024]

    # Black Friday — 4th Friday of November
    def black_friday(year):
        nov1 = pd.Timestamp(f"{year}-11-01")
        fridays = pd.date_range(nov1, periods=30, freq="D")
        fridays = fridays[fridays.day_of_week == 4]
        return fridays[3]   # 4th Friday

    black_fridays = pd.DataFrame({
        "holiday": "black_friday",
        "ds":      [black_friday(y) for y in years],
        "lower_window": -1,   # effect starts 1 day before
        "upper_window":  2,   # effect lasts 2 days after
    })

    christmas = pd.DataFrame({
        "holiday": "christmas",
        "ds": pd.to_datetime([f"{y}-12-25" for y in years]),
        "lower_window": -7,
        "upper_window":  2,
    })

    new_year = pd.DataFrame({
        "holiday": "new_year",
        "ds": pd.to_datetime([f"{y}-01-01" for y in years]),
        "lower_window": -1,
        "upper_window":  1,
    })

    return pd.concat([black_fridays, christmas, new_year], ignore_index=True)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_prophet(df: pd.DataFrame) -> dict:
    print("\n" + "="*55)
    print("  PROPHET MODEL  (by Facebook / Meta)")
    print("="*55)

    # Prophet expects columns named 'ds' and 'y'
    prophet_df = df[["ds", "y"]].copy()

    # ── Train / test split ────────────────────────────────────────────────────
    split = int(len(prophet_df) * TRAIN_RATIO)
    train = prophet_df.iloc[:split]
    test  = prophet_df.iloc[split:]
    print(f"\n Train: {train['ds'].iloc[0].date()} → {train['ds'].iloc[-1].date()} ({len(train)} weeks)")
    print(f"   Test : {test['ds'].iloc[0].date()}  → {test['ds'].iloc[-1].date()} ({len(test)} weeks)")

    # ── Build and fit the model ───────────────────────────────────────────────
    print("\n  Fitting Prophet model...")
    holidays = build_holidays()

    model = Prophet(
        yearly_seasonality=True,    # learn yearly patterns
        weekly_seasonality=True,    # learn day-of-week patterns
        daily_seasonality=False,    # we have weekly data, so skip this
        holidays=holidays,          # our special events
        seasonality_mode="multiplicative",  # seasons multiply the trend
        interval_width=0.95,        # 95% confidence interval
        changepoint_prior_scale=0.05,       # how flexible the trend line is
    )

    # Add a custom monthly seasonality (12 months cycle)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(train)
    print("   [OK] Model fitted!")

    # ── Predict on test period ────────────────────────────────────────────────
    test_future = model.make_future_dataframe(
        periods=len(test) + FORECAST_WEEKS, freq="W"
    )
    forecast_full = model.predict(test_future)

    # Split into test predictions and future forecast
    test_preds   = forecast_full.iloc[len(train): len(train) + len(test)]
    future_preds = forecast_full.iloc[len(train) + len(test):]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(test["y"].values, test_preds["yhat"].values)
    print(f"\n Test-set metrics:")
    print(f"   MAE  : ${metrics['MAE']:>10,.0f}")
    print(f"   RMSE : ${metrics['RMSE']:>10,.0f}")
    print(f"   MAPE : {metrics['MAPE']:>9.2f}%")

    # ── Plot 1: Forecast vs Actual ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Prophet Sales Forecast", fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.plot(train["ds"], train["y"],  color="#378ADD", linewidth=1.5, label="Training data")
    ax.plot(test["ds"],  test["y"],   color="#378ADD", linewidth=1.5, linestyle="--", alpha=0.6, label="Actual (test)")
    ax.plot(test_preds["ds"], test_preds["yhat"],   color="#E24B4A", linewidth=2, label="Prophet prediction (test)")
    ax.plot(future_preds["ds"], future_preds["yhat"], color="#1D9E75", linewidth=2, linestyle="--", label=f"Forecast ({FORECAST_WEEKS} weeks)")
    ax.fill_between(future_preds["ds"],
                    future_preds["yhat_lower"],
                    future_preds["yhat_upper"],
                    color="#EF9F27", alpha=0.25, label="95% confidence band")
    ax.axvline(test["ds"].iloc[0], color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_title("Forecast vs Actual Sales")
    ax.set_ylabel("Weekly Sales ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Plot 2: Seasonal Decomposition ───────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(forecast_full["ds"], forecast_full["trend"], color="#7F77DD", linewidth=2, label="Trend")
    ax2.plot(forecast_full["ds"],
             forecast_full["trend"] + forecast_full["yearly"],
             color="#1D9E75", linewidth=1.5, linestyle="--", alpha=0.8, label="Trend + Yearly season")
    if "holidays" in forecast_full.columns:
        holiday_effect = forecast_full[["ds", "holidays"]].copy()
        spikes = holiday_effect[holiday_effect["holidays"].abs() > 1000]
        ax2.scatter(spikes["ds"], forecast_full.loc[spikes.index, "trend"] + spikes["holidays"],
                    color="#E24B4A", zorder=5, s=40, label="Holiday effect", marker="^")
    ax2.set_title("Trend + Seasonality Decomposition")
    ax2.set_ylabel("Sales ($)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "prophet_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n Plot saved → {path}")

    # ── Plot 3: Prophet components (trend, weekly, yearly) ────────────────────
    comp_fig = model.plot_components(forecast_full)
    comp_fig.suptitle("Prophet — Sales Components", fontsize=14, fontweight="bold")
    comp_path = os.path.join(PLOT_DIR, "prophet_components.png")
    comp_fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(comp_fig)
    print(f" Components plot saved → {comp_path}")

    return {
        "model": "Prophet",
        "metrics": metrics,
        "forecast": future_preds[["ds", "yhat", "yhat_lower", "yhat_upper"]],
    }


if __name__ == "__main__":
    df = load_data()
    result = run_prophet(df)
