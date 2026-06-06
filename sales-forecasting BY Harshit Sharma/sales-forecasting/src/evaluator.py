"""
evaluator.py
------------
Compare all 3 models side by side.

Metrics we use:
  MAE  (Mean Absolute Error)        → average $ error per week
  RMSE (Root Mean Squared Error)    → punishes big mistakes more
  MAPE (Mean Absolute % Error)      → error as a percentage

  Lower = better for all three!
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.append(os.path.dirname(__file__))
from data_generator import load_data
from arima_model   import run_arima
from prophet_model import run_prophet
from lstm_model    import run_lstm

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def compare_models(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  RUNNING ALL 3 MODELS — this will take a few minutes!")
    print("="*55)

    results = {}
    results["ARIMA"]   = run_arima(df)
    results["Prophet"] = run_prophet(df)
    results["LSTM"]    = run_lstm(df)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  FINAL COMPARISON")
    print("="*55)
    print(f"\n{'Model':<12} {'MAE':>12} {'RMSE':>12} {'MAPE':>8}")
    print("-"*46)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<12} ${m['MAE']:>10,.0f} ${m['RMSE']:>10,.0f} {m['MAPE']:>7.2f}%")

    best = min(results, key=lambda k: results[k]["metrics"]["MAPE"])
    print(f"\n Best model by MAPE: {best}")

    # ── Plot 1: Metric comparison bar chart ───────────────────────────────────
    names  = list(results.keys())
    maes   = [results[n]["metrics"]["MAE"]  / 1000 for n in names]
    rmses  = [results[n]["metrics"]["RMSE"] / 1000 for n in names]
    mapes  = [results[n]["metrics"]["MAPE"] for n in names]
    colors = ["#378ADD", "#1D9E75", "#7F77DD"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison — Lower is Better", fontsize=15, fontweight="bold")

    for ax, vals, title, unit in zip(
        axes,
        [maes, rmses, mapes],
        ["MAE ($k)", "RMSE ($k)", "MAPE (%)"],
        ["$", "$", "%"],
    ):
        bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{unit}{v:.1f}" if unit == "$" else f"{v:.1f}{unit}",
                    ha="center", va="bottom", fontsize=10, fontweight="500")
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n Comparison chart saved → {path}")

    # ── Plot 2: All forecasts on one chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical line
    split = int(len(df) * 0.80)
    ax.plot(df["ds"].values[:split], df["y"].values[:split],
            color="#888780", linewidth=1.5, label="Historical sales", alpha=0.7)
    ax.plot(df["ds"].values[split:], df["y"].values[split:],
            color="#888780", linewidth=1.5, linestyle="--", alpha=0.5, label="Actual (test period)")

    # Each model's future forecast
    forecast_colors = {"ARIMA": "#378ADD", "Prophet": "#1D9E75", "LSTM": "#7F77DD"}
    for name, res in results.items():
        fc = res["forecast"]
        if isinstance(fc, pd.Series):
            dates = fc.index
            vals  = fc.values
        else:
            dates = fc["ds"].values if "ds" in fc.columns else fc.index
            vals  = fc["yhat"].values if "yhat" in fc.columns else fc.values
        ax.plot(dates, vals,
                color=forecast_colors[name], linewidth=2.5,
                linestyle="--", label=f"{name} forecast",
                marker="o", markersize=4)

    ax.axvline(df["ds"].values[split], color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(df["ds"].values[split], ax.get_ylim()[0], " test start", fontsize=9, color="gray")
    ax.set_title("All Models — Future Sales Forecast", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weekly Sales ($)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path2 = os.path.join(PLOT_DIR, "all_forecasts.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" All-forecasts chart saved → {path2}")

    print("\n[OK] All done! Check the plots/ folder for your charts.")
    return results


if __name__ == "__main__":
    df = load_data()
    compare_models(df)
