"""
data_generator.py
-----------------
Creates realistic retail sales data with:
- Long-term growth trend
- Monthly seasonality (holiday peaks, summer dips)
- Promotional spikes (Black Friday, Christmas)
- Random noise (because real data is never perfect!)

Think of this as building a fake "history" for an imaginary shop.
"""

import pandas as pd
import numpy as np
import os

# ── Reproducible randomness ──────────────────────────────────────────────────
np.random.seed(42)


def generate_sales_data(
    start_date: str = "2021-01-01",
    periods: int = 156,          # 3 years of weekly data
    base_sales: float = 50_000,  # average weekly sales in $
    trend_rate: float = 0.003,   # ~15% annual growth
    save_path: str = "data/sales_data.csv",
) -> pd.DataFrame:
    """
    Generate a realistic weekly sales time series.

    Parameters
    ----------
    start_date  : first date in the series
    periods     : number of weeks
    base_sales  : starting average weekly sales ($)
    trend_rate  : weekly growth multiplier
    save_path   : where to save the CSV

    Returns
    -------
    pd.DataFrame with columns [ds, y, trend, seasonal, promo, noise]
    """

    dates = pd.date_range(start=start_date, periods=periods, freq="W")

    # 1. TREND — slow, steady growth over time
    trend = base_sales * (1 + trend_rate) ** np.arange(periods)

    # 2. SEASONALITY — sales vary by month
    #    Scale: 1.0 = average month
    monthly_factors = {
        1: 0.75,   # January  — post-holiday slump
        2: 0.72,   # February — still quiet
        3: 0.80,   # March    — picking up
        4: 0.88,   # April    — spring shopping
        5: 0.92,   # May      — steady
        6: 0.95,   # June     — summer starts
        7: 1.02,   # July     — back-to-school prep
        8: 1.08,   # August   — back-to-school peak
        9: 0.98,   # September
        10: 1.05,  # October  — pre-holiday build
        11: 1.35,  # November — Black Friday!
        12: 1.70,  # December — Christmas rush
    }
    seasonal = np.array([monthly_factors[d.month] for d in dates])

    # 3. PROMOTIONAL SPIKES — special events
    #    Black Friday (last Friday of November) → 3x normal
    #    Christmas week (Dec 20-26) → 2.5x normal
    promo = np.ones(periods)
    for i, d in enumerate(dates):
        if d.month == 11 and d.day >= 22:          # Black Friday week
            promo[i] = 3.0
        elif d.month == 12 and 19 <= d.day <= 26:  # Christmas week
            promo[i] = 2.5
        elif d.month == 12 and d.day >= 26:        # Post-Christmas clearance
            promo[i] = 1.4

    # 4. NOISE — real sales are never perfectly smooth
    noise = np.random.normal(loc=1.0, scale=0.05, size=periods)
    noise = np.clip(noise, 0.85, 1.20)  # keep within ±15%

    # 5. COMBINE everything
    sales = trend * seasonal * promo * noise

    df = pd.DataFrame({
        "ds":       dates,               # date (Prophet uses 'ds')
        "y":        sales.round(2),      # total sales $ (Prophet uses 'y')
        "trend":    trend.round(2),
        "seasonal": seasonal.round(4),
        "promo":    promo.round(2),
        "noise":    noise.round(4),
    })

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[OK] Sales data saved → {save_path}")
    print(f"   Rows : {len(df)}")
    print(f"   Range: {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"   Sales: ${df['y'].min():,.0f} – ${df['y'].max():,.0f} per week")

    return df


def load_data(path: str = "data/sales_data.csv") -> pd.DataFrame:
    """Load the generated CSV; generate it first if missing."""
    if not os.path.exists(path):
        print("Data file not found — generating now...")
        return generate_sales_data(save_path=path)
    df = pd.read_csv(path, parse_dates=["ds"])
    print(f"[OK] Loaded {len(df)} rows from {path}")
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    print("\nFirst 5 rows:")
    print(df.head())
