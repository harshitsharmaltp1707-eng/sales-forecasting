"""
lstm_model.py
-------------
LSTM = Long Short-Term Memory (a type of neural network)

Imagine your brain remembers what you had for breakfast last week —
LSTM works the same way. It's a "recurrent" network with a memory cell
that can remember patterns from many steps back.

Key concepts:
   Sequence length   — how many past weeks we show the model at once
   Epochs            — how many times the model re-reads all training data
   Batch size        — how many training examples to process together
   LSTM units        — the "memory cells" (more = smarter but slower)

LSTM needs the data scaled to 0–1 range first (it learns better that way).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TensorFlow logs

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.get_logger().setLevel("ERROR")
except ImportError:
    raise ImportError("Run:  pip install tensorflow")

import sys
sys.path.append(os.path.dirname(__file__))
from data_generator import load_data

# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO    = 0.80
SEQ_LENGTH     = 12      # look back 12 weeks to predict the next week
FORECAST_WEEKS = 12      # weeks to forecast into the future
EPOCHS         = 100     # max training rounds (early stopping kicks in)
BATCH_SIZE     = 16
LSTM_UNITS_1   = 64      # neurons in first LSTM layer
LSTM_UNITS_2   = 32      # neurons in second LSTM layer
DROPOUT        = 0.20    # randomly disable 20% of neurons (prevents overfitting)
PLOT_DIR       = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)


def make_sequences(data: np.ndarray, seq_len: int):
    """
    Convert a 1-D array into (X, y) pairs for LSTM.

    Example with seq_len=3:
      data = [10, 20, 30, 40, 50]
      X[0] = [10, 20, 30]  →  y[0] = 40
      X[1] = [20, 30, 40]  →  y[1] = 50
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def build_lstm_model(seq_len: int) -> tf.keras.Model:
    """
    Build a 2-layer LSTM network.

    Layer 1: LSTM(64) — learns complex long-range patterns
    Layer 2: LSTM(32) — refines those patterns
    Layer 3: Dense(16) — combines everything
    Layer 4: Dense(1)  — outputs a single number (next week's sales)
    """
    model = Sequential([
        LSTM(LSTM_UNITS_1, input_shape=(seq_len, 1),
             return_sequences=True),   # pass output to next LSTM
        Dropout(DROPOUT),
        LSTM(LSTM_UNITS_2, return_sequences=False),
        Dropout(DROPOUT),
        Dense(16, activation="relu"),
        Dense(1),                     # output: one sales prediction
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_lstm(df: pd.DataFrame) -> dict:
    print("\n" + "="*55)
    print("  LSTM MODEL  (Deep Learning)")
    print("="*55)

    sales = df["y"].values.reshape(-1, 1)
    dates = df["ds"].values

    # ── Scale to [0, 1] ───────────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(sales)

    # ── Train / test split (before creating sequences) ────────────────────────
    split = int(len(scaled) * TRAIN_RATIO)
    train_data = scaled[:split]
    test_data  = scaled[split - SEQ_LENGTH:]   # include lookback window

    print(f"\n Train weeks: {split}  |  Test weeks: {len(df) - split}")

    # ── Create sequences ──────────────────────────────────────────────────────
    X_train, y_train = make_sequences(train_data, SEQ_LENGTH)
    X_test,  y_test  = make_sequences(test_data,  SEQ_LENGTH)

    # LSTM expects shape: (samples, timesteps, features)
    X_train = X_train.reshape(-1, SEQ_LENGTH, 1)
    X_test  = X_test.reshape(-1, SEQ_LENGTH, 1)

    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test  shape: {X_test.shape}")

    # ── Build and train model ─────────────────────────────────────────────────
    print("\n  Building LSTM network...")
    model = build_lstm_model(SEQ_LENGTH)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=0
        ),
    ]

    print(f"\n  Training for up to {EPOCHS} epochs (early stopping enabled)...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
    )
    epochs_run = len(history.history["loss"])
    print(f"   [OK] Stopped at epoch {epochs_run}")
    print(f"   Final val loss: {history.history['val_loss'][-1]:.6f}")

    # ── Predictions on test set ───────────────────────────────────────────────
    test_preds_scaled = model.predict(X_test, verbose=0)
    test_preds = scaler.inverse_transform(test_preds_scaled).flatten()
    actual     = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_dates = dates[split: split + len(actual)]

    # ── Recursive future forecast ─────────────────────────────────────────────
    # Feed each prediction back as input for the next step
    last_sequence = scaled[-SEQ_LENGTH:].copy()
    future_preds  = []

    for _ in range(FORECAST_WEEKS):
        seq_input = last_sequence.reshape(1, SEQ_LENGTH, 1)
        next_val  = model.predict(seq_input, verbose=0)[0, 0]
        future_preds.append(next_val)
        last_sequence = np.append(last_sequence[1:], [[next_val]], axis=0)

    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()
    future_dates = pd.date_range(
        start=pd.Timestamp(dates[-1]) + pd.Timedelta(weeks=1),
        periods=FORECAST_WEEKS,
        freq="W",
    )

    # Simple uncertainty band (±1 std of test residuals)
    residual_std = np.std(actual - test_preds)
    upper = future_preds + 1.96 * residual_std
    lower = future_preds - 1.96 * residual_std

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(actual, test_preds)
    print(f"\n Test-set metrics:")
    print(f"   MAE  : ${metrics['MAE']:>10,.0f}")
    print(f"   RMSE : ${metrics['RMSE']:>10,.0f}")
    print(f"   MAPE : {metrics['MAPE']:>9.2f}%")

    # ── Plot 1: Forecast vs Actual ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("LSTM Sales Forecast", fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.plot(df["ds"].values[:split], df["y"].values[:split],
            color="#378ADD", linewidth=1.5, label="Training data")
    ax.plot(test_dates, actual,
            color="#378ADD", linewidth=1.5, linestyle="--", alpha=0.6, label="Actual (test)")
    ax.plot(test_dates, test_preds,
            color="#E24B4A", linewidth=2, label="LSTM prediction (test)")
    ax.plot(future_dates, future_preds,
            color="#1D9E75", linewidth=2, linestyle="--", label=f"Forecast ({FORECAST_WEEKS} weeks)")
    ax.fill_between(future_dates, lower, upper,
                    color="#EF9F27", alpha=0.25, label="95% confidence band")
    ax.axvline(test_dates[0], color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_title("Forecast vs Actual Sales")
    ax.set_ylabel("Weekly Sales ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Plot 2: Training loss curve ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(history.history["loss"],     color="#378ADD", linewidth=2, label="Training loss")
    ax2.plot(history.history["val_loss"], color="#E24B4A", linewidth=2, linestyle="--", label="Validation loss")
    ax2.set_title("Training Loss Curve  (going down = learning!)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "lstm_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n Plot saved → {path}")

    return {
        "model": "LSTM",
        "metrics": metrics,
        "forecast": pd.Series(future_preds, index=future_dates),
        "confidence": pd.DataFrame({"lower": lower, "upper": upper}, index=future_dates),
    }


if __name__ == "__main__":
    df = load_data()
    result = run_lstm(df)
