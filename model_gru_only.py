"""
model_gru_only.py
=================

MVP Research Model - Final Unified Version
-------------------------------------------

ETHUSDT
5-minute candles
15-minute prediction horizon

Architecture:
    Historical OHLCV
        ↓
    Technical Features
        ↓
    Sequence
        ↓
    GRU
        ↓
    GRU
        ↓
    Multi-Head Attention
        ↓
    Dense
        ↓
    Future Log Return

IMPORTANT:
    Prediction != Signal != Position

این فایل فقط مسئول research / training / backtesting مدل است.
منطق Signal Engine و Position Management بعداً در crypto-signal-api
پیاده‌سازی می‌شود.

Target:
    future_log_return = log(close[t+horizon] / close[t])

Sequence:
    sequence_length = 96  (96 × 5m = 8 hours)

Validation:
    Walk-forward با مرزهای تقویمی واقعی

Scaling:
    Scaler فقط روی train هر fold fit می‌شود.

Risk Management:
    ATR-based SL/TP با در نظر گرفتن high/low واقعی کندل‌های بین‌راه.
"""

import os
import json
import random
import gc
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

from features import (
    add_technical_indicators,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    validate_feature_schema,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

SYMBOL = "ETHUSDT"

TIMEFRAME_MINUTES = 5
HORIZON_MINUTES = 15

assert HORIZON_MINUTES % TIMEFRAME_MINUTES == 0

HORIZON_STEPS = HORIZON_MINUTES // TIMEFRAME_MINUTES

# 96 × 5m = 8 hours of history
SEQUENCE_LENGTH = 96

N_FOLDS = 4

# Trading assumptions
INITIAL_CAPITAL = 1000.0

# One-way fee.
FEE_RATE = 0.0004

# Conservative slippage assumption per side.
SLIPPAGE_RATE = 0.0002

# Model must predict at least this much expected movement.
SIGNAL_THRESHOLD = 0.0010  # 0.10%

# Risk used in simulated position sizing.
RISK_PER_TRADE = 0.01  # 1%

# Maximum leverage for simulation.
MAX_LEVERAGE = 1.0

# Backtest opens at most one new trade per prediction horizon.
ENTRY_INTERVAL_MINUTES = HORIZON_MINUTES

# --- ATR-based risk management ---
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.25  # R:R = 1:1.5

# حداقل و حداکثر فاصله‌ی SL به‌عنوان درصد قیمت
MIN_STOP_LOSS_PCT = 0.0015  # 0.15%
MAX_STOP_LOSS_PCT = 0.0100  # 1.00%

MAX_HOLDING_STEPS = HORIZON_STEPS

EPOCHS = 100
BATCH_SIZE = 128
PATIENCE = 10

OUTPUT_DIR = "/content/drive/MyDrive/model_outputs_gru"

# --- Data filter (اختیاری) ---
# None = کل دیتا (8 سال)
# "2023-01-01" = فقط 3 سال آخر (سریع‌تر، رژیم فعلی)
MIN_DATA_DATE = "2023-01-01"

# --- Quick Test ---
QUICK_TEST = False
QUICK_TEST_ROWS = 20_000
QUICK_TEST_EPOCHS = 5

# --- Hyperparameter Search ---
USE_HYPERPARAMETER_SEARCH = False
HP_SEARCH_MAX_TRIALS = 20
HP_SEARCH_EPOCHS_PER_TRIAL = 30


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA CONFIG
# ============================================================

@dataclass
class DataConfig:
    timeframe_minutes: int = TIMEFRAME_MINUTES
    horizon_minutes: int = HORIZON_MINUTES
    horizon_steps: int = HORIZON_STEPS
    sequence_length: int = SEQUENCE_LENGTH


# ============================================================
# DATA LOADING
# ============================================================

def load_and_preprocess_data(file_path):
    """
    Load raw 5-minute OHLCV CSV.

    Required columns:
        open_time, open, high, low, close, volume
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_csv(file_path)

    required_columns = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["open_time"] = pd.to_datetime(data["open_time"], errors="coerce")
    data = data.dropna(subset=["open_time"])
    data = data.sort_values("open_time")
    data = data.drop_duplicates(subset=["open_time"], keep="last")
    data = data.set_index("open_time")
    data = data[["open", "high", "low", "close", "volume"]].copy()

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()

    invalid_ohlc = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    if invalid_ohlc.any():
        count = int(invalid_ohlc.sum())
        print(f"[WARNING] Removing {count} invalid OHLC rows.")
        data = data.loc[~invalid_ohlc]

    data = data[
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["volume"] >= 0)
    ]

    if data.empty:
        raise ValueError("No valid OHLCV data remains.")

    return data


# ============================================================
# DATA QUALITY
# ============================================================

def check_candle_gaps(data, timeframe_minutes):
    """بررسی gapهای زمانی. فقط گزارش می‌دهد."""
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)
    deltas = data.index.to_series().diff()
    gaps = deltas[deltas > expected_delta]

    print(f"[DATA] Rows: {len(data):,}")
    print(f"[DATA] Start: {data.index.min()}")
    print(f"[DATA] End: {data.index.max()}")
    print(f"[DATA] Gaps larger than {timeframe_minutes}m: {len(gaps):,}")

    if len(gaps) > 0:
        print("[DATA] First gaps:")
        print(gaps.head(10))


# ============================================================
# TARGET
# ============================================================

def create_future_log_return_target(close_series, horizon_steps):
    """
    Target: log(close[t+h] / close[t])

    چک gap:
        اگه بین t و t+h یه gap زمانی باشه (Binance downtime)،
        این target نامعتبره چون بازه‌ی زمانی‌اش h دقیقه نیست.
        این تابع اون ردیف‌ها رو NaN می‌کنه.
    """
    future_close = close_series.shift(-horizon_steps)
    target = np.log(future_close / close_series)

    if isinstance(close_series.index, pd.DatetimeIndex):
        expected_delta = pd.Timedelta(
            minutes=TIMEFRAME_MINUTES * horizon_steps
        )
        actual_delta = (
            close_series.index.to_series().shift(-horizon_steps)
            - close_series.index.to_series()
        )
        target = target.where(actual_delta.eq(expected_delta))

    return target


# ============================================================
# LAZY SEQUENCE (RAM-efficient، سازگار با Keras 3)
# ============================================================

class LazySequence(tf.keras.utils.Sequence):
    """
    Memory-safe sequence generator.

    به جای ساختن کل تانسور N × seq_length × n_features در RAM،
    فقط یک batch در لحظه ساخته می‌شه.

    Anchorها از آخرین timestep = t استفاده می‌کنند (شامل خود t).
    """

    def __init__(
        self,
        features_array,
        target_array,
        seq_length,
        anchor_start=None,
        anchor_end=None,
        batch_size=BATCH_SIZE,
        shuffle=False,
    ):
        self.features = np.asarray(features_array, dtype=np.float32)
        self.target = np.asarray(target_array, dtype=np.float32)
        self.seq_length = int(seq_length)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

        if len(self.features) != len(self.target):
            raise ValueError(
                "features_array and target_array must have equal length"
            )

        lo = (
            self.seq_length - 1
            if anchor_start is None
            else max(self.seq_length - 1, int(anchor_start))
        )
        hi = (
            len(self.features) - 1
            if anchor_end is None
            else min(len(self.features) - 1, int(anchor_end))
        )

        if hi < lo:
            anchors = np.empty(0, dtype=np.int64)
        else:
            anchors = np.arange(lo, hi + 1, dtype=np.int64)
            finite = np.isfinite(self.target[anchors])
            if finite.any():
                anchors = anchors[finite]
            else:
                anchors = np.empty(0, dtype=np.int64)

        self.anchors = anchors
        self.indices = np.arange(len(self.anchors), dtype=np.int64)

    def __len__(self):
        if len(self.indices) == 0:
            return 0
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_idx):
        sl = self.indices[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        anchors = self.anchors[sl]
        X = np.empty(
            (len(anchors), self.seq_length, self.features.shape[1]),
            dtype=np.float32,
        )
        for j, t in enumerate(anchors):
            X[j] = self.features[t - self.seq_length + 1 : t + 1]
        y = self.target[anchors].astype(np.float32, copy=False)
        return X, y

    def on_epoch_end(self):
        if self.shuffle and len(self.indices) > 1:
            np.random.shuffle(self.indices)

    @property
    def n_samples(self):
        return len(self.anchors)
# ============================================================
# FAST TF.DATASET (به جای LazySequence کند)
# ============================================================

def compute_valid_anchor_mask(features_array, target_array, seq_length):
    """
    نسخه‌ی برداری برای پیدا کردن anchorهای معتبر.
    بدون حلقه، بدون کپی سنگین.
    """
    n = len(features_array)

    row_finite = np.all(np.isfinite(features_array), axis=1)
    invalid = (~row_finite).astype(np.int32)
    cumsum = np.concatenate(([0], np.cumsum(invalid)))

    valid_mask = np.zeros(n, dtype=bool)

    if n >= seq_length:
        idx = np.arange(seq_length - 1, n)
        window_start = idx - seq_length + 1
        window_invalid_count = cumsum[idx + 1] - cumsum[window_start]
        window_all_finite = window_invalid_count == 0
        target_finite = np.isfinite(target_array[idx])
        valid_mask[idx] = window_all_finite & target_finite

    return valid_mask


def build_tf_dataset(
    features_array,
    target_array,
    seq_length,
    batch_size,
    anchor_start=None,
    anchor_end=None,
    shuffle=False,
):
    """
    tf.data.Dataset با prefetch برای سرعت.
    برمی‌گردونه: (dataset, valid_anchors)
    """
    features_array = np.asarray(features_array, dtype=np.float32)
    target_array = np.asarray(target_array, dtype=np.float32)

    valid_mask = compute_valid_anchor_mask(
        features_array, target_array, seq_length
    )
    valid_anchors = np.nonzero(valid_mask)[0].astype(np.int64)

    lo = (
        seq_length - 1
        if anchor_start is None
        else max(seq_length - 1, int(anchor_start))
    )
    hi = (
        len(features_array) - 1
        if anchor_end is None
        else min(len(features_array) - 1, int(anchor_end))
    )

    valid_anchors = valid_anchors[
        (valid_anchors >= lo) & (valid_anchors <= hi)
    ]

    n_features = features_array.shape[1]

    def gen():
        for i in valid_anchors:
            window = features_array[i - seq_length + 1 : i + 1]
            yield window, target_array[i]

    output_signature = (
        tf.TensorSpec(
            shape=(seq_length, n_features),
            dtype=tf.float32,
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature,
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=10000,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, valid_anchors    
# ============================================================
# MODEL
# ============================================================

def build_gru_model(
    input_shape,
    gru_units=64,
    dropout_rate=0.20,
    num_heads=4,
    key_dim=16,
    l2_reg=1e-4,
    dense_units=32,
    learning_rate=1e-3,
):
    """
    GRU + GRU + Multi-Head Attention.

    Input:  [batch, sequence_length, features]
    Output: one scalar = predicted future log return
    """
    inputs = Input(shape=input_shape, name="sequence_input")

    x = GRU(
        gru_units,
        activation="tanh",
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg),
        name="gru_1",
    )(inputs)

    x = Dropout(dropout_rate, name="dropout_1")(x)

    x = GRU(
        gru_units,
        activation="tanh",
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg),
        name="gru_2",
    )(x)

    x = Dropout(dropout_rate, name="dropout_2")(x)

    # Self-attention: Query = Key = Value = GRU sequence
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name="self_attention",
    )(x, x)

    # Residual connection
    x = LayerNormalization(name="attention_norm")(x + attention_output)

    # Aggregate across sequence
    x = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    x = Dense(
        dense_units,
        activation="tanh",
        kernel_regularizer=l2(l2_reg),
        name="dense_1",
    )(x)

    x = Dropout(dropout_rate, name="dense_dropout")(x)

    outputs = Dense(1, activation="linear", name="future_log_return")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ETH_GRU_Attention")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=Huber(delta=0.01),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    return model


# ============================================================
# LEARNING RATE
# ============================================================

def build_warmup_cosine_schedule(
    total_epochs,
    warmup_epochs=5,
    base_lr=1e-3,
    min_lr=1e-6,
):
    """Warmup + cosine decay."""

    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs) / max(
            1, total_epochs - warmup_epochs
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))

        return min_lr + (base_lr - min_lr) * cosine_decay

    return LearningRateScheduler(schedule, verbose=0)


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_save_path="best_model.keras",
    verbose=1,
    steps_per_epoch=None,
    validation_steps=None,
):
    """
    Train with validation and early stopping.

    برای tf.data.Dataset باید steps_per_epoch و validation_steps پاس بشن
    وگرنه Keras 3 با خطای "ran out of data" متوقف می‌شه.
    """
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1 if verbose else 0,
    )

    lr_schedule = build_warmup_cosine_schedule(
        total_epochs=epochs,
        warmup_epochs=5,
        base_lr=1e-3,
        min_lr=1e-6,
    )

    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1 if verbose else 0,
    )

    is_sequence = isinstance(
        X_train, (tf.keras.utils.Sequence, tf.data.Dataset)
    )

    if is_sequence:
        history = model.fit(
            X_train,
            validation_data=X_val,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stopping, lr_schedule, checkpoint],
            verbose=verbose,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, lr_schedule, checkpoint],
            verbose=verbose,
            shuffle=False,
        )

    return model, history


# ============================================================
# METRICS
# ============================================================

def directional_accuracy(y_true, y_pred):
    """درصد مواردی که مدل جهت حرکت را درست تشخیص داده."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return np.nan

    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def correlation(y_true, y_pred):
    """Correlation between predicted and actual returns."""
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_return_predictions(y_true, y_pred):
    """Evaluation مخصوص future return."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) == 0:
        return {}

    mse = mean_squared_error(y_true, y_pred)

    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
        "Correlation": correlation(y_true, y_pred),
    }


# ============================================================
# NAIVE BASELINE
# ============================================================

def naive_zero_return_metrics(y_true):
    """
    Naive baseline: future return = 0
    یعنی future price = current price.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.zeros_like(y_true)
    return evaluate_return_predictions(y_true, y_pred)


# ============================================================
# PREDICTION → PRICE
# ============================================================

def log_return_to_future_price(anchor_prices, predicted_log_returns):
    """
    future_price = anchor_price * exp(predicted_log_return)

    این تابع دقیقاً همونیه که crypto-signal-api باید موقع inference
    صدا بزنه تا خروجی مدل رو به یک قیمت واقعی تبدیل کنه.
    """
    return np.asarray(anchor_prices) * np.exp(
        np.asarray(predicted_log_returns)
    )


# ============================================================
# BACKTEST HELPERS
# ============================================================

def calculate_max_drawdown(equity_curve):
    """Max drawdown percentage."""
    equity = np.asarray(equity_curve, dtype=float)

    if len(equity) == 0:
        return np.nan

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    return float(drawdown.min() * 100)


def calculate_profit_factor(trade_returns):
    """Gross profit / gross loss."""
    returns = np.asarray(trade_returns, dtype=float)

    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()

    if gross_loss <= 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return float(gross_profit / gross_loss)


def calculate_sharpe(trade_returns):
    """
    Sharpe approximation based on trade returns.

    این Sharpe نهایی production نیست.
    فقط برای مقایسه‌ی اولیه‌ی مدل‌ها استفاده می‌شود.
    """
    returns = np.asarray(trade_returns, dtype=float)

    if len(returns) < 2:
        return np.nan

    std = returns.std(ddof=1)
    if std == 0:
        return np.nan

    return float(returns.mean() / std * np.sqrt(len(returns)))


# ============================================================
# ATR-BASED STOP-LOSS / TAKE-PROFIT
# ============================================================

def compute_atr_based_risk_distances(
    atr_values,
    anchor_prices,
    atr_multiplier_sl=ATR_MULTIPLIER_SL,
    atr_multiplier_tp=ATR_MULTIPLIER_TP,
    min_stop_loss_pct=MIN_STOP_LOSS_PCT,
    max_stop_loss_pct=MAX_STOP_LOSS_PCT,
):
    """
    فاصله‌ی SL/TP هر معامله را از ATR همان لحظه (anchor) محاسبه می‌کند.

    خروجی: دو آرایه (stop_loss_pct_array, take_profit_pct_array)
    """
    atr_values = np.asarray(atr_values, dtype=float)
    anchor_prices = np.asarray(anchor_prices, dtype=float)

    atr_pct = np.where(
        anchor_prices > 0,
        atr_values / anchor_prices,
        np.nan,
    )

    stop_loss_pct = atr_multiplier_sl * atr_pct
    stop_loss_pct = np.clip(
        stop_loss_pct, min_stop_loss_pct, max_stop_loss_pct
    )

    # TP با همون risk/reward ratio نسبت به SL کلیپ‌شده
    risk_reward_ratio = atr_multiplier_tp / atr_multiplier_sl
    take_profit_pct = stop_loss_pct * risk_reward_ratio

    return stop_loss_pct, take_profit_pct


# ============================================================
# REALISTIC MVP BACKTEST
# ============================================================

def backtest_strategy(
    anchor_prices,
    actual_future_returns,
    predicted_returns,
    atr_values,
    future_highs=None,
    future_lows=None,
    timestamps=None,
    initial_capital=INITIAL_CAPITAL,
    fee_rate=FEE_RATE,
    slippage_rate=SLIPPAGE_RATE,
    signal_threshold=SIGNAL_THRESHOLD,
    risk_per_trade=RISK_PER_TRADE,
    atr_multiplier_sl=ATR_MULTIPLIER_SL,
    atr_multiplier_tp=ATR_MULTIPLIER_TP,
    max_leverage=MAX_LEVERAGE,
    entry_interval_minutes=ENTRY_INTERVAL_MINUTES,
):
    """
    MVP trading simulation.

    Signal:
        prediction > threshold → LONG
        prediction < -threshold → SHORT
        otherwise → NO TRADE

    Stop/TP:
        نسخه‌ی مسیر-محور (path-dependent):
        اگه future_highs/future_lows داده بشه، کندل‌به‌کندل چک می‌کنه
        SL/TP کجا واقعاً لمس شده.

        اگه در یک کندل *هم* SL *هم* TP لمس بشن، فرض بدبینانه:
        SL زودتر خورده.

    Position sizing:
        risk_per_trade / stop_loss_pct

    Entry interval:
        بین دو trade حداقل entry_interval_minutes فاصله.
    """
    anchor_prices = np.asarray(anchor_prices, dtype=float)
    actual_future_returns = np.asarray(actual_future_returns, dtype=float)
    predicted_returns = np.asarray(predicted_returns, dtype=float)
    atr_values = np.asarray(atr_values, dtype=float)

    if not (
        len(anchor_prices)
        == len(actual_future_returns)
        == len(predicted_returns)
        == len(atr_values)
    ):
        raise ValueError("Backtest input arrays must have equal length.")

    stop_loss_pct_arr, take_profit_pct_arr = compute_atr_based_risk_distances(
        atr_values,
        anchor_prices,
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
    )

    capital = float(initial_capital)
    equity_curve = [capital]
    trades = []
    gross_trade_returns = []
    last_entry_timestamp = None

    for i in range(len(predicted_returns)):
        prediction = predicted_returns[i]
        actual_return = actual_future_returns[i]
        anchor = anchor_prices[i]
        stop_loss_pct = stop_loss_pct_arr[i]
        take_profit_pct = take_profit_pct_arr[i]

        current_timestamp = (
            timestamps[i]
            if timestamps is not None and i < len(timestamps)
            else None
        )

        # Entry interval check
        if (
            last_entry_timestamp is not None
            and current_timestamp is not None
            and pd.Timestamp(current_timestamp)
            - pd.Timestamp(last_entry_timestamp)
            < pd.Timedelta(minutes=entry_interval_minutes)
        ):
            equity_curve.append(capital)
            continue

        # Validity check
        if not (
            np.isfinite(prediction)
            and np.isfinite(actual_return)
            and np.isfinite(anchor)
            and np.isfinite(stop_loss_pct)
            and np.isfinite(take_profit_pct)
            and anchor > 0
        ):
            continue

        # Signal
        if prediction >= signal_threshold:
            side = "LONG"
        elif prediction <= -signal_threshold:
            side = "SHORT"
        else:
            side = "NO TRADE"

        if side == "NO TRADE":
            equity_curve.append(capital)
            continue

        last_entry_timestamp = current_timestamp

        # Position size
        position_fraction = min(
            risk_per_trade / stop_loss_pct,
            max_leverage,
        )

        # Exit rule
        exit_reason = "TIME_EXIT"
        realized_move = (
            actual_return if side == "LONG" else -actual_return
        )

        if (
            future_highs is not None
            and future_lows is not None
            and i < len(future_highs)
        ):
            highs = np.asarray(future_highs[i], dtype=float)
            lows = np.asarray(future_lows[i], dtype=float)
            highs = highs[np.isfinite(highs)]
            lows = lows[np.isfinite(lows)]

            if len(highs) and len(lows):
                if side == "LONG":
                    sl_price = anchor * (1.0 - stop_loss_pct)
                    tp_price = anchor * (1.0 + take_profit_pct)
                    for h, l in zip(highs, lows):
                        hit_sl = l <= sl_price
                        hit_tp = h >= tp_price
                        if hit_sl and hit_tp:
                            realized_move = -stop_loss_pct
                            exit_reason = "STOP_LOSS_SAME_CANDLE"
                            break
                        if hit_sl:
                            realized_move = -stop_loss_pct
                            exit_reason = "STOP_LOSS"
                            break
                        if hit_tp:
                            realized_move = take_profit_pct
                            exit_reason = "TAKE_PROFIT"
                            break
                    else:
                        realized_move = actual_return
                else:  # SHORT
                    sl_price = anchor * (1.0 + stop_loss_pct)
                    tp_price = anchor * (1.0 - take_profit_pct)
                    for h, l in zip(highs, lows):
                        hit_sl = h >= sl_price
                        hit_tp = l <= tp_price
                        if hit_sl and hit_tp:
                            realized_move = -stop_loss_pct
                            exit_reason = "STOP_LOSS_SAME_CANDLE"
                            break
                        if hit_sl:
                            realized_move = -stop_loss_pct
                            exit_reason = "STOP_LOSS"
                            break
                        if hit_tp:
                            realized_move = take_profit_pct
                            exit_reason = "TAKE_PROFIT"
                            break
                    else:
                        realized_move = -actual_return
        else:
            # Fallback: فقط بر پایه‌ی بازده‌ی نهایی افق
            directional_return = (
                actual_return if side == "LONG" else -actual_return
            )
            if directional_return <= -stop_loss_pct:
                realized_move = -stop_loss_pct
                exit_reason = "STOP_LOSS"
            elif directional_return >= take_profit_pct:
                realized_move = take_profit_pct
                exit_reason = "TAKE_PROFIT"

        # Costs
        round_trip_slippage = 2 * slippage_rate
        round_trip_fee = 2 * fee_rate

        net_return_on_position = (
            realized_move - round_trip_slippage - round_trip_fee
        )

        account_return = position_fraction * net_return_on_position
        capital *= (1 + account_return)

        trade_record = {
            "timestamp": (
                timestamps[i]
                if timestamps is not None and i < len(timestamps)
                else None
            ),
            "side": side,
            "anchor_price": float(anchor),
            "predicted_return": float(prediction),
            "actual_future_return": float(actual_return),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "realized_directional_move": float(realized_move),
            "position_fraction": float(position_fraction),
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slippage_rate),
            "net_return_on_position": float(net_return_on_position),
            "account_return": float(account_return),
            "capital_after_trade": float(capital),
            "exit_reason": exit_reason,
        }

        trades.append(trade_record)
        gross_trade_returns.append(account_return)
        equity_curve.append(capital)

    # Summary
    trade_returns = np.asarray(gross_trade_returns, dtype=float)
    total_return_pct = ((capital / initial_capital) - 1) * 100

    if len(trade_returns) > 0:
        wins = trade_returns[trade_returns > 0]
        win_rate = len(wins) / len(trade_returns) * 100
        average_trade_pct = trade_returns.mean() * 100
    else:
        win_rate = np.nan
        average_trade_pct = np.nan

    if len(actual_future_returns) > 0:
        buy_hold_return_pct = (
            np.exp(np.sum(actual_future_returns)) - 1
        ) * 100
    else:
        buy_hold_return_pct = np.nan

    return {
        "initial_capital": float(initial_capital),
        "final_capital": float(capital),
        "total_return_pct": float(total_return_pct),
        "buy_hold_return_pct": float(buy_hold_return_pct),
        "n_trades": int(len(trades)),
        "win_rate_pct": float(win_rate),
        "average_trade_pct": float(average_trade_pct),
        "profit_factor": calculate_profit_factor(trade_returns),
        "sharpe": calculate_sharpe(trade_returns),
        "max_drawdown_pct": calculate_max_drawdown(equity_curve),
        "equity_curve": equity_curve,
        "trades": trades,
    }


# ============================================================
# PLOT
# ============================================================

def plot_prediction_returns(timestamps, y_true, y_pred, output_path):
    """Plot actual vs predicted future log return."""
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, y_true, label="Actual future log return")
    plt.plot(
        timestamps, y_pred, label="Predicted future log return",
        linestyle="--",
    )
    plt.axhline(0, linestyle=":")
    plt.title("GRU + Attention - 15m Future Return")
    plt.xlabel("Anchor time")
    plt.ylabel("Log return")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_equity_curve(equity_curve, output_path):
    """Plot simulated equity."""
    plt.figure(figsize=(12, 5))
    plt.plot(equity_curve, label="Strategy equity")
    plt.title("MVP Strategy Equity Curve")
    plt.xlabel("Trade / Decision")
    plt.ylabel("Capital")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    # ============================================================
# WALK FORWARD (با مرزهای تقویمی + tz-aware)
# ============================================================

def generate_walk_forward_folds(n_samples, n_folds=N_FOLDS, timestamps=None):
    """
    Expanding-window walk-forward با مرزهای تقویمی واقعی.

    برای دیتای 2018-2026:
        Fold 1: train 2018-2021 | val 2021-2022 | test 2022-2023
        Fold 2: train 2018-2022 | val 2022-2023 | test 2023-2024
        Fold 3: train 2018-2023 | val 2023-2024 | test 2024-2025
        Fold 4: train 2018-2024 | val 2024-2025 | test 2025-2026

    نکته: tz-aware timestamps پشتیبانی می‌شه (UTC).
    """
    if n_samples < 1000:
        raise ValueError(
            "Dataset is too small for reliable walk-forward validation."
        )
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    folds = []

    if timestamps is not None:
        idx = pd.DatetimeIndex(timestamps)
        if len(idx) != n_samples:
            raise ValueError("timestamps length must equal n_samples")
        if not idx.is_monotonic_increasing:
            raise ValueError("timestamps must be sorted ascending")

        start_year = int(idx[0].year)
        tz = idx.tz

        # طول دیتا رو حساب کن
        data_years = (idx[-1] - idx[0]).days / 365.25

        # اگه دیتا 6+ ساله → 3 سال train اولیه (استاندارد research)
        # اگه دیتا 4-6 ساله → 2 سال train
        # اگه دیتا کمتر از 4 ساله → 1 سال train
        if data_years >= 6:
            min_train_years = 3
        elif data_years >= 4:
            min_train_years = 2
        else:
            min_train_years = 1

        print(f"[WALK-FORWARD] Data span: {data_years:.1f} years, "
              f"min_train_years: {min_train_years}")

        for fold in range(n_folds):
            train_end_ts = pd.Timestamp(
                year=start_year + min_train_years + fold, month=1, day=1
            )
            val_end_ts = train_end_ts + pd.DateOffset(years=1)
            test_end_ts = val_end_ts + pd.DateOffset(years=1)

            if tz is not None:
                train_end_ts = train_end_ts.tz_localize(tz)
                val_end_ts = val_end_ts.tz_localize(tz)
                test_end_ts = test_end_ts.tz_localize(tz)

            train_end = int(idx.searchsorted(train_end_ts, side="left"))
            val_end = int(idx.searchsorted(val_end_ts, side="left"))
            test_end = int(idx.searchsorted(test_end_ts, side="left"))

            if train_end <= 0 or val_end <= train_end or test_end <= val_end:
                break
            if test_end > n_samples:
                test_end = n_samples
            if val_end >= n_samples or test_end <= val_end:
                break

            folds.append((0, train_end, val_end, test_end))

        return folds

    # Fallback اگه timestamps ندادی
    block = n_samples // (n_folds + 2)
    min_train = block * 3
    for fold in range(n_folds):
        train_end = min_train + fold * block
        val_end = train_end + block
        test_end = min(val_end + block, n_samples)
        if test_end <= val_end or train_end <= 0:
            break
        folds.append((0, train_end, val_end, test_end))
    return folds


# ============================================================
# PURGE (رفع نشتی مرزی بین Train/Val/Test)
# ============================================================

def purge_target_tail(target_series, horizon_steps):
    """
    چون target[i] = log(close[i+horizon_steps] / close[i])،
    آخرین چند ردیف هر split از قیمت‌هایی محاسبه می‌شن که در واقع
    در بازه‌ی زمانی split بعدی اتفاق افتادن.

    این تابع آخرین horizon_steps مقدار target رو NaN می‌کنه؛
    LazySequence مقادیر non-finite رو حذف می‌کنه.
    """
    if horizon_steps <= 0 or len(target_series) == 0:
        return target_series

    purged = target_series.copy()
    n_to_purge = min(horizon_steps, len(purged))
    purged.iloc[-n_to_purge:] = np.nan
    return purged


# ============================================================
# BUILD FOLD DATA
# ============================================================

def prepare_fold_data(
    features_df,
    target_series,
    train_end,
    val_end,
    test_end,
    sequence_length,
    horizon_steps=0,
):
    """
    برای validation و test، history قبل از split رو نگه می‌داریم
    تا اولین sequenceها context واقعی داشته باشن.

    اما scaler فقط روی train fit می‌شود.
    """
    # Train
    train_df = features_df.iloc[:train_end]
    train_target = target_series.iloc[:train_end]
    train_target = purge_target_tail(train_target, horizon_steps)

    # Validation (با context از قبل train)
    val_context_start = max(0, train_end - sequence_length)
    val_df = features_df.iloc[val_context_start:val_end]
    val_target = target_series.iloc[val_context_start:val_end]
    val_target = purge_target_tail(val_target, horizon_steps)

    # Test (با context از قبل val)
    test_context_start = max(0, val_end - sequence_length)
    test_df = features_df.iloc[test_context_start:test_end]
    test_target = target_series.iloc[test_context_start:test_end]

    return (
        train_df,
        train_target,
        val_df,
        val_target,
        test_df,
        test_target,
        val_context_start,
        test_context_start,
    )


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def build_gru_model_tunable(hp, input_shape):
    """Optional KerasTuner model. برای MVP خاموش است."""
    gru_units = hp.Choice("gru_units", values=[32, 64, 128])
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.4, step=0.1
    )
    num_heads = hp.Choice("num_heads", values=[2, 4])
    key_dim = hp.Choice("key_dim", values=[8, 16, 32])
    l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])
    dense_units = hp.Choice("dense_units", values=[16, 32, 64])
    learning_rate = hp.Choice("learning_rate", values=[1e-3, 3e-4, 1e-4])

    return build_gru_model(
        input_shape=input_shape,
        gru_units=gru_units,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        key_dim=key_dim,
        l2_reg=l2_reg,
        dense_units=dense_units,
        learning_rate=learning_rate,
    )


def run_hyperparameter_search(
    train_dataset,
    val_dataset,
    input_shape,
    output_dir,
    max_trials=20,
    epochs_per_trial=30,
):
    """Optional Bayesian Optimization."""
    import keras_tuner as kt

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_gru_model_tunable(hp, input_shape),
        objective="val_loss",
        max_trials=max_trials,
        directory=os.path.join(output_dir, "kt_search"),
        project_name="gru_attention",
        overwrite=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    is_sequence = isinstance(
        train_dataset, (tf.keras.utils.Sequence, tf.data.Dataset)
    )
    if is_sequence:
        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs_per_trial,
            callbacks=[early_stopping],
            verbose=1,
        )
    else:
        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs_per_trial,
            callbacks=[early_stopping],
            verbose=1,
        )

    return tuner.get_best_hyperparameters(num_trials=1)[0]


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    output_path,
    fold_idx,
    scaler,
    train_date_range=None,
    test_date_range=None,
):
    """
    Save model metadata.

    این metadata بعداً در crypto-signal-api برای جلوگیری از mismatch
    بین model و feature pipeline استفاده خواهد شد.
    """
    metadata = {
        "model_name": "eth_gru_attention",
        "model_version": "gru_attention_mvp_v1",
        "symbol": SYMBOL,
        "timeframe_minutes": TIMEFRAME_MINUTES,
        "horizon_minutes": HORIZON_MINUTES,
        "horizon_steps": HORIZON_STEPS,
        "sequence_length": SEQUENCE_LENGTH,
        "target": "future_log_return",
        "features_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "architecture": {
            "type": "GRU_GRU_MultiHeadAttention",
            "gru_units": 64,
            "num_heads": 4,
            "key_dim": 16,
            "dropout": 0.20,
        },
        "training": {
            "loss": "Huber",
            "optimizer": "Adam",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "backtest": {
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
            "signal_threshold": SIGNAL_THRESHOLD,
            "risk_per_trade": RISK_PER_TRADE,
            "atr_multiplier_sl": ATR_MULTIPLIER_SL,
            "atr_multiplier_tp": ATR_MULTIPLIER_TP,
            "min_stop_loss_pct": MIN_STOP_LOSS_PCT,
            "max_stop_loss_pct": MAX_STOP_LOSS_PCT,
            "max_leverage": MAX_LEVERAGE,
            "entry_interval_minutes": ENTRY_INTERVAL_MINUTES,
            "execution_path": "5m_high_low_path_when_available",
        },
        "fold": fold_idx,
        "train_date_range": {
            "start": train_date_range[0] if train_date_range else None,
            "end": train_date_range[1] if train_date_range else None,
        },
        "test_date_range": {
            "start": test_date_range[0] if test_date_range else None,
            "end": test_date_range[1] if test_date_range else None,
        },
        "scaler": {
            "type": "StandardScaler",
            "n_features": int(scaler.n_features_in_),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed()

    # --- Mixed precision (نصف کردن VRAM) ---
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # --------------------------------------------------------
    # DATA PATH
    # --------------------------------------------------------

    file_path = "/content/drive/MyDrive/binance_data_5min.csv"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ETH GRU + ATTENTION MVP")
    print("=" * 70)

    print(f"Symbol       : {SYMBOL}")
    print(f"Timeframe    : {TIMEFRAME_MINUTES}m")
    print(f"Horizon      : {HORIZON_MINUTES}m")
    print(f"Horizon steps: {HORIZON_STEPS}")
    print(f"Sequence     : {SEQUENCE_LENGTH}")
    print(f"Feature ver. : {FEATURE_VERSION}")
    print(f"Target       : future_log_return")
    print(
        f"Risk model   : ATR-based SL "
        f"({ATR_MULTIPLIER_SL}x) / TP ({ATR_MULTIPLIER_TP}x)"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_and_preprocess_data(file_path)

    print(
        f"\n[DATA] Full range loaded: {data.index.min()} -> "
        f"{data.index.max()} ({len(data):,} rows)"
    )

    if MIN_DATA_DATE is not None:
        data = data[data.index >= MIN_DATA_DATE]
        print(
            f"[DATA] Filtered to >= {MIN_DATA_DATE}: "
            f"{len(data):,} rows remaining"
        )

    # --- Quick Test ---
    effective_epochs = EPOCHS
    if QUICK_TEST:
        data = data.iloc[-QUICK_TEST_ROWS:]
        effective_epochs = QUICK_TEST_EPOCHS
        print(
            f"[QUICK TEST MODE] Using only last {len(data):,} rows, "
            f"epochs limited to {effective_epochs}."
        )

    check_candle_gaps(data, TIMEFRAME_MINUTES)

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_data = add_technical_indicators(data.copy())

    if feature_data.empty:
        raise ValueError("Feature dataframe is empty.")

    validate_feature_schema(feature_data)

    missing = [c for c in FEATURE_COLUMNS if c not in feature_data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # ATR رو جدا نگه می‌داریم (برای backtest)
    atr_series_full = feature_data["ATR"].copy()

    feature_data = (
        feature_data[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    atr_series_full = atr_series_full.reindex(feature_data.index)

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target = create_future_log_return_target(
        feature_data["close"], HORIZON_STEPS
    )
    target = target.reindex(feature_data.index)

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    n_samples = len(feature_data)
       
    folds = generate_walk_forward_folds(
        n_samples,
        N_FOLDS,
        timestamps=feature_data.index,
    )

    print(f"\n[DATA] Total usable rows: {n_samples:,}")
    print(f"[WALK-FORWARD] Folds: {len(folds)}")

    fold_results = []
    backtest_results = []

    # ========================================================
    # FOLD LOOP
    # ========================================================

    for fold_idx, (
        train_start,
        val_start,
        test_start,
        test_end,
    ) in enumerate(folds, start=1):

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold_idx}/{len(folds)}")

        # بازه‌ی زمانی واقعی هر fold
        effective_train_end = max(0, val_start - HORIZON_STEPS)
        effective_val_end = max(val_start, test_start - HORIZON_STEPS)

        train_start_date = feature_data.index[train_start]
        train_end_date = feature_data.index[max(0, effective_train_end - 1)]
        val_start_date = feature_data.index[val_start]
        val_end_date = feature_data.index[max(0, effective_val_end - 1)]
        test_start_date = feature_data.index[test_start]
        test_end_date = feature_data.index[
            min(test_end, len(feature_data)) - 1
        ]

        print(
            f"Train: 0 -> {val_start}  "
            f"({train_start_date} -> {train_end_date})"
        )
        print(
            f"Val  : {val_start} -> {test_start}  "
            f"({val_start_date} -> {val_end_date})"
        )
        print(
            f"Test : {test_start} -> {test_end}  "
            f"({test_start_date} -> {test_end_date})"
        )
        print("=" * 70)

        (
            train_df,
            train_target,
            val_df,
            val_target,
            test_df,
            test_target,
            val_context_start,
            test_context_start,
        ) = prepare_fold_data(
            feature_data,
            target,
            train_end=val_start,
            val_end=test_start,
            test_end=test_end,
            sequence_length=SEQUENCE_LENGTH,
            horizon_steps=HORIZON_STEPS,
        )

        # ----------------------------------------------------
        # SCALER (فقط روی train)
        # ----------------------------------------------------

        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(train_df.values)
        val_scaled = scaler.transform(val_df.values)
        test_scaled = scaler.transform(test_df.values)

        scaler_path = os.path.join(
            OUTPUT_DIR, f"scaler_fold{fold_idx}.pkl"
        )
        joblib.dump(scaler, scaler_path)

        train_seq, train_anchors = build_tf_dataset(
            train_scaled,
            train_target.values,
            SEQUENCE_LENGTH,
            BATCH_SIZE,
            shuffle=False,
        )

        val_anchor_start = max(
            SEQUENCE_LENGTH - 1,
            val_start - val_context_start,
        )
        val_seq, val_anchors_local = build_tf_dataset(
            val_scaled,
            val_target.values,
            SEQUENCE_LENGTH,
            BATCH_SIZE,
            anchor_start=val_anchor_start,
            shuffle=False,
        )

        test_anchor_start = max(
            SEQUENCE_LENGTH - 1,
            test_start - test_context_start,
        )
        test_seq, test_anchors_local = build_tf_dataset(
            test_scaled,
            test_target.values,
            SEQUENCE_LENGTH,
            BATCH_SIZE,
            anchor_start=test_anchor_start,
            shuffle=False,
        )

        print(f"[SEQUENCES] Train: {len(train_anchors):,}")
        print(f"[SEQUENCES] Val  : {len(val_anchors_local):,}")
        print(f"[SEQUENCES] Test : {len(test_anchors_local):,}")

        if (
            len(train_anchors) == 0
            or len(val_anchors_local) == 0
            or len(test_anchors_local) == 0
        ):
            print(
                "[WARNING] Not enough sequence data. Skipping fold."
            )
            continue

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        input_shape = (
            SEQUENCE_LENGTH,
            train_scaled.shape[1],
        )

        if USE_HYPERPARAMETER_SEARCH and fold_idx == len(folds):
            print("\n[HYPERPARAMETER SEARCH]")
            best_hp = run_hyperparameter_search(
                train_seq,
                val_seq,
                input_shape,
                OUTPUT_DIR,
                HP_SEARCH_MAX_TRIALS,
                HP_SEARCH_EPOCHS_PER_TRIAL,
            )
            model = build_gru_model(
                input_shape=input_shape,
                gru_units=best_hp.get("gru_units"),
                dropout_rate=best_hp.get("dropout_rate"),
                num_heads=best_hp.get("num_heads"),
                key_dim=best_hp.get("key_dim"),
                l2_reg=best_hp.get("l2_reg"),
                dense_units=best_hp.get("dense_units"),
                learning_rate=best_hp.get("learning_rate"),
            )
        else:
            model = build_gru_model(input_shape=input_shape)

        model_path = os.path.join(
            OUTPUT_DIR, f"best_gru_attention_fold{fold_idx}.keras"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        steps_per_epoch = max(1, len(train_anchors) // BATCH_SIZE)
        validation_steps = max(1, len(val_anchors_local) // BATCH_SIZE)

        model, history = train_model(
            model,
            train_seq,
            None,
            val_seq,
            None,
            epochs=effective_epochs,
            batch_size=BATCH_SIZE,
            model_save_path=model_path,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

        # ----------------------------------------------------
        # TEST PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(test_seq, verbose=0).reshape(-1)
        y_test_flat = test_target.values[test_anchors_local]

        # ----------------------------------------------------
        # PREDICTION METRICS
        # ----------------------------------------------------

        metrics = evaluate_return_predictions(y_test_flat, predictions)
        naive_metrics = naive_zero_return_metrics(y_test_flat)

        print("\n[MODEL METRICS]")
        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")

        print("\n[NAIVE ZERO-RETURN BASELINE]")
        for key in [
            "RMSE", "MAE", "R2", "Directional_Accuracy", "Correlation",
        ]:
            value = naive_metrics.get(key, np.nan)
            print(f"{key}: {value:.6f}")

        # ----------------------------------------------------
        # ANCHOR PRICES + ATR + TIMESTAMPS
        # ----------------------------------------------------

        global_test_indices = test_context_start + test_anchors_local
        test_times = test_df.index[test_anchors_local]

        anchor_prices = (
            feature_data.iloc[global_test_indices]["close"].values
        )
        anchor_atr = atr_series_full.iloc[global_test_indices].values

        # ----------------------------------------------------
        # FUTURE HIGH/LOW (برای بک‌تست مسیر-محور)
        # ----------------------------------------------------

        raw_high = feature_data["high"].values
        raw_low = feature_data["low"].values

        future_highs = np.stack(
            [
                raw_high[global_test_indices + j]
                for j in range(1, HORIZON_STEPS + 1)
            ],
            axis=1,
        )
        future_lows = np.stack(
            [
                raw_low[global_test_indices + j]
                for j in range(1, HORIZON_STEPS + 1)
            ],
            axis=1,
        )

        # ----------------------------------------------------
        # BACKTEST
        # ----------------------------------------------------

        bt = backtest_strategy(
            anchor_prices=anchor_prices,
            actual_future_returns=y_test_flat,
            predicted_returns=predictions,
            atr_values=anchor_atr,
            future_highs=future_highs,
            future_lows=future_lows,
            timestamps=test_times,
            initial_capital=INITIAL_CAPITAL,
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
            signal_threshold=SIGNAL_THRESHOLD,
            risk_per_trade=RISK_PER_TRADE,
            atr_multiplier_sl=ATR_MULTIPLIER_SL,
            atr_multiplier_tp=ATR_MULTIPLIER_TP,
            max_leverage=MAX_LEVERAGE,
            entry_interval_minutes=ENTRY_INTERVAL_MINUTES,
        )

        print("\n[BACKTEST]")
        print(f"Final capital       : {bt['final_capital']:.2f}")
        print(f"Strategy return     : {bt['total_return_pct']:.2f}%")
        print(f"Buy & Hold          : {bt['buy_hold_return_pct']:.2f}%")
        print(f"Trades              : {bt['n_trades']}")
        print(f"Win rate            : {bt['win_rate_pct']:.2f}%")
        print(f"Average trade       : {bt['average_trade_pct']:.4f}%")
        print(f"Profit factor       : {bt['profit_factor']}")
        print(f"Sharpe              : {bt['sharpe']}")
        print(f"Max drawdown        : {bt['max_drawdown_pct']:.2f}%")

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        final_model_path = os.path.join(
            OUTPUT_DIR, f"model_GRU_Attention_fold{fold_idx}.keras"
        )
        model.save(final_model_path)

        # ----------------------------------------------------
        # SAVE METADATA
        # ----------------------------------------------------

        metadata_path = os.path.join(
            OUTPUT_DIR, f"metadata_fold{fold_idx}.json"
        )
        save_metadata(
            metadata_path,
            fold_idx,
            scaler,
            train_date_range=(
                str(train_start_date),
                str(train_end_date),
            ),
            test_date_range=(
                str(test_start_date),
                str(test_end_date),
            ),
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        prediction_df = pd.DataFrame({
            "timestamp": test_times,
            "anchor_price": anchor_prices,
            "anchor_atr": anchor_atr,
            "actual_future_log_return": y_test_flat,
            "predicted_future_log_return": predictions,
            "actual_future_return_pct": (np.exp(y_test_flat) - 1) * 100,
            "predicted_future_return_pct": (
                np.exp(predictions) - 1
            ) * 100,
        })
        prediction_path = os.path.join(
            OUTPUT_DIR, f"predictions_fold{fold_idx}.csv"
        )
        prediction_df.to_csv(prediction_path, index=False)

        # ----------------------------------------------------
        # SAVE TRADES
        # ----------------------------------------------------

        trades_df = pd.DataFrame(bt["trades"])
        trades_path = os.path.join(
            OUTPUT_DIR, f"trades_fold{fold_idx}.csv"
        )
        trades_df.to_csv(trades_path, index=False)

        # ----------------------------------------------------
        # PLOTS
        # ----------------------------------------------------

        plot_prediction_returns(
            test_times,
            y_test_flat,
            predictions,
            os.path.join(
                OUTPUT_DIR, f"prediction_returns_fold{fold_idx}.png"
            ),
        )
        plot_equity_curve(
            bt["equity_curve"],
            os.path.join(
                OUTPUT_DIR, f"equity_curve_fold{fold_idx}.png"
            ),
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        fold_result = {
            "fold": fold_idx,
            "train_start_date": str(train_start_date),
            "train_end_date": str(train_end_date),
            "val_start_date": str(val_start_date),
            "val_end_date": str(val_end_date),
            "test_start_date": str(test_start_date),
            "test_end_date": str(test_end_date),
            **metrics,
            "Naive_RMSE": naive_metrics["RMSE"],
            "Naive_MAE": naive_metrics["MAE"],
            "Naive_R2": naive_metrics["R2"],
            "Naive_Directional_Accuracy": naive_metrics[
                "Directional_Accuracy"
            ],
            "Strategy_Return_%": bt["total_return_pct"],
            "Buy_Hold_Return_%": bt["buy_hold_return_pct"],
            "Trades": bt["n_trades"],
            "Win_Rate_%": bt["win_rate_pct"],
            "Average_Trade_%": bt["average_trade_pct"],
            "Profit_Factor": bt["profit_factor"],
            "Sharpe": bt["sharpe"],
            "Max_Drawdown_%": bt["max_drawdown_pct"],
        }

        fold_results.append(fold_result)

        backtest_results.append({
            "fold": fold_idx,
            "test_start_date": str(test_start_date),
            "test_end_date": str(test_end_date),
            "strategy_return_%": bt["total_return_pct"],
            "buy_hold_return_%": bt["buy_hold_return_pct"],
            "n_trades": bt["n_trades"],
            "win_rate_%": bt["win_rate_pct"],
            "profit_factor": bt["profit_factor"],
            "sharpe": bt["sharpe"],
            "max_drawdown_%": bt["max_drawdown_pct"],
        })

        # ----------------------------------------------------
        # CLEANUP (رفع OOM بین Foldها)
        # ----------------------------------------------------

        del model
        del train_seq, val_seq, test_seq
        del train_scaled, val_scaled, test_scaled
        del future_highs, future_lows
        del predictions, y_test_flat

        tf.keras.backend.clear_session()
        gc.collect()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(fold_results)

    if not results_df.empty:
        print(results_df.to_string(index=False))
        results_path = os.path.join(
            OUTPUT_DIR, "gru_metrics_summary.csv"
        )
        results_df.to_csv(results_path, index=False)
        print(f"\nMetrics saved to:\n{results_path}")
    else:
        print("No fold results generated.")

    bt_df = pd.DataFrame(backtest_results)

    if not bt_df.empty:
        print("\n[BACKTEST SUMMARY]")
        print(bt_df.to_string(index=False))
        bt_path = os.path.join(
            OUTPUT_DIR, "backtest_summary.csv"
        )
        bt_df.to_csv(bt_path, index=False)
        print(f"\nBacktest saved to:\n{bt_path}")

    print("\nProcessing complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    