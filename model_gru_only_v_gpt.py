"""
model_gru_only.py
=================

MVP Research Model
------------------

ETHUSDT
5-minute candles
15-minute prediction horizon

Architecture:
    Historical OHLCV
        ?
    Technical Features
        ?
    Sequence
        ?
    GRU
        ?
    GRU
        ?
    Multi-Head Attention
        ?
    Dense
        ?
    Future Log Return

IMPORTANT:
    Prediction != Signal != Position

??? ???? ??? ????? research / training / backtesting ??? ???.
???? Signal Engine ? Position Management ????? ?? crypto-signal-api
?????????? ??????.

Target:
    future_log_return = log(close[t+horizon] / close[t])

?????:
    timeframe = 5m
    horizon = 15m
    horizon_steps = 3

???? ??? ?? sequence ?????? ????? ???? ?????? 15 ????? ????? ??
???????? ??????.

Sequence:
    sequence_length = 96

???? ???? ?? prediction ??? 96 ???? 5 ???????? ?? ???????:
    96 × 5m = 8 hours

Attention:
    Attention ??? Sequence ?? ????????.
    GRU ??? sequence ???? ?????? ? hidden state ????? ??????.
    Attention ??? ??????? ???? timestep??? sequence ???? prediction
    ???? ?????? ?????.

Validation:
    Walk-forward validation

Scaling:
    Scaler ??? ??? train ?? fold fit ??????.
    Validation ? Test ??? transform ???????.

Backtest:
    - LONG
    - SHORT
    - NO TRADE
    - fee
    - slippage
    - threshold
    - fixed-risk position sizing
    - max position loss protection
    - equity curve
    - drawdown
    - Sharpe ??????
    - profit factor
"""

import os
import json
import random
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
# Example:
# 0.0004 = 0.04%
FEE_RATE = 0.0004

# Conservative slippage assumption per side.
SLIPPAGE_RATE = 0.0002

# Model must predict at least this much expected movement
# before entering a trade.
#
# This is NOT necessarily the final production threshold.
# It is an MVP research parameter.
SIGNAL_THRESHOLD = 0.0010  # 0.10%

# Risk used in simulated position sizing.
RISK_PER_TRADE = 0.01  # 1%

# Maximum leverage for simulation.
# We keep this low for the MVP.
MAX_LEVERAGE = 1.0

# Stop distance used only by the MVP backtest.
# Later this should be replaced by ATR / volatility / structure based risk.
STOP_LOSS_PCT = 0.0030  # 0.30%

TAKE_PROFIT_PCT = 0.0045  # 0.45%

MAX_HOLDING_STEPS = HORIZON_STEPS

EPOCHS = 100
BATCH_SIZE = 64

PATIENCE = 10

OUTPUT_DIR = "/content/drive/MyDrive/model_outputs_gru"

# Set True only when you intentionally want tuner.
USE_HYPERPARAMETER_SEARCH = False

HP_SEARCH_MAX_TRIALS = 20
HP_SEARCH_EPOCHS_PER_TRIAL = 30


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    """
    ???? ???? reproducible ???? training.
    """
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
        open_time
        open
        high
        low
        close
        volume
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required_columns
        if c not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["open_time"] = pd.to_datetime(
        data["open_time"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["open_time"]
    )

    data = data.sort_values("open_time")

    # Remove duplicated timestamps
    data = data.drop_duplicates(
        subset=["open_time"],
        keep="last",
    )

    data = data.set_index("open_time")

    data = data[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    # Numeric conversion
    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna()

    # Basic OHLC sanity checks
    invalid_ohlc = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    if invalid_ohlc.any():
        count = int(invalid_ohlc.sum())

        print(
            f"[WARNING] Removing {count} invalid OHLC rows."
        )

        data = data.loc[~invalid_ohlc]

    # Price / volume must be positive
    data = data[
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["volume"] >= 0)
    ]

    if data.empty:
        raise ValueError(
            "No valid OHLCV data remains."
        )

    return data


# ============================================================
# DATA QUALITY
# ============================================================

def check_candle_gaps(data, timeframe_minutes):
    """
    ????? gap??? ?????.

    Gap ??????? ?? ???? ???? ???? ???? ????.
    ??? ????? ?????? ?? ?????? dataset ?? ???? ????.
    """

    expected_delta = pd.Timedelta(
        minutes=timeframe_minutes
    )

    deltas = data.index.to_series().diff()

    gaps = deltas[
        deltas > expected_delta
    ]

    print(
        f"[DATA] Rows: {len(data):,}"
    )

    print(
        f"[DATA] Start: {data.index.min()}"
    )

    print(
        f"[DATA] End: {data.index.max()}"
    )

    print(
        f"[DATA] Gaps larger than {timeframe_minutes}m: {len(gaps):,}"
    )

    if len(gaps) > 0:
        print(
            "[DATA] First gaps:"
        )
        print(gaps.head(10))


# ============================================================
# TARGET
# ============================================================

def create_future_log_return_target(
    close_series,
    horizon_steps,
):
    """
    Target:

        log(close[t+h] / close[t])

    ????? ?? index ???? t ???? ????.

    ????:

        t = 12:00
        horizon = 3 candles
        target = log(close[12:15] / close[12:00])
    """

    future_close = close_series.shift(
        -horizon_steps
    )

    target = np.log(
        future_close / close_series
    )

    return target


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences(
    features_array,
    target_array,
    seq_length,
):
    """
    ???? sequence ???? supervised learning.

    X:
        [i-seq_length : i]

    ????? timestep ?? X ????? ?? ???? t ???.

    y:
        target[t]

    ????????:

        X -> history ending at t
        y -> future return from t to t+horizon

    ????? ???? target ???? ??? ????.
    """

    X = []
    y = []

    if len(features_array) != len(target_array):
        raise ValueError(
            "features_array and target_array "
            "must have the same length."
        )

    for i in range(seq_length, len(features_array)):
        target_value = target_array[i]

        if not np.isfinite(target_value):
            continue

        sequence = features_array[
            i - seq_length:i
        ]

        if not np.all(np.isfinite(sequence)):
            continue

        X.append(sequence)
        y.append(target_value)

    if not X:
        return (
            np.empty(
                (0, seq_length, features_array.shape[1]),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


# ============================================================
# SEQUENCE CREATION WITH INDEX
# ============================================================

def create_sequences_with_indices(
    features_array,
    target_array,
    timestamps,
    seq_length,
):
    """
    ???? create_sequences ??? timestamp ????? ?? anchor ?? ?? ???????????.

    ??? ???? evaluation ? backtest ????? ??? ???.
    """

    X = []
    y = []
    anchor_indices = []
    anchor_times = []

    for i in range(seq_length, len(features_array)):
        target_value = target_array[i]

        if not np.isfinite(target_value):
            continue

        sequence = features_array[
            i - seq_length:i
        ]

        if not np.all(np.isfinite(sequence)):
            continue

        X.append(sequence)
        y.append(target_value)

        anchor_indices.append(i)
        anchor_times.append(timestamps[i])

    if not X:
        return (
            np.empty(
                (0, seq_length, features_array.shape[1]),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.int64,
            ),
            pd.DatetimeIndex([]),
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(anchor_indices, dtype=np.int64),
        pd.DatetimeIndex(anchor_times),
    )


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

    Sequence remains essential.

    Input:
        [batch, sequence_length, features]

    GRU #1:
        processes temporal sequence

    GRU #2:
        processes temporal representation

    Attention:
        learns which timesteps are important

    Global mean pooling:
        converts sequence representation to fixed-size vector

    Output:
        one scalar = predicted future log return
    """

    inputs = Input(
        shape=input_shape,
        name="sequence_input",
    )

    x = GRU(
        gru_units,
        activation="tanh",
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg),
        name="gru_1",
    )(inputs)

    x = Dropout(
        dropout_rate,
        name="dropout_1",
    )(x)

    x = GRU(
        gru_units,
        activation="tanh",
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg),
        name="gru_2",
    )(x)

    x = Dropout(
        dropout_rate,
        name="dropout_2",
    )(x)

    # Self-attention:
    # Query = Key = Value = GRU sequence
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name="self_attention",
    )(x, x)

    # Residual connection
    x = LayerNormalization(
        name="attention_norm",
    )(x + attention_output)

    # Instead of selecting only the final timestep,
    # aggregate information across the sequence.
    x = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling",
    )(x)

    x = Dense(
        dense_units,
        activation="tanh",
        kernel_regularizer=l2(l2_reg),
        name="dense_1",
    )(x)

    x = Dropout(
        dropout_rate,
        name="dense_dropout",
    )(x)

    outputs = Dense(
        1,
        activation="linear",
        name="future_log_return",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ETH_GRU_Attention",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=Huber(
            delta=0.01
        ),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
        ],
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
    """
    Warmup + cosine decay.
    """

    def schedule(epoch, lr):

        if epoch < warmup_epochs:

            return (
                base_lr
                * (epoch + 1)
                / warmup_epochs
            )

        progress = (
            epoch - warmup_epochs
        ) / max(
            1,
            total_epochs - warmup_epochs,
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine_decay = (
            0.5
            * (
                1
                + np.cos(
                    np.pi * progress
                )
            )
        )

        return (
            min_lr
            + (
                base_lr - min_lr
            )
            * cosine_decay
        )

    return LearningRateScheduler(
        schedule,
        verbose=0,
    )


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
):
    """
    Train with validation and early stopping.
    """

    os.makedirs(
        os.path.dirname(
            model_save_path
        ) or ".",
        exist_ok=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1 if verbose else 0,
    )

    lr_schedule = (
        build_warmup_cosine_schedule(
            total_epochs=epochs,
            warmup_epochs=5,
            base_lr=1e-3,
            min_lr=1e-6,
        )
    )

    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1 if verbose else 0,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            early_stopping,
            lr_schedule,
            checkpoint,
        ],
        verbose=verbose,
        shuffle=False,
    )

    return model, history


# ============================================================
# METRICS
# ============================================================

def directional_accuracy(
    y_true,
    y_pred,
):
    """
    ???? ?????? ?? ??? ??? ???? ?? ???? ????? ????.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return np.nan

    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)

    return float(
        np.mean(
            true_direction
            == pred_direction
        )
    )


def correlation(
    y_true,
    y_pred,
):
    """
    Correlation between predicted and actual returns.
    """

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0:
        return np.nan

    if np.std(y_pred) == 0:
        return np.nan

    return float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )


def evaluate_return_predictions(
    y_true,
    y_pred,
):
    """
    Evaluation ????? future return.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) == 0:
        return {}

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    direction_acc = directional_accuracy(
        y_true,
        y_pred,
    )

    corr = correlation(
        y_true,
        y_pred,
    )

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Directional_Accuracy": float(
            direction_acc
        ),
        "Correlation": float(corr),
    }


# ============================================================
# NAIVE BASELINE
# ============================================================

def naive_zero_return_metrics(
    y_true,
):
    """
    Naive baseline:

        future return = 0

    ????:

        future price = current price
    """

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.zeros_like(
        y_true
    )

    metrics = evaluate_return_predictions(
        y_true,
        y_pred,
    )

    return metrics


# ============================================================
# PREDICTION ? PRICE
# ============================================================

def log_return_to_future_price(
    anchor_prices,
    predicted_log_returns,
):
    """
    Convert predicted log return to future price.

        future_price =
            anchor_price * exp(predicted_log_return)
    """

    return (
        np.asarray(anchor_prices)
        * np.exp(
            np.asarray(
                predicted_log_returns
            )
        )
    )


# ============================================================
# BACKTEST HELPERS
# ============================================================

def calculate_max_drawdown(
    equity_curve,
):
    """
    Max drawdown percentage.
    """

    equity = np.asarray(
        equity_curve,
        dtype=float,
    )

    if len(equity) == 0:
        return np.nan

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    return float(
        drawdown.min() * 100
    )


def calculate_profit_factor(
    trade_returns,
):
    """
    Gross profit / gross loss.
    """

    returns = np.asarray(
        trade_returns,
        dtype=float,
    )

    gross_profit = returns[
        returns > 0
    ].sum()

    gross_loss = -returns[
        returns < 0
    ].sum()

    if gross_loss <= 0:
        if gross_profit > 0:
            return np.inf

        return np.nan

    return float(
        gross_profit
        / gross_loss
    )


def calculate_sharpe(
    trade_returns,
):
    """
    Sharpe approximation based on trade returns.

    ??? Sharpe ????? production ????.
    ??? ???? ???????? ??????? ?????? ??????? ??????.
    """

    returns = np.asarray(
        trade_returns,
        dtype=float,
    )

    if len(returns) < 2:
        return np.nan

    std = returns.std(
        ddof=1
    )

    if std == 0:
        return np.nan

    return float(
        returns.mean()
        / std
        * np.sqrt(len(returns))
    )


# ============================================================
# REALISTIC MVP BACKTEST
# ============================================================

def backtest_strategy(
    anchor_prices,
    actual_future_returns,
    predicted_returns,
    timestamps=None,
    initial_capital=INITIAL_CAPITAL,
    fee_rate=FEE_RATE,
    slippage_rate=SLIPPAGE_RATE,
    signal_threshold=SIGNAL_THRESHOLD,
    risk_per_trade=RISK_PER_TRADE,
    stop_loss_pct=STOP_LOSS_PCT,
    take_profit_pct=TAKE_PROFIT_PCT,
    max_leverage=MAX_LEVERAGE,
):
    """
    MVP trading simulation.

    Important:
        ??? backtest ???? execution engine ????? ????.

    ???? ?? prediction:

        predicted_return > threshold
            ? LONG

        predicted_return < -threshold
            ? SHORT

        otherwise
            ? NO TRADE

    ??? actual future return ???? horizon ?? ???? ???? ????? ??????? ???????.

    Stop/TP:
        ?? ??? ???? ??? ?? future horizon return ???????? ???????.
        ???? backtest tick-level ?????? ????? ???? ?? high/low ????????
        ??? entry ? exit ??????? ????.

    Position sizing:
        risk_per_trade / stop_loss_pct

    ??? leverage ????? ??????.
    """

    anchor_prices = np.asarray(
        anchor_prices,
        dtype=float,
    )

    actual_future_returns = np.asarray(
        actual_future_returns,
        dtype=float,
    )

    predicted_returns = np.asarray(
        predicted_returns,
        dtype=float,
    )

    if not (
        len(anchor_prices)
        == len(actual_future_returns)
        == len(predicted_returns)
    ):
        raise ValueError(
            "Backtest input arrays must have equal length."
        )

    capital = float(
        initial_capital
    )

    equity_curve = [
        capital
    ]

    trades = []

    gross_trade_returns = []

    for i in range(
        len(predicted_returns)
    ):

        prediction = predicted_returns[i]

        actual_return = (
            actual_future_returns[i]
        )

        anchor = anchor_prices[i]

        if not (
            np.isfinite(prediction)
            and np.isfinite(actual_return)
            and np.isfinite(anchor)
            and anchor > 0
        ):
            continue

        # -----------------------------------------
        # SIGNAL
        # -----------------------------------------

        if prediction >= signal_threshold:

            side = "LONG"

        elif prediction <= -signal_threshold:

            side = "SHORT"

        else:

            side = "NO TRADE"

        if side == "NO TRADE":

            equity_curve.append(
                capital
            )

            continue

        # -----------------------------------------
        # POSITION SIZE
        # -----------------------------------------

        position_fraction = (
            risk_per_trade
            / stop_loss_pct
        )

        position_fraction = min(
            position_fraction,
            max_leverage,
        )

        # -----------------------------------------
        # DIRECTIONAL RETURN
        # -----------------------------------------

        directional_return = (
            actual_return
            if side == "LONG"
            else -actual_return
        )

        # -----------------------------------------
        # EXIT RULE
        # -----------------------------------------

        exit_reason = "TIME_EXIT"

        if directional_return <= -stop_loss_pct:

            realized_move = -stop_loss_pct

            exit_reason = "STOP_LOSS"

        elif directional_return >= take_profit_pct:

            realized_move = take_profit_pct

            exit_reason = "TAKE_PROFIT"

        else:

            realized_move = directional_return

        # -----------------------------------------
        # COSTS
        # -----------------------------------------

        # Entry slippage
        # Exit slippage
        round_trip_slippage = (
            2 * slippage_rate
        )

        round_trip_fee = (
            2 * fee_rate
        )

        net_return_on_position = (
            realized_move
            - round_trip_slippage
            - round_trip_fee
        )

        # Apply position fraction
        account_return = (
            position_fraction
            * net_return_on_position
        )

        capital *= (
            1 + account_return
        )

        trade_record = {
            "timestamp": (
                timestamps[i]
                if timestamps is not None
                and i < len(timestamps)
                else None
            ),
            "side": side,
            "anchor_price": float(anchor),
            "predicted_return": float(
                prediction
            ),
            "actual_future_return": float(
                actual_return
            ),
            "realized_directional_move": float(
                realized_move
            ),
            "position_fraction": float(
                position_fraction
            ),
            "fee_rate": float(
                fee_rate
            ),
            "slippage_rate": float(
                slippage_rate
            ),
            "net_return_on_position": float(
                net_return_on_position
            ),
            "account_return": float(
                account_return
            ),
            "capital_after_trade": float(
                capital
            ),
            "exit_reason": exit_reason,
        }

        trades.append(
            trade_record
        )

        gross_trade_returns.append(
            account_return
        )

        equity_curve.append(
            capital
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    trade_returns = np.asarray(
        gross_trade_returns,
        dtype=float,
    )

    total_return_pct = (
        (
            capital
            / initial_capital
        )
        - 1
    ) * 100

    if len(trade_returns) > 0:

        wins = trade_returns[
            trade_returns > 0
        ]

        losses = trade_returns[
            trade_returns < 0
        ]

        win_rate = (
            len(wins)
            / len(trade_returns)
            * 100
        )

        average_trade_pct = (
            trade_returns.mean()
            * 100
        )

    else:

        win_rate = np.nan
        average_trade_pct = np.nan

    if len(actual_future_returns) > 0:

        buy_hold_return_pct = (
            np.exp(
                np.sum(
                    actual_future_returns
                )
            )
            - 1
        ) * 100

    else:

        buy_hold_return_pct = np.nan

    result = {
        "initial_capital": float(
            initial_capital
        ),
        "final_capital": float(
            capital
        ),
        "total_return_pct": float(
            total_return_pct
        ),
        "buy_hold_return_pct": float(
            buy_hold_return_pct
        ),
        "n_trades": int(
            len(trades)
        ),
        "win_rate_pct": float(
            win_rate
        ),
        "average_trade_pct": float(
            average_trade_pct
        ),
        "profit_factor": calculate_profit_factor(
            trade_returns
        ),
        "sharpe": calculate_sharpe(
            trade_returns
        ),
        "max_drawdown_pct": calculate_max_drawdown(
            equity_curve
        ),
        "equity_curve": equity_curve,
        "trades": trades,
    }

    return result


# ============================================================
# PLOT
# ============================================================

def plot_prediction_returns(
    timestamps,
    y_true,
    y_pred,
    output_path,
):
    """
    Plot actual vs predicted future log return.
    """

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        timestamps,
        y_true,
        label="Actual future log return",
    )

    plt.plot(
        timestamps,
        y_pred,
        label="Predicted future log return",
        linestyle="--",
    )

    plt.axhline(
        0,
        linestyle=":",
    )

    plt.title(
        "GRU + Attention - 15m Future Return"
    )

    plt.xlabel(
        "Anchor time"
    )

    plt.ylabel(
        "Log return"
    )

    plt.legend()

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_equity_curve(
    equity_curve,
    output_path,
):
    """
    Plot simulated equity.
    """

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        equity_curve,
        label="Strategy equity",
    )

    plt.title(
        "MVP Strategy Equity Curve"
    )

    plt.xlabel(
        "Trade / Decision"
    )

    plt.ylabel(
        "Capital"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# ============================================================
# WALK FORWARD
# ============================================================

def generate_walk_forward_folds(
    n_samples,
    n_folds=N_FOLDS,
):
    """
    Expanding-window walk-forward.

    Example:

        Fold 1:
            train | val | test

        Fold 2:
            train -------- | val | test

        Fold 3:
            train ---------------- | val | test

    ?? fold ??? ?? ????? ???? train ??????? ??????.
    """

    if n_samples < 1000:
        raise ValueError(
            "Dataset is too small for reliable walk-forward validation."
        )

    folds = []

    test_size = n_samples // (
        n_folds + 1
    )

    min_train_size = test_size * 2

    for fold in range(
        n_folds
    ):

        test_start = (
            min_train_size
            + fold * test_size
        )

        val_start = (
            test_start
            - test_size
        )

        test_end = min(
            test_start
            + test_size,
            n_samples,
        )

        train_start = 0

        if test_end <= test_start:
            break

        folds.append(
            (
                train_start,
                val_start,
                test_start,
                test_end,
            )
        )

    return folds


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
):
    """
    ????? ???:

    ???? validation ? test? history ????? ?? ??? ??? ?? ??? ????????.

    ??? scaler ??? ??? train fit ??????.

    ??? ??? ???? ?????? ????? sequence??? validation/test ???????
    history ????? ??? ?? split ?? ??????.

    ??? future target ???? sequence ???????.
    """

    # -------------------------
    # Train
    # -------------------------

    train_df = features_df.iloc[
        :train_end
    ]

    train_target = target_series.iloc[
        :train_end
    ]

    # -------------------------
    # Validation
    #
    # Include sequence_length rows
    # before val start as context.
    # -------------------------

    val_context_start = max(
        0,
        train_end - sequence_length,
    )

    val_df = features_df.iloc[
        val_context_start:val_end
    ]

    val_target = target_series.iloc[
        val_context_start:val_end
    ]

    # -------------------------
    # Test
    #
    # Include context from before
    # test start.
    # -------------------------

    test_context_start = max(
        0,
        val_end - sequence_length,
    )

    test_df = features_df.iloc[
        test_context_start:test_end
    ]

    test_target = target_series.iloc[
        test_context_start:test_end
    ]

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

def build_gru_model_tunable(
    hp,
    input_shape,
):
    """
    Optional KerasTuner model.

    ???? MVP ????? ???.
    """

    gru_units = hp.Choice(
        "gru_units",
        values=[
            32,
            64,
            128,
        ],
    )

    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.1,
        max_value=0.4,
        step=0.1,
    )

    num_heads = hp.Choice(
        "num_heads",
        values=[
            2,
            4,
        ],
    )

    key_dim = hp.Choice(
        "key_dim",
        values=[
            8,
            16,
            32,
        ],
    )

    l2_reg = hp.Choice(
        "l2_reg",
        values=[
            1e-5,
            1e-4,
            1e-3,
        ],
    )

    dense_units = hp.Choice(
        "dense_units",
        values=[
            16,
            32,
            64,
        ],
    )

    learning_rate = hp.Choice(
        "learning_rate",
        values=[
            1e-3,
            3e-4,
            1e-4,
        ],
    )

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
    X_train,
    y_train,
    X_val,
    y_val,
    input_shape,
    output_dir,
    max_trials=20,
    epochs_per_trial=30,
):
    """
    Optional Bayesian Optimization.
    """

    import keras_tuner as kt

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp:
            build_gru_model_tunable(
                hp,
                input_shape,
            ),
        objective="val_loss",
        max_trials=max_trials,
        directory=os.path.join(
            output_dir,
            "kt_search",
        ),
        project_name="gru_attention",
        overwrite=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    tuner.search(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=epochs_per_trial,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping
        ],
        verbose=1,
    )

    best_hp = (
        tuner.get_best_hyperparameters(
            num_trials=1
        )[0]
    )

    return best_hp


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    output_path,
    fold_idx,
    scaler,
):
    """
    Save model metadata.

    ??? metadata ????? ?? crypto-signal-api ???? ??????? ?? mismatch
    ??? model ? feature pipeline ??????? ????? ??.
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
            "stop_loss_pct": STOP_LOSS_PCT,
            "take_profit_pct": TAKE_PROFIT_PCT,
            "max_leverage": MAX_LEVERAGE,
        },

        "fold": fold_idx,

        "scaler": {
            "type": "StandardScaler",
            "n_features": int(
                scaler.n_features_in_
            ),
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed()

    # --------------------------------------------------------
    # DATA PATH
    # --------------------------------------------------------

    file_path = (
        "/content/drive/MyDrive/"
        "binance_data_5min.csv"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("=" * 70)
    print("ETH GRU + ATTENTION MVP")
    print("=" * 70)

    print(
        f"Symbol       : {SYMBOL}"
    )

    print(
        f"Timeframe    : {TIMEFRAME_MINUTES}m"
    )

    print(
        f"Horizon      : {HORIZON_MINUTES}m"
    )

    print(
        f"Horizon steps: {HORIZON_STEPS}"
    )

    print(
        f"Sequence     : {SEQUENCE_LENGTH}"
    )

    print(
        f"Feature ver. : {FEATURE_VERSION}"
    )

    print(
        f"Target       : future_log_return"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_and_preprocess_data(
        file_path
    )

    check_candle_gaps(
        data,
        TIMEFRAME_MINUTES,
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_data = add_technical_indicators(
        data.copy()
    )

    if feature_data.empty:
        raise ValueError(
            "Feature dataframe is empty."
        )

    validate_feature_schema(
        feature_data
    )

    missing = [
        c
        for c in FEATURE_COLUMNS
        if c not in feature_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    feature_data = (
        feature_data[
            FEATURE_COLUMNS
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target = create_future_log_return_target(
        feature_data["close"],
        HORIZON_STEPS,
    )

    # Keep feature rows even though the last horizon rows
    # have no target. They are useful only as context, but
    # target will be filtered during sequence creation.
    target = target.reindex(
        feature_data.index
    )

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    n_samples = len(
        feature_data
    )

    folds = generate_walk_forward_folds(
        n_samples,
        N_FOLDS,
    )

    print(
        f"\n[DATA] Total usable rows: {n_samples:,}"
    )

    print(
        f"[WALK-FORWARD] Folds: {len(folds)}"
    )

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
    ) in enumerate(
        folds,
        start=1,
    ):

        print("\n")
        print("=" * 70)

        print(
            f"FOLD {fold_idx}/{len(folds)}"
        )

        print(
            f"Train: 0 -> {val_start}"
        )

        print(
            f"Val  : {val_start} -> {test_start}"
        )

        print(
            f"Test : {test_start} -> {test_end}"
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
        )

        # ----------------------------------------------------
        # SCALER
        # ----------------------------------------------------

        scaler = StandardScaler()

        # IMPORTANT:
        # Fit ONLY on training data.
        train_scaled = scaler.fit_transform(
            train_df.values
        )

        val_scaled = scaler.transform(
            val_df.values
        )

        test_scaled = scaler.transform(
            test_df.values
        )

        scaler_path = os.path.join(
            OUTPUT_DIR,
            f"scaler_fold{fold_idx}.pkl",
        )

        joblib.dump(
            scaler,
            scaler_path,
        )

        # ----------------------------------------------------
        # SEQUENCES
        # ----------------------------------------------------

        X_train, y_train = (
            create_sequences(
                train_scaled,
                train_target.values,
                SEQUENCE_LENGTH,
            )
        )

        (
            X_val,
            y_val,
            val_anchor_indices,
            val_times,
        ) = create_sequences_with_indices(
            val_scaled,
            val_target.values,
            val_df.index,
            SEQUENCE_LENGTH,
        )

        (
            X_test,
            y_test,
            test_anchor_indices,
            test_times,
        ) = create_sequences_with_indices(
            test_scaled,
            test_target.values,
            test_df.index,
            SEQUENCE_LENGTH,
        )

        print(
            f"[SEQUENCES] Train: {len(X_train):,}"
        )

        print(
            f"[SEQUENCES] Val  : {len(X_val):,}"
        )

        print(
            f"[SEQUENCES] Test : {len(X_test):,}"
        )

        if (
            len(X_train) == 0
            or len(X_val) == 0
            or len(X_test) == 0
        ):

            print(
                "[WARNING] Not enough sequence data. "
                "Skipping fold."
            )

            continue

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        input_shape = (
            X_train.shape[1],
            X_train.shape[2],
        )

        if (
            USE_HYPERPARAMETER_SEARCH
            and fold_idx == len(folds)
        ):

            print(
                "\n[HYPERPARAMETER SEARCH]"
            )

            best_hp = (
                run_hyperparameter_search(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    input_shape,
                    OUTPUT_DIR,
                    HP_SEARCH_MAX_TRIALS,
                    HP_SEARCH_EPOCHS_PER_TRIAL,
                )
            )

            model = build_gru_model(
                input_shape=input_shape,
                gru_units=best_hp.get(
                    "gru_units"
                ),
                dropout_rate=best_hp.get(
                    "dropout_rate"
                ),
                num_heads=best_hp.get(
                    "num_heads"
                ),
                key_dim=best_hp.get(
                    "key_dim"
                ),
                l2_reg=best_hp.get(
                    "l2_reg"
                ),
                dense_units=best_hp.get(
                    "dense_units"
                ),
                learning_rate=best_hp.get(
                    "learning_rate"
                ),
            )

        else:

            model = build_gru_model(
                input_shape=input_shape
            )

        model_path = os.path.join(
            OUTPUT_DIR,
            f"best_gru_attention_fold{fold_idx}.keras",
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model, history = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_save_path=model_path,
            verbose=1,
        )

        # ----------------------------------------------------
        # TEST PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(
            X_test,
            verbose=0,
        ).reshape(-1)

        y_test_flat = (
            y_test.reshape(-1)
        )

        # ----------------------------------------------------
        # PREDICTION METRICS
        # ----------------------------------------------------

        metrics = (
            evaluate_return_predictions(
                y_test_flat,
                predictions,
            )
        )

        naive_metrics = (
            naive_zero_return_metrics(
                y_test_flat
            )
        )

        print("\n[MODEL METRICS]")

        for key, value in metrics.items():

            print(
                f"{key}: {value:.6f}"
            )

        print("\n[NAIVE ZERO-RETURN BASELINE]")

        for key in [
            "RMSE",
            "MAE",
            "R2",
            "Directional_Accuracy",
            "Correlation",
        ]:

            value = naive_metrics.get(
                key,
                np.nan,
            )

            print(
                f"{key}: {value:.6f}"
            )

        # ----------------------------------------------------
        # ANCHOR PRICES
        # ----------------------------------------------------

        # test_anchor_indices are local indices inside test_df.
        #
        # test_df starts at test_context_start.
        #
        # Convert local anchor index to global feature index.

        global_test_indices = (
            test_context_start
            + test_anchor_indices
        )

        anchor_prices = (
            feature_data.iloc[
                global_test_indices
            ]["close"]
            .values
        )

        # ----------------------------------------------------
        # BACKTEST
        # ----------------------------------------------------

        bt = backtest_strategy(
            anchor_prices=anchor_prices,
            actual_future_returns=y_test_flat,
            predicted_returns=predictions,
            timestamps=test_times,
            initial_capital=INITIAL_CAPITAL,
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
            signal_threshold=SIGNAL_THRESHOLD,
            risk_per_trade=RISK_PER_TRADE,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            max_leverage=MAX_LEVERAGE,
        )

        print("\n[BACKTEST]")

        print(
            f"Final capital       : "
            f"{bt['final_capital']:.2f}"
        )

        print(
            f"Strategy return     : "
            f"{bt['total_return_pct']:.2f}%"
        )

        print(
            f"Buy & Hold          : "
            f"{bt['buy_hold_return_pct']:.2f}%"
        )

        print(
            f"Trades              : "
            f"{bt['n_trades']}"
        )

        print(
            f"Win rate            : "
            f"{bt['win_rate_pct']:.2f}%"
        )

        print(
            f"Average trade       : "
            f"{bt['average_trade_pct']:.4f}%"
        )

        print(
            f"Profit factor       : "
            f"{bt['profit_factor']}"
        )

        print(
            f"Sharpe              : "
            f"{bt['sharpe']}"
        )

        print(
            f"Max drawdown        : "
            f"{bt['max_drawdown_pct']:.2f}%"
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        final_model_path = os.path.join(
            OUTPUT_DIR,
            f"model_GRU_Attention_fold{fold_idx}.keras",
        )

        model.save(
            final_model_path
        )

        # ----------------------------------------------------
        # SAVE METADATA
        # ----------------------------------------------------

        metadata_path = os.path.join(
            OUTPUT_DIR,
            f"metadata_fold{fold_idx}.json",
        )

        save_metadata(
            metadata_path,
            fold_idx,
            scaler,
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        prediction_df = pd.DataFrame(
            {
                "timestamp": test_times,
                "anchor_price": anchor_prices,
                "actual_future_log_return": y_test_flat,
                "predicted_future_log_return": predictions,
                "actual_future_return_pct": (
                    np.exp(y_test_flat) - 1
                ) * 100,
                "predicted_future_return_pct": (
                    np.exp(predictions) - 1
                ) * 100,
            }
        )

        prediction_path = os.path.join(
            OUTPUT_DIR,
            f"predictions_fold{fold_idx}.csv",
        )

        prediction_df.to_csv(
            prediction_path,
            index=False,
        )

        # ----------------------------------------------------
        # SAVE TRADES
        # ----------------------------------------------------

        trades_df = pd.DataFrame(
            bt["trades"]
        )

        trades_path = os.path.join(
            OUTPUT_DIR,
            f"trades_fold{fold_idx}.csv",
        )

        trades_df.to_csv(
            trades_path,
            index=False,
        )

        # ----------------------------------------------------
        # PLOTS
        # ----------------------------------------------------

        prediction_plot_path = os.path.join(
            OUTPUT_DIR,
            f"prediction_returns_fold{fold_idx}.png",
        )

        plot_prediction_returns(
            test_times,
            y_test_flat,
            predictions,
            prediction_plot_path,
        )

        equity_plot_path = os.path.join(
            OUTPUT_DIR,
            f"equity_curve_fold{fold_idx}.png",
        )

        plot_equity_curve(
            bt["equity_curve"],
            equity_plot_path,
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        fold_result = {
            "fold": fold_idx,
            **metrics,
            "Naive_RMSE": naive_metrics[
                "RMSE"
            ],
            "Naive_MAE": naive_metrics[
                "MAE"
            ],
            "Naive_R2": naive_metrics[
                "R2"
            ],
            "Naive_Directional_Accuracy":
                naive_metrics[
                    "Directional_Accuracy"
                ],
            "Strategy_Return_%":
                bt[
                    "total_return_pct"
                ],
            "Buy_Hold_Return_%":
                bt[
                    "buy_hold_return_pct"
                ],
            "Trades":
                bt[
                    "n_trades"
                ],
            "Win_Rate_%":
                bt[
                    "win_rate_pct"
                ],
            "Average_Trade_%":
                bt[
                    "average_trade_pct"
                ],
            "Profit_Factor":
                bt[
                    "profit_factor"
                ],
            "Sharpe":
                bt[
                    "sharpe"
                ],
            "Max_Drawdown_%":
                bt[
                    "max_drawdown_pct"
                ],
        }

        fold_results.append(
            fold_result
        )

        backtest_results.append(
            {
                "fold": fold_idx,
                "strategy_return_%":
                    bt[
                        "total_return_pct"
                    ],
                "buy_hold_return_%":
                    bt[
                        "buy_hold_return_pct"
                    ],
                "n_trades":
                    bt[
                        "n_trades"
                    ],
                "win_rate_%":
                    bt[
                        "win_rate_pct"
                    ],
                "profit_factor":
                    bt[
                        "profit_factor"
                    ],
                "sharpe":
                    bt[
                        "sharpe"
                    ],
                "max_drawdown_%":
                    bt[
                        "max_drawdown_pct"
                    ],
            }
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(
        fold_results
    )

    if not results_df.empty:

        print(
            results_df.to_string(
                index=False
            )
        )

        results_path = os.path.join(
            OUTPUT_DIR,
            "gru_metrics_summary.csv",
        )

        results_df.to_csv(
            results_path,
            index=False,
        )

        print(
            f"\nMetrics saved to:"
            f"\n{results_path}"
        )

    else:

        print(
            "No fold results generated."
        )

    bt_df = pd.DataFrame(
        backtest_results
    )

    if not bt_df.empty:

        print("\n[BACKTEST SUMMARY]")

        print(
            bt_df.to_string(
                index=False
            )
        )

        bt_path = os.path.join(
            OUTPUT_DIR,
            "backtest_summary.csv",
        )

        bt_df.to_csv(
            bt_path,
            index=False,
        )

        print(
            f"\nBacktest saved to:"
            f"\n{bt_path}"
        )

    print("\nProcessing complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # ??? ?? Colab ????? ??? ?? ?? ?? ?? ??? ???? ??:
    #
    # from google.colab import drive
    # drive.mount('/content/drive')

    main()