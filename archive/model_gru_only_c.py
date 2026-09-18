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

مثلاً:
    timeframe = 5m
    horizon = 15m
    horizon_steps = 3

یعنی مدل با sequence گذشته، بازده مورد انتظار 15 دقیقه آینده را
پیش‌بینی می‌کند.

چرا log-return به‌جای قیمت خام:
    1) جمع‌پذیری زمانی: log(P2/P1) + log(P3/P2) = log(P3/P1)
    2) تقارن: +x% و -x% به قیمت اولیه برمی‌گردونه (با درصد ساده اینطور نیست)
    3) توزیع نزدیک‌تر به نرمال - برای loss function و scaler مناسب‌تره

Sequence:
    sequence_length = 96

یعنی برای هر prediction مدل 96 کندل 5 دقیقه‌ای را می‌بیند:
    96 × 5m = 8 hours

Attention:
    Attention جای Sequence را نمی‌گیرد.
    GRU روی sequence حرکت می‌کند و hidden state تولید می‌کند.
    Attention یاد می‌گیرد کدام timestepهای sequence برای prediction
    فعلی مهم‌تر هستند.

Validation:
    Walk-forward validation

Scaling:
    Scaler فقط روی train هر fold fit می‌شود.
    Validation و Test فقط transform می‌شوند.

Risk Management در Backtest (اصلاح‌شده نسبت به نسخه‌ی قبلی این فایل):
    نسخه‌ی قبلی از STOP_LOSS_PCT / TAKE_PROFIT_PCT به‌صورت درصد *ثابت*
    استفاده می‌کرد - یعنی در بازار پرنوسان حد ضرر خیلی زود فعال می‌شد و
    در بازار کم‌نوسان خیلی گشاد بود. این نسخه SL/TP را بر پایه‌ی ATR
    لحظه‌ای هر anchor محاسبه می‌کند (هماهنگ با منطق trade_decision.py
    در crypto-signal-api) تا با نوسان واقعی بازار در هر لحظه سازگار باشد.

Backtest شامل:
    - LONG / SHORT / NO TRADE
    - fee (round-trip)
    - slippage (round-trip)
    - threshold سیگنال
    - ATR-based stop-loss / take-profit (نه درصد ثابت)
    - fixed-risk position sizing
    - equity curve
    - drawdown
    - Sharpe تقریبی
    - profit factor
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

# --- ATR-based risk management (جایگزین STOP_LOSS_PCT/TAKE_PROFIT_PCT ثابت) ---
# فاصله‌ی حد ضرر = ATR_MULTIPLIER_SL × (ATR در لحظه‌ی anchor / anchor_price)
# فاصله‌ی حد سود = ATR_MULTIPLIER_TP × همون نسبت
# این باعث می‌شه SL/TP با نوسان واقعی بازار در هر لحظه (نه یک درصد ثابت
# برای کل تاریخچه) سازگار باشه - دقیقاً همون منطقی که در trade_decision.py
# برای معاملات دستی هم استفاده می‌کنیم، تا بک‌تست و اجرای واقعی هم‌خوان باشن.
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.25  # نسبت ریسک/ریوارد پیش‌فرض 1:1.5

# حداقل و حداکثر فاصله‌ی SL به‌عنوان درصد قیمت - محافظ در برابر ATR غیرعادی
# (مثلاً داده‌ی خراب یا نوسان لحظه‌ای extreme)
MIN_STOP_LOSS_PCT = 0.0015  # 0.15%
MAX_STOP_LOSS_PCT = 0.0100  # 1.00%

MAX_HOLDING_STEPS = HORIZON_STEPS

EPOCHS = 100
BATCH_SIZE = 32

PATIENCE = 10

OUTPUT_DIR = "/content/drive/MyDrive/model_outputs_gru"

# --- Quick Test (برای اجرای سریع/سنجش سلامت pipeline قبل از اجرای کامل) ---
# وقتی True باشه: فقط آخرین QUICK_TEST_ROWS ردیف دیتا استفاده می‌شه و
# epochs به QUICK_TEST_EPOCHS محدود می‌شه - برای این‌که ظرف چند دقیقه
# مطمئن بشید کل pipeline (از load تا save) بدون خطا اجرا می‌شه، قبل از
# اینکه با ۸ سال داده و epochs کامل (که ساعت‌ها طول می‌کشه) ریسک کنید.
# بعد از موفقیت‌آمیز بودن این تست، حتماً QUICK_TEST = False کنید.
QUICK_TEST = False
QUICK_TEST_ROWS = 20_000
QUICK_TEST_EPOCHS = 5

# Set True only when you intentionally want tuner.
USE_HYPERPARAMETER_SEARCH = False

HP_SEARCH_MAX_TRIALS = 20
HP_SEARCH_EPOCHS_PER_TRIAL = 30


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    """
    تلاش برای reproducible کردن training.
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
    بررسی gapهای زمانی.

    Gap الزاماً به معنی خراب بودن داده نیست.
    فقط گزارش می‌شود تا بدانیم dataset چه وضعی دارد.
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

    خروجی به index همان t تعلق دارد.

    مثال:

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
    ساخت sequence برای supervised learning.

    اصلاح مهم (رفع باگ alignment نسخه‌ی قبلی):
    قبلاً sequence به‌صورت features[i-seq_length:i] بود (یعنی آخرین ردیف
    دیده‌شده i-1 بود) در حالی که target و anchor به ردیف i تعلق داشتن.
    این یعنی مدل دقیقاً همون کندلی که ازش anchor/target ساخته می‌شد رو
    در ورودی نمی‌دید - یک off-by-one واقعی که هم یادگیری رو سخت‌تر از
    حد لازم می‌کرد و هم با نحوه‌ی inference واقعی (که سکانس باید به
    آخرین کندل بسته‌شده ختم بشه) ناهماهنگ بود.

    حالا:
        X[k] = features[i-seq_length+1 : i+1]   (شامل خود ردیف i، anchor)
        y[k] = target[i]                         (log(close[i+h]/close[i]))

    یعنی آخرین ردیف sequence دقیقاً همون کندلیه که target ازش محاسبه شده.

    X:
        [i-seq_length+1 : i+1]  (inclusive of anchor row i)

    y:
        target[t] که t = i (آخرین ردیف X)

    بنابراین:

        X -> history ending at t (شامل خود t)
        y -> future return from t to t+horizon

    اینجا دیگر target قیمت خام نیست.
    """

    X = []
    y = []

    if len(features_array) != len(target_array):
        raise ValueError(
            "features_array and target_array "
            "must have the same length."
        )

    for i in range(seq_length - 1, len(features_array)):
        target_value = target_array[i]

        if not np.isfinite(target_value):
            continue

        sequence = features_array[
            i - seq_length + 1: i + 1
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
    همان create_sequences ولی timestamp مربوط به anchor را هم برمی‌گرداند.

    همون اصلاح off-by-one که در create_sequences اعمال شد اینجا هم اعمال
    شده: X شامل خود ردیف anchor (i) به‌عنوان آخرین timestep است.

    این برای evaluation و backtest بسیار مهم است.
    """

    X = []
    y = []
    anchor_indices = []
    anchor_times = []

    for i in range(seq_length - 1, len(features_array)):
        target_value = target_array[i]

        if not np.isfinite(target_value):
            continue

        sequence = features_array[
            i - seq_length + 1: i + 1
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
# LAZY SEQUENCE DATASET (رفع OOM)
# ============================================================
#
# مشکل نسخه‌ی eager (create_sequences/create_sequences_with_indices):
# کل تانسور N × seq_length × n_features رو یکجا در RAM می‌سازه. با ۸ سال
# داده‌ی ۵ دقیقه‌ای (~۸۴۰,۰۰۰ ردیف) و seq_length=96، این یعنی چیزی حدود
# 840000 × 96 × 20 × 4 بایت ≈ 6.4 گیگابایت فقط برای X_train - که به‌راحتی
# باعث OOM می‌شه، مخصوصاً روی Colab.
#
# راه‌حل: به‌جای ساختن این آرایه‌ی عظیم، دو مرحله انجام می‌دیم:
#   ۱) یک اسکن سریع و کاملاً برداری (vectorized, بدون حلقه‌ی پایتونی روی
#      هر پنجره) که فقط مشخص می‌کنه کدوم anchor ها معتبرن (target و کل
#      پنجره‌ی ورودی‌شون finite هستن) - این فقط آرایه‌های O(N) می‌سازه،
#      نه O(N × seq_length).
#   ۲) یک tf.data.Dataset مبتنی بر generator که پنجره‌ها رو یکی‌یکی (یا
#      batch به batch) در لحظه‌ی نیاز می‌سازه و بلافاصله بعد از استفاده
#      آزادشون می‌کنه - هیچ‌وقت همه رو همزمان در حافظه نگه نمی‌داره.
# ============================================================

def compute_valid_anchor_mask(features_array, target_array, seq_length):
    """
    نسخه‌ی برداری (vectorized) از منطق اعتبارسنجی که در create_sequences
    با حلقه‌ی پایتونی انجام می‌شد - همون نتیجه رو می‌ده ولی بدون ساختن
    هیچ آرایه‌ی O(N × seq_length)ای.

    خروجی: آرایه‌ی boolean به طول len(features_array)، که در ایندکس i
    مقدار True یعنی: هم target[i] finite است، هم تمام seq_length ردیف
    قبل از i (شامل خودش) finite هستن - یعنی می‌شه یک sequence معتبر با
    anchor=i ساخت.
    """
    n = len(features_array)

    row_finite = np.all(
        np.isfinite(features_array), axis=1
    )

    # شمارش تجمعی ردیف‌های غیر-finite برای چک سریع "همه‌ی پنجره finite است؟"
    invalid = (~row_finite).astype(np.int32)
    cumsum = np.concatenate(([0], np.cumsum(invalid)))

    valid_mask = np.zeros(n, dtype=bool)

    if n >= seq_length:
        idx = np.arange(seq_length - 1, n)
        window_start = idx - seq_length + 1
        # تعداد ردیف‌های نامعتبر در پنجره‌ی [window_start, idx] (inclusive)
        window_invalid_count = (
            cumsum[idx + 1] - cumsum[window_start]
        )
        window_all_finite = window_invalid_count == 0

        target_finite = np.isfinite(
            target_array[idx]
        )

        valid_mask[idx] = window_all_finite & target_finite

    return valid_mask


def build_lazy_sequence_dataset(
    features_array,
    target_array,
    seq_length,
    batch_size,
    shuffle_buffer=None,
):
    """
    ساخت یک tf.data.Dataset که sequenceها رو lazy (به‌ازای هر batch، نه
    یکجا برای کل دیتاست) تولید می‌کنه.

    برمی‌گردونه:
        dataset:      tf.data.Dataset از (X_batch, y_batch)
        valid_anchors: آرایه‌ی ایندکس anchor های معتبر (به همون ترتیبی که
                       generator تولیدشون می‌کنه) - برای بازسازی anchor
                       price/ATR/timestamp در evaluation/backtest لازمه.

    نکته: shuffle_buffer فقط باید برای train استفاده بشه؛ برای val/test
    باید None بمونه چون ترتیب زمانی برای ساخت anchor/backtest لازمه.
    """

    valid_mask = compute_valid_anchor_mask(
        features_array, target_array, seq_length
    )
    valid_anchors = np.nonzero(valid_mask)[0].astype(np.int64)

    n_features = features_array.shape[1]

    def gen():
        for i in valid_anchors:
            window = features_array[
                i - seq_length + 1: i + 1
            ]
            yield window, target_array[i]

    output_signature = (
        tf.TensorSpec(
            shape=(seq_length, n_features),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.float32,
        ),
    )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature,
    )

    if shuffle_buffer:
        dataset = dataset.shuffle(
            shuffle_buffer,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    ).prefetch(
        tf.data.AUTOTUNE
    )

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

    این تابع دو حالت ورودی رو پشتیبانی می‌کنه (برای حفظ سازگاری با
    فراخوانی‌های قدیمی/eager، طبق اصل «کد اصلی حذف نمی‌شود»):

    ۱) حالت جدید (lazy، برای اجرای اصلی با دیتای بزرگ): X_train و X_val
       یک tf.data.Dataset هستن که هرکدوم از قبل (X, y) رو batch شده
       برمی‌گردونن - در این حالت y_train/y_val باید None باشن.

    ۲) حالت قدیمی (eager، برای دیتای کوچیک/تست دستی): X_train/y_train و
       X_val/y_val آرایه‌ی numpy هستن - رفتار دقیقاً مثل قبل.
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

    is_dataset = isinstance(
        X_train, tf.data.Dataset
    )

    if is_dataset:
        # حالت lazy: دیتاست از قبل batch شده، دیگه نه y جدا لازمه نه
        # batch_size/shuffle (چون خودش batching رو مدیریت می‌کنه و برای
        # سری زمانی شافل نمی‌شه)
        history = model.fit(
            X_train,
            validation_data=X_val,
            epochs=epochs,
            callbacks=[
                early_stopping,
                lr_schedule,
                checkpoint,
            ],
            verbose=verbose,
        )
    else:
        # حالت قدیمی eager - دقیقاً همون رفتار قبلی
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
    درصد مواردی که مدل جهت حرکت را درست تشخیص داده.
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
    Evaluation مخصوص future return.
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

    یعنی:

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
# PREDICTION → PRICE
# ============================================================

def log_return_to_future_price(
    anchor_prices,
    predicted_log_returns,
):
    """
    Convert predicted log return to future price.

        future_price =
            anchor_price * exp(predicted_log_return)

    این تابع دقیقاً همونیه که crypto-signal-api باید موقع inference صدا
    بزنه تا خروجی مدل (بازده‌ی لگاریتمی) رو به یک قیمت واقعی قابل‌نمایش
    تبدیل کنه - وگرنه API عددی مثل 0.0032 رو به‌جای قیمت برمی‌گردونه.
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

    این Sharpe نهایی production نیست.
    فقط برای مقایسه‌ی اولیه‌ی مدل‌ها استفاده می‌شود.
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
    اصلاح اصلی این نسخه نسبت به قبلی: به‌جای STOP_LOSS_PCT/TAKE_PROFIT_PCT
    ثابت برای کل بک‌تست، فاصله‌ی SL/TP هر معامله را از ATR همان لحظه
    (anchor) محاسبه می‌کند - یعنی در کندل‌های پرنوسان حد ضرر گشادتر و در
    کندل‌های کم‌نوسان تنگ‌تر می‌شود؛ دقیقاً منطقی که یک تریدر واقعی هم
    برای مدیریت ریسک استفاده می‌کند.

    خروجی: دو آرایه (stop_loss_pct_array, take_profit_pct_array) هم‌طول
    با anchor_prices - هرکدام به‌صورت کسر مثبت (مثلاً 0.003 = 0.3%).

    clip بین min/max_stop_loss_pct به‌عنوان محافظ در برابر مقادیر ATR
    غیرعادی (مثلاً به‌خاطر gap قیمتی یا داده‌ی خراب) اعمال می‌شود.
    """

    atr_values = np.asarray(atr_values, dtype=float)
    anchor_prices = np.asarray(anchor_prices, dtype=float)

    # نسبت ATR به قیمت - این عدد "نوسان نسبی لحظه‌ای" است
    atr_pct = np.where(
        anchor_prices > 0,
        atr_values / anchor_prices,
        np.nan,
    )

    stop_loss_pct = atr_multiplier_sl * atr_pct
    take_profit_pct = atr_multiplier_tp * atr_pct

    stop_loss_pct = np.clip(
        stop_loss_pct, min_stop_loss_pct, max_stop_loss_pct
    )
    # TP را با همون نسبت نسبت به SL کلیپ‌شده تنظیم می‌کنیم تا risk/reward حفظ بشه
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
):
    """
    MVP trading simulation.

    Important:
        این backtest هنوز execution engine واقعی نیست.

    برای هر prediction:

        predicted_return > threshold
            → LONG

        predicted_return < -threshold
            → SHORT

        otherwise
            → NO TRADE

    سپس actual future return همان horizon را برای سنجش نتیجه استفاده می‌کنیم.

    Stop/TP (اصلاح‌شده - نسخه‌ی مسیر-محور/path-dependent):
        دیگر درصد ثابت نیستند - از compute_atr_based_risk_distances با
        ATR واقعی هر anchor محاسبه می‌شوند.

        اصلاح مهم نسبت به نسخه‌ی قبلی: قبلاً SL/TP فقط با مقایسه‌ی بازده‌ی
        نهایی کل افق (close-to-close بعد از horizon_steps کندل) تشخیص داده
        می‌شد - یعنی اگه قیمت وسط راه به SL می‌خورد ولی در پایان افق به
        محدوده‌ی سود برمی‌گشت، backtest اشتباهاً اون رو TAKE_PROFIT/TIME_EXIT
        حساب می‌کرد، در حالی که در واقعیت معامله خیلی زودتر با ضرر بسته
        می‌شد.

        حالا اگه future_highs/future_lows داده بشه (آرایه‌ی shape
        (n_samples, horizon_steps) از high/low هر کندل بین anchor تا پایان
        افق)، این تابع کندل‌به‌کندل جلو می‌ره و اولین لحظه‌ای که SL یا TP
        واقعاً لمس شده رو پیدا می‌کنه - دقیقاً مثل یک معامله‌ی واقعی.
        اگه در یک کندل *هم* SL *هم* TP لمس بشن (چون فقط high/low داریم، نه
        ترتیب دقیق تیک‌به‌تیک)، قانون محافظه‌کارانه‌ی ثابت اعمال می‌شه:
        فرض می‌کنیم SL زودتر خورده (بدبینانه، برای جلوگیری از خوش‌بینی
        کاذب در نتایج بک‌تست).

        اگه future_highs/future_lows داده نشه (None)، رفتار قبلی (فقط
        بر پایه‌ی بازده‌ی نهایی افق) به‌عنوان fallback حفظ می‌شه - برای
        سازگاری با فراخوانی‌های قدیمی.

    Position sizing:
        risk_per_trade / stop_loss_pct  (حالا per-trade، نه ثابت)

    اما leverage محدود می‌شود.
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

    atr_values = np.asarray(
        atr_values,
        dtype=float,
    )

    if not (
        len(anchor_prices)
        == len(actual_future_returns)
        == len(predicted_returns)
        == len(atr_values)
    ):
        raise ValueError(
            "Backtest input arrays must have equal length."
        )

    # --- محاسبه‌ی SL/TP اختصاصی هر anchor بر پایه‌ی ATR ---
    stop_loss_pct_arr, take_profit_pct_arr = compute_atr_based_risk_distances(
        atr_values,
        anchor_prices,
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
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

        stop_loss_pct = stop_loss_pct_arr[i]
        take_profit_pct = take_profit_pct_arr[i]

        if not (
            np.isfinite(prediction)
            and np.isfinite(actual_return)
            and np.isfinite(anchor)
            and np.isfinite(stop_loss_pct)
            and np.isfinite(take_profit_pct)
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
        # POSITION SIZE (بر پایه‌ی SL اختصاصی این معامله)
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
        # EXIT RULE (مسیر-محور در صورت وجود High/Low بین‌راه)
        # -----------------------------------------

        realized_move = None
        exit_reason = "TIME_EXIT"

        if future_highs is not None and future_lows is not None:

            if side == "LONG":
                stop_loss_price = anchor * (1 - stop_loss_pct)
                take_profit_price = anchor * (1 + take_profit_pct)
            else:
                stop_loss_price = anchor * (1 + stop_loss_pct)
                take_profit_price = anchor * (1 - take_profit_pct)

            for j in range(future_highs.shape[1]):

                h = future_highs[i, j]
                l = future_lows[i, j]

                if not (np.isfinite(h) and np.isfinite(l)):
                    # داده‌ی ناقص در این کندل - متوقف می‌شیم و به fallback
                    # زیر (بازده‌ی نهایی افق) واگذار می‌کنیم
                    break

                if side == "LONG":
                    hit_tp = h >= take_profit_price
                    hit_sl = l <= stop_loss_price
                else:
                    hit_tp = l <= take_profit_price
                    hit_sl = h >= stop_loss_price

                if hit_tp and hit_sl:
                    # قانون محافظه‌کارانه‌ی ثابت: چون ترتیب دقیق تیک‌به‌تیک
                    # داخل کندل رو نداریم، فرض بدبینانه می‌کنیم که SL
                    # زودتر از TP لمس شده
                    realized_move = -stop_loss_pct
                    exit_reason = "STOP_LOSS_AMBIGUOUS_SAME_CANDLE"
                    break
                elif hit_sl:
                    realized_move = -stop_loss_pct
                    exit_reason = "STOP_LOSS"
                    break
                elif hit_tp:
                    realized_move = take_profit_pct
                    exit_reason = "TAKE_PROFIT"
                    break

        if realized_move is None:
            # یا future_highs/future_lows داده نشده (fallback به رفتار
            # قدیمی)، یا در طول افق هیچ‌کدوم از SL/TP لمس نشدن (خروج زمانی
            # واقعی) - در هر دو حالت از بازده‌ی نهایی افق استفاده می‌کنیم
            if directional_return <= -stop_loss_pct:
                realized_move = -stop_loss_pct
                exit_reason = "STOP_LOSS"
            elif directional_return >= take_profit_pct:
                realized_move = take_profit_pct
                exit_reason = "TAKE_PROFIT"
            else:
                realized_move = directional_return
                exit_reason = "TIME_EXIT"

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
            "stop_loss_pct": float(
                stop_loss_pct
            ),
            "take_profit_pct": float(
                take_profit_pct
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

    هر fold فقط از گذشته برای train استفاده می‌کند.
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
# PURGE (رفع نشتی مرزی بین Train/Val/Test برای horizon چندکندلی)
# ============================================================

def purge_target_tail(target_series, horizon_steps):
    """
    مشکلی که این تابع حل می‌کنه:

    چون target[i] = log(close[i+horizon_steps] / close[i])، آخرین چند
    ردیف هر split (مثلاً train) از قیمت‌هایی محاسبه می‌شن که در واقع در
    بازه‌ی زمانی split *بعدی* (val) اتفاق افتادن. یعنی مدل در حین train،
    به‌طور غیرمستقیم اطلاعاتی از آینده‌ی نزدیک (داده‌ی validation) رو در
    برچسب‌هاش می‌بینه - این دقیقاً همون چیزیه که purge/embargo در
    time-series ML بهش می‌گن و باید حذف بشه.

    این تابع آخرین horizon_steps مقدار target رو NaN می‌کنه؛ چون
    create_sequences مقادیر non-finite رو حذف می‌کنه، این عملاً یعنی
    نمونه‌های نزدیک به مرز از train/val کنار گذاشته می‌شن (purge واقعی).
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
    بسیار مهم:

    برای validation و test، history مربوط به بخش قبل را نگه می‌داریم.

    اما scaler فقط روی train fit می‌شود.

    این کار باعث می‌شود اولین sequenceهای validation/test بتوانند
    history واقعی قبل از split را ببینند.

    هیچ future target وارد sequence نمی‌شود.

    Purge (اصلاح جدید): horizon_steps > 0 باعث می‌شه آخرین horizon_steps
    مقدار target در train و val (که به بازه‌ی زمانی split بعدی نشت
    می‌کردن) حذف بشن - جلوگیری از leakage مرزی که در نسخه‌ی قبلی این
    فایل وجود نداشت.
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

    train_target = purge_target_tail(
        train_target, horizon_steps
    )

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

    val_target = purge_target_tail(
        val_target, horizon_steps
    )

    # -------------------------
    # Test
    #
    # Include context from before
    # test start.
    #
    # نکته: تیل test عمداً purge نمی‌شه چون هیچ split دیگه‌ای داخل همین
    # fold بعد از test نمیاد که ازش نشت کنه (هر fold مستقل از fold بعدیه).
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

    برای MVP خاموش است.
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
    train_dataset,
    val_dataset,
    input_shape,
    output_dir,
    max_trials=20,
    epochs_per_trial=30,
):
    """
    Optional Bayesian Optimization.

    اصلاح: حالا train_dataset/val_dataset، tf.data.Dataset (از
    build_lazy_sequence_dataset) هستن، نه آرایه‌ی eager X/y جدا - چون
    KerasTuner از Dataset پشتیبانی می‌کنه (دقیقاً مثل model.fit)، فقط
    batch_size/y جدا لازم نیست چون دیتاست خودش batch شده و (X,y) رو با
    هم برمی‌گردونه.
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
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_per_trial,
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
    train_date_range=None,
    test_date_range=None,
):
    """
    Save model metadata.

    این metadata بعداً در crypto-signal-api برای جلوگیری از mismatch
    بین model و feature pipeline استفاده خواهد شد.

    train_date_range / test_date_range: تاپل (start, end) به‌صورت رشته -
    مشخص می‌کنه این مدل روی کدوم بازه‌ی زمانی واقعی train/test شده، تا
    بعداً بشه فهمید مدل برای کدوم رژیم بازار مناسب‌تره.
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

    # --- کاهش مصرف رم GPU با mixed precision (float16 برای محاسبات، float32 برای وزن‌ها) ---
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # --------------------------------------------------------
    # DATA PATH
    # --------------------------------------------------------

    file_path = (
        "/content/drive/MyDrive/"
        "binance_data_5min.csv"
    )

    # محدود کردن دیتا به بازه‌ی زمانی اخیر - هم رم رو کنترل می‌کنه هم
    # مدل رو روی رژیم فعلی بازار متمرکز می‌کنه (نه رژیم‌های قدیمی که
    # دیگه تکرار نمی‌شن). None یعنی بدون محدودیت (کل دیتا از ابتدا).
    MIN_DATA_DATE = "2023-01-01"  # حدود ۳ سال آخر - در صورت نیاز تغییر بدید

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

    print(
        f"Risk model   : ATR-based SL "
        f"({ATR_MULTIPLIER_SL}x) / TP ({ATR_MULTIPLIER_TP}x)"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_and_preprocess_data(
        file_path
    )

    print(f"\n[DATA] Full range loaded: {data.index.min()} -> {data.index.max()} ({len(data):,} rows)")

    if MIN_DATA_DATE is not None:
        data = data[data.index >= MIN_DATA_DATE]
        print(f"[DATA] Filtered to >= {MIN_DATA_DATE}: {len(data):,} rows remaining")

    # --- Quick Test: محدود کردن به آخرین ردیف‌ها برای اجرای سریع/سلامت‌سنجی ---
    effective_epochs = EPOCHS
    if QUICK_TEST:
        data = data.iloc[-QUICK_TEST_ROWS:]
        effective_epochs = QUICK_TEST_EPOCHS
        print(
            f"[QUICK TEST MODE] Using only last {len(data):,} rows, "
            f"epochs limited to {effective_epochs}. "
            f"بعد از موفقیت، QUICK_TEST = False کنید و اجرای کامل بگیرید."
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

    # ATR رو جدا از FEATURE_COLUMNS هم نگه می‌داریم چون برای بک‌تست لازمه
    # (حتی اگه به هر دلیلی از FEATURE_COLUMNS حذفش کنید، بک‌تست همچنان
    # کار می‌کنه چون از خودِ feature_data قبل از reindex می‌خونتش)
    atr_series_full = feature_data["ATR"].copy()

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

    # هماهنگ کردن ATR با ایندکس نهایی feature_data (بعد از dropna)
    atr_series_full = atr_series_full.reindex(feature_data.index)

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

        # --- بازه‌ی زمانی واقعی هر fold (نه فقط ایندکس ردیف) ---
        # این برای دیتای طولانی‌مدت (مثلاً از 2018) حیاتیه: بدون این،
        # نمی‌شه فهمید هر fold دقیقاً کدوم رژیم بازار (خرسی/گاوی/رنج) رو
        # پوشش می‌ده، و مقایسه‌ی عملکرد مدل بین fold های مختلف بی‌معنی
        # می‌مونه.
        train_start_date = feature_data.index[train_start]
        train_end_date = feature_data.index[val_start - 1]
        val_start_date = feature_data.index[val_start]
        val_end_date = feature_data.index[test_start - 1]
        test_start_date = feature_data.index[test_start]
        test_end_date = feature_data.index[min(test_end, len(feature_data)) - 1]

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
        # SEQUENCES (lazy - رفع OOM)
        # ----------------------------------------------------
        #
        # به‌جای create_sequences/create_sequences_with_indices (که کل
        # تانسور N×seq_length×features رو eager می‌سازن)، از
        # build_lazy_sequence_dataset استفاده می‌کنیم که فقط anchor های
        # معتبر رو (با اسکن برداری ارزون) پیدا می‌کنه و پنجره‌ها رو در
        # لحظه‌ی نیاز (batch به batch) تولید می‌کنه.

        train_dataset, train_anchors = build_lazy_sequence_dataset(
            train_scaled,
            train_target.values,
            SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            shuffle_buffer=None,  # سری زمانی - شافل نمی‌کنیم (رفتار قبلی shuffle=False حفظ شد)
        )

        val_dataset, val_anchors_local = build_lazy_sequence_dataset(
            val_scaled,
            val_target.values,
            SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            shuffle_buffer=None,
        )

        test_dataset, test_anchors_local = build_lazy_sequence_dataset(
            test_scaled,
            test_target.values,
            SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            shuffle_buffer=None,
        )

        print(
            f"[SEQUENCES] Train: {len(train_anchors):,}"
        )

        print(
            f"[SEQUENCES] Val  : {len(val_anchors_local):,}"
        )

        print(
            f"[SEQUENCES] Test : {len(test_anchors_local):,}"
        )

        if (
            len(train_anchors) == 0
            or len(val_anchors_local) == 0
            or len(test_anchors_local) == 0
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
            SEQUENCE_LENGTH,
            train_scaled.shape[1],
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
                    train_dataset,
                    val_dataset,
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
            train_dataset,
            None,
            val_dataset,
            None,
            epochs=effective_epochs,
            batch_size=BATCH_SIZE,
            model_save_path=model_path,
            verbose=1,
        )

        # ----------------------------------------------------
        # TEST PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(
            test_dataset,
            verbose=0,
        ).reshape(-1)

        # y_test_flat: چون generator داخل build_lazy_sequence_dataset دقیقاً
        # به همون ترتیب test_anchors_local مقدار target رو yield می‌کنه،
        # بازسازی مستقیم از test_target.values هم دقیقاً همون ترتیب رو
        # می‌ده - بدون نیاز به عبور دوباره از دیتاست.
        y_test_flat = (
            test_target.values[test_anchors_local]
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
        # ANCHOR PRICES + ANCHOR ATR + TIMESTAMPS
        # ----------------------------------------------------

        # test_anchors_local ایندکس‌های محلی داخل test_df هستن.
        #
        # test_df از test_context_start شروع می‌شه.
        #
        # تبدیل ایندکس محلی به ایندکس global در feature_data.

        global_test_indices = (
            test_context_start
            + test_anchors_local
        )

        test_times = test_df.index[test_anchors_local]

        anchor_prices = (
            feature_data.iloc[
                global_test_indices
            ]["close"]
            .values
        )

        # ATR در لحظه‌ی anchor - برای محاسبه‌ی SL/TP اختصاصی هر معامله
        anchor_atr = (
            atr_series_full.iloc[
                global_test_indices
            ]
            .values
        )

        # ----------------------------------------------------
        # FUTURE HIGH/LOW (برای بک‌تست مسیر-محور)
        # ----------------------------------------------------
        #
        # برای هر anchor، high/low کندل‌های 1 تا HORIZON_STEPS بعد از اون
        # (نه شامل خود anchor) رو استخراج می‌کنیم تا backtest_strategy
        # بتونه کندل‌به‌کندل چک کنه SL/TP کجا واقعاً لمس شده.
        #
        # نکته: چون target قبلاً برای همین انکرها finite بوده (یعنی
        # close[anchor+HORIZON_STEPS] در feature_data وجود داشته)، این
        # ایندکس‌ها همیشه در محدوده‌ی feature_data معتبرن - نیازی به
        # کلمپ کردن نیست.

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
            train_date_range=(str(train_start_date), str(train_end_date)),
            test_date_range=(str(test_start_date), str(test_end_date)),
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        prediction_df = pd.DataFrame(
            {
                "timestamp": test_times,
                "anchor_price": anchor_prices,
                "anchor_atr": anchor_atr,
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
            "train_start_date": str(train_start_date),
            "train_end_date": str(train_end_date),
            "val_start_date": str(val_start_date),
            "val_end_date": str(val_end_date),
            "test_start_date": str(test_start_date),
            "test_end_date": str(test_end_date),
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
                "test_start_date": str(test_start_date),
                "test_end_date": str(test_end_date),
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

        # ----------------------------------------------------
        # CLEANUP (رفع OOM بین Foldها)
        # ----------------------------------------------------
        #
        # بدون این، حافظه‌ی مدل قبلی (وزن‌ها، optimizer state، گراف
        # محاسباتی TensorFlow) و آرایه‌های سنگین این fold (train_scaled,
        # val_scaled, test_scaled, future_highs/lows, ...) تا پایان کل
        # اجرا در RAM باقی می‌مونن و روی هم انباشته می‌شن - با ۴ fold و
        # دیتای چندصدهزار ردیفی، این خودش می‌تونه باعث OOM بشه حتی اگه
        # هر fold به‌تنهایی مشکلی نداشته باشه.

        del model
        del train_dataset, val_dataset, test_dataset
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

    # اگر در Colab هستی، این دو خط را یک بار اجرا کن:
    #
    # from google.colab import drive
    # drive.mount('/content/drive')

    main()
