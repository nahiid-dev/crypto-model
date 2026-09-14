"""
مدل GRU برای پیش‌بینی قیمت - نسخه‌ی Direct Multi-Step

تصمیم معماری کلیدی: مدل روی دیتای با گرانولاریتی ریز (مثلاً ۵ دقیقه) train
می‌شه، ولی هدف (target) قیمت close در افق دورتر (مثلاً ۱۵ دقیقه = ۳ کندل
جلوتر) است - نه کندل بعدی. این روش «Direct Multi-Step Forecasting» نام
داره و طبق شواهد پژوهشی، از دو روش دیگه بهتره:
  - Recursive/Iterated: پیش‌بینی یک‌گام رو به‌عنوان ورودی گام بعد می‌ده،
    خطا در هر گام تشدید می‌شه.
  - Resample به تایم‌فریم درشت‌تر: جزئیات قیمتی داخل هر بازه از دست میره.
Direct هم granularity کامل ۵ دقیقه‌ای رو حفظ می‌کنه، هم مستقیماً افق
تصمیم‌گیری معاملاتی (۱۵ دقیقه) رو هدف می‌گیره، بدون انباشت خطا.

نکته‌ی مهم معماری: چون هدف الان "قیمت ۱۵ دقیقه بعد" است، نه "کندل بعدی"،
baseline نایو و بک‌تست هم باید متناسب اصلاح بشن (این نسخه این اصلاح رو
داره؛ نسخه‌ی قبلی این فایل این مشکل رو داشت):
  - Baseline نایو حالا یعنی "قیمت ۱۵ دقیقه دیگه = همون قیمت الانه"
    (persistence روی افق)، نه "کندل بعدی = کندل الان".
  - بک‌تست حالا معاملات را به‌صورت غیرهمپوشان (non-overlapping) با فاصله‌ی
    HORIZON_STEPS شبیه‌سازی می‌کنه - یعنی دقیقاً مثل رفتار واقعی شما: یک
    پوزیشن باز می‌کنید، تا افق پیش‌بینی (۱۵ دقیقه) نگهش می‌دارید، بعد
    دوباره تصمیم می‌گیرید. بک‌تست نسخه‌ی قبلی فرض می‌کرد هر کندل یک تصمیم
    جدیده و پوزیشن‌ها را هم‌پوشان حساب می‌کرد که با این معماری هماهنگ نبود.

شامل:
- اندیکاتورهای تکنیکال: SMA, RSI (وایلدر), Bollinger Bands کامل, ATR,
  MACD, حجم نسبی, زمان چرخه‌ای (ساعت/روز هفته)
- Target با افق مشخص (HORIZON_STEPS) به‌جای کندل بعدی
- اعتبارسنجی Walk-Forward (به‌جای یک split ثابت)
- بک‌تست غیرهم‌پوشان با کارمزد رفت‌وبرگشت واقعی صرافی
- مقایسه با baseline نایوی هماهنگ با افق (نه کندل بعدی)

معماری مدل: GRU دولایه + Multi-Head Attention (لایه‌ی آماده‌ی Keras) +
Huber Loss + L2 Regularization + Learning Rate Warmup/Cosine Decay.

برای جستجوی خودکار بهترین hyperparameter ها، تابع run_hyperparameter_search
را در main فعال کنید (نیاز به نصب: pip install keras-tuner --break-system-packages).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
import tensorflow as tf
import joblib
import os

# --- فایل مشترک فیچرها - باید عیناً با نسخه‌ی crypto-signal-api یکی باشه ---
# (کپی همین features.py را کنار این فایل قرار دهید)
from features import add_technical_indicators, FEATURE_COLUMNS, FEATURE_VERSION


# ----------------------------------------------------------------------
# بارگذاری و پیش‌پردازش داده
# ----------------------------------------------------------------------
def load_and_preprocess_data(file_path):
    """
    Loads data from a CSV file, converts 'open_time' to datetime,
    sets it as index, selects relevant columns, and drops NA values.

    نکته: این فایل باید دقیقاً با تایم‌فریم ریز موردنظر (مثلاً ۵ دقیقه)
    از قبل دانلود شده باشه - این تابع resample نمی‌کنه.
    """
    data = pd.read_csv(file_path)
    data["open_time"] = pd.to_datetime(data["open_time"])
    data.set_index("open_time", inplace=True)
    data = data[["open", "high", "low", "close", "volume"]]
    data.dropna(inplace=True)
    return data


# ----------------------------------------------------------------------
# ساخت توالی‌ها با هدف Direct Multi-Step (نکته‌ی معماری اصلی این نسخه)
# ----------------------------------------------------------------------
def create_sequences(data, seq_length, horizon_steps):
    """
    برخلاف نسخه‌ی single-step قبلی (که y = data[i, 0]، یعنی کندل بلافاصله
    بعد از توالی بود)، اینجا:

        X[k] = data[i - seq_length : i]         (آخرین کندل داخلش، ایندکس i-1، "لنگر"/anchor است)
        y[k] = data[i + horizon_steps - 1, 0]   (قیمت close دقیقاً horizon_steps کندل بعد از انکر)

    یعنی با horizon_steps=3 روی دیتای ۵دقیقه‌ای، مدل قیمت ۱۵ دقیقه بعد از
    آخرین کندلی که دیده رو پیش‌بینی می‌کنه - نه کندل بعدی.

    نکته‌ی مهم: چون anchor (قیمت لحظه‌ی تصمیم) همیشه ستون 0 از آخرین ردیف
    ورودی (X[k][-1, 0]) است، لازم نیست جدا ذخیره‌اش کنیم - همیشه از X قابل
    استخراجه (تابع extract_anchor_prices این کار رو می‌کنه).
    """
    X, y = [], []
    max_i = len(data) - horizon_steps + 1  # آخرین i مجاز که i+horizon_steps-1 از داده خارج نشه
    if max_i <= seq_length:
        return np.array(X), np.array(y)
    for i in range(seq_length, max_i):
        X.append(data[i - seq_length: i])
        y.append(data[i + horizon_steps - 1, 0])
    return np.array(X), np.array(y)


def extract_anchor_prices_normalized(X):
    """
    قیمت close انکر (لحظه‌ی تصمیم = آخرین کندل ورودی) رو از خود X استخراج
    می‌کنه، هنوز نرمال‌شده (باید بعداً inverse_transform بشه).
    """
    return X[:, -1, 0]


def rescale_column0(values_normalized, scaler):
    """
    کمکی برای inverse_transform یک بردار تک‌بعدی که فقط مقدار ستون 0
    (close) رو داره - بقیه‌ی ستون‌ها رو صفر می‌ذاریم چون MinMaxScaler
    خطیه و فقط ستون 0 برامون مهمه.
    """
    num_features = scaler.n_features_in_
    dummy = np.zeros((len(values_normalized), num_features))
    dummy[:, 0] = values_normalized.reshape(-1)
    return scaler.inverse_transform(dummy)[:, 0]


# ----------------------------------------------------------------------
# مدل GRU با Multi-Head Attention
# ----------------------------------------------------------------------
def build_gru_model(input_shape, gru_units=64, dropout_rate=0.2, num_heads=4,
                     key_dim=16, l2_reg=1e-4, dense_units=32, learning_rate=1e-3):
    """
    GRU دولایه + Multi-Head Attention + L2 Regularization + Huber Loss.
    (بدون تغییر نسبت به نسخه‌ی قبلی - این بخش مشکلی نداشت، فقط target و
    ارزیابی/بک‌تست بودن که نیاز به اصلاح داشتن.)
    """
    inputs = Input(shape=input_shape)
    x = GRU(gru_units, activation="tanh", return_sequences=True,
            kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))(inputs)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units, activation="tanh", return_sequences=True,
            kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization()(x + attn_output)
    x = tf.reduce_mean(x, axis=1)

    x = Dense(dense_units, activation="tanh", kernel_regularizer=l2(l2_reg))(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=Huber(delta=1.0))
    return model


def build_warmup_cosine_schedule(total_epochs, warmup_epochs=5, base_lr=1e-3, min_lr=1e-6):
    """Learning Rate Warmup (خطی) + Cosine Decay - جایگزین ReduceLROnPlateau."""
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay
    return LearningRateScheduler(schedule, verbose=0)


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64,
                 model_save_path="best_model.keras", verbose=1):
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr_schedule = build_warmup_cosine_schedule(total_epochs=epochs, warmup_epochs=5)
    model_checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=verbose)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_schedule, model_checkpoint],
        verbose=verbose,
    )
    return model, history


# ----------------------------------------------------------------------
# Hyperparameter Search خودکار (اختیاری)
# ----------------------------------------------------------------------
def build_gru_model_tunable(hp, input_shape):
    """نسخه‌ی قابل‌جستجوی build_gru_model برای KerasTuner."""
    gru_units = hp.Choice("gru_units", values=[32, 64, 128])
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.4, step=0.1)
    num_heads = hp.Choice("num_heads", values=[2, 4, 8])
    key_dim = hp.Choice("key_dim", values=[8, 16, 32])
    l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])
    dense_units = hp.Choice("dense_units", values=[16, 32, 64])
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    inputs = Input(shape=input_shape)
    x = GRU(gru_units, activation="tanh", return_sequences=True,
            kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))(inputs)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units, activation="tanh", return_sequences=True,
            kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization()(x + attn_output)
    x = tf.reduce_mean(x, axis=1)

    x = Dense(dense_units, activation="tanh", kernel_regularizer=l2(l2_reg))(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=Huber(delta=1.0),
    )
    return model


def run_hyperparameter_search(X_train, y_train, X_val, y_val, input_shape,
                               output_dir=".", max_trials=20, epochs_per_trial=30):
    """Bayesian Optimization برای پیدا کردن بهترین ترکیب hyperparameter."""
    import keras_tuner as kt

    tuner = kt.BayesianOptimization(
        lambda hp: build_gru_model_tunable(hp, input_shape),
        objective="val_loss",
        max_trials=max_trials,
        directory=os.path.join(output_dir, "kt_search"),
        project_name="gru_attention_search",
        overwrite=True,
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_per_trial,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n[Hyperparameter Search] بهترین ترکیب پیدا‌شده:")
    for param_name in ["gru_units", "dropout_rate", "num_heads", "key_dim",
                        "l2_reg", "dense_units", "learning_rate"]:
        print(f"    {param_name}: {best_hp.get(param_name)}")
    return best_hp


def build_gru_model_from_hp(input_shape, best_hp):
    return build_gru_model(
        input_shape,
        gru_units=best_hp.get("gru_units"),
        dropout_rate=best_hp.get("dropout_rate"),
        num_heads=best_hp.get("num_heads"),
        key_dim=best_hp.get("key_dim"),
        l2_reg=best_hp.get("l2_reg"),
        dense_units=best_hp.get("dense_units"),
        learning_rate=best_hp.get("learning_rate"),
    )


# ----------------------------------------------------------------------
# ارزیابی - حالا شامل anchor price هم هست (لازم برای baseline/بک‌تست درست)
# ----------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, scaler):
    """
    برمی‌گردونه: y_test_rescaled (قیمت واقعی در افق هدف)،
    predictions_rescaled (پیش‌بینی مدل برای همون افق)،
    anchor_prices_rescaled (قیمت لحظه‌ی تصمیم = آخرین کندل ورودی)،
    و معیارهای خطا.

    anchor_prices لازمه چون این‌جا دیگه "کندل قبلی در آرایه‌ی y_test" معنای
    "همین الان" رو نداره (چون y_test الان قیمت‌های horizon-قدم-جلوتره،
    نه کندل‌های متوالی) - باید مستقیم از X استخراج بشه.
    """
    predictions = model.predict(X_test, verbose=0)
    predictions_rescaled = rescale_column0(predictions.reshape(-1), scaler)
    y_test_rescaled = rescale_column0(y_test.reshape(-1), scaler)
    anchor_prices_rescaled = rescale_column0(extract_anchor_prices_normalized(X_test), scaler)

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)
    next_predicted_price = predictions_rescaled[-1] if len(predictions_rescaled) > 0 else np.nan

    return y_test_rescaled, predictions_rescaled, anchor_prices_rescaled, mse, rmse, mae, r2, next_predicted_price


def naive_baseline_metrics(anchor_prices_rescaled, y_test_rescaled):
    """
    Naive forecast هماهنگ با افق: فرض می‌کنیم قیمت در طول افق (مثلاً ۱۵
    دقیقه) بدون تغییر می‌مونه، یعنی پیش‌بینی = قیمت انکر (لحظه‌ی الان).

    این جایگزین باگ نسخه‌ی قبلی شد که baseline رو با "کندل قبلی در آرایه‌ی
    y_test" می‌سنجید - که چون y_test الان مقادیر horizon-قدم-جلوتره (نه
    کندل‌های متوالی)، آن مقایسه اصلاً معنای "فردا=امروز" رو نداشت.
    """
    if len(y_test_rescaled) == 0:
        return {"RMSE": np.nan, "R²": np.nan}
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, anchor_prices_rescaled))
    r2 = r2_score(y_test_rescaled, anchor_prices_rescaled)
    return {"RMSE": rmse, "R²": r2}


def plot_results(y_test_rescaled, predictions_rescaled, timeframe, dates, fold=None, output_dir="."):
    suffix = f"_fold{fold}" if fold is not None else ""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test_rescaled, label="قیمت واقعی (در افق هدف)")
    plt.plot(dates, predictions_rescaled, label="قیمت پیش‌بینی‌شده", linestyle="--")
    plt.title(f"GRU Model - {timeframe} Price Prediction{suffix}")
    plt.xlabel("زمان")
    plt.ylabel("قیمت")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"GRU_{timeframe}{suffix}_prediction.png"))
    plt.close()


# ----------------------------------------------------------------------
# اعتبارسنجی Walk-Forward
# ----------------------------------------------------------------------
def generate_walk_forward_folds(n_samples, n_folds=4, train_ratio=0.7, val_ratio=0.15):
    folds = []
    for i in range(1, n_folds + 1):
        end_idx = int(n_samples * i / n_folds)
        train_end = int(end_idx * train_ratio)
        val_end = int(end_idx * (train_ratio + val_ratio))
        folds.append((0, train_end, val_end, end_idx))
    return folds


# ----------------------------------------------------------------------
# بک‌تست غیرهم‌پوشان هماهنگ با افق (اصلاح اصلی نسبت به نسخه‌ی قبلی)
# ----------------------------------------------------------------------
def horizon_aware_backtest(anchor_prices, y_true, y_pred, horizon_steps,
                            fee_rate=0.0004, initial_capital=1000):
    """
    نسخه‌ی قبلی این بک‌تست فرض می‌کرد هر کندل یک معامله‌ی جدیده و پوزیشن
    فقط یک کندل نگه داشته می‌شه - این با معماری Direct Multi-Step ناسازگار
    بود، چون الان هر پیش‌بینی مربوط به ۱۵ دقیقه (horizon_steps کندل) بعده،
    نه کندل بعدی. نگه‌داشتن پوزیشن فقط برای یک کندل در حالی که مدل برای
    ۱۵ دقیقه پیش‌بینی کرده، باعث سنجش نادرست سودآوری می‌شد.

    این نسخه‌ی اصلاح‌شده معاملات رو به‌صورت غیرهم‌پوشان (non-overlapping)
    شبیه‌سازی می‌کنه: با گام‌های horizon_steps روی داده حرکت می‌کنه (دقیقاً
    مثل رفتار واقعی شما - وارد می‌شید، ۱۵ دقیقه نگه می‌دارید، دوباره تصمیم
    می‌گیرید)، و برای هر معامله یک کارمزد رفت‌وبرگشت کامل (ورود+خروج) اعمال
    می‌کنه.
    """
    capital = initial_capital
    n_trades = 0
    equity_curve = [capital]

    for j in range(0, len(y_true), horizon_steps):
        anchor = anchor_prices[j]
        true_target = y_true[j]
        pred_target = y_pred[j]

        predicted_direction = 1 if pred_target > anchor else -1
        actual_return = (true_target - anchor) / anchor

        # کارمزد رفت‌وبرگشت کامل (ورود + خروج) چون این یک معامله‌ی کامل و مجزاست
        capital *= (1 - 2 * fee_rate)
        n_trades += 1

        capital *= (1 + predicted_direction * actual_return)
        equity_curve.append(capital)

    total_return_pct = (capital - initial_capital) / initial_capital * 100

    # Buy & hold روی کل بازه‌ی تست (از اولین انکر تا آخرین هدف) برای مقایسه‌ی منصفانه
    buy_hold_return_pct = (y_true[-1] - anchor_prices[0]) / anchor_prices[0] * 100 if len(y_true) > 0 else np.nan

    return {
        "final_capital": capital,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "n_trades": n_trades,
        "equity_curve": equity_curve,
    }


def main():
    file_path = "/content/drive/MyDrive/binance_data_5min.csv"  # دیتای خام ۵ دقیقه‌ای شما
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure the file path is correct and you have mounted Google Drive if using Colab.")
        return

    output_dir = "/content/drive/MyDrive/model_outputs_gru"
    os.makedirs(output_dir, exist_ok=True)

    # ================= تنظیمات افق پیش‌بینی (Direct Multi-Step) =================
    # داده‌ی خام باید دقیقاً با TIMEFRAME_MINUTES دانلود شده باشه (مثلاً از
    # Binance با interval="5m"). HORIZON_MINUTES افق واقعی تصمیم معاملاتی
    # شماست. HORIZON_STEPS خودکار محاسبه می‌شه تا از ناهماهنگی دستی
    # (مثلاً یادتون بره وقتی TIMEFRAME رو عوض کردید HORIZON_STEPS رو هم
    # عوض کنید) جلوگیری بشه.
    TIMEFRAME_MINUTES = 5
    HORIZON_MINUTES = 15
    HORIZON_STEPS = HORIZON_MINUTES // TIMEFRAME_MINUTES
    assert HORIZON_MINUTES % TIMEFRAME_MINUTES == 0, (
        "HORIZON_MINUTES باید مضرب صحیحی از TIMEFRAME_MINUTES باشه."
    )
    print(f"Direct Multi-Step: train روی {TIMEFRAME_MINUTES} دقیقه، "
          f"target = {HORIZON_STEPS} کندل جلوتر ({HORIZON_MINUTES} دقیقه)")

    timeframe_label = f"{TIMEFRAME_MINUTES}min_to_{HORIZON_MINUTES}min"
    seq_length = 50
    n_folds = 4
    fee_rate = 0.0004  # کارمزد یک‌طرفه‌ی تیکر فیوچرز - با کارمزد صرافی خودتون تنظیم کنید

    USE_HYPERPARAMETER_SEARCH = False
    HP_SEARCH_MAX_TRIALS = 20
    HP_SEARCH_EPOCHS_PER_TRIAL = 30

    fold_results = []
    backtest_summary = []

    base_data = load_and_preprocess_data(file_path)
    if base_data.empty:
        print("No data loaded. Exiting.")
        return

    tf_data_with_indicators = add_technical_indicators(base_data)
    if tf_data_with_indicators.empty:
        print("No data after adding technical indicators.")
        return

    # از features.py وارد شده - عیناً همینه که crypto-signal-api هم استفاده می‌کنه
    feature_cols = FEATURE_COLUMNS
    missing_cols = [c for c in feature_cols if c not in tf_data_with_indicators.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    tf_data_ordered = tf_data_with_indicators[feature_cols]
    features_np = tf_data_ordered.values
    n_samples = len(features_np)

    folds = generate_walk_forward_folds(n_samples, n_folds=n_folds)

    for fold_idx, (start, train_end, val_end, test_end) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_idx}/{n_folds} | train: 0-{train_end}, val: {train_end}-{val_end}, test: {val_end}-{test_end} ===")

        train_features = features_np[start:train_end]
        val_features = features_np[train_end:val_end]
        test_features = features_np[val_end:test_end]

        min_required = seq_length + HORIZON_STEPS
        if not (len(train_features) > min_required and len(val_features) > min_required and len(test_features) > min_required):
            print(f"Insufficient data in fold {fold_idx} for seq_length+horizon={min_required}. Skipping.")
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_normalized = scaler.fit_transform(train_features)
        val_data_normalized = scaler.transform(val_features)
        test_data_normalized = scaler.transform(test_features)

        scaler_path = os.path.join(output_dir, f"scaler_{timeframe_label}_fold{fold_idx}.pkl")
        joblib.dump(scaler, scaler_path)

        X_train, y_train = create_sequences(train_data_normalized, seq_length, HORIZON_STEPS)
        X_val, y_val = create_sequences(val_data_normalized, seq_length, HORIZON_STEPS)
        X_test, y_test = create_sequences(test_data_normalized, seq_length, HORIZON_STEPS)

        if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"Skipping fold {fold_idx} due to insufficient data after sequencing.")
            continue

        # آفست تاریخ برای نمودار: هدف k-ام مربوط به ایندکس global زیره
        # (فرمول با توجه به offset افق تعمیم داده شده - برای horizon=1 دقیقاً
        # با نسخه‌ی قبلی یکسانه)
        test_dates_start_index = val_end + seq_length + HORIZON_STEPS - 1
        dates_for_plot = tf_data_ordered.index[test_dates_start_index: test_dates_start_index + len(y_test)]
        input_shape = (X_train.shape[1], X_train.shape[2])
        is_last_fold = (fold_idx == n_folds)

        if USE_HYPERPARAMETER_SEARCH and is_last_fold:
            print(f"\n[Hyperparameter Search] شروع جستجو روی fold {fold_idx}...")
            best_hp = run_hyperparameter_search(
                X_train, y_train, X_val, y_val, input_shape,
                output_dir=output_dir,
                max_trials=HP_SEARCH_MAX_TRIALS,
                epochs_per_trial=HP_SEARCH_EPOCHS_PER_TRIAL,
            )
            build_fn = lambda shape: build_gru_model_from_hp(shape, best_hp)
        else:
            build_fn = lambda shape: build_gru_model(shape)

        model = build_fn(input_shape)
        model_path = os.path.join(output_dir, f"best_gru_{timeframe_label}_fold{fold_idx}.keras")
        model, _ = train_model(model, X_train, y_train, X_val, y_val,
                                epochs=100, model_save_path=model_path, verbose=0)

        y_test_r, preds_r, anchors_r, mse, rmse, mae, r2, next_p = evaluate_model(model, X_test, y_test, scaler)
        model.save(os.path.join(output_dir, f"model_GRU_{timeframe_label}_fold{fold_idx}.keras"))

        # --- Metadata (طبق سند معماری - برای جلوگیری از ناهماهنگی مدل/فیچر در آینده) ---
        metadata = {
            "model_name": "eth_gru",
            "symbol": "ETHUSDT",  # با نماد واقعی خودتون جایگزین کنید
            "timeframe_minutes": TIMEFRAME_MINUTES,
            "horizon_minutes": HORIZON_MINUTES,
            "horizon_steps": HORIZON_STEPS,
            "sequence_length": seq_length,
            "target": "close_price_at_horizon",  # وقتی به log-return تغییر کردید، اینجا هم آپدیت کنید
            "features_version": FEATURE_VERSION,
            "feature_columns": feature_cols,
            "fold": fold_idx,
            "fee_rate_used_in_backtest": fee_rate,
        }
        import json as _json
        with open(os.path.join(output_dir, f"metadata_{timeframe_label}_fold{fold_idx}.json"), "w") as f:
            _json.dump(metadata, f, indent=2, ensure_ascii=False)

        naive = naive_baseline_metrics(anchors_r, y_test_r)
        if rmse >= naive["RMSE"]:
            print(f"[WARNING] fold{fold_idx}: RMSE مدل ({rmse:.2f}) بهتر از baseline نایوی "
                  f"هم‌افق ({naive['RMSE']:.2f}) نیست - یعنی مدل عملاً بهتر از فرض "
                  f"'قیمت {HORIZON_MINUTES} دقیقه بعد = قیمت الان' عمل نمی‌کنه.")
        else:
            improvement = (1 - rmse / naive["RMSE"]) * 100
            print(f"[OK] fold{fold_idx}: RMSE مدل {improvement:.1f}% بهتر از baseline نایوی هم‌افقه.")

        fold_results.append({
            "fold": fold_idx, "MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2,
            "Next_Predicted_Price": next_p,
            "Naive_RMSE": naive["RMSE"], "Naive_R²": naive["R²"],
        })

        if len(dates_for_plot) == len(y_test_r):
            plot_results(y_test_r, preds_r, timeframe_label, dates_for_plot, fold=fold_idx, output_dir=output_dir)

        bt = horizon_aware_backtest(anchors_r, y_test_r, preds_r, HORIZON_STEPS, fee_rate=fee_rate)
        backtest_summary.append({
            "fold": fold_idx,
            "strategy_return_%": round(bt["total_return_pct"], 2),
            "buy_hold_return_%": round(bt["buy_hold_return_pct"], 2),
            "n_trades": bt["n_trades"],
        })
        print(f"[Backtest] fold{fold_idx}: strategy={bt['total_return_pct']:.2f}% "
              f"vs buy&hold={bt['buy_hold_return_pct']:.2f}% ({bt['n_trades']} معامله‌ی غیرهم‌پوشان)")

    print("\n--- خلاصه معیارهای ارزیابی (هر fold) ---")
    results_df = pd.DataFrame(fold_results)
    if not results_df.empty:
        print(results_df.to_string(index=False))
        results_df.to_csv(os.path.join(output_dir, "gru_metrics_summary.csv"), index=False)

    print("\n--- خلاصه بک‌تست (استراتژی در مقابل خرید-و-نگه‌داری) ---")
    bt_df = pd.DataFrame(backtest_summary)
    if not bt_df.empty:
        print(bt_df.to_string(index=False))
        bt_df.to_csv(os.path.join(output_dir, "backtest_summary.csv"), index=False)
        print(f"\nنتایج در {output_dir} ذخیره شدن.")
    else:
        print("No backtest results generated.")

    print("\nProcessing complete.")


if __name__ == "__main__":
    # from google.colab import drive
    # drive.mount('/content/drive')
    main()
