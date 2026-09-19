"""
features.py - ماژول مشترک Feature Engineering

این فایل باید عیناً (بدون هیچ تفاوتی) هم در crypto-model (برای train) و هم
در crypto-signal-api (برای inference) وجود داشته باشه. دلیل جدا کردنش در
یک فایل مستقل دقیقاً همینه: تضمین می‌کنه فیچرهایی که مدل باهاشون train شده
و فیچرهایی که موقع پیش‌بینی زنده بهش داده می‌شه، دقیقاً یکی باشن.

نحوه‌ی استفاده در دو پروژه:
- در crypto-model: مستقیم import می‌شه و روی داده‌ی تاریخی اعمال می‌شه.
- در crypto-signal-api: همین فایل (کپی عیناً یکسان) روی آخرین کندل‌های
  زنده‌ی گرفته‌شده از صرافی اعمال می‌شه.

هر تغییری در این فایل باید FEATURE_VERSION را افزایش بده و در هر دو
پروژه هم‌زمان اعمال بشه - وگرنه مدلی که با یک نسخه از فیچرها train شده
با ورودی نسخه‌ی دیگه‌ای در Production تغذیه می‌شه (دقیقاً همون خطایی که
سند معماری بهش اشاره کرد: "Model trained with A,B,C / Production
receives A,B,C,D,E").
"""

import numpy as np

# هر تغییری در منطق فیچرها (اضافه/حذف/تغییر فرمول) باید این عدد رو افزایش بده.
FEATURE_VERSION = "v2"

# ترتیب دقیق فیچرها - این ترتیب باید در train و inference عیناً یکی باشه،
# چون هم scaler و هم مدل بر همین اساس train شدن.
#
# v2: اضافه شدن ۵ فیچر جدید (VWAP, VWAP_Distance_Pct, Stochastic %K/%D,
# ROC, HL_Range_Pct) - همه از خود OHLCV محاسبه می‌شن، بدون نیاز به دیتای
# خارجی (مثل funding rate که بعداً جدا اضافه می‌شه).
FEATURE_COLUMNS = [
    "close", "open", "high", "low", "volume",
    "SMA", "RSI",
    "Bollinger_Mid", "Bollinger_Upper", "Bollinger_Lower", "Bollinger_PctB",
    "ATR",
    "MACD", "MACD_Signal", "MACD_Hist",
    "Relative_Volume",
    "Hour_Sin", "Hour_Cos", "DOW_Sin", "DOW_Cos",
    "VWAP", "VWAP_Distance_Pct",
    "Stoch_K", "Stoch_D",
    "ROC",
    "HL_Range_Pct",
]
NUM_FEATURES = len(FEATURE_COLUMNS)


def wilder_smoothing(series, period):
    """میانگین‌گیری نمایی وایلدر - فرمول استاندارد RSI."""
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_technical_indicators(data, rsi_period=14, bb_period=20, bb_std=2, atr_period=14):
    """
    اضافه کردن تمام اندیکاتورهای موردنیاز مدل به دیتافریم.

    ورودی: دیتافریم با ستون‌های open, high, low, close, volume و ایندکس
    زمانی (datetime).
    خروجی: همون دیتافریم با تمام ستون‌های FEATURE_COLUMNS اضافه‌شده،
    ردیف‌های ابتدایی (که به‌خاطر rolling window ها NaN دارن) حذف‌شده.
    """
    data["SMA"] = data["close"].rolling(window=14).mean()

    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = wilder_smoothing(gain, rsi_period)
    avg_loss = wilder_smoothing(loss, rsi_period)
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].where(avg_loss != 0, 100.0)

    bb_mid = data["close"].rolling(window=bb_period).mean()
    bb_std_val = data["close"].rolling(window=bb_period).std()
    data["Bollinger_Mid"] = bb_mid
    data["Bollinger_Upper"] = bb_mid + bb_std * bb_std_val
    data["Bollinger_Lower"] = bb_mid - bb_std * bb_std_val
    band_width = (data["Bollinger_Upper"] - data["Bollinger_Lower"]).replace(0, np.nan)
    data["Bollinger_PctB"] = (data["close"] - data["Bollinger_Lower"]) / band_width

    high_low = data["high"] - data["low"]
    high_close = np.abs(data["high"] - data["close"].shift())
    low_close = np.abs(data["low"] - data["close"].shift())
    import pandas as pd
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    data["ATR"] = tr.rolling(window=atr_period).mean()

    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    vol_ma20 = data["volume"].rolling(window=20).mean()
    data["Relative_Volume"] = data["volume"] / vol_ma20.replace(0, np.nan)

    hour = data.index.hour
    dow = data.index.dayofweek
    data["Hour_Sin"] = np.sin(2 * np.pi * hour / 24)
    data["Hour_Cos"] = np.cos(2 * np.pi * hour / 24)
    data["DOW_Sin"] = np.sin(2 * np.pi * dow / 7)
    data["DOW_Cos"] = np.cos(2 * np.pi * dow / 7)

    # --- v2: فیچرهای جدید (بدون نیاز به دیتای خارجی) ---

    # VWAP غلتان (نه VWAP تجمعی از ابتدای روز، چون برای دیتای پیوسته‌ی
    # چندماهه معنی نداره) - میانگین وزنی قیمت بر اساس حجم در یک پنجره‌ی
    # ۲۰ کندلی، مشابه پنجره‌ی Bollinger برای هماهنگی مقیاس زمانی
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    vwap_window = 20
    pv = typical_price * data["volume"]
    data["VWAP"] = (
        pv.rolling(window=vwap_window).sum()
        / data["volume"].rolling(window=vwap_window).sum().replace(0, np.nan)
    )
    # فاصله‌ی نسبی قیمت از VWAP - سیگنال mean-reversion (مثبت = قیمت بالاتر از VWAP)
    data["VWAP_Distance_Pct"] = (data["close"] - data["VWAP"]) / data["VWAP"]

    # Stochastic Oscillator (%K, %D) - پنجره‌ی ۱۴ کندلی (استاندارد رایج)،
    # عمداً متفاوت از پنجره‌ی RSI (که وایلدر-محوره) تا مکمل باشه نه تکراری
    stoch_period = 14
    lowest_low = data["low"].rolling(window=stoch_period).min()
    highest_high = data["high"].rolling(window=stoch_period).max()
    stoch_range = (highest_high - lowest_low).replace(0, np.nan)
    data["Stoch_K"] = 100 * (data["close"] - lowest_low) / stoch_range
    data["Stoch_D"] = data["Stoch_K"].rolling(window=3).mean()  # میانگین‌گیری هموارساز استاندارد

    # Rate of Change - مومنتوم ساده نسبت به N کندل قبل، مکمل MACD
    roc_period = 10
    data["ROC"] = (
        (data["close"] - data["close"].shift(roc_period))
        / data["close"].shift(roc_period)
    )

    # دامنه‌ی نوسان درون‌کندلی نسبی - مکمل ATR (که میانگین‌گیری‌شده روی
    # چند کندله)؛ این فیچر لحظه‌ایه و فقط به همون یک کندل مربوطه
    data["HL_Range_Pct"] = (data["high"] - data["low"]) / data["close"]

    data.dropna(inplace=True)
    return data


def validate_feature_schema(df):
    """
    چک می‌کنه که دیتافریم دقیقاً همون ستون‌هایی رو داره که مدل انتظارشون
    رو داره - این دقیقاً همون "Feature Schema Validation" است که در سند
    معماری آمده. در crypto-signal-api باید قبل از هر inference صدا زده بشه.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature schema mismatch (FEATURE_VERSION={FEATURE_VERSION}): "
            f"missing columns {missing}. Model and data pipeline are out of sync."
        )
    return True
