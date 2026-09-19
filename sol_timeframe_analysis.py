"""
sol_timeframe_analysis.py
==========================

هدف: دانلود ۶ ماه دیتای SOLUSDT از Binance و محاسبه‌ی آمار نوسان/ریسک/
سود در تایم‌فریم‌های مختلف.

خروجی‌ها:
    - sol_6month_5min.csv        : دیتای خام ۵ دقیقه‌ای
    - sol_timeframe_analysis.csv : جدول مقایسه‌ی تایم‌فریم‌ها
"""

import requests
import pandas as pd
import numpy as np
import time
import os


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "SOLUSDT"
MONTHS_BACK = 6
BASE_TIMEFRAME = "5m"

# مسیرهای ذخیره‌سازی - کنار خود اسکریپت ذخیره می‌شن
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "sol_6month_5min.csv")
ANALYSIS_CSV = os.path.join(BASE_DIR, "sol_timeframe_analysis.csv")

# فرضیات هزینه‌ی معامله
FEE_RATE_ONE_WAY = 0.00045
SLIPPAGE_ESTIMATE = 0.0002
ROUND_TRIP_COST = 2 * (FEE_RATE_ONE_WAY + SLIPPAGE_ESTIMATE)

TIMEFRAMES_TO_ANALYZE = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
}


# ============================================================
# DOWNLOAD
# ============================================================

def download_binance_klines(symbol, interval, start_time_ms, end_time_ms):
    """
    دانلود کامل دیتای کندل از Binance با pagination.
    فقط درصد پیشرفت را نمایش می‌دهد، نه هر رکورد را.
    """
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    current_start = start_time_ms
    total_span = end_time_ms - start_time_ms

    while current_start < end_time_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        last_close_time = data[-1][6]
        current_start = last_close_time + 1

        # فقط یک خط progress که خودش را بازنویسی می‌کند
        progress = min((current_start - start_time_ms) / total_span * 100, 100)
        print(f"\r  پیشرفت دانلود: {progress:5.1f}%  |  {len(all_rows):,} کندل", end="", flush=True)

        time.sleep(0.3)

        if len(data) < 1000:
            break

    print()  # خط آخر برای تمیز موندن خروجی
    return all_rows


def fetch_and_save_sol_data():
    print(f"شروع دانلود {SYMBOL} - تایم‌فریم پایه: {BASE_TIMEFRAME}, بازه: {MONTHS_BACK} ماه اخیر\n")

    end_time = pd.Timestamp.now(tz="UTC")
    start_time = end_time - pd.DateOffset(months=MONTHS_BACK)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    rows = download_binance_klines(SYMBOL, BASE_TIMEFRAME, start_ms, end_ms)

    if not rows:
        raise RuntimeError("هیچ دیتایی دانلود نشد - نماد یا بازه رو چک کنید.")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")

    # ذخیره‌ی دیتای خام در CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ دیتای خام ذخیره شد: {OUTPUT_CSV}")
    print(f"  تعداد ردیف: {len(df):,}")
    print(f"  بازه‌ی زمانی: {df['open_time'].min()} -> {df['open_time'].max()}")

    return df


# ============================================================
# RESAMPLE
# ============================================================

def resample_ohlcv(df_5min, timeframe_minutes):
    df = df_5min.set_index("open_time")
    rule = f"{timeframe_minutes}min"
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


# ============================================================
# ANALYSIS
# ============================================================

def analyze_timeframe(df, timeframe_label, round_trip_cost):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    log_returns = np.log(close / close.shift(1)).dropna()
    intracandle_range_pct = ((high - low) / close).dropna()
    atr_pct = intracandle_range_pct.rolling(14).mean().dropna()

    mean_abs_return = log_returns.abs().mean()
    std_return = log_returns.std()
    movement_to_cost_ratio = mean_abs_return / round_trip_cost
    pct_candles_exceeding_cost = (log_returns.abs() > round_trip_cost).mean() * 100

    return {
        "تایم‌فریم": timeframe_label,
        "تعداد_کندل": len(log_returns),
        "میانگین_قدرمطلق_بازده_%": mean_abs_return * 100,
        "انحراف_معیار_بازده_%": std_return * 100,
        "میانگین_ATR_%": atr_pct.mean() * 100 if len(atr_pct) > 0 else np.nan,
        "نسبت_حرکت_به_هزینه": movement_to_cost_ratio,
        "درصد_کندل‌های_فراتر_از_هزینه": pct_candles_exceeding_cost,
        "بیشینه_افت_%": log_returns.min() * 100,
        "بیشینه_رشد_%": log_returns.max() * 100,
    }


def run_full_analysis(df_5min, round_trip_cost=ROUND_TRIP_COST):
    print(f"\nهزینه‌ی تخمینی رفت‌وبرگشت هر معامله: {round_trip_cost * 100:.3f}%")
    print("(بر پایه‌ی fee={:.3f}% + slippage={:.3f}%, دو طرفه)\n".format(
        FEE_RATE_ONE_WAY * 100, SLIPPAGE_ESTIMATE * 100
    ))

    results = []
    for label, minutes in TIMEFRAMES_TO_ANALYZE.items():
        print(f"در حال محاسبه برای تایم‌فریم {label}...")
        resampled = resample_ohlcv(df_5min, minutes)
        stats = analyze_timeframe(resampled, label, round_trip_cost)
        results.append(stats)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("نتیجه‌ی نهایی - مقایسه‌ی تایم‌فریم‌ها")
    print("=" * 100)
    print(results_df.to_string(index=False))

    print("\n--- تفسیر ---")
    print("نسبت_حرکت_به_هزینه: هرچی بزرگ‌تر از ۱ باشه بهتره.")
    print("درصد_کندل‌های_فراتر_از_هزینه: هرچی بیشتر، فرصت معاملاتی بیشتره.")

    results_df.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ نتایج تحلیل ذخیره شد: {ANALYSIS_CSV}")

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"دیتای از قبل دانلودشده پیدا شد: {OUTPUT_CSV}")
        print("برای دانلود مجدد، این فایل رو حذف کنید.\n")
        df_5min = pd.read_csv(OUTPUT_CSV, parse_dates=["open_time"])
    else:
        df_5min = fetch_and_save_sol_data()

    run_full_analysis(df_5min)