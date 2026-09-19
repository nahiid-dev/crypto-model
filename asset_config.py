"""
asset_config.py
================

تنظیمات مخصوص هر دارایی معاملاتی (symbol) - جدا از منطق مدل، API، و
تصمیم‌گیری معاملاتی.

چرا این فایل جدا شد:
قبل از این، threshold سیگنال، ضرایب ATR برای SL/TP، کارمزد، و لوریج در
سه فایل جدا (model_gru_only.py, app_updated.py, trade_decision.py) به
شکل تکراری تعریف شده بودن. وقتی دارایی معاملاتی عوض می‌شه (مثلاً از ETH
به SOL که نوسان پایه‌ش بیشتره)، باید هر سه فایل رو دستی و هماهنگ عوض
می‌کردیم - که خیلی مستعد خطا و ناهماهنگی بود (دقیقاً مثل مشکلی که با
features.py حل کردیم).

نحوه‌ی استفاده در سه پروژه:
    from asset_config import ACTIVE_ASSET
    threshold = ACTIVE_ASSET.signal_threshold

این فایل باید عیناً (بدون تفاوت) در هر سه پروژه (crypto-model,
crypto-predict-api) وجود داشته باشه - دقیقاً مثل features.py.

برای سوییچ دارایی: فقط خط ACTIVE_ASSET پایین فایل رو عوض کنید.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetConfig:
    # --- هویت دارایی و تایم‌فریم ---
    symbol: str
    timeframe_minutes: int   # تایم‌فریم کندلی که مدل train می‌شه (مثلاً 5 دقیقه)
    horizon_minutes: int     # افق پیش‌بینی (مثلاً 15 دقیقه) - باید مضرب timeframe_minutes باشه

    # --- هزینه‌ی معامله (باید با کارمزد واقعی صرافی/DEX هماهنگ باشه) ---
    fee_rate: float          # کارمزد یک‌طرفه (تیکر)
    slippage_rate: float     # تخمین اسلیپیج یک‌طرفه

    # --- سیگنال ---
    # حداقل بازده‌ی پیش‌بینی‌شده (log-return) برای این‌که سیگنال معتبر
    # در نظر گرفته بشه. باید متناسب با نوسان پایه‌ی دارایی باشه - دارایی
    # پرنوسان‌تر (مثل SOL نسبت به ETH) باید threshold بالاتری داشته باشه
    # وگرنه نویز عادی بازار به‌اشتباه سیگنال تشخیص داده می‌شه.
    signal_threshold: float

    # --- فاصله‌ی SL/TP بر پایه‌ی ATR (نسبت به نوسان دارایی کالیبره می‌شه) ---
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    min_stop_loss_pct: float     # کف امنیتی - جلوگیری از SL بیش‌ازحد تنگ
    max_stop_loss_pct: float     # سقف امنیتی - جلوگیری از SL بیش‌ازحد گشاد

    # --- فیلتر قدرت سیگنال (چقدر باید از نویز عادی این دارایی فاصله داشته باشه) ---
    min_signal_atr_multiple: float

    # --- داده‌ی train ---
    min_data_months_back: int    # چند ماه اخیر برای train استفاده بشه

    @property
    def horizon_steps(self) -> int:
        assert self.horizon_minutes % self.timeframe_minutes == 0, (
            f"horizon_minutes ({self.horizon_minutes}) باید مضرب صحیح "
            f"timeframe_minutes ({self.timeframe_minutes}) باشه."
        )
        return self.horizon_minutes // self.timeframe_minutes


# نکته‌ی مهم درباره‌ی چیزهایی که عمداً اینجا نیستن:
# risk_per_trade_pct، max_leverage، initial_capital، min_risk_reward
# این‌ها به نوسان دارایی ربطی ندارن - انتخاب استراتژی/ریسک‌پذیری تریدرن
# (چه ETH چه SOL، ممکنه بخواید همیشه همون درصد ریسک رو داشته باشید).
# این‌ها همچنان مستقل در هر فایل (model_gru_only.py برای بک‌تست،
# trade_decision.py برای تصمیم زنده) تعریف می‌مونن - عمداً می‌تونن باهم
# فرق کنن چون بک‌تست و معامله‌ی زنده می‌تونن فلسفه‌ی ریسک متفاوتی داشته
# باشن.


# ============================================================
# پیکربندی هر دارایی
# ============================================================

ETHUSDT = AssetConfig(
    symbol="ETHUSDT",
    timeframe_minutes=5,
    horizon_minutes=15,
    fee_rate=0.0004,
    slippage_rate=0.0002,
    signal_threshold=0.0010,
    atr_multiplier_sl=1.5,
    atr_multiplier_tp=2.25,
    min_stop_loss_pct=0.0015,
    max_stop_loss_pct=0.0100,
    min_signal_atr_multiple=0.5,
    min_data_months_back=6,
)

SOLUSDT = AssetConfig(
    symbol="SOLUSDT",
    timeframe_minutes=5,
    # نکته: بعد از اجرای sol_timeframe_analysis.py با دیتای واقعی، اگه
    # نسبت_حرکت_به_هزینه در تایم‌فریم‌های کوتاه‌تر (مثلاً 5m) به‌اندازه‌ی
    # کافی بالا بود، می‌شه horizon_minutes رو به 5 کاهش داد (یعنی 5m->5m
    # به‌جای 5m->15m) - این عدد فعلاً یک نقطه‌ی شروع منطقیه، نه قطعی.
    horizon_minutes=15,
    fee_rate=0.00045,   # Hyperliquid taker fee (طبق بررسی قبلی)
    slippage_rate=0.0002,
    # بالاتر از ETH چون نوسان پایه‌ی SOL بیشتره (طبق داده‌های بررسی‌شده:
    # HV سالانه‌ی SOL در محدوده‌ی ~68% در برابر ~45% برای ETH) - با
    # threshold یکسان با ETH، مدل مدام نویز عادی SOL رو به‌اشتباه سیگنال
    # تشخیص می‌داد.
    signal_threshold=0.0015,
    atr_multiplier_sl=1.5,
    atr_multiplier_tp=2.25,
    # بالاتر از ETH - همون دلیل نوسان بیشتر
    min_stop_loss_pct=0.0025,
    max_stop_loss_pct=0.0150,
    min_signal_atr_multiple=0.5,
    min_data_months_back=6,
)


# ============================================================
# انتخاب دارایی فعال
# ============================================================
# فقط همین خط رو عوض کنید تا کل pipeline (train + API + تصمیم معاملاتی)
# به دارایی دیگه سوییچ کنه. بعد از عوض کردنش، لازمه:
#   ۱) crypto-model: دوباره train کنید (چون scaler/مدل برای دارایی قبلی
#      کالیبره شده بودن)
#   ۲) crypto-predict-api: فایل‌های مدل/اسکیلر جدید رو در models/ جایگزین
#      کنید
# هیچ تغییر دیگه‌ای در کد سه پروژه لازم نیست.

ACTIVE_ASSET: AssetConfig = SOLUSDT