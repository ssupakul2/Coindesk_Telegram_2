import os
import time
import logging
import requests
import pandas as pd
import numpy as np

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==========================================
# Environment Variables
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "FLOKI", "SHIB", "EIGEN", "OP", "DOGE", "NEAR",
    "TRX", "AVAX", "SUI"
]

# ==========================================
# Constants & Hyperparameters
# ==========================================
API_RATE_LIMIT_DELAY = 0.35
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2.0
HISTOHOUR_LIMIT = 2000
RSI_PERIOD = 14
EMA_SHORT = 50
EMA_LONG = 200
RSI_OVERSOLD = 32
RSI_OVERBOUGHT = 70

# --- RSI Recovery & Pullback Configuration ---
RSI_RECOVERY_THRESHOLD = 45
RSI_PULLBACK_THRESHOLD = 55
RSI_RECOVERY_LOOKBACK = 5

# --- Divergence Configuration ---
RSI_BULL_DIV_MAX = 45
RSI_BEAR_DIV_MIN = 55
LOOKBACK_BARS = 15
LOOKBACK_SKIP_BARS = 3

# --- Trend Continuity Configuration ---
TREND_SLOPE_BARS = 5          
TREND_MIN_CONSECUTIVE = 3     

# --- RSI Bounce Configuration ---
RSI_BOUNCE_CONFIRM_BARS = 2   
RSI_BOUNCE_MIN_RISE = 3.0     

# --- Order Block (SMC) & FVG Configuration ---
OB_LOOKBACK = 20              
OB_IMBALANCE_RATIO = 1.5      
FVG_THRESHOLD_PCT = 0.2       

# --- Take Profit & Stop Loss Tiers (Optimized for 4H Timeframe) ---
TP_TIERS = {
    "major":  {"tp1": 0.06, "tp2": 0.12, "sl_buffer": 0.025},  # SL ~2.5%
    "mid":    {"tp1": 0.12, "tp2": 0.20, "sl_buffer": 0.050},  # SL ~5.0%
    "small":  {"tp1": 0.20, "tp2": 0.35, "sl_buffer": 0.080},  # SL ~8.0%
}

COIN_TIER = {
    "BTC": "major", "ETH": "major",
    "BNB": "mid",   "SOL": "mid",   "XRP": "mid",
    "ADA": "mid",   "NEAR": "mid",  "OP": "mid",
    "TRX": "mid",   "AVAX": "mid",
    "FLOKI": "small","SHIB": "small","EIGEN": "small","DOGE": "small",
    "SUI": "small"
}


# ==========================================
# Telegram Integration
# ==========================================
def send_telegram_messages(chunks: list) -> None:
    token = str(TELEGRAM_BOT_TOKEN or "").strip()
    chat_id = str(TELEGRAM_CHAT_ID or "").strip()

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID ไม่ได้ตั้งค่าใน Environment Variables")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for idx, chunk in enumerate(chunks, start=1):
        if not chunk.strip():
            continue

        payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"Telegram ส่งสำเร็จ (ส่วน {idx}/{len(chunks)})")
            else:
                logger.warning(f"Telegram ส่งล้มเหลว (ส่วน {idx}): {resp.text}")
        except Exception as e:
            logger.error(f"Exception ขณะส่ง Telegram (ส่วน {idx}): {e}")

        if idx < len(chunks):
            time.sleep(0.5)


# ==========================================
# Data Fetching & Core Technical Analysis
# ==========================================
def get_historical_data(coin: str) -> pd.DataFrame | None:
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": HISTOHOUR_LIMIT,
        "api_key": CRYPTOCOMPARE_API_KEY,
    }

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            if data.get("Response") == "Success":
                df = pd.DataFrame(data["Data"]["Data"])
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.set_index("time", inplace=True)

                df_4h = df.resample("4h").agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volumeto": "sum",
                    }
                ).dropna()

                logger.info(f"{coin}: ดึงข้อมูล 4H สำเร็จ ({len(df_4h)} แท่ง)")
                return df_4h
            else:
                logger.warning(f"{coin} attempt {attempt}: API ตอบกลับผิดปกติ – {data.get('Message')}")

        except requests.exceptions.Timeout:
            logger.warning(f"{coin} attempt {attempt}: Request timeout")
        except Exception as e:
            logger.warning(f"{coin} attempt {attempt}: {e}")

        if attempt < API_MAX_RETRIES:
            time.sleep(API_RETRY_DELAY * attempt)

    logger.error(f"{coin}: ดึงข้อมูล 4H ล้มเหลวทั้ง {API_MAX_RETRIES} ครั้ง")
    return None


def analyze_weekly_context(coin: str) -> dict:
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": 1000,  
        "api_key": CRYPTOCOMPARE_API_KEY,
    }
    
    result = {
        "rsi_weekly": None,
        "weekly_bullish_div": False,
        "weekly_status_label": "↔️ ไม่พบข้อมูลระบุระดับสัปดาห์ชัดเจน",
        "fibo_618": None,
        "fibo_786": None
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("Response") == "Success":
            df = pd.DataFrame(data["Data"]["Data"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            df_w = df.resample("W").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last"
                }
            ).dropna()

            if len(df_w) < RSI_PERIOD + LOOKBACK_BARS + 5:
                return result

            fibo_window = df_w.iloc[-52:]
            w_max = fibo_window["high"].max()
            w_min = fibo_window["low"].min()
            w_diff = w_max - w_min
            
            result["fibo_618"] = w_max - (0.618 * w_diff)
            result["fibo_786"] = w_max - (0.786 * w_diff)

            close = df_w["close"]
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df_w["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

            curr_rsi_w = round(df_w["RSI"].iloc[-1], 2)
            result["rsi_weekly"] = curr_rsi_w

            prev_window = df_w.iloc[-(LOOKBACK_BARS + 1) : -(LOOKBACK_SKIP_BARS)]
            if len(prev_window) > 0:
                min_low_idx = prev_window["low"].argmin()
                prev_low_price = prev_window["low"].iloc[min_low_idx]
                prev_low_rsi   = prev_window["RSI"].iloc[min_low_idx]

                curr_price_w = df_w["low"].iloc[-1]
                curr_rsi_w_now = df_w["RSI"].iloc[-1]

                if (prev_low_rsi <= RSI_BULL_DIV_MAX) and (curr_price_w < prev_low_price) and (curr_rsi_w_now > prev_low_rsi):
                    result["weekly_bullish_div"] = True

            if result["weekly_bullish_div"]:
                result["weekly_status_label"] = f"👑 <b>เกิด Weekly Bullish Divergence ในระดับภาพใหญ่!</b> (RSI 1W: {curr_rsi_w})"
            elif curr_rsi_w <= RSI_OVERSOLD:
                result["weekly_status_label"] = f"🔥 <b>ภาพใหญ่เข้าเขต Oversold รุนแรง ({curr_rsi_w})</b> มีโอกาสเด้งกลับระยะยาว"
            elif curr_rsi_w <= 45:
                result["weekly_status_label"] = f"📥 ภาพใหญ่สะสมพลังอยู่ในโซนต่ำ ({curr_rsi_w})"
            elif curr_rsi_w >= RSI_OVERBOUGHT:
                result["weekly_status_label"] = f"⚠️ ภาพใหญ่เกิด Overbought ({curr_rsi_w}) ระวังความเสี่ยงการปรับฐาน"
            else:
                result["weekly_status_label"] = f"↔️ ภาพใหญ่ทรงตัวปกติ (RSI 1W: {curr_rsi_w})"

            logger.info(f"{coin}: วิเคราะห์ความเคลื่อนไหวระดับรายสัปดาห์สำเร็จ")
            return result

    except Exception as e:
        logger.error(f"{coin} (Weekly): ระบบขัดข้องระหว่างคำนวณข้อมูลระดับสัปดาห์: {e}")
        
    return result


# ==========================================
# Updated 1M Module: Including Cycle Top Prediction
# ==========================================
def analyze_monthly_targets(coin: str) -> dict:
    """ดึงข้อมูลรายวันย้อนหลังเพื่อแปลงเป็นแท่ง 1M และคำนวณเป้าหมายราคาจุดจบ Cycle (Fibonacci Extension)"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": 2000,  # ข้อมูลประมาณ 5.4 ปี ครอบคลุมอดีตวัฏจักรใหญ่ก่อนหน้า
        "api_key": CRYPTOCOMPARE_API_KEY,
    }
    
    result = {
        "m_resistance_target": None,
        "m_support_target": None,
        "monthly_summary_label": "⏳ ไม่สามารถคำนวณเป้าหมายระดับเดือนได้",
        "monthly_trend": "sideways",
        # เพิ่มข้อมูลทำนายวัฏจักร
        "cycle_target_1618": None,
        "cycle_target_2618": None,
        "cycle_target_4236": None,
        "cycle_upside_pct": 0.0
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("Response") == "Success":
            df = pd.DataFrame(data["Data"]["Data"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            df_m = df.resample("ME").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last"
                }
            ).dropna()

            if len(df_m) < 12:
                return result

            # 1. ค้นหาจุดสูงสุดประวัติศาสตร์ (All-Time High) และจุดต่ำสุดของรอบปรับฐานในอดีต (Cycle Low)
            macro_high = df_m["high"].max()
            macro_low = df_m["low"].min()
            macro_diff = macro_high - macro_low

            # 2. คำนวณราคาสุดทาง Cycle ด้วย Fibonacci Extension จากฐานวัฏจักรใหญ่
            result["cycle_target_1618"] = macro_high + (0.618 * macro_diff)   # เป้าหมายเซฟตี้
            result["cycle_target_2618"] = macro_high + (1.618 * macro_diff)   # เป้าหมายหลักขยายตัวสถาบัน
            result["cycle_target_4236"] = macro_high + (3.236 * macro_diff)   # เป้าหมายสุดท้าย (โซนฟองสบู่แตก)

            last_month = df_m.iloc[-2]
            current_month = df_m.iloc[-1]
            curr_price = current_month["close"]

            # คำนวณสัดส่วนศักยภาพการเติบโต (Upside) ไปยังเป้าหมายหลักประเมินวัฏจักร
            if curr_price < result["cycle_target_2618"]:
                upside = ((result["cycle_target_2618"] - curr_price) / curr_price) * 100
                result["cycle_upside_pct"] = round(upside, 1)

            m_high = last_month["high"]
            m_low = last_month["low"]
            m_close = last_month["close"]
            
            pivot = (m_high + m_low + m_close) / 3
            r1 = (2 * pivot) - m_low
            s1 = (2 * pivot) - m_high
            
            df_m["EMA_12"] = df_m["close"].ewm(span=12, adjust=False).mean()
            m_ema12 = df_m["EMA_12"].iloc[-1]

            if curr_price >= m_ema12:
                target_up = max(r1, m_high)
                target_down = pivot
                trend_status = "bullish"
                status_text = f"🔮 <b>ภาพ 1M (Bullish):</b> ตลาดภาพใหญ่คุมเทรนด์ขาขึ้นอย่างมั่นคง"
            else:
                target_up = pivot
                target_down = min(s1, m_low)
                trend_status = "bearish"
                status_text = f"🔮 <b>ภาพ 1M (Bearish):</b> ตลาดหลักอยู่ในช่วงสะสมพลัง/ปรับฐานใหญ่"

            result["m_resistance_target"] = target_up
            result["m_support_target"] = target_down
            result["monthly_summary_label"] = status_text
            result["monthly_trend"] = trend_status
            
            logger.info(f"{coin}: วิเคราะห์เป้าหมายมูลค่าขยายตัวและจุดจบ Cycle (1M) สำเร็จ")
            return result

    except Exception as e:
        logger.error(f"{coin} (Monthly): ระบบขัดข้องระหว่างคำนวณแนวโน้มวัฏจักร: {e}")
        
    return result


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df["EMA_50"] = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=EMA_LONG, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

    df["VOL_MA20"] = df["volumeto"].rolling(20).mean()
    return df


# ==========================================
# Advanced Analysis Modules (4H)
# ==========================================
def find_fair_value_gaps(df: pd.DataFrame) -> dict:
    fvg_result = {"has_fvg_support": False, "fvg_top": None, "fvg_bottom": None}
    if len(df) < 4:
        return fvg_result
        
    for i in range(len(df) - 1, 2, -1):
        high_minus2 = df["high"].iloc[i - 2]
        low_current = df["low"].iloc[i]
        close_minus1 = df["close"].iloc[i - 1]
        open_minus1 = df["open"].iloc[i - 1]
        
        if low_current > high_minus2 and close_minus1 > open_minus1:
            gap_pct = ((low_current - high_minus2) / high_minus2) * 100
            if gap_pct >= FVG_THRESHOLD_PCT:
                curr_price = df["close"].iloc[-1]
                if curr_price > high_minus2:
                    fvg_result["has_fvg_support"] = True
                    fvg_result["fvg_top"] = low_current
                    fvg_result["fvg_bottom"] = high_minus2
                    break
    return fvg_result


def analyze_trend_continuity(df: pd.DataFrame) -> dict:
    result = {
        "ema50_slope_pct": 0.0,
        "ema200_slope_pct": 0.0,
        "ema50_trending_up": False,
        "ema200_trending_up": False,
        "consecutive_up": 0,
        "consecutive_down": 0,
        "trend_strength": "sideways",
        "trend_label": "↔️ ไม่ชัดเจน",
    }

    n = TREND_SLOPE_BARS
    if len(df) < n + 2:
        return result

    ema50_now  = df["EMA_50"].iloc[-1]
    ema50_prev = df["EMA_50"].iloc[-(n + 1)]
    ema200_now  = df["EMA_200"].iloc[-1]
    ema200_prev = df["EMA_200"].iloc[-(n + 1)]

    slope50  = ((ema50_now  - ema50_prev)  / ema50_prev)  * 100 if ema50_prev  != 0 else 0
    slope200 = ((ema200_now - ema200_prev) / ema200_prev) * 100 if ema200_prev != 0 else 0

    result["ema50_slope_pct"]  = round(slope50,  4)
    result["ema200_slope_pct"] = round(slope200, 4)
    result["ema50_trending_up"]  = slope50  > 0
    result["ema200_trending_up"] = slope200 > 0

    closes = df["close"].iloc[-20:]
    diffs  = closes.diff().iloc[1:]

    up_streak = 0
    dn_streak = 0
    for val in reversed(diffs.values):
        if val > 0:
            if dn_streak == 0:
                up_streak += 1
            else:
                break
        elif val < 0:
            if up_streak == 0:
                dn_streak += 1
            else:
                break
        else:
            break

    result["consecutive_up"]   = up_streak
    result["consecutive_down"] = dn_streak

    both_up   = result["ema50_trending_up"] and result["ema200_trending_up"]
    both_down = (not result["ema50_trending_up"]) and (not result["ema200_trending_up"])
    strong_streak = TREND_MIN_CONSECUTIVE

    if both_up and up_streak >= strong_streak:
        strength = "strong_up"
        label = f"🚀 ขาขึ้นต่อเนื่องแข็งแกร่ง ({up_streak} แท่ง, EMA ชันขึ้นทั้งคู่)"
    elif result["ema50_trending_up"] and up_streak >= 1:
        strength = "moderate_up"
        label = f"📈 ขาขึ้นปานกลาง ({up_streak} แท่ง, EMA50 ชันขึ้น)"
    elif both_down and dn_streak >= strong_streak:
        strength = "strong_down"
        label = f"🔻 ขาลงต่อเนื่องแข็งแกร่ง ({dn_streak} แท่ง, EMA ชันลงทั้งคู่)"
    elif (not result["ema50_trending_up"]) and dn_streak >= 1:
        strength = "moderate_down"
        label = f"📉 ขาลงปานกลาง ({dn_streak} แท่ง, EMA50 ชันลง)"
    else:
        strength = "sideways"
        label = "↔️ Sideways / แนวโน้มไม่ชัด"

    result["trend_strength"] = strength
    result["trend_label"]    = label
    return result


def analyze_rsi_bounce(df: pd.DataFrame) -> dict:
    window = LOOKBACK_BARS
    result = {
        "touched_oversold": False,
        "rsi_low": None,
        "rsi_rise": 0.0,
        "consecutive_rise": 0,
        "below_midline": False,
        "quality": "none",
        "quality_label": "⬜ ไม่มีสัญญาณดีดกลับ",
        "entry_timing": "",
    }

    if len(df) < window + RSI_BOUNCE_CONFIRM_BARS + 2:
        return result

    rsi_series = df["RSI"].iloc[-(window + 1):-1]
    rsi_curr   = df["RSI"].iloc[-1]

    rsi_min = rsi_series.min()
    touched_oversold = rsi_min <= RSI_OVERSOLD

    result["touched_oversold"] = touched_oversold
    result["rsi_low"] = round(rsi_min, 2)

    if not touched_oversold:
        return result

    rsi_rise = rsi_curr - rsi_min
    result["rsi_rise"] = round(rsi_rise, 2)

    recent_rsi = df["RSI"].iloc[-(RSI_BOUNCE_CONFIRM_BARS + 3):]
    rsi_diffs  = recent_rsi.diff().iloc[1:]
    consec = 0
    for val in reversed(rsi_diffs.values):
        if val > 0:
            consec += 1
        else:
            break
    result["consecutive_rise"] = consec

    below_midline = rsi_curr < 50
    result["below_midline"] = below_midline

    recent_recovery_zone = df["RSI"].iloc[-RSI_RECOVERY_LOOKBACK:]
    has_recovered = (recent_recovery_zone >= RSI_RECOVERY_THRESHOLD).any()

    score = 0
    if rsi_rise >= RSI_BOUNCE_MIN_RISE:     score += 1
    if consec >= RSI_BOUNCE_CONFIRM_BARS:   score += 1
    if below_midline or has_recovered:       score += 1

    if score == 3:
        quality = "strong"
        label   = f"✅ ดีดกลับแข็งแกร่ง (จากต่ำสุด {result['rsi_low']:.1f} → ขึ้น {rsi_rise:.1f} จุด)"
        timing  = "⭐ จังหวะเข้าซื้อดีที่สุด: RSI ดีดกลับผ่านเกณฑ์ฟื้นตัวแล้ว"
    elif score == 2:
        quality = "moderate"
        label   = f"🟡 ดีดกลับปานกลาง (จากต่ำสุด {result['rsi_low']:.1f} → ขึ้น {rsi_rise:.1f} จุด)"
        timing  = "⚡ พิจารณาเข้าซื้อได้ แต่ควรรอยืนยันแท่งถัดไป"
    else:
        quality = "none"
        label   = f"⬜ RSI แตะ Oversold แต่ยังไม่เกิดแรงดีดฟื้นตัวชัดเจน"
        timing  = "🚫 ยังไม่ควรเข้า: รอสัญญาณฟื้นตัวหน้างาน"

    result["quality"]       = quality
    result["quality_label"] = label
    result["entry_timing"]  = timing
    return result


def find_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK) -> dict:
    ob_result = {
        "bullish_ob_price": None,
        "bearish_ob_price": None,
        "has_bullish_ob": False,
        "has_bearish_ob": False
    }
    
    if len(df) < lookback + 5:
        return ob_result

    body_sizes = (df["close"] - df["open"]).abs()
    avg_body = body_sizes.rolling(20).mean().iloc[-1]

    curr_close = df["close"].iloc[-1]
    curr_open = df["open"].iloc[-1]
    curr_body = abs(curr_close - curr_open)

    past_df = df.iloc[-(lookback + 1):-(LOOKBACK_SKIP_BARS)]
    recent_high = past_df["high"].max()
    recent_low = past_df["low"].min()

    if curr_close > recent_high and curr_body > (avg_body * OB_IMBALANCE_RATIO):
        for i in range(2, min(15, len(df))):
            idx = -i
            p_open = df["open"].iloc[idx]
            p_close = df["close"].iloc[idx]
            p_low = df["low"].iloc[idx]
            
            if p_close < p_open: 
                subsequent_lows = df["low"].iloc[idx+1:]
                if not (subsequent_lows < p_low).any(): 
                    ob_result["has_bullish_ob"] = True
                    ob_result["bullish_ob_price"] = p_low
                    break

    elif curr_close < recent_low and curr_body > (avg_body * OB_IMBALANCE_RATIO):
        for i in range(2, min(15, len(df))):
            idx = -i
            p_open = df["open"].iloc[idx]
            p_close = df["close"].iloc[idx]
            p_high = df["high"].iloc[idx]
            
            if p_close > p_open: 
                subsequent_highs = df["high"].iloc[idx+1:]
                if not (subsequent_highs > p_high).any():
                    ob_result["has_bearish_ob"] = True
                    ob_result["bearish_ob_price"] = p_high
                    break

    return ob_result


def check_bullish_divergence(df: pd.DataFrame) -> bool:
    if len(df) < LOOKBACK_BARS + 2:
        return False

    prev_window = df.iloc[-(LOOKBACK_BARS + 1) : -(LOOKBACK_SKIP_BARS)]
    if len(prev_window) == 0:
        return False
        
    min_low_idx = prev_window["low"].argmin()
    prev_low_price = prev_window["low"].iloc[min_low_idx]
    prev_low_rsi   = prev_window["RSI"].iloc[min_low_idx]

    if prev_low_rsi > RSI_BULL_DIV_MAX:
        return False

    curr_price = df["low"].iloc[-1]
    curr_rsi   = df["RSI"].iloc[-1]
    return (curr_price < prev_low_price) and (curr_rsi > prev_low_rsi)


def is_volume_confirmed(row: pd.Series) -> bool:
    if pd.isna(row.get("VOL_MA20")) or row["VOL_MA20"] == 0:
        return False
    return row["volumeto"] > row["VOL_MA20"]


def format_price(price: float) -> str:
    if price is None:
        return "N/A"
    if price < 0.0001:
        return f"{price:.8f}"
    elif price < 0.001:
        return f"{price:.6f}"
    elif price < 1:
        return f"{price:.4f}"
    else:
        return f"{price:.2f}"


# ==========================================
# Market Scanner
# ==========================================
def scan_market():
    buy_signals   = []
    sell_signals  = []
    bullish_coins = 0
    bearish_coins = 0
    total_valid_coins  = 0
    coin_trends_summary = []

    for coin in COINS:
        df = get_historical_data(coin)
        time.sleep(API_RATE_LIMIT_DELAY)

        if df is None or len(df) < EMA_LONG + 10:
            continue

        weekly_ctx = analyze_weekly_context(coin)
        time.sleep(API_RATE_LIMIT_DELAY)

        monthly_ctx = analyze_monthly_targets(coin)
        time.sleep(API_RATE_LIMIT_DELAY)

        df = calculate_indicators(df)
        row = df.iloc[-1]

        current_price = row["close"]
        rsi           = row["RSI"]
        ema_50        = row["EMA_50"]
        ema_200       = row["EMA_200"]
        vol_confirmed = is_volume_confirmed(row)

        total_valid_coins += 1
        is_divergence = check_bullish_divergence(df)
        rsi_rounded   = round(rsi, 2)

        trend_info = analyze_trend_continuity(df)
        bounce_info = analyze_rsi_bounce(df)
        ob_info = find_order_blocks(df)
        fvg_info = find_fair_value_gaps(df)

        fibo_4h_max = df["high"].iloc[-60:].max()
        fibo_4h_min = df["low"].iloc[-60:].min()
        fibo_4h_618 = fibo_4h_max - (0.618 * (fibo_4h_max - fibo_4h_min))

        tier    = COIN_TIER.get(coin, "mid")
        tp1_pct = TP_TIERS[tier]["tp1"]
        tp2_pct = TP_TIERS[tier]["tp2"]
        sl_buf  = TP_TIERS[tier]["sl_buffer"]
        vol_tag = " 🔊" if vol_confirmed else ""

        in_fibo_zone = (weekly_ctx["fibo_618"] is not None) and (current_price <= weekly_ctx["fibo_618"] * 1.02)
        in_4h_fibo_zone = current_price <= (fibo_4h_618 * 1.01)
        in_ob_zone = ob_info["has_bullish_ob"] and (current_price <= ob_info["bullish_ob_price"] * 1.03)
        in_fvg_zone = fvg_info["has_fvg_support"] and (current_price <= fvg_info["fvg_top"]) and (current_price >= fvg_info["fvg_bottom"] * 0.99)

        signal_type = ""

        if current_price > ema_200:
            coin_trend = "🟢 ขาขึ้น (Above EMA 200)"
            bullish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🟢 ขาขึ้น (RSI: {rsi_rounded}) | Upside เหลือ {monthly_ctx['cycle_upside_pct']}%"
            )

            if in_fibo_zone or in_4h_fibo_zone or in_ob_zone or in_fvg_zone:
                if current_price > (ema_50 * 0.98) and (rsi <= RSI_OVERSOLD or rsi <= RSI_PULLBACK_THRESHOLD):
                    if bounce_info["quality"] in ["strong", "moderate"]:
                        signal_type = f"Institution Dip & Rebound 📉{vol_tag}"
                    elif rsi <= RSI_OVERSOLD:
                        signal_type = f"Golden Fib / OB Zone Oversold 📉{vol_tag}"
                        
                if is_divergence and not signal_type:
                    signal_type = f"Confluence Bullish Divergence 📈{vol_tag}"
                    
                if ob_info["has_bullish_ob"] and not signal_type:
                    signal_type = f"Smart Money OB Reversal 🚀{vol_tag}"
        else:
            coin_trend = "🔴 ขาลง (Below EMA 200)"
            bearish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🔴 ขาลง (RSI: {rsi_rounded}) | แถวฐานวัฏจักรลึก"
            )

            if in_fibo_zone or in_ob_zone:
                if rsi <= RSI_OVERSOLD:
                    signal_type = f"Deep Retracement Buy 📉{vol_tag}"
                elif is_divergence:
                    signal_type = f"Macro Support Divergence 📈{vol_tag}"

        if signal_type:
            if weekly_ctx["weekly_bullish_div"]:
                signal_type = f"⭐ {signal_type} + [1W Bullish Div]"
            elif monthly_ctx["monthly_trend"] == "bullish" and in_fvg_zone:
                signal_type = f"🔥 {signal_type} + [1M Trend + FVG Filled]"

            entry_min      = format_price(current_price * 0.98)
            entry_max      = format_price(current_price * 1.01)
            target_tp1     = current_price * (1 + tp1_pct)
            target_tp2     = current_price * (1 + tp2_pct)
            
            sl_reference = ema_200
            if ob_info["has_bullish_ob"]: sl_reference = ob_info["bullish_ob_price"]
            elif fvg_info["has_fvg_support"]: sl_reference = fvg_info["fvg_bottom"]
            
            sl_val         = sl_reference * (1 - sl_buf) if current_price > sl_reference else current_price * (1 - sl_buf)
            stop_loss      = format_price(sl_val)

            buy_signals.append(
                {
                    "coin":          coin,
                    "trend":         coin_trend,
                    "price":         format_price(current_price),
                    "rsi":           rsi_rounded,
                    "type":          signal_type,
                    "entry":         f"${entry_min} - ${entry_max}",
                    "tp1":           f"${format_price(target_tp1)} (+{tp1_pct*100:.0f}%)",
                    "tp2":           f"${format_price(target_tp2)} (+{tp2_pct*100:.0f}%)",
                    "sl":            f"${stop_loss}",
                    "vol_confirmed": vol_confirmed,
                    "trend_info":    trend_info,
                    "bounce_info":   bounce_info,
                    "ob_info":       ob_info,
                    "fvg_info":      fvg_info,
                    "weekly_ctx":    weekly_ctx,
                    "monthly_ctx":   monthly_ctx,
                }
            )

        if rsi >= RSI_OVERBOUGHT or ob_info["has_bearish_ob"]:
            tp_min      = format_price(current_price * 1.00)
            tp_max      = format_price(current_price * (1 + tp1_pct * 0.4))
            exit_val    = ema_50 if current_price > ema_50 else current_price * (1 - sl_buf)
            safety_exit = format_price(exit_val)

            sell_signals.append(
                {
                    "coin":          coin,
                    "trend":         coin_trend,
                    "price":         format_price(current_price),
                    "rsi":           rsi_rounded,
                    "tp_zone":       f"${tp_min} - ${tp_max}",
                    "exit":          f"${safety_exit}",
                    "vol_confirmed": vol_confirmed,
                    "trend_info":    trend_info,
                    "ob_info":       ob_info,
                    "weekly_ctx":    weekly_ctx,
                    "monthly_ctx":   monthly_ctx,
                }
            )

    if total_valid_coins > 0:
        bullish_ratio = (bullish_coins / total_valid_coins) * 100
        summary_msg = f"📊 <b>[Market Trend Summary]</b>\n"
        summary_msg += f"📈 ขาขึ้น: {bullish_coins} เหรียญ | 📉 ขาลง: {bearish_coins} เหรียญ\n"
        summary_msg += f"📋 <b>สรุปแนวโน้มและระยะคำนวณ Upside สู่เป้าหมายวัฏจักร:</b>\n"
        summary_msg += "\n".join(coin_trends_summary)
    else:
        summary_msg = "⚠️ ไม่สามารถวิเคราะห์ภาพรวมตลาดได้"

    return buy_signals, sell_signals, summary_msg


# ==========================================
# Message Builder with Cycle Terminus Report
# ==========================================
def build_messages(buy_list: list, sell_list: list, market_summary: str) -> list:
    message_blocks = []
    message_blocks.append(market_summary)

    if buy_list:
        buy_header = "🎯 <b>[Crypto Screener 4H - สัญญาณซื้อ + พยากรณ์กรอบสุดรอบวัฏจักร]</b>"
        current_block = buy_header

        for opt in buy_list:
            vol_note = "🔊 Volume: ยืนยันสัญญาณ" if opt["vol_confirmed"] else "🔇 Volume: ไม่ยืนยัน"
            ti = opt["trend_info"]
            bi = opt["bounce_info"]
            ob = opt["ob_info"]
            fvg = opt["fvg_info"]
            w_ctx = opt.get("weekly_ctx", {})
            m_ctx = opt.get("monthly_ctx", {})

            confluence_report = "\n🛡️ <b>การทดสอบแนวรับสถาบัน (4H):</b>"
            if fvg.get("has_fvg_support"):
                confluence_report += f"\n   ⚡พบช่องว่าง FVG: <code>${format_price(fvg['fvg_bottom'])} - ${format_price(fvg['fvg_top'])}</code>"
            if ob.get("has_bullish_ob"):
                confluence_report += f"\n   🐳 Smart Money OB: <code>${format_price(ob['bullish_ob_price'])}</code>"

            # บล็อกรายงานจุดสิ้นสุดของ Cycle (Macro Cycle Terminus)
            cycle_report = ""
            if m_ctx and m_ctx.get("cycle_target_2618"):
                tier = COIN_TIER.get(opt["coin"], "mid")
                main_target = m_ctx["cycle_target_2618"] if tier != "small" else m_ctx["cycle_target_4236"]
                
                cycle_report = (
                    f"\n🛸 <b>พยากรณ์เป้าหมายจุดสิ้นสุดรอบ (Cycle Terminus):</b>"
                    f"\n   🎯 เป้าหมายแรกสุด (Fibo 1.618): <code>${format_price(m_ctx['cycle_target_1618'])}</code>"
                    f"\n   🚀 เป้าหมายหลักสถาบัน (Fibo 2.618): <code>${format_price(m_ctx['cycle_target_2618'])}</code>"
                    f"\n   🔥เป้าหมายเก็งกำไรคลั่ง (Fibo 4.236): <code>${format_price(m_ctx['cycle_target_4236'])}</code>"
                    f"\n   📈 <b>โอกาสขยายตัว (Upside เหลือ):</b> <code>+{m_ctx['cycle_upside_pct']}%</code> ไปยังเป้าหลักรอบนี้"
                )

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b> ({opt['trend']})"
                f"\n🚨 รูปแบบ: <b>{opt['type']}</b>"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']} | {vol_note}"
                f"{confluence_report}"
                f"{cycle_report}"
                f"\n🟢 ช่วงเข้าซื้อพิจารณา: <code>{opt['entry']}</code>"
                f"\n💰 เป้า TP1 / TP2: <code>{opt['tp1']}</code> / <code>{opt['tp2']}</code>"
                f"\n❌ จุดตัดขาดทุนหลุดต้องยอม (SL): <code>{opt['sl']}</code>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = buy_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    if sell_list:
        sell_header = "⚠️ <b>[Crypto Screener 4H - เตือนโซนทำกำไรระยะสั้น]</b>"
        current_block = sell_header

        for opt in sell_list:
            m_ctx = opt.get("monthly_ctx", {})
            cycle_note = ""
            if m_ctx and m_ctx.get("cycle_target_2618"):
                cycle_note = f"\n🚀 เป้าหมายปลายทางรอบใหญ่ (Fibo 2.618): <code>${format_price(m_ctx['cycle_target_2618'])}</code>"

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b>"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']} | RSI (4H): {opt['rsi']}"
                f"{cycle_note}"
                f"\n🔴 ช่วงราคาทยอยแบ่งขาย: <code>{opt['tp_zone']}</code>"
                f"\n❌ จุดล็อกกำไรหลุดต้องหนี (Safety Exit): <code>{opt['exit']}</code>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = sell_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    return message_blocks


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    logger.info("เริ่มต้นใช้งาน Crypto Screener มหาวัฏจักร v8 (Fibonacci Extension Cycle Targets)...")

    buy_list, sell_list, market_summary = scan_market()
    final_messages = build_messages(buy_list, sell_list, market_summary)
    send_telegram_messages(final_messages)

    logger.info("บอททำงานและแจ้งเตือนข้อมูลเป้าหมายวัฏจักรผ่าน Telegram สำเร็จ!")
