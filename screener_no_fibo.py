Import os
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

# Watchlist ของคุณ
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

# --- Order Block (SMC) Configuration ---
OB_LOOKBACK = 20              
OB_IMBALANCE_RATIO = 1.5      

# --- Take Profit Tiers ---
TP_TIERS = {
    "major":  {"tp1": 0.08, "tp2": 0.12, "sl_buffer": 0.02},
    "mid":    {"tp1": 0.12, "tp2": 0.15, "sl_buffer": 0.025},
    "small":  {"tp1": 0.18, "tp2": 0.25, "sl_buffer": 0.03},
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
    """ดึงข้อมูลรายวันย้อนหลังเพื่อแปลงเป็น 1W และหา RSI + Weekly Bullish Divergence"""
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
        "weekly_status_label": "↔️ ไม่พบข้อมูลระบุระดับสัปดาห์ชัดเจน"
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
# New Features: Monthly (1M) Analysis Module
# ==========================================
def analyze_monthly_targets(coin: str) -> dict:
    """ดึงข้อมูลรายวันย้อนหลังมาแปลงเป็นแท่ง 1M เพื่อหาแนวรับ-แนวต้านหลัก และคำนวณเป้าหมายราคาตามโมเมนตัม"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": coin,
        "tsym": "USD",
        "limit": 2000,  # ดึงข้อมูลยาวประมาณ 5.4 ปี เพื่อแปลงเป็นแท่งเดือนที่มีเสถียรภาพในการคำนวณ
        "api_key": CRYPTOCOMPARE_API_KEY,
    }
    
    result = {
        "m_resistance_target": None,
        "m_support_target": None,
        "monthly_summary_label": "⏳ ไม่สามารถคำนวณเป้าหมายระดับเดือนได้",
        "monthly_trend": "sideways"
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("Response") == "Success":
            df = pd.DataFrame(data["Data"]["Data"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # แปลงข้อมูลรายวัน (1D) เป็นระดับรายเดือน (1M)
            df_m = df.resample("ME").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last"
                }
            ).dropna()

            if len(df_m) < 5:
                return result

            # ใช้แท่งเดือนก่อนหน้าที่ปิดสมบูรณ์แล้ว (แท่งรองสุดท้าย) ในการหา Key Levels
            last_month = df_m.iloc[-2]
            current_month = df_m.iloc[-1]
            
            m_high = last_month["high"]
            m_low = last_month["low"]
            m_close = last_month["close"]
            
            # คำนวณกรอบมูลค่าด้วย Classic Pivot Points ระดับเดือน
            pivot = (m_high + m_low + m_close) / 3
            r1 = (2 * pivot) - m_low
            s1 = (2 * pivot) - m_high
            
            # คำนวณเส้นค่าเฉลี่ย 12 แท่งเดือน (EMA 12 เดือน เสมือนตัวแทนแนวโน้มหลักรอบ 1 ปี)
            if len(df_m) >= 12:
                df_m["EMA_12"] = df_m["close"].ewm(span=12, adjust=False).mean()
                m_ema12 = df_m["EMA_12"].iloc[-1]
            else:
                m_ema12 = pivot

            curr_price = current_month["close"]
            
            # ประเมินแนวโน้มและกรอบเป้าหมายราคาตามโมเมนตัมระดับเดือน
            if curr_price >= m_ema12:
                target_up = max(r1, m_high)
                target_down = pivot
                trend_status = "bullish"
                status_text = "🔮 <b>ภาพ 1M (Bullish):</b> ทิศทางหลักในภาพใหญ่เป็นขาขึ้น มีเป้าหมายราคาวิ่งทดสอบกรอบบน"
            else:
                target_up = pivot
                target_down = min(s1, m_low)
                trend_status = "bearish"
                status_text = "🔮 <b>ภาพ 1M (Bearish):</b> ทิศทางหลักในภาพใหญ่เป็นขาลง/พักฐาน มีแนวโน้มไหลลงหาแนวรับกรอบล่าง"

            result["m_resistance_target"] = target_up
            result["m_support_target"] = target_down
            result["monthly_summary_label"] = status_text
            result["monthly_trend"] = trend_status
            
            logger.info(f"{coin}: วิเคราะห์เป้าหมายมูลค่าระดับเดือน (1M) สำเร็จ")
            return result

    except Exception as e:
        logger.error(f"{coin} (Monthly): ระบบขัดข้องระหว่างคำนวณกรอบเป้าหมายราคา: {e}")
        
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
        label   = (
            f"✅ ดีดกลับแข็งแกร่ง (จากต่ำสุด {result['rsi_low']:.1f} → ขึ้น {rsi_rise:.1f} จุด, "
            f"{consec} แท่งติด, ยืนยันโซนฟื้นตัว)"
        )
        timing  = "⭐ จังหวะเข้าซื้อดีที่สุด: RSI ดีดกลับจาก Oversold อย่างมีคุณภาพและผ่านเกณฑ์ฟื้นตัว"
    elif score == 2:
        quality = "moderate"
        label   = (
            f"🟡 ดีดกลับปานกลาง (จากต่ำสุด {result['rsi_low']:.1f} → ขึ้น {rsi_rise:.1f} จุด, "
            f"{consec} แท่งติด)"
        )
        timing  = "⚡ พิจารณาเข้าซื้อได้ แต่ควรรอยืนยันแท่งเพิ่มเติม"
    elif score == 1:
        quality = "weak"
        label   = f"🟠 ดีดกลับอ่อน (ขึ้นเพียง {rsi_rise:.1f} จุด, {consec} แท่งติด)"
        timing  = "⚠️ ยังไม่แนะนำ: สัญญาณดีดกลับยังไม่ชัดเจนพอ"
    else:
        quality = "none"
        label   = f"⬜ RSI แตะ Oversold แต่ยังไม่ดีดกลับ (ต่ำสุด {result['rsi_low']:.1f})"
        timing  = "🚫 ยังไม่ควรเข้า: รอให้ RSI ดีดกลับก่อน"

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

    # 1. Bullish Order Block (BOS Breakout)
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

    # 2. Bearish Order Block (BOS Breakdown)
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
        # 1. ดึงข้อมูล 4H หลัก
        df = get_historical_data(coin)
        time.sleep(API_RATE_LIMIT_DELAY)

        if df is None or len(df) < EMA_LONG + 10:
            logger.warning(f"{coin}: ข้อมูลไม่พอ (ต้องการ > {EMA_LONG + 10} แท่ง) – ข้ามเหรียญนี้")
            continue

        # 2. ดึงข้อมูล 1W มาวิเคราะห์ภาพใหญ่รอบปานกลาง
        weekly_ctx = analyze_weekly_context(coin)
        time.sleep(API_RATE_LIMIT_DELAY)

        # 3. ดึงข้อมูล 1M มาประเมินกรอบเป้าหมายมูลค่ารอบใหญ่ยักษ์
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

        tier    = COIN_TIER.get(coin, "mid")
        tp1_pct = TP_TIERS[tier]["tp1"]
        tp2_pct = TP_TIERS[tier]["tp2"]
        sl_buf  = TP_TIERS[tier]["sl_buffer"]
        vol_tag = " 🔊" if vol_confirmed else ""

        signal_type = ""

        # --- CASE ขาขึ้น (Above EMA 200) ---
        if current_price > ema_200:
            coin_trend = "🟢 ขาขึ้น (Above EMA 200)"
            bullish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🟢 ขาขึ้น (RSI 4H: {rsi_rounded}) | {trend_info['trend_label']}"
            )

            if current_price > (ema_50 * 0.98) and (rsi <= RSI_OVERSOLD or rsi <= RSI_PULLBACK_THRESHOLD):
                if bounce_info["quality"] in ["strong", "moderate"]:
                    signal_type = f"RSI Pullback & Rebound 📉{vol_tag}"
                elif rsi <= RSI_OVERSOLD:
                    signal_type = f"RSI Oversold + Pullback 📉{vol_tag}"
                    
            if is_divergence and not signal_type:
                signal_type = f"Bullish Divergence 📈{vol_tag}"
                
            if ob_info["has_bullish_ob"] and not signal_type:
                signal_type = f"Bullish OB Breakout (SMC) 🚀{vol_tag}"

        # --- CASE ขาลง (Below EMA 200) ---
        else:
            coin_trend = "🔴 ขาลง (Below EMA 200)"
            bearish_coins += 1
            coin_trends_summary.append(
                f"• {coin}: 🔴 ขาลง (RSI 4H: {rsi_rounded}) | {trend_info['trend_label']}"
            )

            if rsi <= RSI_OVERSOLD:
                signal_type = f"RSI Oversold (ขาลง-เสี่ยงสูง) 📉{vol_tag}"
            elif is_divergence:
                signal_type = f"Bullish Divergence (สวนเทรนด์) 📈{vol_tag}"
            elif ob_info["has_bullish_ob"]:
                signal_type = f"Bullish OB (สวนเทรนด์-ระวัง) 🚀{vol_tag}"

        if signal_type:
            # เพิ่มการติดตราประทับหากเทรนด์ 1M สนับสนุนหนุนหลังรอบใหญ่ด้วย
            if weekly_ctx["weekly_bullish_div"]:
                signal_type = f"⭐ {signal_type} + [1W Bullish Divergence คอนเฟิร์มภาพใหญ่]"
            elif weekly_ctx["rsi_weekly"] and weekly_ctx["rsi_weekly"] <= 35:
                signal_type = f"💎 {signal_type} + [1W โซนแนวรับก้นหลุม]"
            elif monthly_ctx["monthly_trend"] == "bullish":
                signal_type = f"🔥 {signal_type} + [1M มหาเทรนด์เดือนขาขึ้นคุม]"

            entry_min      = format_price(current_price * 0.97)
            entry_max      = format_price(current_price * 1.00)
            target_tp1     = format_price(current_price * (1 + tp1_pct))
            target_tp2     = format_price(current_price * (1 + tp2_pct))
            sl_val         = ema_200 * (1 - sl_buf) if current_price > ema_200 else current_price * (1 - sl_buf)
            stop_loss      = format_price(sl_val)

            buy_signals.append(
                {
                    "coin":          coin,
                    "trend":         coin_trend,
                    "price":         format_price(current_price),
                    "rsi":           rsi_rounded,
                    "type":          signal_type,
                    "ema_50":        format_price(ema_50),
                    "ema_200":       format_price(ema_200),
                    "entry":         f"${entry_min} - ${entry_max}",
                    "tp1":           f"${target_tp1} (+{tp1_pct*100:.0f}%)",
                    "tp2":           f"${target_tp2} (+{tp2_pct*100:.0f}%)",
                    "sl":            f"${stop_loss}",
                    "vol_confirmed": vol_confirmed,
                    "trend_info":    trend_info,
                    "bounce_info":   bounce_info,
                    "ob_info":       ob_info,
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
                    "ema_50":        format_price(ema_50),
                    "ema_200":       format_price(ema_200),
                    "tp_zone":       f"${tp_min} - ${tp_max}",
                    "exit":          f"${safety_exit}",
                    "vol_confirmed": vol_confirmed,
                    "trend_info":    trend_info,
                    "ob_info":       ob_info,
                    "weekly_ctx":    weekly_ctx,
                    "monthly_ctx":   monthly_ctx,
                }
            )

    # Market summary calculation
    if total_valid_coins > 0:
        bullish_ratio = (bullish_coins / total_valid_coins) * 100
        summary_msg = f"📊 <b>[Market Trend Summary]</b>\n"
        summary_msg += f"📈 ขาขึ้น: {bullish_coins} เหรียญ | 📉 ขาลง: {bearish_coins} เหรียญ\n"

        if bullish_ratio >= 65:
            summary_msg += "🔥 ภาพรวม: <b>🟢 ขาขึ้นชัดเจน (Strong Bullish)</b>\n<i>กลยุทธ์: เน้นดักซื้อเมื่อเกิดการย่อตัว (Buy on Dip)</i>"
        elif bullish_ratio >= 40:
            summary_msg += "🔥 ภาพรวม: <b>🟡 ไซด์เวย์ / เลือกทาง (Sideways)</b>\n<i>กลยุทธ์: ตลาดก้ำกึ่ง ควรเลือกเทรดเฉพาะตัวที่มีสัญญาณชัดเจน</i>"
        else:
            summary_msg += "🔥 ภาพรวม: <b>🔴 ขาลง / พักฐานแรง (Bearish)</b>\n<i>กลยุทธ์: ตลาดมีความเสี่ยงสูง เน้นถือเงินสดหรือลดขนาดไม้ลง</i>"

        summary_msg += "\n\n📋 <b>สรุปแนวโน้มรายเหรียญ:</b>\n"
        summary_msg += "\n".join(coin_trends_summary)
    else:
        summary_msg = "⚠️ ไม่สามารถดึงข้อมูลเหรียญเพื่อวิเคราะห์ภาพรวมได้"

    return buy_signals, sell_signals, summary_msg


# ==========================================
# Message Builder
# ==========================================
def build_messages(buy_list: list, sell_list: list, market_summary: str) -> list:
    message_blocks = []

    # 1. ภาพรวมตลาด
    message_blocks.append(market_summary)

    # 2. รายงานสัญญาณซื้อ
    if buy_list:
        buy_header = "🎯 <b>[Crypto Screener 4H - สัญญาณช้อนซื้อ]</b>"
        current_block = buy_header

        for opt in buy_list:
            vol_note = (
                "\n🔊 Volume: <b>ยืนยันสัญญาณ (สูงกว่า MA20)</b>"
                if opt["vol_confirmed"]
                else "\n🔇 Volume: ไม่ยืนยัน (ต่ำกว่า MA20)"
            )

            ti = opt["trend_info"]
            bi = opt["bounce_info"]
            ob = opt["ob_info"]
            w_ctx = opt.get("weekly_ctx", {})
            m_ctx = opt.get("monthly_ctx", {})

            trend_block = (
                f"\n📐 <b>แนวโน้มต่อเนื่อง (4H):</b> {ti['trend_label']}"
                f"\n   EMA50 slope: {ti['ema50_slope_pct']:+.3f}% | EMA200 slope: {ti['ema200_slope_pct']:+.3f}%"
            )

            bounce_block = (
                f"\n🔄 <b>RSI Bounce Check (4H):</b> {bi['quality_label']}"
                + (f"\n   {bi['entry_timing']}" if bi["entry_timing"] else "")
            )

            ob_block = ""
            if ob.get("has_bullish_ob"):
                ob_price_formatted = format_price(ob["bullish_ob_price"])
                ob_block = f"\n🛡️ <b>Smart Money OB Support:</b> แนวรับราคาก้อนใหญ่ย้อนหลังที่ ${ob_price_formatted}"

            weekly_block = ""
            if w_ctx and w_ctx.get("rsi_weekly"):
                weekly_block = f"\n🗓️ <b>ภาพรวมระดับสัปดาห์ (1W):</b> {w_ctx['weekly_status_label']}"

            # ดึงข้อมูลการประเมินกรอบราคาจากแท่ง Month (1M) มาต่อสายตาผู้ใช้งานก่อนเคาะออเดอร์
            monthly_block = ""
            if m_ctx and m_ctx.get("m_resistance_target"):
                target_up_str = format_price(m_ctx["m_resistance_target"])
                target_down_str = format_price(m_ctx["m_support_target"])
                monthly_block = (
                    f"\n🔮 <b>กรอบเป้าหมายมูลค่า (1M):</b>"
                    f"\n   🔼 โซนมูลค่าสูงสุด/เป้าหมายขึ้น: <code>${target_up_str}</code>"
                    f"\n   🔽 โซนมูลค่าต่ำสุด/รับลึกถ้าหลุด: <code>${target_down_str}</code>"
                    f"\n   👉 {m_ctx['monthly_summary_label']}"
                )

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b>"
                f"\n📊 เทรนด์: {opt['trend']}"
                f"\n🚨 รูปแบบ: <b>{opt['type']}</b>"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']}"
                f"\n📉 RSI (4H): {opt['rsi']}"
                f"\n📈 เส้น EMA 50 / 200: ${opt['ema_50']} / ${opt['ema_200']}"
                f"{vol_note}"
                f"{trend_block}"
                f"{bounce_block}"
                f"{ob_block}"
                f"{weekly_block}"
                f"{monthly_block}"
                f"\n🟢 ช่วงเข้าซื้อ: <code>{opt['entry']}</code>"
                f"\n💰 เป้าหมายขาย 1 (TP1): <code>{opt['tp1']}</code>"
                f"\n💰 เป้าหมายขาย 2 (TP2): <code>{opt['tp2']}</code>"
                f"\n❌ จุดตัดขาดทุน (SL): <code>{opt['sl']}</code>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = buy_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    # 3. เตือนโซน Overbought / Bearish OB
    if sell_list:
        sell_header = (
            "⚠️ <b>[Crypto Screener 4H - เตือนโซนทำกำไร / แนวต้านยักษ์]</b>\n"
            "<i>คำแนะนำ: ราคาถึงแนวต้านหรือซื้อมากเกินไป ควรพิจารณาแบ่งขายทำกำไร</i>"
        )
        current_block = sell_header

        for opt in sell_list:
            vol_note = (
                "\n🔊 Volume: <b>ยืนยันแรงซื้อ (ระวังเกิดการพักตัวแรง)</b>"
                if opt["vol_confirmed"]
                else "\n🔇 Volume: ไม่ผิดปกติ"
            )

            ti = opt["trend_info"]
            ob = opt["ob_info"]
            w_ctx = opt.get("weekly_ctx", {})
            m_ctx = opt.get("monthly_ctx", {})
            
            trend_block = (
                f"\n📐 <b>แนวโน้มต่อเนื่อง (4H):</b> {ti['trend_label']}"
                f"\n   EMA50 slope: {ti['ema50_slope_pct']:+.3f}% | EMA200 slope: {ti['ema200_slope_pct']:+.3f}%"
            )

            ob_block = ""
            if ob.get("has_bearish_ob"):
                ob_price_formatted = format_price(ob["bearish_ob_price"])
                ob_block = f"\n🚨 <b>Smart Money Bearish OB:</b> ตรวจพบกำแพงขายของสถาบันที่ ${ob_price_formatted}"

            weekly_block = ""
            if w_ctx and w_ctx.get("rsi_weekly"):
                weekly_block = f"\n🗓️ <b>ภาพรวมระดับสัปดาห์ (1W):</b> {w_ctx['weekly_status_label']}"

            monthly_block = ""
            if m_ctx and m_ctx.get("m_resistance_target"):
                target_up_str = format_price(m_ctx["m_resistance_target"])
                target_down_str = format_price(m_ctx["m_support_target"])
                monthly_block = (
                    f"\n🔮 <b>กรอบเป้าหมายมูลค่า (1M):</b>"
                    f"\n   🔼 โซนมูลค่าสูงสุด/เป้าหมายขึ้น: <code>${target_up_str}</code>"
                    f"\n   🔽 โซนมูลค่าต่ำสุด/รับลึกถ้าหลุด: <code>${target_down_str}</code>"
                    f"\n   👉 {m_ctx['monthly_summary_label']}"
                )

            coin_msg = (
                f"\n\n🪙 <b>เหรียญ: {opt['coin']}</b>"
                f"\n📊 เทรนด์: {opt['trend']}"
                f"\n💵 ราคาปัจจุบัน: ${opt['price']}"
                f"\n📈 RSI (4H): {opt['rsi']}"
                f"\n📈 เส้น EMA 50 / 200: ${opt['ema_50']} / ${opt['ema_200']}"
                f"{vol_note}"
                f"{trend_block}"
                f"{ob_block}"
                f"{weekly_block}"
                f"{monthly_block}"
                f"\n🔴 ช่วงราคาที่ควรทยอยขาย: <code>{opt['tp_zone']}</code>"
                f"\n❌ จุดล็อกกำไรหลุดตรงนี้ต้องหนี (Safety Exit): <code>{opt['exit']}</code>"
            )

            if len(current_block) + len(coin_msg) > 3500:
                message_blocks.append(current_block)
                current_block = sell_header + coin_msg
            else:
                current_block += coin_msg
        message_blocks.append(current_block)

    if not buy_list and not sell_list:
        message_blocks.append(
            "\n=========================\n😴 <i>ตลาดนิ่งสนิท: ไม่มีสัญญาณซื้อ/ขายที่เข้าเงื่อนไขใหม่ในรอบนี้</i>"
        )

    return message_blocks


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    logger.info("เริ่มต้นใช้งาน Crypto Screener 4H + 1W + 1M Multi-Timeframe v6...")

    # 1. ทำการสแกนตลาดและวิเคราะห์แนวโน้มควบคู่ทั้ง 4H, 1W และ 1M
    buy_list, sell_list, market_summary = scan_market()

    logger.info(f"สแกนระบบเสร็จสมบูรณ์ → พบสัญญาณซื้อ: {len(buy_list)} ตัว | พบสัญญาณขาย/ระวัง: {len(sell_list)} ตัว")

    # 2. แปลงผลลัพธ์ออกเป็นบล็อกข้อความ HTML
    final_messages = build_messages(buy_list, sell_list, market_summary)
    
    # 3. ยิงแจ้งเตือนเข้าแอป Telegram
    send_telegram_messages(final_messages)

    logger.info("บอททำงานและรายงานผลสมบูรณ์!")
