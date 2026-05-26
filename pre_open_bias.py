"""
Pre-Open Bias Poster v3 — runs ~9:25 ET, posts overall consensus + REASONING
mapped to the 9 conditions from the "Otaku Trader Futures" manual.

Timeframe mapping (v3):
  - Macro       16180 → 15-min
  - Structural  3820  → 3-min
  - Execution   618   → 1-min
  - Hard filter: max risk <= 5 pts from 618 zone entry to stop
  - Window: 9:30 - 11:00 ET, Nasdaq futures (NQ=F)

9 conditions evaluated (LONG / SHORT mirror):
  1. Macro favors direction      (15m EMAs stacked + MACD agrees)
  2. 3820 structural agrees      (3m EMAs stacked + MACD agrees)
  3. 3-min trend clean           (UT Bot proxy + LinReg slope on 3m)
  4. Momentum rising/falling     (Impulse MACD on 3m)
  5. Valid 618 zone              (needs user-input — pre-open skips; flagged as N/A)
  6. Clean break of zone         (same — N/A pre-open)
  7. Retest of preferred half    (same — N/A pre-open)
  8. SMA50 slope agrees          (1m SMA50 slope)
  9. Micro-high/low break entry  (1m last-swing break)

NOTE: pre-open at 9:25 cannot evaluate L5-L7 because those depend on the user's
zone inputs from their morning chart. The script reports "objective" score
(out of 6 evaluable conditions: 1,2,3,4,8,9) plus a "best-case" score that
assumes 5-7 will line up if the user draws a valid zone.

Required env vars:
  DISCORD_WEBHOOK_URL   — Discord channel webhook
Optional:
  FINNHUB_API_KEY       — high-impact econ calendar context

Deploy: GitHub Actions cron at 13:23 UTC (9:23 ET EDT) — posts ~9:25.
"""
import os
import datetime as dt
import requests
import numpy as np
import pandas as pd

WEBHOOK = os.environ["DISCORD_WEBHOOK_URL"]
FINNHUB = os.environ.get("FINNHUB_API_KEY", "")

ET = dt.timezone(dt.timedelta(hours=-4))  # EDT; switch to -5 for EST
now = dt.datetime.now(ET)
today = now.date()


# ============================================================================
# Indicator helpers
# ============================================================================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def sma(series, length):
    return series.rolling(length).mean()


def macd_pair(series, fast=12, slow=26, sig=9):
    line = ema(series, fast) - ema(series, slow)
    signal = ema(line, sig)
    return line, signal


def linreg(series, length=11, offset=0):
    """Linear regression endpoint value, matching TradingView ta.linreg."""
    vals = series.values
    out = np.full(len(vals), np.nan)
    for i in range(length - 1, len(vals)):
        y = vals[i - length + 1 : i + 1]
        x = np.arange(length)
        m, b = np.polyfit(x, y, 1)
        out[i] = m * (length - 1 - offset) + b
    return pd.Series(out, index=series.index)


def ut_bot_state(df, atr_period=1, sensitivity=2.0):
    """UT Bot proxy: ATR-based trailing stop direction."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    band = sensitivity * atr
    trail = pd.Series(index=df.index, dtype=float)
    prev = np.nan
    for i, (close, b) in enumerate(zip(c, band)):
        if np.isnan(prev) or np.isnan(b):
            prev = close - b if not np.isnan(b) else np.nan
        else:
            if close > prev and c.iloc[i - 1] > prev:
                prev = max(prev, close - b)
            elif close < prev and c.iloc[i - 1] < prev:
                prev = min(prev, close + b)
            elif close > prev:
                prev = close - b
            else:
                prev = close + b
        trail.iloc[i] = prev
    return c.iloc[-1] > trail.iloc[-1], c.iloc[-1] < trail.iloc[-1]


def impulse_macd(df, length=34, sig=9):
    """Simplified Impulse MACD: MACD line on EMA(length)."""
    c = df["Close"]
    md = ema(c, length) - ema(ema(c, length), length)
    signal = sma(md, sig)
    return md.iloc[-1], signal.iloc[-1]


# ============================================================================
# Multi-timeframe state evaluation
# ============================================================================
def get_state():
    import yfinance as yf
    tk = yf.Ticker("NQ=F")

    # --- 15m macro ---
    m15 = tk.history(period="5d", interval="15m", prepost=True)
    if m15.empty:
        raise RuntimeError("No 15m data")
    c15 = m15["Close"]
    ema14_15 = ema(c15, 14).iloc[-1]
    ema21_15 = ema(c15, 21).iloc[-1]
    sma50_15 = sma(c15, 50).iloc[-1]
    ema200_15 = ema(c15, 200).iloc[-1]
    last15 = c15.iloc[-1]
    macd15_line, macd15_sig = macd_pair(c15)
    macd15_line_v, macd15_sig_v = macd15_line.iloc[-1], macd15_sig.iloc[-1]

    macroBull = (ema14_15 > ema21_15 > sma50_15) and (last15 > sma50_15) and (macd15_line_v > macd15_sig_v)
    macroBear = (ema14_15 < ema21_15 < sma50_15) and (last15 < sma50_15) and (macd15_line_v < macd15_sig_v)

    # --- 3m structural + trend + momentum ---
    m3 = tk.history(period="3d", interval="2m", prepost=True)  # yfinance 3m not native; closest is 2m; resample
    if m3.empty:
        raise RuntimeError("No 2m data")
    # Resample 2m → 3m
    m3 = m3.resample("3min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    c3 = m3["Close"]
    ema14_3 = ema(c3, 14).iloc[-1]
    ema21_3 = ema(c3, 21).iloc[-1]
    sma50_3 = sma(c3, 50).iloc[-1]
    last3 = c3.iloc[-1]
    macd3_line, macd3_sig = macd_pair(c3)
    macd3_line_v, macd3_sig_v = macd3_line.iloc[-1], macd3_sig.iloc[-1]

    structBull = (ema14_3 > ema21_3 > sma50_3) and (last3 > sma50_3) and (macd3_line_v > macd3_sig_v)
    structBear = (ema14_3 < ema21_3 < sma50_3) and (last3 < sma50_3) and (macd3_line_v < macd3_sig_v)

    # Trend clarity (UT Bot proxy + LinReg slope on 3m)
    utBull, utBear = ut_bot_state(m3)
    lr_now = linreg(c3, 11, 0).iloc[-1]
    lr_prev = linreg(c3, 11, 1).iloc[-1]
    linregUp = lr_now > lr_prev
    linregDown = lr_now < lr_prev
    trendCleanBull = utBull and linregUp
    trendCleanBear = utBear and linregDown

    # Momentum (Impulse MACD on 3m)
    md_v, sig_v = impulse_macd(m3)
    momBull = md_v > sig_v and md_v > 0
    momBear = md_v < sig_v and md_v < 0

    # --- 1m execution: SMA50 slope + micro break ---
    m1 = tk.history(period="1d", interval="1m", prepost=True)
    if m1.empty:
        raise RuntimeError("No 1m data")
    c1 = m1["Close"]
    sma50_1 = sma(c1, 50)
    slope = sma50_1.iloc[-1] - sma50_1.iloc[-6] if len(sma50_1) >= 6 else 0.0
    slopeUp = slope > 0
    slopeDown = slope < 0
    micro_high = m1["High"].iloc[-6:-1].max() if len(m1) >= 6 else np.nan
    micro_low = m1["Low"].iloc[-6:-1].min() if len(m1) >= 6 else np.nan
    last_close = c1.iloc[-1]
    microHighBreak = last_close > micro_high if not np.isnan(micro_high) else False
    microLowBreak = last_close < micro_low if not np.isnan(micro_low) else False

    # Overnight range
    on_high = float(m15["High"].max())
    on_low = float(m15["Low"].min())

    state = {
        # Numerics
        "last_price": float(last_close),
        "ema21_15m": float(ema21_15),
        "sma50_15m": float(sma50_15),
        "macro_dist_pct": float((last15 - sma50_15) / sma50_15 * 100),
        "sma50_1m_slope_5b": float(slope),
        "on_high": on_high,
        "on_low": on_low,
        "on_range": on_high - on_low,
        # Booleans for 6 evaluable conditions
        "L1": bool(macroBull), "S1": bool(macroBear),
        "L2": bool(structBull), "S2": bool(structBear),
        "L3": bool(trendCleanBull), "S3": bool(trendCleanBear),
        "L4": bool(momBull), "S4": bool(momBear),
        "L8": bool(slopeUp), "S8": bool(slopeDown),
        "L9": bool(microHighBreak), "S9": bool(microLowBreak),
    }
    return state


# ============================================================================
# Econ calendar
# ============================================================================
def econ_events():
    if not FINNHUB:
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/economic",
            params={"from": today.isoformat(), "to": today.isoformat(), "token": FINNHUB},
            timeout=10,
        ).json()
        hits = []
        for e in r.get("economicCalendar", []):
            if e.get("country") != "US":
                continue
            if (e.get("impact") or "").lower() != "high":
                continue
            try:
                t = dt.datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).astimezone(ET)
                if t.date() == today and dt.time(9, 30) <= t.time() <= dt.time(11, 0):
                    hits.append(f"{t.strftime('%H:%M')} {e.get('event','')}")
            except Exception:
                pass
        return hits
    except Exception:
        return []


# ============================================================================
# Scoring & reasoning
# ============================================================================
EVALUABLE = ["1", "2", "3", "4", "8", "9"]   # L5/L6/L7 depend on user-drawn zones
COND_NAMES = {
    "1": "Macro 15m",
    "2": "Struct 3m",
    "3": "Trend clean 3m",
    "4": "Momentum 3m",
    "5": "Valid 618 zone",
    "6": "Clean break of zone",
    "7": "Retest preferred half",
    "8": "SMA50 slope 1m",
    "9": "Micro break 1m",
}


def score(state, side):
    """Side = 'L' or 'S'. Returns (count_met, breakdown_list)."""
    met = 0
    parts = []
    for n in EVALUABLE:
        key = f"{side}{n}"
        ok = state.get(key, False)
        parts.append(f"{key} {COND_NAMES[n]}: {'✓' if ok else '✗'}")
        if ok:
            met += 1
    return met, parts


def verdict(state):
    L_met, L_parts = score(state, "L")
    S_met, S_parts = score(state, "S")
    # objective: out of 6 evaluable; best-case adds 3 zone-conditions
    if L_met > S_met:
        return ("LONG BIAS", "🟢", 3066993, L_met, S_met, L_parts, S_parts)
    if S_met > L_met:
        return ("SHORT BIAS", "🔴", 15158332, L_met, S_met, L_parts, S_parts)
    return ("NEUTRAL / NO EDGE", "⚪", 9807270, L_met, S_met, L_parts, S_parts)


def build_reasoning(state, news, L_met, S_met, L_parts, S_parts, side):
    out = []
    out.append(
        f"**16180 Macro (15m):** price {state['last_price']:.2f} vs SMA50 {state['sma50_15m']:.2f} "
        f"(distance {state['macro_dist_pct']:+.2f}%). "
        f"Macro: {'BULL' if state['L1'] else 'BEAR' if state['S1'] else 'FLAT (stack not aligned)'}."
    )
    out.append(
        f"**3820 Structural (3m):** "
        f"{'BULL stack + MACD up' if state['L2'] else 'BEAR stack + MACD down' if state['S2'] else 'no clean stack yet'}."
    )
    out.append(
        f"**3m Trend (UT Bot + LinReg):** "
        f"{'clean UP' if state['L3'] else 'clean DOWN' if state['S3'] else 'choppy / not clean'}."
    )
    out.append(
        f"**3m Momentum (Impulse MACD):** "
        f"{'rising above signal & above 0' if state['L4'] else 'falling below signal & below 0' if state['S4'] else 'flat / no clear push'}."
    )
    out.append(
        f"**1m SMA50 slope (5-bar):** {state['sma50_1m_slope_5b']:+.2f} → "
        f"{'UP' if state['L8'] else 'DOWN' if state['S8'] else 'flat'}."
    )
    out.append(
        f"**1m Micro break:** "
        f"{'broke last 5-bar high' if state['L9'] else 'broke last 5-bar low' if state['S9'] else 'inside last 5-bar range'}."
    )
    out.append(
        "**Zone conditions (L5/L6/L7):** require your manually-drawn 618 demand/supply zone — "
        "evaluate live on chart at the open. Hard filter: risk ≤ 5 pts from entry to stop."
    )
    out.append(
        f"**Overnight range:** {state['on_low']:.2f} → {state['on_high']:.2f} "
        f"({state['on_range']:.1f} pts)."
    )
    if news:
        out.append("**News risk in window:** " + " · ".join(news))
    else:
        out.append("**News risk in window:** none flagged.")
    return "\n\n".join(out)


# ============================================================================
# Compose & post
# ============================================================================
try:
    state = get_state()
    news = econ_events()
    label, emoji, color, L_met, S_met, L_parts, S_parts = verdict(state)
    side = "L" if "LONG" in label else "S" if "SHORT" in label else "N"
    reasoning = build_reasoning(state, news, L_met, S_met, L_parts, S_parts, side)

    if side == "L":
        best_objective = L_met
        breakdown = "\n".join(L_parts)
    elif side == "S":
        best_objective = S_met
        breakdown = "\n".join(S_parts)
    else:
        best_objective = max(L_met, S_met)
        breakdown = "LONG side:\n" + "\n".join(L_parts) + "\n\nSHORT side:\n" + "\n".join(S_parts)

    best_case = best_objective + 3  # if zones line up: L5+L6+L7 / S5+S6+S7

    payload = {
        "username": "NQ Desk",
        "embeds": [{
            "title": f"{emoji} 9:25 PRE-OPEN BIAS — {label} — {today.strftime('%a %b %d')}",
            "description": (
                f"**Window:** 9:30–11:00 ET   ·   **Last:** {state['last_price']:.2f}\n"
                f"**Objective score:** {best_objective}/6 evaluable   ·   "
                f"**Best-case w/ zone:** up to {best_case}/9\n"
                f"_(7/9 = urgent — zone conditions L5/L6/L7 require your chart input)_"
            ),
            "color": color,
            "fields": [
                {"name": "Why (9 conditions from your manual)", "value": reasoning, "inline": False},
                {"name": f"{'LONG' if side=='L' else 'SHORT' if side=='S' else 'BOTH'} breakdown",
                 "value": breakdown, "inline": False},
                {"name": "Plan",
                 "value": (
                     "• Draw the 618 demand/supply zone on your 1m chart now\n"
                     "• Wait for clean break above demand / below supply\n"
                     "• Retest into bottom-half demand / top-half supply\n"
                     "• Enter on micro-high / micro-low break (≤ 5 pts risk)\n"
                     "• 3-contract management: C1 +2pts, C2 trail to fib 50–75%, C3 runner\n"
                     "• Stop trading at 11:00 ET"
                 ),
                 "inline": False},
            ],
            "footer": {"text": f"Auto-posted {now.strftime('%H:%M')} ET · v3 (15m/3m/1m · 9-condition)"}
        }]
    }

    r = requests.post(WEBHOOK, json=payload, timeout=10)
    r.raise_for_status()
    print(f"✅ Pre-open bias posted — {label}, L={L_met}/6 S={S_met}/6")

except Exception as e:
    requests.post(WEBHOOK, json={
        "username": "NQ Desk",
        "embeds": [{
            "title": "⚠️ 9:25 Pre-Open Bias — data fetch failed",
            "description": f"Error: `{e}`\nFall back to manual chart read before 9:30 ET.",
            "color": 15844367,
        }]
    }, timeout=10)
    print(f"❌ Posted fallback. Error: {e}")
    raise
