"""
Live NQ Alert Loop  —  Python-only, no TradingView/NinjaTrader needed.

Runs from script start until 11:05 ET, polling NQ price data every 60 seconds.
Fires:
  • Snapshots at 9:30, 10:00, 10:30, 11:00 ET
  • Urgent @here when score >= URGENT_THRESHOLD inside the 9:30-11:00 window
  • Throttle: max 1 urgent every 5 minutes; re-arms when score drops well below

Evaluates 6 of 9 conditions from the manual (the ones that don't require manual
chart zone input):
  L1/S1  Macro 15m       — EMA stack + price vs SMA50 + MACD
  L2/S2  Struct 3m       — EMA stack + price vs SMA50 + MACD
  L3/S3  Trend clean 3m  — UT Bot proxy + LinReg slope
  L4/S4  Momentum 3m     — MACD proxy for Impulse MACD
  L8/S8  SMA50 slope 1m  — 5-bar slope
  L9/S9  Micro break 1m  — close beyond last 5-bar high/low

Conditions L5 (valid 618 zone), L6 (clean break), L7 (retest preferred half)
require zones drawn on your chart — they're not evaluable here. The bot tells
you the objective score out of 6 and reminds you to validate zone setup on
your chart.

Required env vars:
  DISCORD_WEBHOOK_URL   — Discord channel webhook
Optional:
  URGENT_THRESHOLD      — int 4-6, default 5

Deploy: GitHub Actions cron at 13:28 UTC (9:28 ET EDT).
"""
import os
import sys
import time
import json
import datetime as dt
import requests
import numpy as np
import pandas as pd

WEBHOOK = os.environ["DISCORD_WEBHOOK_URL"]
URGENT_THRESHOLD = int(os.environ.get("URGENT_THRESHOLD", "5"))

ET = dt.timezone(dt.timedelta(hours=-4))   # EDT; switch to -5 for EST


# =============================================================================
# Indicator helpers
# =============================================================================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def sma(series, length):
    return series.rolling(length).mean()


def macd_pair(series, fast=12, slow=26, sig=9):
    line = ema(series, fast) - ema(series, slow)
    signal = ema(line, sig)
    return line, signal


def linreg_endpoint(series, length=11, offset=0):
    vals = series.values
    out = np.full(len(vals), np.nan)
    for i in range(length - 1, len(vals)):
        y = vals[i - length + 1 : i + 1]
        x = np.arange(length)
        m, b = np.polyfit(x, y, 1)
        out[i] = m * (length - 1 - offset) + b
    return pd.Series(out, index=series.index)


def ut_bot_state(df, atr_period=1, sensitivity=2.0):
    """UT Bot proxy: ATR-based trailing stop direction (bull/bear booleans)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    band = sensitivity * atr
    trail_val = np.nan
    for i, (close, b) in enumerate(zip(c, band)):
        if np.isnan(trail_val) or np.isnan(b):
            trail_val = close - b if not np.isnan(b) else np.nan
            continue
        if close > trail_val and c.iloc[i - 1] > trail_val:
            trail_val = max(trail_val, close - b)
        elif close < trail_val and c.iloc[i - 1] < trail_val:
            trail_val = min(trail_val, close + b)
        elif close > trail_val:
            trail_val = close - b
        else:
            trail_val = close + b
    last_close = c.iloc[-1]
    return last_close > trail_val, last_close < trail_val


# =============================================================================
# Evaluate 6 conditions per side at current time
# =============================================================================
def evaluate():
    import yfinance as yf
    tk = yf.Ticker("NQ=F")

    # 15m macro
    m15 = tk.history(period="5d", interval="15m", prepost=True)
    if m15.empty or len(m15) < 200:
        raise RuntimeError("Insufficient 15m data")
    c15 = m15["Close"]
    ema14_15, ema21_15 = ema(c15, 14).iloc[-1], ema(c15, 21).iloc[-1]
    sma50_15 = sma(c15, 50).iloc[-1]
    macd15_line, macd15_sig = macd_pair(c15)
    last15 = c15.iloc[-1]
    macroBull = (ema14_15 > ema21_15 > sma50_15) and (last15 > sma50_15) and (macd15_line.iloc[-1] > macd15_sig.iloc[-1])
    macroBear = (ema14_15 < ema21_15 < sma50_15) and (last15 < sma50_15) and (macd15_line.iloc[-1] < macd15_sig.iloc[-1])

    # 3m struct (yfinance has no native 3m → resample 2m to 3m)
    m2 = tk.history(period="3d", interval="2m", prepost=True)
    if m2.empty:
        raise RuntimeError("No 2m data")
    m3 = (m2
          .resample("3min")
          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
          .dropna())
    c3 = m3["Close"]
    ema14_3, ema21_3 = ema(c3, 14).iloc[-1], ema(c3, 21).iloc[-1]
    sma50_3 = sma(c3, 50).iloc[-1]
    macd3_line, macd3_sig = macd_pair(c3)
    last3 = c3.iloc[-1]
    structBull = (ema14_3 > ema21_3 > sma50_3) and (last3 > sma50_3) and (macd3_line.iloc[-1] > macd3_sig.iloc[-1])
    structBear = (ema14_3 < ema21_3 < sma50_3) and (last3 < sma50_3) and (macd3_line.iloc[-1] < macd3_sig.iloc[-1])

    # 3m trend (UT Bot + LinReg)
    utBull, utBear = ut_bot_state(m3)
    lr_now = linreg_endpoint(c3, 11, 0).iloc[-1]
    lr_prev = linreg_endpoint(c3, 11, 1).iloc[-1]
    trendCleanBull = utBull and (lr_now > lr_prev)
    trendCleanBear = utBear and (lr_now < lr_prev)

    # 3m momentum (MACD proxy for Impulse MACD)
    momBull = macd3_line.iloc[-1] > macd3_sig.iloc[-1] and macd3_line.iloc[-1] > 0
    momBear = macd3_line.iloc[-1] < macd3_sig.iloc[-1] and macd3_line.iloc[-1] < 0

    # 1m execution: SMA50 slope + micro break
    m1 = tk.history(period="1d", interval="1m", prepost=True)
    if m1.empty or len(m1) < 60:
        raise RuntimeError("Insufficient 1m data")
    c1 = m1["Close"]
    sma50_1 = sma(c1, 50)
    slope = sma50_1.iloc[-1] - sma50_1.iloc[-6]
    slopeUp = slope > 0
    slopeDown = slope < 0
    micro_high = m1["High"].iloc[-6:-1].max()
    micro_low = m1["Low"].iloc[-6:-1].min()
    last_close = c1.iloc[-1]
    microHighBreak = last_close > micro_high
    microLowBreak = last_close < micro_low

    return {
        # Booleans
        "L1": bool(macroBull),       "S1": bool(macroBear),
        "L2": bool(structBull),      "S2": bool(structBear),
        "L3": bool(trendCleanBull),  "S3": bool(trendCleanBear),
        "L4": bool(momBull),         "S4": bool(momBear),
        "L8": bool(slopeUp),         "S8": bool(slopeDown),
        "L9": bool(microHighBreak),  "S9": bool(microLowBreak),
        # Numbers for context
        "price": float(last_close),
        "ema21_15m": float(ema21_15),
        "sma50_15m": float(sma50_15),
        "macro_dist_pct": float((last15 - sma50_15) / sma50_15 * 100),
        "slope_5b": float(slope),
        "micro_high": float(micro_high),
        "micro_low": float(micro_low),
    }


# =============================================================================
# Scoring + Discord payloads
# =============================================================================
EVAL = ["1", "2", "3", "4", "8", "9"]
NAMES = {
    "1": "Macro 15m",
    "2": "Struct 3m",
    "3": "Trend 3m",
    "4": "Momentum 3m",
    "8": "SMA50 slope 1m",
    "9": "Micro break 1m",
}


def score_side(state, side):
    met = 0
    parts = []
    for n in EVAL:
        ok = state.get(f"{side}{n}", False)
        parts.append(f"{side}{n} {NAMES[n]}:{'✓' if ok else '✗'}")
        if ok:
            met += 1
    return met, "  ".join(parts)


def post_snapshot(state, time_label):
    L_met, L_break = score_side(state, "L")
    S_met, S_break = score_side(state, "S")
    direction = "LONG" if L_met > S_met else "SHORT" if S_met > L_met else "NEUTRAL"

    payload = {
        "username": "NQ Desk · 15m/3m/1m time",
        "embeds": [{
            "title": f"⏱ {time_label} ET Snapshot — NQ",
            "color": 3447003,
            "fields": [
                {"name": "Direction",   "value": direction, "inline": True},
                {"name": "LONG score",  "value": f"{L_met}/6 evaluable", "inline": True},
                {"name": "SHORT score", "value": f"{S_met}/6 evaluable", "inline": True},
                {"name": "Price",       "value": f"{state['price']:.2f}", "inline": True},
                {"name": "Macro 15m",   "value": "BULL" if state["L1"] else "BEAR" if state["S1"] else "FLAT", "inline": True},
                {"name": "Struct 3m",   "value": "BULL" if state["L2"] else "BEAR" if state["S2"] else "FLAT", "inline": True},
                {"name": "LONG breakdown",  "value": L_break, "inline": False},
                {"name": "SHORT breakdown", "value": S_break, "inline": False},
                {"name": "Zone check",
                 "value": "L5/L6/L7 (valid zone · clean break · retest) require manual chart confirmation. Best case w/ zone: up to 9/9.",
                 "inline": False},
            ],
            "footer": {"text": f"Python live · {time_label} ET snapshot"}
        }]
    }
    return _post(payload)


def post_urgent(state, side, score):
    direction = "LONG" if side == "L" else "SHORT"
    _, breakdown = score_side(state, side)
    payload = {
        "username": "NQ Desk · 15m/3m/1m time",
        "content": "@here",
        "embeds": [{
            "title": f"🚨 {score}/6 EVALUABLE READY — NQ {direction}",
            "color": 15158332,
            "fields": [
                {"name": "Price",        "value": f"{state['price']:.2f}", "inline": True},
                {"name": "Score",        "value": f"{score}/6 evaluable", "inline": True},
                {"name": "Risk filter",  "value": "≤ 5 pts from 618 zone", "inline": True},
                {"name": "Breakdown",    "value": breakdown, "inline": False},
                {"name": "ACT",
                 "value": ("• Check your chart for a valid 618 zone (L5)\n"
                           "• Confirm clean break (L6) and retest in preferred half (L7)\n"
                           "• Enter on micro-high/low break with ≤ 5 pts risk\n"
                           "• If zone is invalid or risk > 5 pts → SKIP"),
                 "inline": False},
            ],
            "footer": {"text": "Python live · score crossed threshold"}
        }]
    }
    return _post(payload)


def post_status(text, color=9807270):
    payload = {
        "username": "NQ Desk · 15m/3m/1m time",
        "embeds": [{"description": text, "color": color}]
    }
    return _post(payload)


def _post(payload):
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[post error] {e}", flush=True)
        return False


# =============================================================================
# Main loop — 9:25 to 11:05 ET
# =============================================================================
def main():
    snaps_sent = {"9:30": False, "10:00": False, "10:30": False, "11:00": False}
    snap_times = {
        "9:30":  dt.time(9, 30),
        "10:00": dt.time(10, 0),
        "10:30": dt.time(10, 30),
        "11:00": dt.time(11, 0),
    }

    last_urgent = dt.datetime.min.replace(tzinfo=ET)
    urgent_armed = True

    start = dt.datetime.now(ET)
    end_time = start.replace(hour=11, minute=5, second=0, microsecond=0)
    if start > end_time:
        print(f"[abort] now {start} is past end_time {end_time} — exiting", flush=True)
        post_status(f"⚠️ Live bot started past 11:05 ET — exiting. (now={start.strftime('%H:%M:%S')})", color=15844367)
        return

    post_status(
        f"▶️ NQ Live Bot online — {start.strftime('%a %b %d')} · running until 11:05 ET. "
        f"Urgent threshold: {URGENT_THRESHOLD}/6 evaluable conditions. "
        f"Snapshots: 9:30 / 10:00 / 10:30 / 11:00.",
        color=3066993,
    )

    loop_count = 0
    while True:
        now = dt.datetime.now(ET)
        if now >= end_time:
            break

        loop_count += 1
        print(f"[loop {loop_count}] {now.strftime('%H:%M:%S')} ET", flush=True)

        try:
            state = evaluate()
        except Exception as e:
            print(f"[eval error] {e}", flush=True)
            time.sleep(60)
            continue

        L_met, _ = score_side(state, "L")
        S_met, _ = score_side(state, "S")
        best = max(L_met, S_met)
        side = "L" if L_met >= S_met else "S"

        in_window = dt.time(9, 30) <= now.time() <= dt.time(11, 0)

        # Snapshots
        for label, t in snap_times.items():
            if not snaps_sent[label] and now.time() >= t:
                print(f"  -> sending {label} snapshot", flush=True)
                post_snapshot(state, label)
                snaps_sent[label] = True

        # Urgent
        if in_window and best >= URGENT_THRESHOLD and urgent_armed:
            if (now - last_urgent).total_seconds() >= 300:   # 5 min throttle
                print(f"  -> URGENT: {best}/6 {('LONG' if side=='L' else 'SHORT')}", flush=True)
                post_urgent(state, side, best)
                last_urgent = now
                urgent_armed = False

        # Re-arm when score drops well below threshold
        if best <= URGENT_THRESHOLD - 2:
            urgent_armed = True

        time.sleep(60)

    print("[done] reached end_time, sending shutdown", flush=True)
    post_status("⏹ NQ Live Bot finished — window closed. See you tomorrow at 9:25 ET.", color=9807270)


if __name__ == "__main__":
    main()
