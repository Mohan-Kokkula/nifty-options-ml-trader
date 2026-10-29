"""
fib_fan.py -- Fibonacci Speed Resistance Fan
=============================================
Calculates dynamic support/resistance levels from Fibonacci
speed resistance fan lines drawn from swing pivots on hourly candles.

Speed Resistance Fan:
  From a swing low: fan lines project upward at Fib-ratio speeds
  From a swing high: fan lines project downward at Fib-ratio speeds

  Fan level at time T from pivot:
    UP fan:   pivot_price + (swing_range * fib_ratio) * (bars_elapsed / swing_duration)
    DOWN fan: pivot_price - (swing_range * fib_ratio) * (bars_elapsed / swing_duration)

  The fan lines rotate with time -- steep fans decay faster,
  flat fans hold longer. This creates dynamic S/R that naturally
  tightens as the move ages.

Fib ratios used: 0.236, 0.382, 0.500, 0.618, 0.786
"""

import numpy as np

FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]
SWING_LOOKBACK = 10  # bars to look back for swing detection on hourly


def detect_swing_high(highs, idx, lookback=SWING_LOOKBACK):
    """A bar is a swing high if its high is the max in the lookback window."""
    if idx < lookback:
        return False
    window = highs[max(0, idx - lookback):idx + 1]
    return highs[idx] == max(window) and sum(1 for h in window if h == highs[idx]) == 1


def detect_swing_low(lows, idx, lookback=SWING_LOOKBACK):
    """A bar is a swing low if its low is the min in the lookback window."""
    if idx < lookback:
        return False
    window = lows[max(0, idx - lookback):idx + 1]
    return lows[idx] == min(window) and sum(1 for l in window if l == lows[idx]) == 1


def find_recent_swings(hourly_highs, hourly_lows, hourly_times, max_swings=3):
    """
    Find the most recent swing highs and swing lows from hourly data.
    Returns lists of (price, bar_index, timestamp) tuples.
    """
    n = len(hourly_highs)
    swing_highs = []
    swing_lows = []

    for i in range(n - 1, max(0, n - 60), -1):
        if len(swing_highs) >= max_swings and len(swing_lows) >= max_swings:
            break
        if detect_swing_high(hourly_highs, i) and len(swing_highs) < max_swings:
            swing_highs.append((hourly_highs[i], i, hourly_times[i]))
        if detect_swing_low(hourly_lows, i) and len(swing_lows) < max_swings:
            swing_lows.append((hourly_lows[i], i, hourly_times[i]))

    return swing_highs, swing_lows


def compute_fan_levels(pivot_price, pivot_idx, swing_range, current_idx,
                       direction="up", swing_duration=None):
    """
    Compute Fib speed fan levels at the current bar.

    Parameters
    ----------
    pivot_price : float, price at the swing pivot
    pivot_idx : int, bar index of the pivot
    swing_range : float, total price range of the swing (high - low)
    current_idx : int, current bar index
    direction : "up" for fan from swing low, "down" for fan from swing high
    swing_duration : int, number of bars the original swing took.
                     If None, uses bars_elapsed as a proxy.

    Returns
    -------
    dict of {fib_ratio: level} for each Fibonacci ratio
    """
    bars_elapsed = current_idx - pivot_idx
    if bars_elapsed <= 0 or swing_range <= 0:
        return {}

    if swing_duration is None:
        swing_duration = max(bars_elapsed, 1)

    time_ratio = bars_elapsed / swing_duration

    levels = {}
    for fib in FIB_RATIOS:
        if direction == "up":
            level = pivot_price + swing_range * fib * time_ratio
        else:
            level = pivot_price - swing_range * fib * time_ratio
        levels[fib] = level

    return levels


class FibFanTracker:
    """
    Maintains Fibonacci Speed Resistance Fan state across bars.
    Recalculates fan lines every hour using swing pivots.
    """

    def __init__(self, hourly_lookback=60, proximity_pts=15):
        self.hourly_highs = []
        self.hourly_lows = []
        self.hourly_opens = []
        self.hourly_closes = []
        self.hourly_times = []
        self.hourly_bar_idx = 0

        self.current_hour_high = 0.0
        self.current_hour_low = 99999.0
        self.current_hour_open = 0.0
        self.current_hour_close = 0.0
        self.current_hour_start = None
        self.bars_in_hour = 0

        self.up_fans = []    # list of (pivot_price, pivot_idx, swing_range, swing_dur)
        self.down_fans = []

        self.proximity_pts = proximity_pts

    def update_bar(self, ts, bar_open, high, low, close):
        """
        Feed a 5-min bar. Returns True if an hourly bar just closed
        and fans were recalculated.
        """
        hour_key = (ts.date(), ts.hour)
        new_hour = False

        if self.current_hour_start is None:
            self.current_hour_start = hour_key
            self.current_hour_open = bar_open
            self.current_hour_high = high
            self.current_hour_low = low

        if hour_key != self.current_hour_start:
            self._close_hourly_bar()
            self.current_hour_start = hour_key
            self.current_hour_open = bar_open
            self.current_hour_high = high
            self.current_hour_low = low
            self.bars_in_hour = 0
            new_hour = True

        self.current_hour_high = max(self.current_hour_high, high)
        self.current_hour_low = min(self.current_hour_low, low)
        self.current_hour_close = close
        self.bars_in_hour += 1

        return new_hour

    def _close_hourly_bar(self):
        """Finalize current hourly bar and recalculate swings."""
        self.hourly_highs.append(self.current_hour_high)
        self.hourly_lows.append(self.current_hour_low)
        self.hourly_opens.append(self.current_hour_open)
        self.hourly_closes.append(self.current_hour_close)
        self.hourly_times.append(self.current_hour_start)
        self.hourly_bar_idx += 1

        if len(self.hourly_highs) > 120:
            trim = len(self.hourly_highs) - 80
            self.hourly_highs = self.hourly_highs[trim:]
            self.hourly_lows = self.hourly_lows[trim:]
            self.hourly_opens = self.hourly_opens[trim:]
            self.hourly_closes = self.hourly_closes[trim:]
            self.hourly_times = self.hourly_times[trim:]

        self._recalculate_fans()

    def _recalculate_fans(self):
        """Find swings and set up fan lines."""
        n = len(self.hourly_highs)
        if n < SWING_LOOKBACK + 2:
            return

        sh_list, sl_list = find_recent_swings(
            self.hourly_highs, self.hourly_lows, self.hourly_times
        )

        self.up_fans = []
        self.down_fans = []

        for sl_price, sl_idx, sl_ts in sl_list:
            highs_after = self.hourly_highs[sl_idx:]
            if not highs_after:
                continue
            swing_high = max(highs_after)
            swing_range = swing_high - sl_price
            sh_rel_idx = sl_idx + highs_after.index(swing_high)
            swing_dur = max(sh_rel_idx - sl_idx, 1)
            if swing_range > 20:
                self.up_fans.append((sl_price, sl_idx, swing_range, swing_dur))

        for sh_price, sh_idx, sh_ts in sh_list:
            lows_after = self.hourly_lows[sh_idx:]
            if not lows_after:
                continue
            swing_low = min(lows_after)
            swing_range = sh_price - swing_low
            sl_rel_idx = sh_idx + lows_after.index(swing_low)
            swing_dur = max(sl_rel_idx - sh_idx, 1)
            if swing_range > 20:
                self.down_fans.append((sh_price, sh_idx, swing_range, swing_dur))

    def get_fan_levels(self, spot):
        """
        Get current fan support and resistance levels near the spot price.

        Returns
        -------
        dict with:
          fan_supports : list of (level, fib_ratio, source) sorted descending
          fan_resists  : list of (level, fib_ratio, source) sorted ascending
          nearest_support : float or None
          nearest_resist  : float or None
          fan_confluence  : int, count of fan lines within proximity of spot
        """
        cur_idx = len(self.hourly_highs)
        supports = []
        resists = []

        for pivot_price, pivot_idx, swing_range, swing_dur in self.up_fans:
            levels = compute_fan_levels(
                pivot_price, pivot_idx, swing_range, cur_idx,
                direction="up", swing_duration=swing_dur
            )
            for fib, level in levels.items():
                if level < spot:
                    supports.append((level, fib, "up_fan"))
                else:
                    resists.append((level, fib, "up_fan"))

        for pivot_price, pivot_idx, swing_range, swing_dur in self.down_fans:
            levels = compute_fan_levels(
                pivot_price, pivot_idx, swing_range, cur_idx,
                direction="down", swing_duration=swing_dur
            )
            for fib, level in levels.items():
                if level < spot:
                    supports.append((level, fib, "down_fan"))
                else:
                    resists.append((level, fib, "down_fan"))

        supports.sort(key=lambda x: -x[0])
        resists.sort(key=lambda x: x[0])

        nearest_sup = supports[0][0] if supports else None
        nearest_res = resists[0][0] if resists else None

        fan_confluence = sum(
            1 for s in supports + resists
            if abs(s[0] - spot) <= self.proximity_pts
        )

        return {
            "fan_supports": supports[:5],
            "fan_resists": resists[:5],
            "nearest_support": nearest_sup,
            "nearest_resist": nearest_res,
            "fan_confluence": fan_confluence,
        }
