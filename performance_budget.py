"""Pure CPU-budget policy for optional, asynchronous, INTERNAL Deep searches.

This is *not* a frame rate governor and never relaxes input/drop validation.
Use evidence of actual GUI capture pressure and whether recent background
results could be applied. PPS alone is not evidence of CPU overload.
"""
from __future__ import annotations
import math

BASE = {'ATTACK': 28.0, 'COUNTER': 22.0, 'DEFENSE': 14.0, 'EMERGENCY': 14.0}


def deep_budget_ms(combat, capture_ms=0.0, observed_pps=None, *,
                   mode='internal', received=0, adopted=0, stale=0):
    """Return bounded background worker wall time in milliseconds.

    This function is called only at task submission. The GUI never waits on
    this budget. External/CC2 modes keep their former allocation. No claimed
    improvement to OS input speed or real-game PPS is implied.
    """
    base = BASE.get(combat, 14.0)
    if mode != 'internal':
        if observed_pps is not None and observed_pps >= 3:
            base = min(base, 18.0)
        if capture_ms > 22:
            base = min(base, 16.0)
        return base
    try:
        capture = float(capture_ms)
        if not math.isfinite(capture) or capture < 0:
            capture = 22.1
    except (TypeError, ValueError):
        capture = 22.1
    try:
        received = max(0, int(received))
        adopted = max(0, min(received, int(adopted)))
        stale = max(0, int(stale))
    except (TypeError, ValueError):
        received = adopted = stale = 0
    samples = received + stale
    # Capture throughput and obsolete worker outcomes are direct signals of
    # CPU contention. They outrank any additional search-quality allowance.
    if capture > 22:
        return min(base, 14.0)
    if samples >= 12 and stale * 2 >= samples:
        return min(base, 12.0)
    # No valid Deep improvements after a significant number of replies:
    # stop spending the *highest* budget on inadmissible late suggestions.
    if received >= 16 and adopted * 20 < received:
        return min(base, 15.0)
    if capture > 15:
        return min(base, 20.0)
    # A useful, nonstale worker on a healthy system can exploit more spare
    # background time. High PPS is not a reason to blindly throttle it.
    if (combat == 'ATTACK' and capture <= 12 and received >= 8
            and adopted * 4 >= received and (samples == 0 or stale * 4 < samples)):
        return 36.0
    return base


def experimental_budget_ms(worker_budget_ms, incoming=0):
    """Subset of the *same* Deep worker deadline for optional PC/6-3 search.

    Never extend the parent budget, especially when incoming is nonzero.
    """
    if incoming:
        return 0.0
    try:
        budget = float(worker_budget_ms)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(budget) or budget <= 0:
        return 0.0
    # 4ms legacy at <=22ms; up to 8ms if enough background slack exists.
    return max(0.0, min(8.0, budget * .20))
