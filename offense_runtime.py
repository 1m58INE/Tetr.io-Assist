"""Runtime gates for *observed*, asynchronous offensive decisions.

Kept pure so the actual GUI control loop can unit-test every release condition.
A short no-input gate is allowed only on an already known, safe stack. The
Fast plan remains executable at timeout, without waiting for a worker result.
"""
from engine import board_metrics

MAX_ATTACK_DEFER_S = 0.032


def conservative_incoming(value, meter_configured, meter_known):
    """Convert an UNKNOWN selected garbage meter to a cautious planning bound.

    This is a risk proxy, NEVER a claim of actual observed incoming lines.
    No selected meter leaves the user's historical no-meter policy unchanged.
    """
    if meter_configured and not meter_known:
        return max(3,int(value or 0))
    return max(0,int(value or 0)) if meter_known else 0


def defer_for_attack(board, plan, mode, phase, now, detected_at,
                     deep_in_flight, foreground=True, vision_ok=True,
                     incoming=0, incoming_known=True, max_defer=MAX_ATTACK_DEFER_S):
    """True = intentionally send NO keys this tick to await fresh Deep advice.

    Does not change a command that was already sent. Unknown pressure explicitly
    disables the gate. The bounded delay is counted from first detection,
    not from task submission (so worker/GUI queueing cannot extend the pause).
    """
    if not (plan is not None and getattr(plan,'move',None) is not None
            and deep_in_flight and foreground and vision_ok and incoming_known
            and mode=='attack' and phase=='idle' and incoming==0):
        return False
    if plan.use_hold or getattr(plan,'continuation_source','') in ('offensive_beam','canonical_fast'):
        return False
    if not (0 <= now-detected_at < min(MAX_ATTACK_DEFER_S, max(0.0,float(max_defer)))):
        return False
    _, height, holes, *rest=board_metrics(board)
    return height <= 10 and holes <= 2
