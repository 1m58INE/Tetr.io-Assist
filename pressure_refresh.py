"""Decide whether a *not-yet-executed* Fast decision needs meter refresh.

Meter evidence may change strategic risk but MUST NEVER invalidate in-flight
movement/drop tickets. This module only gives an idle planner a chance to
reconsider. Reuse the normal pressure conversion from offense_runtime so
Fast and Deep see the same conservative interpretation of UNKNOWN.
"""
from offense_runtime import conservative_incoming

MIN_REFRESH_INTERVAL_S = 0.075


def pressure_change(before_lines, before_confidence, before_known,
                    after_lines, after_confidence, after_known,
                    *, meter_configured):
    """Return (old_risk, new_risk), or None when no Fast risk changed."""
    if not meter_configured:
        return None
    old=conservative_incoming(before_lines,True,
                              bool(before_known and before_confidence>=.55))
    new=conservative_incoming(after_lines,True,
                              bool(after_known and after_confidence>=.55))
    return (old,new) if old!=new else None


def may_refresh_idle(*, enabled, phase, input_busy, key_matches,
                     engine_mode, changed_risk, now, last_refresh):
    """No pending keyboard input, no committed route, and a minimum interval."""
    return bool(changed_risk is not None and enabled and phase=='idle'
                and not input_busy and key_matches and engine_mode=='internal'
                and (last_refresh is None or now-last_refresh>=MIN_REFRESH_INTERVAL_S))


def permit_prepared_commit(*, meter_configured, meter_known, incoming, confidence):
    """A speculative NEXT plan is safe to fast-commit only with no pressure.

    The committed continuation was calculated before the latest meter reading.
    A selected but unreadable meter is UNKNOWN, not a free 0-line queue.
    An incoming attack may change route priorities before the next lock.
    """
    return (not meter_configured or
            (bool(meter_known) and confidence>=.55 and incoming==0))
