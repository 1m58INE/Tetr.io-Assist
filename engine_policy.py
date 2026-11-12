"""Explicit and testable solver-mode policy; no OS/input operations."""
from __future__ import annotations

ENGINE_MODES = ('internal', 'cc2', 'hybrid', 'bundled')
DISPLAY_NAMES = {
    'internal': 'INTERNAL ONLY', 'cc2': 'COLD CLEAR 2 ONLY',
    'hybrid': 'AUTO HYBRID (CC2 → INTERNAL)', 'bundled': 'LEGACY BUNDLED',
}


def configured_mode(settings):
    """Migrate previous checkboxes without turning fallback into CC2-only."""
    value = settings.get('engine_mode')
    if value in ENGINE_MODES:
        return value
    if settings.get('external_tbp') is True:
        return 'hybrid'  # old CC2 checkbox silently fell back to native
    if settings.get('bundled_tbp') is True:
        return 'bundled'
    return 'internal'


def normalized_worker_mode(value):
    # Old code and tests passed bool opt-ins; they were always fallback modes.
    if value is True:
        return 'hybrid'
    if value == 'bundled':
        return 'bundled'
    return value if isinstance(value, str) and value in ENGINE_MODES else 'internal'


def exclusive_cc2_safe(current, proposed, incoming):
    """Independently verify CC2-only proposal without using Fast as judge.

    TBP decoder already verified exact SRS route and projected grounded pose in
    the worker. Here the UI cheaply checks current-state identity and stack
    safety. It never uses the *relative Fast target* as an approval condition.
    """
    if (proposed is None or proposed.continuation_source != 'external_tbp'
            or proposed.move is None):
        return False
    if current is not None and (current.source_board != proposed.source_board or
                                current.source_piece != proposed.source_piece):
        return False
    if incoming < 0 or incoming > 3 or proposed.incoming_before != incoming:
        return False
    if proposed.use_hold and proposed.observed_hold == '?':
        return False
    from engine import board_metrics
    _, height, holes, *_ = board_metrics(proposed.source_board)
    _, after_height, after_holes, *_ = board_metrics(proposed.move.board)
    return (after_height <= 13 and after_height <= height+4 and
            after_holes <= holes+1)


def hybrid_wait_for_cc2(now, detected_at, *, pending, phase, focused, vision_ok,
                         incoming, incoming_known, hold, allow_hold, enabled=True):
    """Bounded, clean-board-only CC2 preference (never wait under pressure).

    This is a control scheduling hint, NOT a guarantee of CC2 adoption. The
    normal Fast plan remains available immediately after the 110ms deadline.
    """
    return bool(enabled and pending and phase=='idle' and focused and vision_ok
                and incoming_known and incoming==0 and hold!='?' and allow_hold
                and detected_at>0 and 0 <= now-detected_at < .110)
