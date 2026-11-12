"""Bounded provisional-LEGACY transition admission (not a vision fallback).

An *otherwise valid* new brightness state is held out of the controller for
one or two independent captures. A held state is NEVER treated as the old
state, never authorizes a drop and never becomes a board-planning input.

Unlike the previous blanket VISION_LOST path, that temporary uncertainty does
not have to tear down a partially completed movement burst. It does NOT apply
to unrecognized pixels, wrong CURRENT identity, HOLD/DROP transitions or
already paused/defocused input. All of those remain fail-closed.
"""

MAX_PROVISIONAL_S = 0.060
PROVISIONAL_REASONS = frozenset({
    'TRANSITION_FIRST_FRAME', 'TRANSITION_NOT_INDEPENDENT',
    'TRANSITION_PROBATION',
})


def allow_provisional_hold(*, gate_reason, phase, observed_piece,
                           planned_piece, paused, input_busy,
                           drop_inflight, now, entered_at=None):
    """True means *defer this frame*, not approve its candidate state.

    The caller MUST refrain from issuing NEW inputs or consuming the frame as
    a lock acknowledgement. Path-only input already in progress can finish,
    because each move will be verified against a subsequent live board before
    a separate hard-drop command can be requested.
    """
    if (gate_reason not in PROVISIONAL_REASONS or paused or drop_inflight
            or phase not in ('idle', 'align', 'rotate_wait')):
        return False
    if phase == 'idle':
        # Do not let an older in-flight burst affect a newly detected turn.
        if input_busy:
            return False
    elif observed_piece != planned_piece:
        return False
    if entered_at is not None and (now < entered_at or now-entered_at>MAX_PROVISIONAL_S):
        return False
    return True
