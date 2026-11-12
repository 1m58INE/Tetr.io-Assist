"""Bounded monotonic control-gap instrumentation (no key injection or networking).

Only the caller's actual input-submission events reset the silence timer.  The
sensor logs a checkpoint at 0.6s and another at 1.5s; it does not assume a
TETR.IO replay's keydown gap equals a visual or control stall.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class StallCheckpoint:
    silence_ms: int
    phase: str
    paused: bool
    reason: str
    capture_valid: bool
    input_busy: bool
    stage: int
    blocker: str = "UNKNOWN"


class StallMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_input_at = None
        self.active_since = None
        self.stage = 0
        self.samples = []
        self.transitions = []
        self.last_blocker = None

    def sent(self, now):
        self.last_input_at = float(now)
        self.active_since = float(now)
        self.stage = 0
        self.last_blocker = None

    def observe(self, now, *, enabled, phase, paused, reason,
                capture_valid, input_busy=False, deep_pending=False, has_plan=True):
        if not enabled:
            self.active_since = None
            self.stage = 0
            self.last_blocker = None
            return None
        now = float(now)
        if not capture_valid:blocker='VISION_UNAVAILABLE'
        elif paused:blocker='CONTROLLER_PAUSED'
        elif input_busy:blocker='INPUT_WORKER_BUSY'
        elif phase=='drop_wait':blocker='LOCK_ACK_WAIT'
        elif phase=='hold_wait':blocker='HOLD_ACK_WAIT'
        elif not has_plan:blocker='NO_PLAN'
        elif deep_pending:blocker='DEEP_PENDING'
        else:blocker='ALIGN_OR_REPLAN'
        if blocker!=self.last_blocker:
            self.transitions.append({'at':round(now,3),'blocker':blocker,
                                     'phase':str(phase)[:32]})
            if len(self.transitions)>128:del self.transitions[:-128]
            self.last_blocker=blocker
        if self.active_since is None:
            self.active_since = now
            return None
        elapsed = max(0, now - self.active_since)
        thresholds = (.60, 1.50)
        if self.stage >= len(thresholds) or elapsed < thresholds[self.stage]:
            return None
        self.stage += 1
        proof = StallCheckpoint(round(elapsed*1000), str(phase)[:32],
                                bool(paused), str(reason)[:140],
                                bool(capture_valid), bool(input_busy), self.stage, blocker)
        self.samples.append(proof)
        if len(self.samples) > 64:
            del self.samples[:-64]
        return proof

    def snapshot(self):
        return [sample.__dict__.copy() for sample in self.samples]

    def trace_snapshot(self):
        """Bounded metadata only; no frames, raw keys or user identifiers."""
        return [event.copy() for event in self.transitions]
