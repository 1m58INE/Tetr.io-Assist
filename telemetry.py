"""Small, UI-independent confidence gate for the experimental garbage-meter ROI.

The meter is not an authoritative server queue. Do not synthesize lines when
pixels cannot be interpreted confidently.
"""

import time


class GarbageFilter:
    MIN_CONFIDENCE = 0.28
    SPIKE_CONFIDENCE = 0.55
    MAX_AGE = 0.65  # ~8 missed 80ms garbage telemetry samples

    def __init__(self):
        self.lines = 0
        self.confidence = 0.0
        self.candidate = None
        self.repeat = 0
        self.last_valid_at = None

    def is_known(self, now=None):
        """True only for fresh, accepted visual evidence (including zero)."""
        if now is None:now=time.perf_counter()
        return (self.last_valid_at is not None and
                now-self.last_valid_at<=self.MAX_AGE and
                self.confidence>=self.MIN_CONFIDENCE)

    def state(self, now=None):
        """(lines|None, confidence): None means unknown, not zero."""
        if not self.is_known(now):return None,0.0
        return self.lines,self.confidence

    def expire(self, now=None):
        """Drop stale meter evidence, without pretending to observe zero lines."""
        if now is None:now=time.perf_counter()
        if self.last_valid_at is not None and now-self.last_valid_at>self.MAX_AGE:
            self.last_valid_at=None
            changed=bool(self.lines or self.confidence)
            self.lines=0;self.confidence=0.0
            self.candidate=None;self.repeat=0
            return self.lines,self.confidence,changed
        return self.lines,self.confidence,False

    def update(self, lines, confidence, now=None):
        """Return (committed lines, accepted confidence, changed).

        Small ambiguous changes need two successive readings; sharp *positive*
        attacks with reliable pixels must not wait another meter cycle.
        """
        if now is None:now=time.perf_counter()
        if lines is None or confidence is None or confidence < self.MIN_CONFIDENCE:
            self.candidate = None
            self.repeat = 0
            return self.expire(now)
        self.last_valid_at=now
        lines = max(0, min(20, int(lines)))
        confidence = max(0.0, min(1.0, float(confidence)))
        if lines == self.lines:
            self.confidence = confidence
            self.candidate = None
            self.repeat = 0
            return self.lines, self.confidence, False
        if lines - self.lines >= 4 and confidence >= self.SPIKE_CONFIDENCE:
            self.lines, self.confidence = lines, confidence
            self.candidate, self.repeat = None, 0
            return self.lines, self.confidence, True
        if self.candidate == lines:
            self.repeat += 1
        else:
            self.candidate, self.repeat = lines, 1
        if self.repeat >= 2:
            self.lines, self.confidence = lines, confidence
            self.candidate, self.repeat = None, 0
            return self.lines, self.confidence, True
        return self.lines, self.confidence, False
