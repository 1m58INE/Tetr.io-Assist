"""Observed-lock attack pace and bounded goal-directed attack shaping.

All numbers are *local rules-model estimates*. There is no TETR.IO server
acknowledgement or way to infer net lines received by the opponent from a
screenshot of our board. In particular spin labels and RNG rounding are not
independently adjudicated. Never expose these as verified net attack.
"""
from collections import deque
from dataclasses import dataclass, asdict
import math

TARGET_MODEL_APM = 100.0
WINDOW_SECONDS = 60.0
MIN_RATE_SECONDS = 10.0
MAX_EVENTS = 4096  # bounded across exceptionally fast runs


@dataclass(frozen=True)
class AttackPace:
    target_apm: float
    modeled_apm: float | None
    modeled_floor_apm: float | None
    modeled_ceiling_apm: float | None
    observed_seconds: float
    observed_locks: int
    modeled_gross: float
    sample_ready: bool
    # This is a planning hint, NOT a measured opponent attack rate.
    urgency: float
    # Hypothetical minimum PPS at this modeled attack-per-confirmed-lock.
    # NOT a measured server/net-attack requirement.
    required_pps_for_target: float | None = None

    def as_dict(self):
        return asdict(self)


class AttackRateWindow:
    """Sliding 60-second window, updated only after a confirmed local lock.

    Time is supplied by the caller's monotonic clock so tests can deterministically
    replay lock bursts, gaps and garbage transitions without sleep or input.
    """
    def __init__(self, target_apm=TARGET_MODEL_APM):
        target = float(target_apm)
        if not math.isfinite(target) or target <= 0 or target > 10000:
            raise ValueError('target_apm outside supported range')
        self.target_apm = target
        self.started_at = None
        self.events = deque()
        self._total_floor = 0
        self._total_ceiling = 0
        self._total_cautious = 0

    def start(self, now):
        self.started_at = float(now)
        self.events.clear()
        self._total_floor = 0
        self._total_ceiling = 0
        self._total_cautious = 0

    def _trim(self, now):
        threshold = float(now) - WINDOW_SECONDS
        while self.events and self.events[0][0] < threshold:
            _,floor,ceiling,cautious = self.events.popleft()
            self._total_floor -= floor
            self._total_ceiling -= ceiling
            self._total_cautious -= cautious

    def record(self, now, estimate, rule_certain=False):
        """Record a lock confirmed by visual transition, NEVER an intended drop.

        'floor' is a model floor under configured rules, NOT network-confirmed;
        'certain' is zero for uncertain spins, never credited as proof of a spin.
        """
        now = float(now)
        if not math.isfinite(now):
            return False
        if self.started_at is None:
            self.start(now)
        # Ignore stale/duplicated out-of-order data; caller owns lock identity.
        if now < self.started_at or (self.events and now < self.events[-1][0]):
            return False
        floor = max(0, int(estimate.guaranteed))
        ceiling = max(floor, int(estimate.ceiling))
        cautious = floor if rule_certain else 0
        self.events.append((now, floor, ceiling, cautious))
        self._total_floor += floor
        self._total_ceiling += ceiling
        self._total_cautious += cautious
        self._trim(now)
        while len(self.events) > MAX_EVENTS:
            _,old_floor,old_ceiling,old_cautious = self.events.popleft()
            self._total_floor -= old_floor
            self._total_ceiling -= old_ceiling
            self._total_cautious -= old_cautious
        return True

    def snapshot(self, now, incoming=0, safe=True):
        now = float(now)
        if self.started_at is None or now < self.started_at:
            return AttackPace(self.target_apm, None, None, None, 0.0, 0, 0.0, False, 0.0)
        self._trim(now)
        observed = min(WINDOW_SECONDS, max(0.0, now - self.started_at))
        modeled = self._total_floor
        ceiling = self._total_ceiling
        cautious = self._total_cautious
        ready = observed >= MIN_RATE_SECONDS
        scale = 60.0 / observed if observed > 0 else 0.0
        if not ready:
            # Early stages: pursue attack, but don't report an unstable APM.
            urgency = 0.75
        else:
            deficit = (self.target_apm - modeled * scale) / self.target_apm
            urgency = max(0.0, min(1.0, deficit))
        if incoming or not safe:
            urgency = 0.0
        return AttackPace(self.target_apm, modeled * scale if ready else None,
                          cautious * scale if ready else None,
                          ceiling * scale if ready else None,
                          observed, len(self.events), float(modeled), ready, urgency,
                          (self.target_apm*len(self.events)/(60.0*modeled)
                           if ready and modeled>0 and self.events else None))


def fast_attack_goal_bias(move, board, b2b, combo, urgency):
    """O(1) additive Fast score: prefers viable difficult attacks, not spam.

    This is deliberately smaller than ordinary height/hole penalties. It cannot
    override an imminent topout. Called only for confirmed no-incoming ATTACK
    mode; the caller also checks that the source board has safety headroom.
    """
    u = max(0.0, min(1.0, float(urgency)))
    if not u or any(board[:9]):
        return 0.0
    if not move.lines:
        return 0.0  # Don't award speculative setups without observed NEXT.
    if move.lines == 4:
        return u * (3.0 + min(8, max(0, int(b2b))) * 0.35)
    if move.spin == 'full' and move.lines >= 2:
        return u * (2.5 + min(6, max(0, int(b2b))) * 0.22)
    if move.spin == 'mini':
        # A local kick label is not independent server spin adjudication.
        return u * 0.5
    if move.lines == 1 and b2b > 0:
        return -u * min(5.0, 1.8 + b2b * .4)
    if move.lines == 1 and combo < 1:
        return -u * 1.0
    return 0.0


from functools import lru_cache
_CUBE_FILL=tuple((n/9.0)**3 for n in range(10))


@lru_cache(maxsize=8192)
def safe_well_progress(board):
    """Cached terminal 9-0 edge-well potential (not an imaginary future Quad).

    A clean stack can receive an incremental platform bonus. Unsafe or badly
    holed boards get zero bonus, and the solver's normal survival rules remain.
    """
    if len(board)!=20 or any(board[:9]):
        return 0.0
    from engine import columns
    cols=columns(board)
    heights=tuple(21-(c & -c).bit_length() if c else 0 for c in cols)
    if max(heights)>11 or sum(heights)-sum(int(r).bit_count() for r in board)>1:
        return 0.0
    four=board[-4:]
    def one_edge(well, mask, middle):
        depth=max(0,middle-heights[well])
        covered=sum(_CUBE_FILL[(r & mask).bit_count()] for r in four)
        obstruction=sum(bool(r & (1<<well)) for r in four)
        return (covered*3.4 + min(depth,4)*3.2 - obstruction*1.25
                + (2.0 if depth and heights[well]==0 else 0.0))
    return max(0.0,one_edge(0,1022,min(heights[1:])),
               one_edge(9,511,min(heights[:9])))
