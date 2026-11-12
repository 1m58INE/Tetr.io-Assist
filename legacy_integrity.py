"""Conservative LEGACY brightness state transition validator.

This guard never synthesizes/edits pixels, never returns an old game state,
and never authorizes a hard drop. It filters first-frame changes of locked
board/piece identity that could be caused by transient brightness effects.

A stable CURRENT position can move under gravity while the fixed board is the
same. We therefore require agreement on (locked board, CURRENT kind), not a
pixel-identical whole frame. Each confirmation must be from a later capture.
"""

class LegacyTransitionGate:
    def __init__(self, min_interval=.010, max_interval=.35):
        self.min_interval=float(min_interval)
        self.max_interval=float(max_interval)
        self.reset()

    def reset(self):
        self.confirmed=None
        self.pending=None
        self.pending_time=None
        self.count=0
        self.rejections=0
        self.confirmed_changes=0
        self.last_reason='RESET'
        self.confirmed_cells=None
        self.pose_jump_rejections=0

    def accept(self, state, now, *, extra_caution=False, independently_verified=False):
        """True only for a new, currently observed full state, never stale data.

        The known board+piece stays hot-path fast. New states receive at least
        two spatially consistent, time-separated captures; unexplained changes
        during an in-flight turn receive a third. No timers can independently
        approve an absent state or stale board.
        """
        if state is None:
            self.pending=None
            self.pending_time=None
            self.count=0
            self.last_reason='NO_STATE'
            return False
        if not isinstance(state,(tuple,list)) or len(state)!=3:
            self.last_reason='INVALID_STATE'
            return False
        board,piece,cells=state
        try:
            valid=(len(board)==20 and all(type(r) is int and 0<=r<=1023 for r in board)
                   and piece in 'IOTJLSZ' and len(cells)==4 and
                   len(set(cells))==4 and
                   all(type(x) is int and type(y) is int and
                       0<=x<10 and 0<=y<20 and not (board[y]&(1<<x))
                       for x,y in cells))
        except (TypeError,ValueError):
            valid=False
        if not valid:
            self.last_reason='INVALID_GEOMETRY'
            self.rejections+=1
            return False
        signature=(tuple(board),piece)
        if self.confirmed is None:
            self.confirmed=signature
            self.confirmed_cells=tuple(sorted(cells))
            self.last_reason='INITIAL'
            return True
        # Another independent verifier already checks a predicted lock or
        # exact physical garbage shift across multiple fresh observations.
        # Duplicating the gate here would add unnecessary STOP/PAUSE frames
        # to every normal lock. Trust only explicitly verified geometry;
        # never a meter reading, old snapshot or arbitrary board change.
        if independently_verified:
            if signature!=self.confirmed:self.confirmed_changes+=1
            self.confirmed=signature
            self.confirmed_cells=tuple(sorted(cells))
            self.pending=None
            self.pending_time=None
            self.count=0
            self.last_reason='INDEPENDENT_GEOMETRY'
            return True
        # A same-board/same-piece candidate can still be the wrong four
        # bright minos. Gravity only descends; known rotation kicks shift a
        # tetromino upwards by at most a few rows. A large unverified upward
        # teleport needs two time-separated captures before it is trusted.
        # Do not impose this restriction after a separately proven lock, HOLD
        # or exact Garbage rise: those are admitted above.
        upstream=self.confirmed_cells
        upward_teleport=(signature==self.confirmed and upstream is not None
                         and min(y for x,y in cells)<min(y for x,y in upstream)-4)
        if signature==self.confirmed and not upward_teleport:
            self.confirmed_cells=tuple(sorted(cells))
            self.pending=None
            self.pending_time=None
            self.count=0
            self.last_reason='UNCHANGED'
            return True
        candidate=(signature, 'UNVERIFIED_UPWARD',tuple(sorted(cells))) if upward_teleport else signature
        # Independent captures separated in time. A single noisy reading must
        # never invalidate the last confirmed snapshot.
        if (self.pending!=candidate or self.pending_time is None or
                not 0<=now-self.pending_time<=self.max_interval):
            self.pending=candidate
            if upward_teleport:self.pose_jump_rejections+=1
            self.pending_time=float(now)
            self.count=1
            self.rejections+=1
            self.last_reason='TRANSITION_FIRST_FRAME'
            return False
        if now-self.pending_time < self.min_interval:
            self.rejections+=1
            self.last_reason='TRANSITION_NOT_INDEPENDENT'
            return False
        self.count+=1
        required=3 if extra_caution else 2
        if self.count<required:
            self.rejections+=1
            self.last_reason='TRANSITION_PROBATION'
            return False
        self.confirmed=signature
        self.confirmed_cells=tuple(sorted(cells))
        self.pending=None
        self.pending_time=None
        self.count=0
        self.confirmed_changes+=1
        self.last_reason='TRANSITION_CONFIRMED'
        return True
