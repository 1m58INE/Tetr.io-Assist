"""Sparse interior samples: all 200 cells in one NumPy operation."""
from functools import lru_cache
from vision_recovery import recover_risen_piece
import numpy as np

@lru_cache(maxsize=8)
def sample_indices(height,width):
    offsets=np.array([.36,.50,.64])
    ys=np.minimum(((np.arange(20)[:,None]+offsets)*height/20).astype(int),height-1)
    xs=np.minimum(((np.arange(10)[:,None]+offsets)*width/10).astype(int),width-1)
    return ys[:,None,:,None],xs[None,:,None,:]

def read_grid(bgr,threshold):
    ys,xs=sample_indices(*bgr.shape[:2])
    # max(B,G,R)>=threshold is exactly any(channel>=threshold) for the
    # integer camera frame. Comparing before reduction avoids a full uint8
    # max temporary and speeds up the 125-Hz LEGACY recognition hot path.
    active=np.any(bgr[ys,xs]>=threshold,axis=-1).reshape(20,10,9)
    # Majority of nine interior pixels, robust to single-pixel effects.
    return (np.count_nonzero(active,axis=-1)>=5).tolist()

def against_board(grid,board,piece=None):
    """Recover four LIVE minos from a referenced board using row bitmasks.

    The former hot path iterated over all 200 coordinates to build the delta,
    even if the source board had only two or forty surplus cells. Check the
    total surplus cardinality first, and materialize coordinates only when it
    is exactly four. Never infer or fill any missing observed pixel.
    """
    from engine import identify
    if len(board)!=20 or len(grid)!=20:
        return None
    try:
        observed=tuple(sum(1<<x for x,v in enumerate(row) if v)
                       for row in grid if len(row)==10)
        if len(observed)!=20 or any(type(r) is not int or r<0 or r>1023 for r in board):
            return None
    except (TypeError,ValueError):
        return None
    surplus=[]
    count=0
    for y,(fixed,seen) in enumerate(zip(board,observed)):
        if fixed & ~seen:
            return None
        extra=seen & ~fixed
        if extra:
            count+=extra.bit_count()
            if count>4:
                return None
            surplus.append((y,extra))
    if count!=4:
        return None
    delta=[]
    for y,mask in surplus:
        while mask:
            bit=mask & -mask
            delta.append((bit.bit_length()-1,y))
            mask^=bit
    found=identify(delta)
    if found is not None and (piece is None or piece==found):
        return board,found,delta
    return None

def locate_piece(grid,previous=None,hints=()):
    from engine import separate
    for board,piece in hints:
        state=against_board(grid,board,piece)
        if state is not None:return state
    if previous is not None:
        state=against_board(grid,*previous)
        if state is not None:return state
    return separate(grid,5)

class BoardTracker:
    """Keep evidence across brief missing frames; never return an old state.

    Only four cells actually present in the current frame can produce a valid
    observation. The normal one-second expiry stays intact; a specifically
    marked short vision dropout permits a bounded, shape-verified recovery.
    """
    def __init__(self):self.reset()
    def reset(self):
        self.previous=None
        self.seen_at=None
        self.dropout_at=None
        self.last_rise=None

    def mark_missing(self, now):
        """Remember that a verified frame was interrupted by an UNKNOWN frame.

        This only extends the period where a previous *locked* board may be
        checked. No old occupancy is ever handed directly to the controller.
        """
        if self.previous is not None and self.dropout_at is None:
            self.dropout_at=float(now)

    def update(self,grid,now,hints=()):
        fresh=self.seen_at is not None and 0 <= now-self.seen_at <= 1.0
        dropout_recovery=(self.dropout_at is not None and self.seen_at is not None
                          and 0 <= now-self.dropout_at <= 2.5
                          and 0 <= now-self.seen_at <= 3.0)
        prior=self.previous if (fresh or dropout_recovery) else None
        self.last_rise=None
        state=locate_piece(grid,prior,hints)
        if state is None:
            # Try only physically evidenced shifts; ordinary failure still
            # waits without supplying speculative input to the controller.
            references=[]
            for base,piece in hints:
                if base is not None and (base,piece) not in references:
                    references.append((base,piece))
            if prior is not None and prior not in references:
                references.append(prior)
            for base,piece in references:
                found=recover_risen_piece(grid,base,piece)
                if found is not None:
                    settled,active,cells,rise=found
                    state=settled,active,cells
                    self.last_rise=(base,settled,rise)
                    break
        if state is not None:
            self.previous=state[:2]
            self.seen_at=now
            self.dropout_at=None
        elif self.dropout_at is not None and now-self.dropout_at > 2.5:
            self.dropout_at=None
        return state
