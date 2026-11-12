"""Exact, bounded Perfect Clear completion solver for *observed* pieces.

Never invent the remaining bag. Used after the early scaffold, when the final
2-4 pieces can actually be seen in CURRENT+NEXT. A clean-looking gap or 40
cells of area alone is NOT proof that the PC is achievable. Future placements
must be independent legal, routable local moves.
"""
from dataclasses import dataclass
import time

@dataclass(frozen=True,slots=True)
class PCFinish:
    moves: tuple
    clears: int
    nodes: int


def find_pc_finish(board,piece,queue=(),*,placed=7,incoming=0,
                   deadline=None,max_nodes=1600,max_depth=4):
    """Find a physical PC in <=4 actually observed consecutive pieces.

    `placed` is the number of verified physical locks since the empty board.
    The normal first four-row PCO needs ten locks, i.e. (10-placed) known
    pieces. Unknown slots and prior Garbage are hard stop conditions.
    """
    from engine import BASE,placements,observed_queue_prefix
    from motion import can_route
    if (incoming or not isinstance(board,(tuple,list)) or len(board)!=20 or
        any(not isinstance(r,int) or r<0 or r>1023 for r in board) or
        piece not in BASE or not 6<=placed<10 or deadline is not None and
        time.perf_counter()>=deadline):return None
    need=10-placed
    known=(piece,)+observed_queue_prefix(queue,limit=4)
    if need>max_depth or len(known)<need:return None
    seq=known[:need]
    if any(board[:12]):return None  # a four-row PC cannot top out to 8+ rows
    # Exact canonical suffix shortcut: this IS a proved SRS/line-clear path,
    # not an area-only PC guess.  The general DFS used to miss even the known
    # 4/3/2-piece suffix because its node budget was exhausted before the
    # correct move was enumerated. Only use a fully OBSERVED piece suffix.
    from canonical_openers import _cheap_book_fields, book
    for mirrored in (False,True):
        rows=_cheap_book_fields('PCO_TEN',mirrored)
        if (rows[placed][0]!=tuple(board) or
                tuple(row[1] for row in rows[placed:])!=seq):
            continue
        if deadline is not None and time.perf_counter()>=deadline:
            return None
        ledger=book('PCO_TEN',mirrored)
        if (len(ledger)==10 and ledger[placed].source==tuple(board) and
                all(ledger[i].piece==seq[i-placed]
                    for i in range(placed,10)) and
                all(ledger[i].move.board==ledger[i+1].source
                    for i in range(placed,9)) and
                not any(ledger[-1].move.board)):
            moves=tuple(row.move for row in ledger[placed:])
            return PCFinish(moves,sum(m.lines for m in moves),0)
    # Four occupied cells are added per piece and every cleared line removes
    # exactly ten. If the residual area cannot become zero modulo ten, no
    # route can possibly yield a PC. Reject before a costly DFS/BFS.
    if (sum(r.bit_count() for r in board)+4*need)%10:
        return None
    nodes=0
    route_checks=0
    def recurse(state,depth,path,clears):
        nonlocal nodes,route_checks
        if deadline is not None and time.perf_counter()>=deadline:return None
        if nodes >= max_nodes:return None
        if depth==need:
            if any(state):return None
            # Fast geometric search is a *candidate generator*, not a license
            # to execute an unreachable placement. Verify EVERY stage's
            # complete SRS route before returning a PC. Failed path candidates
            # are discarded and DFS continues (never assume a soft-drop tuck).
            for i,m in enumerate(path):
                route_checks+=1
                previous=tuple(board) if i==0 else path[i-1].board
                if not can_route(previous,seq[i],tuple(sorted(m.cells)),m.spin):
                    return None
            return PCFinish(tuple(path),clears,nodes)
        # Shortlist by physically valid *hard-drop* geometry first. Repeated
        # full-board SRS BFS on every failed DFS branch was the CPU hotspot.
        # Prefer line clears to find terminal PCs early; return only after
        # checking the exact current/observed-piece route at each step.
        moves=sorted(placements(state,seq[depth]),key=lambda m:-m.lines)
        complete=True
        for m in moves:
            if nodes>=max_nodes:
                complete=False
                break
            nodes+=1
            if any(m.board[:12]):continue
            if depth==need-1 and any(m.board):continue
            done=recurse(m.board,depth+1,path+[m],clears+m.lines)
            if done:return done
            if deadline is not None and time.perf_counter()>=deadline:
                complete=False
                break
        # Do not memoize failures by board alone: route-validated histories
        # may differ even when their final geometry is identical.
        return None
    return recurse(tuple(board),0,[],0)
