"""Bounded, observation-only adaptive opener selector for the internal Deep worker.

This is not a fixed TKI/PCO script.  It validates and ranks *already searched*
current + observed NEXT attack continuations.  It never predicts an unseen bag,
changes input timings, or authorizes a hard drop.  The consumer must revalidate
board/piece/HOLD at every real spawn.
"""
from dataclasses import dataclass
import time


@dataclass(frozen=True, slots=True)
class OpeningProfile:
    family: str  # VERIFIED_TSD, VERIFIED_QUAD, CLEAN_FLEX, or NONE
    steps: int
    generated: int  # guaranteed LOCAL attack estimate, not delivered garbage
    difficult: int
    first_cost: int
    height: int
    holes: int


def _board_metrics(board):
    from engine import board_metrics
    _, height, holes, *_ = board_metrics(board)
    return height, holes


def _finesse(move):
    x = min(col for col, _ in move.cells)
    return abs(x - 3) + min(move.rotation % 4, 4 - (move.rotation % 4)) + (3 if move.spin or move.tactical else 0)


def _physical_successor(board, move):
    """Verify every advertised intermediate board, without speculative input."""
    if not isinstance(board, tuple) or len(board) != 20:
        return False
    cells=tuple(move.cells)
    if len(cells)!=4 or len(set(cells))!=4:
        return False
    after=list(board)
    for x,y in cells:
        if not (isinstance(x,int) and isinstance(y,int) and 0<=x<10 and 0<=y<20):
            return False
        if after[y] & (1<<x):
            return False
        after[y] |= 1<<x
    remaining=[r for r in after if r!=1023]
    lines=20-len(remaining)
    return (lines == move.lines and tuple([0]*lines+remaining)==tuple(move.board))


def inspect_opening(source, piece, observed_queue, path, *, b2b=0, combo=-1,
                    ruleset='standard', combo_table='multiplier', rounding='auto',
                    deadline=None):
    """Return a proven observed-queue profile or None. Never trust unseen T/I.

    Every placement, including lookahead, is tested against exact board physics
    and SRS route reachability.  Reject expired worker tasks instead of doing
    extra work that might starve the image/input pipeline.
    """
    if not path or len(path)>4 or deadline is not None and time.perf_counter()>=deadline:
        return None
    from engine import BASE, observed_queue_prefix
    from combo_rules import attack_estimate
    from motion import can_route
    known=(piece,)+observed_queue_prefix(observed_queue)
    if len(path)>len(known) or not 1<=len(path)<=4:
        return None
    previous=tuple(source)
    height0, holes0 = _board_metrics(previous)
    if height0>9 or holes0>2:
        return None
    generated=0; difficult=0; finishes=[]; progression=[]
    bb=b2b; co=combo
    for index,move in enumerate(path):
        if deadline is not None and time.perf_counter()>=deadline:
            return None
        if move.piece != known[index] or move.piece not in BASE:
            return None
        if not _physical_successor(previous,move):
            return None
        if not can_route(previous,move.piece,tuple(sorted(move.cells)),move.spin):
            return None
        h, holes=_board_metrics(move.board)
        # A verified two-ply T cavity may temporarily add two holes, but a
        # buried stack or tall one-shot build is not an acceptable opener.
        if h>min(12,height0+5) or holes>holes0+2:
            return None
        estimate=attack_estimate(move,bb,co,ruleset,combo_table,rounding)
        generated += estimate.guaranteed
        bb,co=estimate.next_b2b,estimate.next_combo
        if move.lines==4 or move.spin and move.lines or move.lines and not any(move.board):
            difficult+=1
        if move.piece=='T' and move.spin=='full' and move.lines==2:
            finishes.append(('VERIFIED_TSD',index))
        if move.piece=='I' and move.lines==4:
            finishes.append(('VERIFIED_QUAD',index))
        previous=move.board
        progression.append((h,holes))
    h,holes=progression[-1]
    # TSD cavities must actually heal at the *observed* finisher; prospective
    # empty slots alone do not count as proof of an opener.
    valid_finishes=[(kind,index) for kind,index in finishes
                    if progression[index][0] <= min(11,height0+4)
                    and progression[index][1] <= holes0]
    family='CLEAN_FLEX'
    if valid_finishes:
        family=min(valid_finishes,key=lambda v:(v[1],v[0]!='VERIFIED_TSD'))[0]
    elif max(hh for hh,_ in progression)>8 or any(hole>holes0 for _,hole in progression):
        return None
    return OpeningProfile(family,len(path),generated,difficult,
                          _finesse(path[0]),h,holes)


def select_adaptive_opening(source,piece,queue,beam,cradle,*,b2b=0,combo=-1,
                            incoming=0,garbage_present=False,placed=0,
                            deadline=None,ruleset='standard',combo_table='multiplier',
                            rounding='auto'):
    """Compare beam vs TSD-builder without adding search or changing fallback.

    Returns (chosen original beam tuple or None, profile or None).  Only a
    *complete*, validated known-NEXT finisher can displace the beam.  This
    operates once per Deep result; it never sends keys or forces a template.
    """
    if (incoming or garbage_present or placed>=9 or placed<0 or
            deadline is not None and time.perf_counter()>=deadline):
        return beam,None
    from engine import board_metrics
    known=tuple(queue)
    opts=[]
    if beam:
        first,path,score=beam
        profile=inspect_opening(source,piece,known,path,b2b=b2b,combo=combo,
                                deadline=deadline,ruleset=ruleset,
                                combo_table=combo_table,rounding=rounding)
        if profile:
            opts.append(('beam',beam,profile))
    if cradle is not None:
        path=[];walk=cradle
        while walk is not None and len(path)<4:
            path.append(walk);walk=walk.next_move
        profile=inspect_opening(source,piece,known,path,b2b=b2b,combo=combo,
                                deadline=deadline,ruleset=ruleset,
                                combo_table=combo_table,rounding=rounding)
        if profile and profile.family=='VERIFIED_TSD':
            opts.append(('cradle',(cradle,tuple(path),0.0),profile))
    if not opts:
        return beam,None
    original=next((x for x in opts if x[0]=='beam'),None)
    builder=next((x for x in opts if x[0]=='cradle'),None)
    if builder is None:
        return (beam,original[2]) if original else (beam,None)
    if beam is None:
        return builder[1],builder[2]
    if original is None:
        return beam,None  # retain the established validated fallback
    current=original[2]
    proposed=builder[2]
    if current.family in ('VERIFIED_TSD','VERIFIED_QUAD'):
        return beam,current
    old_first=beam[0]
    new_first=builder[1][0]
    old_h,old_holes=_board_metrics(old_first.board)
    new_h,new_holes=_board_metrics(new_first.board)
    # Never trade a real multi-line clear, two extra key inputs, a dangerous
    # initial stack or more unhealed cavities for a predicted spin.
    if (old_first.lines>=2 or new_first.lines<old_first.lines or
            proposed.first_cost>current.first_cost+1 or
            new_h>min(9,old_h+1) or new_holes>old_holes+1 or
            proposed.height>11 or proposed.holes>_board_metrics(source)[1]):
        return beam,current
    return builder[1],proposed
