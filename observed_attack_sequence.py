"""Deadline-bounded, strictly observed HOLD-aware attack continuation search.

This is a *Deep worker only* strategy. It does not infer unseen 7-bag symbols,
change real input timing, or claim that its modeled attack is server-confirmed.
An entire suggested prefix must be independently reachable before it can be
considered for adoption; future moves are rechecked at each real spawn.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from combo_rules import attack_estimate
from engine import BASE, Move, board_metrics, observed_queue_prefix, placements, _deep_obsolete
from motion import can_route_until


@dataclass(frozen=True)
class VerifiedSequence:
    moves: tuple[Move, ...]
    holds: tuple[bool, ...]
    first_uses_hold: bool
    guaranteed_attack: int
    score: float
    explored: int
    # Actual NEXT after the first lock. When a later move uses HOLD, it is
    # intentionally NOT offered as a directly executable speculative prefetch.
    direct_next: bool


@dataclass(slots=True)
class _Node:
    board: tuple[int, ...]
    current: str | None
    preview: tuple[str, ...]
    hold: str | None
    b2b: int
    combo: int
    moves: tuple[Move, ...]
    holds: tuple[bool, ...]
    gross: int
    accrued: float
    rating: float


def _choices(current, preview, held, allow_hold):
    """(played piece, queue after HOLD consumption, new HOLD, uses HOLD).

    HOLD is available at most once *per spawned piece*, not once per entire
    search. After each simulated lock the next CURRENT is drawn from the
    actually observed preview. Empty HOLD consumes NEXT immediately.
    """
    yield current, preview, held, False
    if not allow_hold or held == '?':
        return
    if held is None:
        if preview and preview[0] != current:
            yield preview[0], preview[1:], current, True
    elif held in BASE and held != current:
        yield held, preview, current, True


def _bounded_route(board, piece, cells, spin, deadline):
    """Fast deterministic path first, then deadline-bounded SRS+ proof.

    Never confuse absence of proof due to time exhaustion with a proof of
    impossibility. An accepted result is always executable from spawn.
    """
    return can_route_until(board, piece, cells, spin, deadline=deadline,
                           limit=850, max_depth=14)


def find_verified_sequence(board, piece, queue, hold='?', allow_hold=True, b2b=0, combo=-1,
                           *, deadline, width=6, max_nodes=650, revision=0,
                           ruleset='standard', combo_table='multiplier', rounding='auto'):
    """Find a route-proved 2–4 lock attack prefix using *observed* queue only.

    Two locks are required, even for an empty HOLD. Partial/expired states are
    never exported as plans. The route is verified for every claimed move and
    the result has one HOLD decision per lock, never a speculative input queue.
    """
    if (piece not in BASE or not isinstance(board, tuple) or len(board) != 20
            or any(not isinstance(r, int) or r < 0 or r >= 1024 or r == 1023
                   for r in board)
            or hold not in ('?', None, *BASE) or not isinstance(deadline,(int,float))
            or not math.isfinite(deadline)
            or _deep_obsolete(revision)):
        return None
    preview=observed_queue_prefix(queue, 3)
    if not preview or time.perf_counter() >= deadline:
        return None
    # A known HOLD is needed to consider later swaps; unknown disables every
    # HOLD branch, not just the initial one.
    allow_hold=bool(allow_hold and hold!='?')
    width=max(2,min(int(width),10));max_nodes=max(16,min(int(max_nodes),1500))
    start=_Node(board,piece,preview,hold,max(0,int(b2b)),int(combo),(),(),0,0.0,0.0)
    frontier=[start];finished=[];explored=0
    for depth in range(min(4,len(preview)+1)):
        successors={}
        for node in frontier:
            if time.perf_counter()>=deadline or _deep_obsolete(revision):break
            if node.current is None:continue
            for to_play, remaining, new_hold, swapped in _choices(
                    node.current,node.preview,node.hold,allow_hold):
                if time.perf_counter()>=deadline:break
                # Pure geometry is the cheap candidate superset. Each exported
                # path is route-proven. Avoid expensive BFS on every branch.
                options=list(placements(node.board,to_play))
                if (to_play=='T' and any(node.board) and depth<=1 and
                        deadline-time.perf_counter()>.004):
                    from engine import tactical_moves
                    by_cells={m.cells:m for m in options}
                    for move in tactical_moves(node.board,to_play,95,8,deadline=deadline):
                        if move.spin and move.lines:
                            by_cells[tuple(sorted(move.cells))]=move
                    options=list(by_cells.values())
                for move in options:
                    explored+=1
                    if explored>max_nodes or (explored&15)==0 and time.perf_counter()>=deadline:
                        break
                    _,height,holes,*_=board_metrics(move.board)
                    if height>=15 or holes>=5 or height>max(12,board_metrics(board)[1]+4):
                        continue
                    estimate=attack_estimate(move,node.b2b,node.combo,
                                             ruleset,combo_table,rounding)
                    # Attack earned per placement is NOT interchangeable with
                    # an optimistic full B2B bank payout. Discount distant
                    # earned attack and evaluate terminal stack only once.
                    earned=node.accrued + (0.82**depth)*estimate.guaranteed
                    gross=node.gross + estimate.guaranteed
                    flat=board_metrics(move.board)[4]
                    terminal= -holes*7.0 - height*1.05 - flat*.35
                    if height>=12:terminal-=(height-11)**2*3.0
                    # Prefer B2B-compatible clears without inventing any
                    # attack lines. This is a tiny tie-break, not the objective.
                    rating=earned*7.5+terminal + min(estimate.next_b2b,8)*.65
                    if swapped:rating-=.8
                    next_current=remaining[0] if remaining else None
                    after_preview=remaining[1:] if remaining else ()
                    # In HOLD-empty the played piece was NEXT1 and was already
                    # consumed. Only the remaining observed queue can spawn.
                    new=_Node(move.board,next_current,after_preview,new_hold,
                        estimate.next_b2b,estimate.next_combo,
                        node.moves+(move,),node.holds+(swapped,),gross,earned,rating)
                    key=(new.board,new.current,new.preview,new.hold,new.b2b,new.combo)
                    old=successors.get(key)
                    if old is None or new.rating>old.rating:
                        successors[key]=new
                if explored>=max_nodes:break
            if explored>=max_nodes:break
        if not successors:break
        ordered=sorted(successors.values(),key=lambda n:n.rating,reverse=True)
        frontier=ordered[:width]
        if depth>=1:
            # Keep independently complete 2+ ply states even if the next ply
            # cannot be finished under the same bounded deadline.
            finished.extend(frontier)
        if (explored>=max_nodes or time.perf_counter()>=deadline or
                _deep_obsolete(revision)):
            break
    if _deep_obsolete(revision):return None
    # A complete path with 3 real locks has more objective information than
    # two; do not compare two vs three unnormalised earnings as equal-length.
    finished.sort(key=lambda n:(len(n.moves),n.rating),reverse=True)
    for node in finished:
        if time.perf_counter()>=deadline:return None
        prior=board
        valid=True
        for m in node.moves:
            if time.perf_counter()>=deadline or not _bounded_route(
                    prior,m.piece,m.cells,m.spin,deadline):
                valid=False;break
            prior=m.board
        if not valid:continue
        # Return fresh moves only; chain must stop before a subsequent HOLD
        # (a held successor is NOT the next observed active piece).
        moves=node.moves
        for idx in range(len(moves)-1):
            if not node.holds[idx+1]:
                moves[idx].next_move=moves[idx+1]
            else:
                break
        return VerifiedSequence(moves,node.holds,node.holds[0],node.gross,
                                node.rating,explored,
                                len(node.holds)>1 and not node.holds[1])
    return None


def adopt_sequence(seq, incumbent, *, board, b2b=0, combo=-1,
                   ruleset='standard', combo_table='multiplier', rounding='auto'):
    """Restrict adoption to concrete attack advantage and no stack regression.

    `incumbent` is a (first, route, score) from the existing worker, or None.
    This does not use the uncalibrated score scales to claim a win. If the
    existing worker already has a good route, demand >=2 *model-generated*
    lines of provable two-lock improvement, without extra holes or height.
    """
    if seq is None or len(seq.moves)<2:return False
    first=seq.moves[0]
    _,src_h,src_holes,*_=board_metrics(board)
    _,new_h,new_holes,*_=board_metrics(first.board)
    if new_h>min(12,src_h+3) or new_holes>src_holes+1:return False
    if incumbent is None:return True
    old=incumbent[0]
    if old is None:return True
    _,old_h,old_holes,*_=board_metrics(old.board)
    if new_h>old_h+1 or new_holes>old_holes:return False
    def damage_path(moves):
        bb,co=b2b,combo;dmg=0
        for m in moves[:2]:
            est=attack_estimate(m,bb,co,ruleset,combo_table,rounding)
            dmg+=est.guaranteed;bb,co=est.next_b2b,est.next_combo
        return dmg,bb
    # A second-lock HOLD decision is a *future* action, not a preauthorised
    # speculative key press. The next turn will have to choose it again from
    # fresh vision. Do not replace the present Fast plan solely on such a
    # conditional future improvement.
    if len(seq.holds)>1 and seq.holds[1]:
        return False
    good,chain_good=damage_path(seq.moves)
    older,chain_old=damage_path(incumbent[1])
    # The new branch is a setup only if both physical locks are proven. Do not
    # trade an immediate high-confidence Quad/TSD for a speculative longer tail.
    a=attack_estimate(first,b2b,combo,ruleset,combo_table,rounding).guaranteed
    b=attack_estimate(old,b2b,combo,ruleset,combo_table,rounding).guaranteed
    # Production admission is intentionally stricter than the search score.
    # A future HOLD can be reevaluated after the next actual lock; it is not a
    # reason to give up the baseline's known immediate damage today. Avoid
    # model-only 1-line wins that reduce B2B or require a riskier setup.
    return bool(good>=older+3 and a>=b and chain_good>=chain_old and
                new_h<=old_h and new_holes<=old_holes)
