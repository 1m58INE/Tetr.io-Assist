"""Bounded, observed-next TSD construction for native Fast ATTACK.

Only positive actual SRS paths are used; never infer unseen NEXT or HOLD.
Executed only on healthy, no-Garbage boards with attack urgency and known T.
"""
from __future__ import annotations
import time


def tsd_finishes(board, deadline):
    from engine import tactical_moves, board_metrics, move_executable
    if time.perf_counter() >= deadline:
        return ()
    _, height, holes, *_ = board_metrics(board)
    if height >= 13 or holes > 4 or not holes:
        return ()
    # Bounded BFS. A 240-node limit can miss some real spins but must NEVER
    # return a T spin not actually reachable by the motion module.
    moves = tactical_moves(board, 'T', limit=320, max_depth=12, deadline=deadline)
    out = []
    for m in moves:
        if time.perf_counter() >= deadline:
            break
        if m.spin != 'full' or m.lines < 2:
            continue
        _, after_height, after_holes, *_ = board_metrics(m.board)
        if after_height >= 14 or after_holes > max(2,holes - 1):
            continue
        if move_executable(board, 'T', tuple(sorted(m.cells)), m.spin):
            out.append(m)
    return tuple(out)


def builder_candidates(board, moves, limit=2):
    """Diverse cavity candidates that first-ply attack scoring may discard."""
    from engine import board_metrics
    _, oldh, oldholes, *_ = board_metrics(board)
    if oldh >= 12 or oldholes > 2:
        return ()
    pool=[]; used_columns=set()
    for m in moves:
        if m.lines:
            continue
        _, h, holes, *_ = board_metrics(m.board)
        if h>min(12,oldh+3) or holes<=oldholes or holes>oldholes+2:
            continue
        column=min(x for x,y in m.cells)
        pool.append((holes,h,column,m))
    pool.sort(key=lambda e:(e[0],e[1],e[2]))
    chosen=[]
    for _holes,_h,x,m in pool:
        if x in used_columns:
            continue
        used_columns.add(x);chosen.append(m)
        if len(chosen) >= limit:break
    return tuple(chosen)


def efficiency_reward(move, urgency, source_board, incoming=0):
    """Calibrate Fast's arbitrary rewards toward actual gross attack yield.

    The legacy Fast relative rewards (double=5, triple=10, Quad=28)
    disproportionately choose low-efficiency ordinary Double/Triple clears
    over future difficult clears.  In the goal-driven, safe-only mode reduce
    only these two categories; do not invent attack from quiet placements.
    """
    from engine import fast_attack_reward
    base=fast_attack_reward(move)
    if (incoming or not urgency or not move.lines or move.spin or
            any(source_board[:9])):
        return base
    strength=min(1.0,max(0.0,float(urgency))/0.75)
    if move.lines==2:
        return base*(1.0-0.8*strength)
    if move.lines==3:
        return base*(1.0-0.5*strength)
    return base
