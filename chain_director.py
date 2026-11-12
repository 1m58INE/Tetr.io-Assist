"""Worker-only guarded B2B preservation and verified Mini rescue.

This module doesn't change server counters, infer unseen pieces, or push keys.
A proven Mini line clear extends the local chain; a quiet placement preserves
it without incrementing it. Neither guarantees game-server adjudication.
"""
from __future__ import annotations

def preserve_b2b(plan, board, piece, queue=(), b2b=0, incoming=0,
                 deadline=None, revision=0):
    if plan is None or plan.move is None or b2b<=0 or incoming:
        return plan
    if plan.use_hold or plan.move.piece!=piece:
        return plan
    m=plan.move
    if not m.lines or m.lines==4 or m.spin or not any(m.board):
        return plan
    from engine import (board_metrics, combat_profile, placements, is_difficult,
                        pvp_board_value, move_executable, _deep_obsolete)
    from offensive_beam import attack_structure
    import time
    if combat_profile(board,incoming)[0] not in ('ATTACK','COUNTER'):return plan
    _,old_height,old_holes,*_=board_metrics(board)
    _,target_height,target_holes,*_=board_metrics(m.board)
    if old_height>=12 or old_holes>=6:return plan
    # Try a genuinely executed difficult spin before choosing a no-clear
    # preservation. This runs only inside the existing Deep B2B rescue path
    # with a live deadline; it never fabricates attack from a cosmetic twist.
    if piece in 'TIJLSZ' and (deadline is None or time.perf_counter()+.001<deadline):
        from engine import tactical_moves, is_difficult
        from combo_rules import attack_estimate
        spins=[]
        for candidate in tactical_moves(board,piece,limit=240,max_depth=10,
                                        deadline=deadline,include_i=(piece=='I')):
            if deadline is not None and time.perf_counter()>=deadline:return plan
            if not (candidate.spin=='mini' or candidate.spin=='full') or not is_difficult(candidate):
                continue
            _,height,holes,*_=board_metrics(candidate.board)
            if (height>min(12,max(old_height+2,target_height+1)) or
                    holes>min(5,max(old_holes,target_holes)+1)):
                continue
            estimate=attack_estimate(candidate,b2b)
            if estimate.next_b2b<=b2b:continue
            spins.append((estimate.guaranteed*2.0+pvp_board_value(candidate.board),candidate))
        for _,candidate in sorted(spins,key=lambda pair:pair[0],reverse=True)[:4]:
            if deadline is not None and time.perf_counter()>=deadline:return plan
            if move_executable(board,piece,tuple(sorted(candidate.cells)),candidate.spin):
                candidate.score=m.score
                candidate.next_move=None
                plan.move=candidate
                plan.prepared_safe=False
                plan.continuation_source='all_mini_b2b_rescue'
                return plan
    # Any potential optimization must be strictly subordinate to survival.
    candidates=[]
    for candidate in placements(board,piece):
        if (deadline is not None and time.perf_counter()>=deadline) or _deep_obsolete(revision):
            return plan
        if candidate.lines:continue
        _,height,holes,*_=board_metrics(candidate.board)
        if (height>min(12,max(old_height+2,target_height+2)) or
                holes>min(5,max(old_holes,target_holes))):continue
        structure,_,_=attack_structure(candidate.board)
        score=(pvp_board_value(candidate.board)+structure*.10-
               abs(height-target_height)*.6)
        # If NEXT is an I, prefer a healthy open quad well, never a fake clear.
        if queue and queue[0]=='I':
            from opening import quad_well_potential
            score+=quad_well_potential(candidate.board,queue)*.25
        candidates.append((score,candidate))
    for _,candidate in sorted(candidates,key=lambda pair:pair[0],reverse=True)[:5]:
        if deadline is not None and time.perf_counter()>=deadline:return plan
        if move_executable(board,piece,tuple(sorted(candidate.cells)),candidate.spin):
            candidate.score=m.score
            candidate.next_move=None  # unrelated stale predictions are invalid
            plan.move=candidate
            plan.prepared_safe=False
            plan.continuation_source='b2b_guard'
            return plan
    return plan


def fast_chain_penalty(move,b2b,source_board):
    """Cheap soft preference for the normal Fast fallback; no no-clear freezes."""
    if b2b<=0 or not move.lines or move.spin or move.lines==4 or not any(move.board):
        return 0.0
    from engine import board_metrics
    _,old_height,old_holes,*_=board_metrics(source_board)
    _,new_height,new_holes,*_=board_metrics(move.board)
    if old_height>=12 or new_height>=12 or new_holes>old_holes:
        return 0.0
    return -min(45.0,15.0+2.3*b2b)
