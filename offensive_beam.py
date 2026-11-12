"""Original, bounded observed-queue attack beam (no external code copied).

A secondary Deep-only policy, with explicit chain-aware board shaping. Never
reads unseen pieces. Every selected placement is independently route checked.
"""
import time
from functools import lru_cache
from engine import placements, tactical_moves, board_metrics, observed_queue_prefix, is_difficult, _deep_obsolete
from motion import can_route_until
from combo_rules import attack_estimate

@lru_cache(maxsize=4096)
def attack_structure(board):
    heights, highest, holes, covered, bump, wells, transitions = board_metrics(board)
    # Clean 9-column plateau, with an accessible far-right I well.
    plateau=heights[:9]
    shallow=min(plateau)
    flat=sum(abs(plateau[i]-plateau[i+1]) for i in range(8))
    well=max(0,shallow-heights[9])
    # The shape score is a *terminal state potential*, not a bonus every ply.
    # It must never rationalize dangerous stacks or inaccessible holes.
    safety=-holes*10.7-covered*.16-flat*.68-highest*.85
    stack=-sum(heights)*.23
    preparation=min(well,8)*3.2 if holes<=1 and highest<=12 else 0.0
    if highest>=13:safety-=(highest-12)**2*3.3
    if highest>=17:safety-=180
    return safety+stack+preparation, highest, holes


def plan_attack_beam(board,piece,queue=(),b2b=0,combo=-1,deadline=None,width=8,max_nodes=1000,
                     ruleset='standard',combo_table='multiplier',rounding='auto',revision=0,
                     incoming=0):
    """Return (first move, path, objective) or None under hard search caps.

    No HOLD or opponent model here; integration only permits direct-queue use.
    This avoids silently reordering the preview or inventing a held piece.
    """
    if deadline is None:deadline=time.perf_counter()+.020
    if _deep_obsolete(revision):return None
    sequence=(piece,)+observed_queue_prefix(queue,3)
    width=max(2,min(int(width),16));max_nodes=max(24,min(int(max_nodes),2000))
    # board, b2b, combo, earned_attack, route, objective, remaining incoming
    # The incoming state is tracked explicitly only in counter-offense mode:
    # generated attacks first cancel queued garbage, rather than pretending
    # every generated line reaches the opponent.
    incoming=max(0,min(16,int(incoming)))
    frontier=[(board,b2b,combo,0.0,(),0.0,incoming)]
    source_height=attack_structure(board)[1]
    explored=0
    final=[]
    for depth,p in enumerate(sequence):
        candidates={}
        for current,bb,co,earned,path,prior,remaining in frontier:
            if _deep_obsolete(revision):return None
            if explored>=max_nodes or time.perf_counter()>=deadline:break
            moves=list(placements(current,p))
            # With an established chain, non-T immobile mini clears can
            # preserve B2B without a speculative T queue.  Search only the
            # FIRST surviving branch per ply and under the SAME deadline.
            # T's proven finishing pipeline retains priority.  I is Deep-only;
            # O is excluded because its orientation is visually ambiguous.
            time_left=deadline-time.perf_counter()
            height=attack_structure(current)[1]
            scan_third=(depth==2 and time_left>=.003
                        and 0<attack_structure(current)[2]<=3
                        and height<=12)
            seek_t=(p=='T' and (depth<=1 or scan_third))
            seek_mini=(p in 'IJLSZ' and bb>0 and depth<=1
                       and height<=12 and time_left>=.0025)
            if (path==frontier[0][4] and any(current) and (seek_t or seek_mini)):
                tactical=tactical_moves(current,p,
                                       limit=110 if seek_mini else (120 if depth==2 else 195),
                                       max_depth=8 if seek_mini else 9,
                                       deadline=deadline,include_i=(p=='I'))
                positions={tuple(sorted(m.cells)):m for m in moves}
                for m in tactical:
                    if m.spin and m.lines:positions[tuple(sorted(m.cells))]=m
                moves=list(positions.values())
            for m in moves:
                explored+=1
                if (explored&31)==0 and _deep_obsolete(revision):return None
                if explored>max_nodes or (explored&31)==0 and time.perf_counter()>=deadline:break
                potential,height,holes=attack_structure(m.board)
                if height>=19 or holes>=7:continue
                # Do not repeatedly reward quiet speculative buildup. A
                # real incoming attack may raise the stack before the next
                # predicted finisher appears.
                if not m.lines and height>min(14,source_height+3):continue
                if depth>=1 and not m.lines and height>=12:continue
                source_holes=attack_structure(current)[2]
                estimate=attack_estimate(m,bb,co,ruleset,combo_table,rounding)
                # Attack is earned on the path; a quiet placement never earns B2B.
                # High chains have greater preservation value but survival wins.
                gained=estimate.guaranteed
                pending=remaining
                if incoming:
                    from combat_rules import exchange
                    exchanged=exchange(gained,remaining,early_rule=False)
                    pending=exchanged.remaining
                    # Cancellation is defensive value, not outgoing attack.
                    # Reward it while penalizing projected top-out risk.
                    gained=exchanged.sent+1.5*exchanged.cancelled
                    if height+pending>=19:continue
                # A T-spin Mini creating multiple *new* covered holes is not a
                # sustainable B2B factory. It often causes the next safe move
                # to break the chain; retain real spins without this penalty.
                if bb>=2 and m.spin=='mini' and holes>source_holes+1:
                    gained-=min(20.0,(holes-source_holes-1)*8.0)
                if bb and m.lines and not is_difficult(m):
                    gained-=7.0+min(bb,10)*1.65 if height<=13 else 0
                if is_difficult(m):gained+=2.0+min(bb,12)*.65
                if not m.lines and bb and height<13:gained+=.10
                if not bb and m.lines in (1,2) and not m.spin and height<=11:gained-=2.5
                total=earned+gained
                # Keep one terminal potential, not a repeated board penalty.
                objective=total*4.9+potential+min(estimate.next_b2b,12)*1.2
                if incoming:
                    objective-=pending*5.0+max(0,height+pending-14)**2*8.0
                if depth==0 and not path and not can_route_until(current,p,tuple(sorted(m.cells)),m.spin,deadline=deadline):continue
                key=(m.board,estimate.next_b2b,estimate.next_combo,pending)
                result=(m.board,estimate.next_b2b,estimate.next_combo,total,path+(m,),objective,pending)
                if key not in candidates or result[5]>candidates[key][5]:candidates[key]=result
        if not candidates:break
        ordered=sorted(candidates.values(),key=lambda item:item[5],reverse=True)
        # Preserve option diversity and a Tetris-compatible well structure.
        selected=ordered[:max(2,width-2)]
        for predicate in (
            lambda s:s[1]>b2b,
            lambda s:attack_structure(s[0])[2]==0,
            lambda s:s[0][-1] != 0 and s[0][-1].bit_count()==9,
        ):
            extra=next((x for x in ordered if predicate(x) and x not in selected),None)
            if extra is not None:selected.append(extra)
        frontier=selected[:width]
        final=ordered[:width]
        if explored>=max_nodes or time.perf_counter()>=deadline:break
    if not final:return None
    # Every optimistic future attack must be executable, not only the
    # first and second move. Otherwise an unreachable third-ply T-spin
    # can inflate the score and cause a bad first move in real play.
    # Route checks stay in the Deep worker and are bound by the deadline.
    for state in final:
        path=state[4]
        prior=board
        valid=True
        for move in path:
            if time.perf_counter() >= deadline or not can_route_until(
                    prior,move.piece,tuple(sorted(move.cells)),move.spin,deadline=deadline):
                valid=False;break
            prior=move.board
        if not valid:continue
        first=path[0]
        if len(path)>=2:first.next_move=path[1]
        first.score=state[5]
        return first,path,state[5]
    return None


def beam_continuation_beneficial(previous, candidate, source_board, b2b=0,
                                 combo=-1, incoming=0, ruleset='standard',
                                 combo_table='multiplier', rounding='auto'):
    """Cheap guarded prefetch adoption after worker-verified route.

    Called only after an exact predicted lock board/current piece/queue match.
    No reachability BFS is performed on the GUI thread. This only compares two
    moves for the *same* observed next piece; it cannot infer hidden pieces.
    """
    if candidate is None or previous is None or incoming:
        return False
    if candidate.piece!=previous.piece:
        return False
    _, src_height, src_holes=attack_structure(source_board)
    if src_height>=11 or src_holes>=3:
        return False
    old_shape, old_height, old_holes=attack_structure(previous.board)
    new_shape, new_height, new_holes=attack_structure(candidate.board)
    if (new_height>12 or new_height>old_height+2 or new_holes>old_holes
            or new_holes>src_holes+1):
        return False
    old=attack_estimate(previous,b2b,combo,ruleset,combo_table,rounding)
    new=attack_estimate(candidate,b2b,combo,ruleset,rounding=rounding,combo_table=combo_table)
    if new.guaranteed<old.guaranteed:
        return False
    if b2b and old.next_b2b==0 and new.next_b2b>0:
        return True
    return new_shape>=old_shape+2.0
