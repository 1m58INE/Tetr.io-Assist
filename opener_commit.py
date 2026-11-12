"""Pre-input, observation-bound opener/6-3 commit for the native Fast worker.

No GUI, external process, Windows input, or hidden bag reads.  A verified
reference can become the FIRST executable target instead of racing against a
late Deep proposal.  Never persists a recipe across a changed board.
"""
from __future__ import annotations


def opener_start_possible(board, piece, next_piece, *, canonical=False, pco=False):
    """Cheap initial queue hint for the single bounded NEXT2 capture window."""
    if any(board):
        return False
    return bool((canonical and piece=='O' and next_piece=='I') or
                (pco and piece=='Z' and next_piece=='I'))


def _safe_reference(board, piece, queue, hold, allow_hold, b2b, combo,
                    incoming, placed, *, canonical=False, pco=False,
                    ruleset='standard', combo_table='multiplier', rounding='auto'):
    from canonical_openers import choose_scaffold
    from engine import Plan, guard_fast_stack, board_metrics
    step=choose_scaffold(board,piece,queue,placed=placed,incoming=incoming,
                         enabled=canonical,pco_enabled=pco)
    if step is None:
        return None
    # The source is EXACT, and choose_scaffold independently verifies the
    # current SRS path.  Do not authorize an unsafe high stack for a template.
    _,height,_,*_=board_metrics(step.move.board)
    if height>10:
        return None
    nxt=queue[0] if queue else None
    plan=Plan(step.move,board,piece,False,False,nxt,hold,b2b,combo)
    plan.continuation_source='canonical_fast'
    plan.opening_profile=step.name
    plan.opening_verified_steps=1
    guarded=guard_fast_stack(plan,board,piece,nxt,hold,allow_hold,b2b,combo,
                              incoming,ruleset,combo_table,rounding)
    if guarded is not plan:
        return None
    return plan


def fast_opening_plan(board, piece, queue, *, hold='?', allow_hold=True,
                      b2b=0, combo=-1, incoming=0, placed=20,
                      session_ok=False, canonical=False, pco=False,
                      ruleset='standard',combo_table='multiplier',rounding='auto'):
    """Use a physically checked first target, with no Fast/Deep adoption race.

    Called ONLY on the pre-input Fast executor.  NEXT entries must be actually
    observed; unknown is never filled with a hypothetical 7-bag sequence.
    """
    if (not session_ok or incoming or not 0<=placed<15 or
            not (canonical or pco)):
        return None
    if not isinstance(board,tuple) or len(board)!=20 or piece not in 'IOTSZJL':
        return None
    from engine import observed_queue_prefix
    known=observed_queue_prefix(queue)
    return _safe_reference(board,piece,known,hold,allow_hold,b2b,combo,
                           incoming,placed,canonical=canonical,pco=pco,
                           ruleset=ruleset,combo_table=combo_table,rounding=rounding)


def fast_stacker63_plan(fast,board,piece,*,hold='?',allow_hold=True,
                        b2b=0,combo=-1,incoming=0,placed=20,
                        session_ok=False):
    """Only replace a weak non-clear Fast move with a safer x=6-well move."""
    if (not session_ok or incoming or not 0<=placed<28 or
            fast is None or fast.use_hold or fast.move is None or
            fast.move.lines>=2 or fast.move.spin or
            not isinstance(board,tuple) or len(board)!=20):
        return fast
    from canonical_openers import choose_63, score_63
    from engine import Plan, board_metrics
    if board_metrics(board)[1]>=11:
        return fast
    candidate=choose_63(board,piece,incoming=incoming,max_candidates=36)
    if candidate is None:
        return fast
    move,quality=candidate
    old_quality=score_63(board,fast.move)
    if old_quality is not None and quality<=old_quality+2.0:
        return fast
    if (tuple(sorted(fast.move.cells))==tuple(sorted(move.cells))
            and fast.move.spin==move.spin):
        return fast
    # 6-3 is a style, not a hard request to bury new holes or to abandon
    # an immediate defensive clear. The candidate was independently routed.
    plan=Plan(move,board,piece,False,False,fast.observed_next,hold,b2b,combo)
    plan.continuation_source='stack63_fast'
    plan.opening_profile='SIX_THREE'
    plan.opening_verified_steps=1
    return plan


def observable_reference_stage(board,piece,queue,placed,*,canonical=False,pco=False):
    """Pure cheap prefetch veto: a proven reference may need fresh Fast commit.

    Never returns a move. This avoids the current-turn speculative NEXT
    prefetch outranking a known full opener sequence in turbo mode.
    """
    from canonical_openers import _cheap_book_fields
    from engine import observed_queue_prefix
    seen=observed_queue_prefix(queue)
    for family in (('TKI3_FLAT','DT_CANNON_FULL') if canonical else ()) + \
                  (('PCO_TEN',) if pco else ()):
        for mirrored in (False,True):
            rows=_cheap_book_fields(family,mirrored)
            if not 0<=placed<len(rows):continue
            src,p,*_=rows[placed]
            if src!=tuple(board) or p!=piece:continue
            if placed==0:
                # At stage 0 the ROI worker may still be observing NEXT2;
                # postpone speculative prefetch even with only NEXT1.
                if seen and rows[1][1]==seen[0]:return True
            elif not seen or placed>=len(rows)-1 or rows[placed+1][1]==seen[0]:
                return True
    return False


def confirmed_reference_start(board,piece,queue,*,canonical=False,pco=False):
    """Require CURRENT plus two *observed* NEXT pieces, exact new game."""
    from canonical_openers import _cheap_book_fields
    from engine import observed_queue_prefix
    seen=observed_queue_prefix(queue)
    if any(board) or len(seen)<2:
        return False
    for family in (('TKI3_FLAT','DT_CANNON_FULL') if canonical else ()) + \
                  (('PCO_TEN',) if pco else ()):
        for mirrored in (False,True):
            rows=_cheap_book_fields(family,mirrored)
            if rows[0][1]==piece and (rows[1][1],rows[2][1])==seen[:2]:
                return True
    return False


def confirmed_opener_continuation(board,piece,placed,confirmed,last_profile,*,
                                  canonical=False,pco=False):
    """Allow a *previously confirmed* exact recipe to finish after burst cap.

    The 12-second / 12-generated-line Opening Burst limit must not reopen
    an unstarted recipe on a random board. A physical Garbage rise/resync is
    checked separately by the caller and always cancels this eligibility.
    This is only a cheap candidate gate; fast_opening_plan independently
    rechecks CURRENT, observed NEXT, board and SRS route.
    """
    if not (confirmed>0 and placed>=1 and placed-1<=confirmed<=placed and
            isinstance(last_profile,str) and last_profile):
        return False
    from canonical_openers import _cheap_book_fields
    families=(('TKI3_FLAT','DT_CANNON_FULL') if canonical else ()) + \
             (('PCO_TEN',) if pco else ())
    for family in families:
        for mirror in (False,True):
            name=family+('_M' if mirror else '')
            if last_profile!=name:continue
            rows=_cheap_book_fields(family,mirror)
            if (0<placed<len(rows) and rows[placed][0]==tuple(board)
                    and rows[placed][1]==piece):
                return True
    return False
