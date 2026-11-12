"""Evidence-ranked post-hard-drop transitions (no input or speculative locks).

The board in a tracking observation is independently reconstructed from actual
pixels.  This module never credits a lock merely because a gap or timer elapsed.
A verified garbage rise of the *source* board means a live piece can still be
present after an attempted hard drop: replan it, do not repeat the stale drop.
"""
from dataclasses import dataclass
from garbage_transition import verified_rise

@dataclass(frozen=True)
class PostDrop:
    kind: str   # confirmed_lock, risen_active, fresh_board, wait
    inserted: int = 0


def classify_post_drop(source, predicted, old_piece, old_cells, observed,
                       new_piece, new_cells, *, stable_frames, elapsed_ms,
                       saw_gap=False):
    """Classify only full, independently observed grid/piece states.

    No timer alone turns a predicted lock into a confirmed lock.  A different
    new piece or a witnessed recognition gap can justify *uncredited* resync on
    a changed stack even if the next active piece fell below the spawn rows.
    """
    if (len(source)!=20 or len(predicted)!=20 or len(observed)!=20
            or not isinstance(new_piece,str) or new_piece not in 'IOTJLSZ'
            or len(new_cells)!=4):
        return PostDrop('wait')
    if (any(not isinstance(row,int) or not 0<=row<=1023 for row in observed)
            or len(set(new_cells))!=4):
        return PostDrop('wait')
    from engine import identify
    if (identify(new_cells)!=new_piece or
            any(not 0<=x<10 or not 0<=y<20 or observed[y]&(1<<x)
                for x,y in new_cells)):
        return PostDrop('wait')
    changed=(observed,new_piece,tuple(sorted(new_cells))) != (
        source,old_piece,tuple(sorted(old_cells)))
    expected_rise=verified_rise(predicted,observed)
    if expected_rise is not None and (changed or saw_gap):
        # Perfect clear leaves exactly the original empty settled board.
        # A moved same-kind active piece is NOT proof that hard drop locked.
        # Without a different active kind this observation is ambiguous.
        if predicted==source and observed==source and new_piece==old_piece:
            return PostDrop('wait')
        if stable_frames>=2:
            return PostDrop('confirmed_lock',expected_rise)
    # The expected lock did NOT appear. A visible, exact garbage rise of the
    # source is positive evidence that the old piece needs a new plan.
    if new_piece==old_piece and stable_frames>=2:
        live_rise=verified_rise(source,observed)
        if live_rise is not None and live_rise>0:
            return PostDrop('risen_active',live_rise)
    # A different observed active tetromino on a changed settled board is
    # independently useful evidence for a NEW plan. Do not wait for the old
    # 90ms timeout merely to discard an obsolete hard-drop target. This is
    # NOT lock acknowledgement: missing predicted cells are not credited.
    if (stable_frames >= 2 and elapsed_ms >= 16 and
            new_piece != old_piece and observed != source):
        return PostDrop('fresh_board')
    if stable_frames>=3 and elapsed_ms>=90 and observed!=source:
        # Same-piece transitions are ambiguous: keep the old 3-frame and
        # 90ms gates before an *uncredited* restart.
        return PostDrop('fresh_board')
    return PostDrop('wait')
