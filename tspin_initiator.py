"""Worker-only, observed NEXT=T T-spin Double initiator.

This complements the generic beam: it reserves a small time slice for a
reachable double-spin *finisher* instead of pruning its quieter builder.
No hidden previews, no keyboard input, and no board inference.
"""
import time


def find_verified_tsd_builder(board, piece, next_piece, deadline, max_probes=3):
    if next_piece!='T' or time.perf_counter()>=deadline:
        return None
    from engine import placements, board_metrics, is_difficult
    from tspin_factory import reachable_tspin_finishes
    from motion import can_route
    _h,source_height,source_holes,*_=board_metrics(board)
    if source_height>11 or source_holes>3:
        return None
    possibilities=[]
    for m in placements(board,piece):
        if m.lines or is_difficult(m):
            continue
        _,height,holes,*_=board_metrics(m.board)
        # The T slot is permitted to introduce at most two temporary holes,
        # but never bury the entire stack or require an unsafe top-out.
        if height>min(12,source_height+3) or holes>source_holes+2:
            continue
        if holes==0:continue  # a reachable T-spin cavity must exist
        possibilities.append((holes, height, m))
    # Enumerate low-risk cavitated candidates first rather than high single-ply
    # beam-score candidates (which usually avoid the very T slot we need).
    possibilities.sort(key=lambda x:(x[0],x[1],min(c[0] for c in x[2].cells)))
    for _holes,_height,builder in possibilities[:max_probes]:
        if time.perf_counter()>=deadline:break
        options=reachable_tspin_finishes(builder.board,deadline,limit=240,depth=11)
        for finish in options:
            if finish.spin!='full' or finish.lines<2:continue
            _,final_height,final_holes,*_=board_metrics(finish.board)
            if final_height>min(12,source_height+3) or final_holes>source_holes:
                continue
            if not can_route(board,piece,tuple(sorted(builder.cells)),builder.spin):
                continue
            # The independently validated SRS path is part of the follow-up
            # and will be rechecked against the real next spawn by AutoPlayer.
            builder.next_move=finish
            return builder
    # Fallback: geometric cavity diversity, then independently verified
    # SRS TSD finisher. This never extends the calling Deep deadline, and is
    # inactive whenever the observed NEXT is not T or the board is unsafe.
    if time.perf_counter()<deadline and max_probes>0:
        from attack_structure11113 import builder_candidates, tsd_finishes
        for builder in builder_candidates(board, list(placements(board,piece)), limit=min(4,max_probes)):
            if time.perf_counter()>=deadline:break
            if not can_route(board,piece,tuple(sorted(builder.cells)),builder.spin):continue
            for finish in tsd_finishes(builder.board,deadline):
                _,height,holes,*_=board_metrics(finish.board)
                if (finish.lines==2 and finish.spin=='full' and
                        height<=min(12,source_height+3) and holes<=source_holes):
                    builder.next_move=finish
                    return builder
    return None


def find_verified_tsd_pipeline(board, piece, queue, deadline, max_first=14,
                               max_probes=4):
    """Bounded three-move setup: CURRENT -> observed NEXT1 -> NEXT2 TSD.

    Every placement is geometrically real, and all three execution routes
    are independently checked in the Deep worker. This is an optional
    opportunity search: no inferred queue items, speculative hold or runtime
    input is involved. Finishes only when the T spin is FULL and clears two.
    """
    if (len(queue)<2 or queue[1]!='T' or piece not in 'IOTJLSZ' or
            queue[0] not in 'IOTJLSZ' or time.perf_counter() >= deadline):
        return None
    from engine import placements, board_metrics
    from motion import can_route
    _,h0,holes0,*_=board_metrics(board)
    if h0>11 or holes0>2:return None
    choices=[]
    for first in placements(board,piece):
        if first.lines:continue
        _,h1,holes1,*_=board_metrics(first.board)
        if h1>min(12,h0+3) or holes1>holes0+1:continue
        # Center / edge diversity is more valuable than another copy of
        # a near-identical branch in a short fixed time budget.
        col=min(x for x,y in first.cells)
        choices.append((holes1,h1,col,first))
    # Stable deterministic order. Reserve at least one candidate per column
    # so a narrow beam does not discard every T-slot-enabling side placement.
    choices.sort(key=lambda entry:(entry[0],entry[1],entry[2]))
    columns={}; extras=[]
    for record in choices:
        if record[2] not in columns:columns[record[2]]=record
        else:extras.append(record)
    shortlist=list(columns.values())+extras
    examined=0
    for _holes,_height,_col,first in shortlist:
        if examined >= max(1,min(int(max_first),45)) or time.perf_counter()>=deadline:break
        examined+=1
        if not can_route(board,piece,tuple(sorted(first.cells)),first.spin):continue
        bridge=find_verified_tsd_builder(first.board,queue[0],'T',deadline,
                                         max_probes=max_probes)
        if bridge is None:continue
        finisher=bridge.next_move
        if (finisher is not None and finisher.spin=='full' and finisher.lines==2
                and can_route(first.board,queue[0],tuple(sorted(bridge.cells)),bridge.spin)
                and can_route(bridge.board,'T',tuple(sorted(finisher.cells)),finisher.spin)):
            first.next_move=bridge
            return first
    return None
