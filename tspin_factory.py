"""BUDGETED T-spin setup factory from *observed* NEXT information.

A setup is counted only when placing the builder creates a verified reachable
T-spin line clear on the following T.  The target is never fabricated from
corner heuristics.  This module runs in the Deep worker, never in GUI/Fast.
"""
import time

def reachable_tspin_finishes(board, deadline=None, limit=650, depth=14):
    """Return only reachable T-spins with a real line clear and input route."""
    from engine import tactical_moves, board_metrics, execution_cost
    if deadline is not None and time.perf_counter()>=deadline:
        return ()
    # Generation uses SRS 90-degree kicks and bounded, finite BFS.
    moves=tactical_moves(board,'T',limit=limit,max_depth=depth,deadline=deadline)
    results=[]
    for m in moves:
        if deadline is not None and time.perf_counter()>=deadline:
            break
        if not (m.spin and m.lines):
            continue
        _,height,holes,*_=board_metrics(m.board)
        if height>=17:continue
        # A spin label produced by the geometric search must also be routable
        # from ordinary spawn.  This gate is deliberately worker-only.
        effort=execution_cost(board,'T',tuple(sorted(m.cells)),m.spin)
        if effort>=999:continue
        value=(12.0 if m.spin=='full' else 5.0)+m.lines*5.0 - holes*1.4 - max(0,height-10)*2.0 - effort*.1
        results.append((value,m))
    results.sort(key=lambda pair: pair[0],reverse=True)
    return tuple(m for _,m in results[:4])


def factory_portfolio(board, entries, known_next, deadline, max_probes=5):
    """Find genuine first-piece T-spin setups before beam cuts them off.

    `entries` are already-scored first-ply records; returns records whose
    resulting board has a *verified* T-spin using the observed NEXT=T.
    """
    if known_next!='T' or time.perf_counter()>=deadline:
        return ()
    from engine import board_metrics
    found=[];seen=set();probed=0
    for entry in entries:
        if time.perf_counter()>=deadline or probed>=max_probes:break
        m=entry[1]
        _,height,holes,*_=board_metrics(m.board)
        if height>13 or holes>3:continue
        if m.board in seen:continue
        seen.add(m.board);probed+=1
        spins=reachable_tspin_finishes(m.board,deadline,limit=340,depth=11)
        if spins:
            chosen=spins[0]
            # The preview is an advisory: the controller will revalidate it
            # against the observed subsequent board, active piece and input route.
            found.append((entry, chosen))
    return tuple(found)
