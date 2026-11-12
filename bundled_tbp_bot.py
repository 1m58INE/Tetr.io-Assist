"""Self-contained, OFFLINE Tetris Bot Protocol subprocess.

This is NOT Cold Clear 2. It uses the bundled native attack planner in a
separate, sandboxed-by-protocol process, with no Rust toolchain or downloads.
The independent process is a *usable* fallback while CC2 binaries are absent.
It has no game input, no window access, and never opens a network socket.
"""
from __future__ import annotations
import json
import sys

NAME = 'TETRIO Bundled Strategy'
PIECES = frozenset('IOTJLSZ')
ORIENTATIONS = ('north', 'east', 'south', 'west')
MAX_INPUT = 65536


def decode_snapshot(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get('type') != 'start':
        raise ValueError('not start')
    rows = snapshot.get('board')
    queue = snapshot.get('queue')
    if not isinstance(rows, list) or len(rows) != 40 or not isinstance(queue, list) or not (1 <= len(queue) <= 9):
        raise ValueError('invalid dimensions')
    if any(p not in PIECES for p in queue):
        raise ValueError('invalid queue')
    if any(not isinstance(row, list) or len(row) != 10 for row in rows):
        raise ValueError('invalid board')
    if any(cell is not None for row in rows[20:] for cell in row):
        raise ValueError('unsupported high stack')
    if any(cell is not None and (not isinstance(cell, str) or len(cell) > 12) for row in rows for cell in row):
        raise ValueError('untrusted cell content')
    board = tuple(sum(1 << x for x,c in enumerate(row) if c is not None) for row in rows[:20][::-1])
    hold = snapshot.get('hold')
    if hold is not None and hold not in PIECES: raise ValueError('invalid hold')
    allow_hold = snapshot.get('assist_hold_allowed', False)
    if type(allow_hold) is not bool: raise ValueError('invalid hold permission')
    b2b = snapshot.get('assist_b2b_count', int(bool(snapshot.get('back_to_back', False))))
    combo = snapshot.get('assist_combo', -1)
    incoming = snapshot.get('assist_incoming', 0)
    for n,lo,hi in ((b2b,0,1000),(combo,-1,1000),(incoming,0,20)):
        if type(n) is not int or not lo <= n <= hi: raise ValueError('invalid count')
    return board,queue[0],tuple(queue[1:]),hold,allow_hold,b2b,combo,incoming


def encode_move(move):
    from motion import SHAPES
    from tbp_bridge import ANCHORS
    s=SHAPES[move.piece][move.rotation]
    ax,ay=ANCHORS[move.piece][move.rotation]
    ox=min(x for x,y in move.cells)-min(x for x,y in s)
    oy=min(y for x,y in move.cells)-min(y for x,y in s)
    spin=move.spin if move.spin in ('full','mini') else 'none'
    return {'location':{'type':move.piece,'orientation':ORIENTATIONS[move.rotation],
                        'x':ox+ax,'y':19-(oy+ay)}, 'spin':spin}


def solve(snapshot):
    from engine import plan_move_pvp, plan_move, move_executable
    from motion import can_route
    board,piece,queue,hold,allow_hold,b2b,combo,incoming=decode_snapshot(snapshot)
    # The known queue comes only from the observed state; no unseen block is
    # supplied to the planner. Keep the subprocess bounded even if it fails.
    result=plan_move_pvp(board,piece,queue,hold,allow_hold,b2b,combo,
                         incoming,budget_ms=28.0,pieces_placed=20)
    if result is None or result.move is None:
        result=plan_move(board,piece,queue[0] if queue else None,hold,allow_hold,
                         mode='survival' if incoming >= 4 else 'attack',fast=True,
                         b2b=b2b,combo=combo)
    if result is None or result.move is None: return []
    move=result.move
    if result.use_hold and (not allow_hold or hold not in PIECES and (not queue or move.piece!=queue[0])):
        return []
    if not can_route(board,move.piece,tuple(sorted(move.cells)),move.spin):return []
    return [encode_move(move)]


def run_protocol(stdin=None,stdout=None):
    source=stdin or sys.stdin.buffer
    target=stdout or sys.stdout
    target.write(json.dumps({'type':'info','name':NAME,'version':'1.1.8','features':[]})+'\n')
    target.flush()
    current=None
    cached=None
    for raw in source:
        if len(raw)>MAX_INPUT:
            break  # fail closed on oversized input
        try:
            msg=json.loads(raw)
            if not isinstance(msg,dict):break
            op=msg.get('type')
            if op=='rules':
                current=None;cached=None;response={'type':'ready'}
            elif op=='stop':
                current=None;cached=None;continue
            elif op=='start':
                decode_snapshot(msg)  # no reuse of corrupted board snapshots
                current=msg;cached=None;continue
            elif op=='suggest':
                if current is None: response={'type':'suggestion','moves':[]}
                else:
                    if cached is None:
                        try:cached=solve(current)
                        except (ValueError,RuntimeError,IndexError,KeyError,TypeError,OverflowError):cached=[]
                    response={'type':'suggestion','moves':cached}
            else:continue
            target.write(json.dumps(response,separators=(',',':'))+'\n')
            target.flush()
        except (ValueError,TypeError,UnicodeError):
            break

if __name__=='__main__':
    run_protocol()
