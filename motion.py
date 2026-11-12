"""Bounded SRS+ paths and conservative TETR.IO All-Mini+ spin recognition.

180-degree rotations use six documented historic T/J/L kick tests, but I/S/Z
180 orientations are excluded from the live route generator because the visual
acknowledgment is ambiguous. This is not proof of current server/room parity.
Partial soft drops remain unsupported. Screen-y increases downward.
"""
from collections import deque
import time
from engine import BASE, norm

KICKS = {
 (0,1): ((0,0),(-1,0),(-1,-1),(0,2),(-1,2)),
 (1,0): ((0,0),(1,0),(1,1),(0,-2),(1,-2)),
 (1,2): ((0,0),(1,0),(1,1),(0,-2),(1,-2)),
 (2,1): ((0,0),(-1,0),(-1,-1),(0,2),(-1,2)),
 (2,3): ((0,0),(1,0),(1,-1),(0,2),(1,2)),
 (3,2): ((0,0),(-1,0),(-1,1),(0,-2),(-1,-2)),
 (3,0): ((0,0),(-1,0),(-1,1),(0,-2),(-1,-2)),
 (0,3): ((0,0),(1,0),(1,-1),(0,2),(1,2)),
}
# I 90-degree kicks: SRS+ mirrors the left-side transitions of standard SRS
# horizontally from their right-side counterparts (screen-y grows downward).
# Source for SRS+ symmetry: https://tetris.wiki/TETR.IO#SRS.2B
# Guideline right-side kick entries: https://tetris.wiki/SRS
# Do not confuse with 180 kicks or JLSTZ kicks.
I_KICKS = {
 (0,1): ((0,0),(-2,0),(1,0),(-2,1),(1,-2)),
 (1,0): ((0,0),(2,0),(-1,0),(2,-1),(-1,2)),
 (1,2): ((0,0),(-1,0),(2,0),(-1,-2),(2,1)),
 (2,1): ((0,0),(1,0),(-2,0),(1,2),(-2,-1)),
 (2,3): ((0,0),(2,0),(-1,0),(2,-1),(-1,2)),
 (3,2): ((0,0),(-2,0),(1,0),(-2,1),(1,-2)),
 (3,0): ((0,0),(1,0),(-2,0),(1,2),(-2,-1)),
 (0,3): ((0,0),(-1,0),(2,0),(-1,-2),(2,1)),
}
# Default SRS+ retains the four right-side transitions (R = 1), and
# mirrors them for the corresponding left-side rotations (L = 3).
# Build rather than duplicate the table to guarantee horizontal symmetry.
for _left, _right in (((0,3),(0,1)), ((3,0),(1,0)),
                      ((2,3),(2,1)), ((3,2),(1,2))):
    I_KICKS[_left] = tuple((-dx,dy) for dx,dy in I_KICKS[_right])

# Source: sfinder SRSPlus 180 reference (historical reconstruction; NOT a
# server-level or modded-room parity guarantee). Source uses y-up; invert dy.
# These are pivot offsets for J/L/T, NOT I offsets. For T/J/L our SHAPES
# rotation pivot is at (1,1), so the vectors apply without translation.
# 180 must be a SINGLE bound key; it is not two 90-degree operations.
ROT180_KICKS = {
    0: ((0,0), (0,-1), (1,-1), (-1,-1), (1,0), (-1,0)),
    1: ((0,0), (1,0), (1,-2), (1,-1), (0,-2), (0,-1)),
    2: ((0,0), (0,1), (-1,1), (1,1), (-1,0), (1,0)),
    3: ((0,0), (-1,0), (-1,-2), (-1,-1), (0,-2), (0,-1)),
}

# A no-kick half turn changes screen orientation for T/J/L, which can be
# acknowledged by vision. For I/S/Z it is visually ambiguous: 0 and 2 share
# the same normalized occupied cells, likewise 1 and 3. The geometry helper
# step() still models A for compatibility, but live candidate routes must
# never send an unprovable 180 for these pieces.
VISIBLE_180_PIECES = frozenset(('T','J','L'))

def route_actions(piece):
    return ('left','right','cw','ccw','soft','rot180') if piece in VISIBLE_180_PIECES else ('left','right','cw','ccw','soft')

SHAPES = {}
for p, base in BASE.items():
    shape = tuple((x,y+1) for x,y in base) if p == 'I' else base
    shapes = []
    for _ in range(4):
        shapes.append(shape)
        size = 3 if p == 'I' else 2
        shape = shape if p == 'O' else tuple((size-y,x) for x,y in shape)
    SHAPES[p] = tuple(shapes)

# Each orientation occupies 1-4 row masks. The motion search calls fits()
# repeatedly while following SRS/SRS+ kicks, so check 2-3 rows instead of
# allocating/repeating four individual-cell tests. Board masks remain 10-bit.
_SHAPE_ROWS = {}
for _piece, _states in SHAPES.items():
    for _r, _shape in enumerate(_states):
        _rows = {}
        for _dx, _dy in _shape:
            _rows[_dy] = _rows.get(_dy, 0) | (1 << _dx)
        _SHAPE_ROWS[_piece, _r] = (
            min(x for x,y in _shape), max(x for x,y in _shape),
            min(y for x,y in _shape), max(y for x,y in _shape),
            tuple(sorted(_rows.items())),
        )

def cells_at(piece, pose):
    x,y,r = pose
    return tuple(sorted((x+a,y+b) for a,b in SHAPES[piece][r]))

def pose_from(piece, cells):
    shape = norm(cells)
    for r,s in enumerate(SHAPES[piece]):
        if norm(s) == shape:
            return (min(x for x,y in cells)-min(x for x,y in s),
                    min(y for x,y in cells)-min(y for x,y in s), r)
    return None

def fits(board, piece, pose):
    x,y,r = pose
    low_x,high_x,low_y,high_y,rows=_SHAPE_ROWS[piece,r]
    if x+low_x<0 or x+high_x>=10 or y+low_y<0 or y+high_y>=20:
        return False
    for dy,mask in rows:
        shifted=mask<<x if x>=0 else mask>>(-x)
        if board[y+dy] & shifted:
            return False
    return True

def step(board, piece, pose, action):
    x,y,r = pose
    if action in ('left','right'):
        n=(x+(1 if action=='right' else -1),y,r)
        return (n,0) if fits(board,piece,n) else None
    if action == 'soft':
        while fits(board,piece,(x,y+1,r)): y+=1
        return ((x,y,r),0)
    if action == 'rot180':
        nr=(r+2)%4
        if piece=='O':return None
        # T/J/L have observably distinct 180 states so the input controller
        # can acknowledge a single-key rotated outcome. For I/S/Z keep the
        # historical geometry-only (0,0) compatibility behavior; do NOT
        # route these visually ambiguous rotations to Windows input.
        kicks=ROT180_KICKS[r] if piece in VISIBLE_180_PIECES else ((0,0),)
        for k,(dx,dy) in enumerate(kicks):
            n=(x+dx,y+dy,nr)
            if fits(board,piece,n):return n,k
        return None
    if action not in ('cw','ccw'):
        return None
    nr=(r+(1 if action=='cw' else -1))%4
    for k,(dx,dy) in enumerate(I_KICKS[r,nr] if piece=='I' else KICKS[r,nr] if piece!='O' else ((0,0),)):
        n=(x+dx,y+dy,nr)
        if fits(board,piece,n):return n,k
    return None

def spin_kind(board, pose, rotated, kick):
    """Legacy T three-corner predicate (kept for existing test compatibility)."""
    if not rotated:return ''
    x,y,r=pose
    def occupied(a,b):return a<0 or a>=10 or b>=20 or (b>=0 and bool(board[b]&(1<<a)))
    corners=[occupied(x,y),occupied(x+2,y),occupied(x+2,y+2),occupied(x,y+2)]
    if sum(corners)<3:return ''
    front=((0,1),(1,2),(2,3),(3,0))[r]
    return 'full' if all(corners[i] for i in front) or kick==4 else 'mini'


def is_immobile(board, piece, pose):
    """Immobile test at lock BEFORE line removal (up, left, right, down).

    The down check is explicit to avoid treating a floating, unreachable
    rotation as a lock-spin.  We test translations, not rotations: successful
    SRS wall kicks do not make an already-rotated piece mobile.
    """
    x,y,r=pose
    return (not fits(board,piece,(x,y-1,r))
            and not fits(board,piece,(x-1,y,r))
            and not fits(board,piece,(x+1,y,r))
            and not fits(board,piece,(x,y+1,r)))


def all_mini_plus_spin(board, piece, pose, last_rotation, kick=0):
    """Classify a proven last-rotation lock under default All-Mini+ rules.

    T retains the 3-corner/full-vs-mini test, with immobile fallback to mini.
    Other pieces have ONLY immobile mini detection; no full non-T spins.
    No rotation action alone is sufficient. Unknown 180 wall kicks remain
    excluded by step(), not guessed as legal moves.
    """
    if not last_rotation:return ''
    if piece=='T':
        three_corner=spin_kind(board,pose,True,kick)
        if three_corner:return three_corner
    if piece in ('T','I','J','L','S','Z') and is_immobile(board,piece,pose):
        return 'mini'
    return ''

def reachable(board,piece,cells=None,limit=850,max_depth=14):
    """Yield grounded poses, exact input paths, and All-Mini+ spin labels.

    Soft means a full sonic drop, not one row. Each path step includes its
    expected cells so the controller can verify before the next segment.
    Search is intentionally bounded; it does not enumerate every legal tuck.
    """
    start=pose_from(piece,cells) if cells else (3,0,0)
    if start is None or not fits(board,piece,start):return
    queue=deque([(start,(),False,0)])
    actions=route_actions(piece)
    seen={(start,False,False)}
    expanded=0
    while queue and expanded<limit:
        pose,path,rotated,kick=queue.popleft();expanded+=1
        x,y,r=pose
        if not fits(board,piece,(x,y+1,r)):
            yield pose,path,all_mini_plus_spin(board,piece,pose,rotated,kick)
        if len(path)>=max_depth:continue
        for action in actions:
            if piece=='O' and action in ('cw','ccw','rot180'):continue
            result=step(board,piece,pose,action)
            if result is None:continue
            new,k=result
            if new==pose:continue
            rotated=action in ('cw','ccw','rot180')
            # Only the 5th *90-degree* kick has the historical T-spin full
            # upgrade. A 180 kick index is NOT the same adjudication rule.
            k=k if action!='rot180' else 0
            key=(new,rotated if piece!='O' else False,k==4 if piece=='T' else False)
            if key in seen:continue
            seen.add(key)
            queue.append((new,path+((action,cells_at(piece,new)),),rotated,k))


def reachable_states(board,piece,limit=850,max_depth=14,deadline=None):
    """Yield grounded poses/spins without building path tuples.

    Used only by the planner. route_to()/reachable() still build exact paths for
    the one selected target.
    """
    start=(3,0,0)
    if not fits(board,piece,start):return
    queue=deque([(start,False,0,0)])
    actions=route_actions(piece)
    seen={(start,False,False)}
    expanded=0
    while queue and expanded<limit:
        # Deep may supply a wall-clock deadline. Fast caches only completed,
        # no-deadline pose enumerations keyed by the entire board and piece.
        if deadline is not None and (expanded & 15)==0 and time.perf_counter()>=deadline:
            break
        pose,rotated,kick,depth=queue.popleft();expanded+=1
        x,y,r=pose
        if not fits(board,piece,(x,y+1,r)):
            yield pose,all_mini_plus_spin(board,piece,pose,rotated,kick)
        if depth>=max_depth:continue
        for action in actions:
            if piece=='O' and action in ('cw','ccw','rot180'):continue
            result=step(board,piece,pose,action)
            if result is None:continue
            new,k=result
            if new==pose:continue
            was_rotated=action in ('cw','ccw','rot180')
            k=k if action!='rot180' else 0
            key=(new,was_rotated if piece!='O' else False,k==4 if piece=='T' else False)
            if key in seen:continue
            seen.add(key)
            queue.append((new,was_rotated,k,depth+1))


def can_route_until(board, piece, target, spin='', *, deadline, limit=850, max_depth=14):
    """Strict, deadline-bounded reachability proof for Deep candidate screening.

    An accepted route is either the same deterministic spawn -> rotation ->
    horizontal taps -> sonic drop used by can_route(), or a grounded pose
    produced by the same bounded SRS+ reachable_states() search.
    Incomplete/expired searches can yield false negatives, NEVER positives.
    No keyboard commands or unobserved pieces are produced here.
    """
    if time.perf_counter() >= deadline:
        return False
    target = tuple(sorted(target))
    start = (3, 0, 0)
    if not fits(board, piece, start):
        return False
    # Ordinary placements should not each pay for 850 BFS expansions merely
    # to prove a simple path. This exactly mirrors can_route's cheap path.
    if not spin:
        turns_options = ((), ('cw',), ('ccw',), ('cw', 'cw'))
        if piece in VISIBLE_180_PIECES:
            turns_options += (('rot180',),)
        target_norm = norm(target)
        for turns in turns_options:
            if time.perf_counter() >= deadline:
                return False
            pose = start
            valid = True
            for action in turns:
                result = step(board, piece, pose, action)
                if result is None:
                    valid = False
                    break
                pose, _ = result
            if not valid or norm(cells_at(piece, pose)) != target_norm:
                continue
            delta = min(x for x, _ in target) - min(x for x, _ in cells_at(piece, pose))
            for _ in range(abs(delta)):
                if time.perf_counter() >= deadline:
                    return False
                result = step(board, piece, pose, 'right' if delta > 0 else 'left')
                if result is None:
                    valid = False
                    break
                pose, _ = result
            if not valid:
                continue
            final = step(board, piece, pose, 'soft')
            if final is not None and cells_at(piece, final[0]) == target:
                return True
    for pose, kind in reachable_states(board, piece, limit, max_depth,
                                       deadline=deadline):
        if time.perf_counter() >= deadline:
            return False
        if cells_at(piece, pose) == target and (not spin or kind == spin):
            return True
    return False


def can_route(board,piece,target,spin=''):
    """Fast reachability predicate without allocating an input path."""
    target=tuple(sorted(target));start=(3,0,0)
    if not fits(board,piece,start):return False
    if not spin:
        for turns in ((),('cw',),('ccw',),('cw','cw')) + ((('rot180',),) if piece in VISIBLE_180_PIECES else ()):
            pose=start;valid=True
            for action in turns:
                result=step(board,piece,pose,action)
                if result is None:valid=False;break
                pose,_=result
            if not valid or norm(cells_at(piece,pose))!=norm(target):continue
            delta=min(x for x,y in target)-min(x for x,y in cells_at(piece,pose))
            for _ in range(abs(delta)):
                result=step(board,piece,pose,'right' if delta>0 else 'left')
                if result is None:valid=False;break
                pose,_=result
            if valid:
                result=step(board,piece,pose,'soft')
                if result is not None and cells_at(piece,result[0])==target:return True
    for pose,kind in reachable_states(board,piece):
        if cells_at(piece,pose)==target and (not spin or kind==spin):return True
    return False


def _route_parent_search(board,piece,cells,target,spin='',limit=850,max_depth=14):
    """BFS using parent links; allocate an input route only for the winner.

    The older reachable() stored an ever-growing tuple of actions for *every*
    explored node.  This has the same action/visited/limit order and preserves
    spin last-rotation state for every non-O piece, keeping one parent per node.
    """
    start=pose_from(piece,cells)
    if start is None or not fits(board,piece,start):
        return None
    first=(start,False,False)
    actions=route_actions(piece)
    queue=deque([(first,0,0)])
    parents={first:None}
    expanded=0
    while queue and expanded<limit:
        node,depth,kick=queue.popleft()
        expanded+=1
        pose,rotated,_=node
        x,y,r=pose
        if (not fits(board,piece,(x,y+1,r)) and
                cells_at(piece,pose)==target and
                (not spin or all_mini_plus_spin(board,piece,pose,rotated,kick)==spin)):
            route=[]
            while parents[node] is not None:
                prev,action=parents[node]
                route.append((action,cells_at(piece,node[0])))
                node=prev
            route.reverse()
            return tuple(route)
        if depth>=max_depth:continue
        for action in actions:
            if piece=='O' and action in ('cw','ccw','rot180'):continue
            result=step(board,piece,pose,action)
            if result is None:continue
            new,k=result
            if new==pose:continue
            was_rotated=action in ('cw','ccw','rot180')
            k=k if action!='rot180' else 0
            child=(new,was_rotated if piece!='O' else False,k==4 if piece=='T' else False)
            if child in parents:continue
            parents[child]=(node,action)
            queue.append((child,depth+1,k))
    return None


def route_to(board,piece,cells,target,spin=''):
    """Prefer short airborne rotation/translation paths for ordinary drops."""
    target=tuple(sorted(target));start=pose_from(piece,cells)
    if start is None:return None
    if not spin:
        candidates=[]
        for turns in ((),('cw',),('ccw',),('cw','cw')) + ((('rot180',),) if piece in VISIBLE_180_PIECES else ()):
            pose=start;path=();valid=True
            for action in turns:
                result=step(board,piece,pose,action)
                if result is None:valid=False;break
                pose,_=result;path+=((action,cells_at(piece,pose)),)
            if not valid or norm(cells_at(piece,pose))!=norm(target):continue
            delta=min(x for x,y in target)-min(x for x,y in cells_at(piece,pose))
            for _ in range(abs(delta)):
                action='right' if delta>0 else 'left'
                result=step(board,piece,pose,action)
                if result is None:valid=False;break
                pose,_=result;path+=((action,cells_at(piece,pose)),)
            if valid and cells_at(piece,step(board,piece,pose,'soft')[0])==target:candidates.append(path)
        if candidates:return min(candidates,key=len)
    return _route_parent_search(board,piece,cells,target,spin)
