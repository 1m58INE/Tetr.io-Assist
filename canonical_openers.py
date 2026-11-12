"""Strict, observation-driven early game pattern book (INTERNAL workers).

The library separates stage-by-stage route-verified first-bag scaffolds
from complete ten-placement Perfect Clear recipes. Unseen NEXT/HOLD symbols
must never be assumed. Only observed NEXT prefixes authorize a stage.

Pattern proposals carry the actual first Move, never an instruction to bypass
AutoPlayer's per-frame movement/garbage/final hard-drop authorization.

TKI-3 base: includes an independently SRS-proven TSD and the 7th J placement.
DT Cannon full: 15-step 7-bag reference with verified TSD at 14 / TST at 15.
PCO ten: ten placements, clearing 1+2+1 on locks 8-10, ending empty.
"""
from dataclasses import dataclass
from functools import lru_cache
import time

EMPTY=(0,)*20

# x,y are zero-based board cells, screen-y increasing downward. These are
# reference *placements*, not directives to synthesize unseen NEXT/HOLD.
TKI_FLAT=(
 ('O', ((8,18),(8,19),(9,18),(9,19)),''),
 ('I', ((3,19),(4,19),(5,19),(6,19)),''),
 ('S', ((6,17),(6,18),(7,18),(7,19)),''),
 ('L', ((0,17),(0,18),(0,19),(1,19)),''),
 ('Z', ((3,17),(4,17),(4,18),(5,18)),''),
 ('T', ((1,18),(2,18),(2,19),(3,18)),'full'),
 ('J', ((3,18),(4,18),(5,18),(5,19)),''),
)
DT_BASE=(
 ('O', ((0,18),(0,19),(1,18),(1,19)),''),
 ('I', ((6,16),(6,17),(6,18),(6,19)),''),
 ('L', ((3,19),(4,19),(5,18),(5,19)),''),
 ('J', ((7,18),(7,19),(8,19),(9,19)),''),
 ('S', ((3,18),(4,17),(4,18),(5,17)),''),
 ('T', ((3,16),(4,15),(4,16),(5,16)),''),
 ('Z', ((7,17),(8,17),(8,18),(9,18)),''),
)
# Complete first Perfect Clear reference (Z I J O L T S / J T I).
# Move eight requires a real SRS mini-kick rather than phantom placement.
PCO_TEN=(
 ('Z', ((6,18),(7,18),(7,19),(8,19)),''),
 ('I', ((3,16),(3,17),(3,18),(3,19)),''),
 ('J', ((0,18),(0,19),(1,19),(2,19)),''),
 ('O', ((1,17),(1,18),(2,17),(2,18)),''),
 ('L', ((0,16),(0,17),(1,16),(2,16)),''),
 ('T', ((8,18),(9,17),(9,18),(9,19)),''),
 ('S', ((7,17),(8,16),(8,17),(9,16)),''),
 ('J', ((4,18),(4,19),(5,19),(6,19)),'mini'),
 ('T', ((4,18),(5,18),(5,19),(6,18)),''),
 ('I', ((4,19),(5,19),(6,19),(7,19)),''),
)
# Complete 15-placement DT Cannon reference: two true 7-bags, then a T.
# First seven reuse DT_BASE; second bag is J L O I Z S T, third-bag T.
# The last two moves are independent FULL TSD and FULL TST with real SRS
# reachability. No queued/Garbage state is invented for unavailable NEXT.
DT_FULL=DT_BASE+(
 ('J', ((0,15),(1,15),(0,16),(0,17)),''),
 ('L', ((2,13),(3,13),(3,14),(3,15)),''),
 ('O', ((7,15),(8,15),(7,16),(8,16)),''),
 ('I', ((9,14),(9,15),(9,16),(9,17)),''),
 ('Z', ((2,11),(3,11),(3,12),(4,12)),''),
 ('S', ((5,15),(6,14),(6,15),(7,14)),''),
 ('T', ((1,17),(2,17),(2,18),(3,17)),'full'),
 ('T', ((1,18),(2,17),(2,18),(2,19)),'full'),
)
MIRROR={'I':'I','O':'O','T':'T','J':'L','L':'J','S':'Z','Z':'S'}

@dataclass(frozen=True, slots=True)
class BookStep:
    name: str
    index: int
    source: tuple
    piece: str
    move: object
    remaining: int


def _mirror_spec(spec):
    return tuple((MIRROR[p],tuple((9-x,y) for x,y in cells),spin)
                 for p,cells,spin in spec)


def _find_actual_move(board,piece,cells,spin):
    from engine import placements,tactical_moves
    from motion import can_route
    target=tuple(sorted(cells))
    # Locked spin can only come from a legal last rotation, never by stamping
    # an artificial spin flag onto an ordinary hard-drop placement.
    pool=(tactical_moves(board,piece,limit=850,max_depth=14)
          if spin else placements(board,piece))
    for m in pool:
        if tuple(sorted(m.cells))==target and (not spin or m.spin==spin):
            if can_route(board,piece,target,spin):return m
    return None


@lru_cache(maxsize=8)
def book(name,mirror=False):
    if name not in ('TKI3_FLAT','DT_CANNON_BASE','DT_CANNON_FULL','PCO_TEN'):return ()
    spec={'TKI3_FLAT':TKI_FLAT,'DT_CANNON_BASE':DT_BASE,'DT_CANNON_FULL':DT_FULL,'PCO_TEN':PCO_TEN}[name]
    if mirror:spec=_mirror_spec(spec)
    board=EMPTY;entries=[]
    for i,(piece,cells,spin) in enumerate(spec):
        move=_find_actual_move(board,piece,cells,spin)
        if move is None:return ()  # fail-closed on SRS/physics mismatch
        entries.append(BookStep(name+('_M' if mirror else ''),i,board,piece,move,len(spec)-i))
        board=move.board
    if name=='TKI3_FLAT' and (entries[5].move.lines!=2 or entries[5].move.spin!='full'):
        return ()
    if name=='DT_CANNON_BASE' and any(s.move.lines for s in entries):
        return ()
    if name=='DT_CANNON_FULL' and (any(s.move.lines for s in entries[:13])
            or tuple((s.move.lines,s.move.spin) for s in entries[13:])
                   !=((2,'full'),(3,'full'))):
        return ()
    if name=='PCO_TEN' and (tuple(s.move.lines for s in entries[7:])!=(1,2,1)
                             or any(entries[-1].move.board)):
        return ()
    return tuple(entries)


@lru_cache(maxsize=8)
def _cheap_book_fields(name,mirror):
    """Exact 20-row snapshots without invoking route BFS for absent books."""
    if name not in ('TKI3_FLAT','DT_CANNON_BASE','DT_CANNON_FULL','PCO_TEN'):return ()
    spec={'TKI3_FLAT':TKI_FLAT,'DT_CANNON_BASE':DT_BASE,'DT_CANNON_FULL':DT_FULL,'PCO_TEN':PCO_TEN}[name]
    if mirror:spec=_mirror_spec(spec)
    source=EMPTY
    out=[]
    for i,(p,cells,spin) in enumerate(spec):
        out.append((source,p,cells,spin))
        nxt=list(source)
        for x,y in cells:
            if not (0<=x<10 and 0<=y<20) or nxt[y]&(1<<x):return ()
            nxt[y]|=1<<x
        keep=[row for row in nxt if row!=1023]
        source=tuple([0]*(20-len(keep))+keep)
    return tuple(out)


def choose_scaffold(board,piece,queue=(),*,placed=0,incoming=0,garbage=False,
                    deadline=None, enabled=True, pco_enabled=False):
    """Offer an exact next step in a physically verified reference.

    TKI is seven steps, DT full fifteen and PCO ten, each with verified
    local physical outcomes. Unknown NEXT or unavailable HOLD is never guessed.
    """
    if (not (enabled or pco_enabled) or incoming or garbage or not 0<=placed<15 or
            deadline is not None and time.perf_counter()>=deadline):return None
    from engine import observed_queue_prefix
    seen=observed_queue_prefix(queue)
    alternatives=[]
    families=(('TKI3_FLAT','DT_CANNON_FULL') if enabled else ())
    if pco_enabled:families+=('PCO_TEN',)
    for name in families:
        for mirrored in (False,True):
            rows=_cheap_book_fields(name,mirrored)
            if not rows or placed>=len(rows):continue
            src,p,cells,spin=rows[placed]
            if src!=tuple(board) or p!=piece:continue
            # Only start a template when CURRENT+two real consecutive NEXT
            # symbols match; an unreadable slot never becomes a fantasy bag.
            if placed==0 and (len(seen)<2 or
                tuple(row[1] for row in rows[1:3])!=seen[:2]):continue
            if placed>0 and seen and placed<len(rows)-1 and rows[placed+1][1]!=seen[0]:continue
            m=_find_actual_move(tuple(board),p,cells,spin)
            if m is None:continue
            entry=BookStep(name+('_M' if mirrored else ''),placed,src,p,m,len(rows)-placed)
            alternatives.append((name,mirrored,m,entry))
    if not alternatives:return None
    # Prefer an earlier verified attack, not merely a distant future plan.
    # Each DT stage has a route-proven complete 15-step physical reference.
    alternatives.sort(key=lambda v:({'TKI3_FLAT':0,'PCO_TEN':1,'DT_CANNON_FULL':2}[v[0]],v[1]))
    return alternatives[0][3]


def matches_reference_target(plan, placed):
    """Cheap static ledger gate for a Deep plan on the GUI callback path.

    No BFS or expensive compute in the UI.  Actual route validation occurs
    in choose_scaffold on the worker and in AutoPlayer before key dispatch.
    """
    name=getattr(plan,'opening_profile','')
    if not isinstance(name,str):return False
    mirror=name.endswith('_M')
    family=name[:-2] if mirror else name
    if family not in ('TKI3_FLAT','DT_CANNON_FULL','PCO_TEN'):
        return False
    rows=_cheap_book_fields(family,mirror)
    if not 0<=placed<len(rows):return False
    source,piece,cells,spin=rows[placed]
    move=getattr(plan,'move',None)
    return bool(move is not None and
                tuple(plan.source_board)==source and plan.source_piece==piece and
                move.piece==piece and tuple(sorted(move.cells))==tuple(sorted(cells)) and
                move.spin==spin)


def signature_63(board):
    """6–3 layout inspection. x=6 (seventh column) is the I-well.

    This is a stack *style*, not a guaranteed T-spin slot. The right bank
    cannot be called a finished J/L roof without a separately proven spin.
    """
    from engine import board_metrics
    heights,highest,holes,covered,bump,*_=board_metrics(tuple(board))
    left=heights[:6];right=heights[7:]
    depth=min(min(left),min(right))-heights[6]
    balances=(max(left)-min(left),max(right)-min(right))
    return highest,holes,covered,depth,balances,heights


def score_63(source,move,*,incoming=0):
    """Conservative surface-only 6–3 scorer; no positive reward for holes."""
    from engine import execution_cost,board_metrics
    if incoming or move is None:return None
    highest,holes,covered,depth,(rough_left,rough_right),heights=signature_63(move.board)
    src=signature_63(source)
    if (highest>=13 or holes>src[1] or covered>src[2]+1 or
            (depth<0 and any(source))):return None
    if move.lines>=2:return None  # deliberate clears are handled by attack beam
    # Keep the central I shaft *strictly* open while constructing a 6-3 bank;
    # filling its first floor cell is not a 6-3 opening, even when smooth.
    # Quad finishing is handled by the normal attack engine, not this scorer.
    if any(x==6 for x,y in move.cells):return None
    left=sum(heights[:6])/6
    right=sum(heights[7:])/3
    balance=abs(left-right)
    finesse=execution_cost(source,move.piece,tuple(sorted(move.cells)),move.spin)
    # Keep a shallow central well. An observed I is useful, but never force
    # I-well geometry if it builds cavities or makes top-out likely.
    benefit=(min(4,max(0,depth))*3.0 -
             rough_left*1.8-rough_right*2.3-balance*1.5-
             highest*.75-holes*12-covered*.18-finesse*.3)
    return benefit


def choose_63(board,piece,*,incoming=0,deadline=None,max_candidates=36):
    """Pick a reachable next move preserving the 6–3 channel, if justified.

    Searches only CURRENT hard-drop destinations, independently verifies the
    winning path, and never incurs time in GUI/Windows key threads.
    """
    if incoming or deadline is not None and time.perf_counter()>=deadline:return None
    from engine import placements,board_metrics
    from motion import can_route
    if piece not in 'IOTSZJL' or board_metrics(tuple(board))[1]>=11:return None
    ranked=[]
    for i,m in enumerate(placements(tuple(board),piece)):
        if i>=max_candidates or deadline is not None and time.perf_counter()>=deadline:break
        score=score_63(board,m,incoming=incoming)
        if score is not None:ranked.append((score,m))
    for score,m in sorted(ranked,key=lambda v:v[0],reverse=True)[:4]:
        if deadline is not None and time.perf_counter()>=deadline:break
        if can_route(tuple(board),piece,tuple(sorted(m.cells)),m.spin):
            return m,score
    return None
