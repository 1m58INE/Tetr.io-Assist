"""Low-latency two-piece solver with optional bounded tactical refinement.

The normal path stays intentionally small.  A deeper second-ply tactical pass can
run out-of-band and replace/augment the fast result only when it arrives in time.
"""
from dataclasses import dataclass, field
import math
from functools import lru_cache
import time
from combat_rules import exchange
from combo_rules import attack_estimate, surge_lines
from opening import opening_burst_bonus, verified_i_well_bonus, quad_well_potential, OPENING_PIECES
from garbage_tracking import advance_mask, garbage_progress
from b2b_search import chain_increment, verified_pair_bonus, preserve_portfolio, search_observed_chain
from competitive_policy import combat_tempo_value
from attack_rate import fast_attack_goal_bias, safe_well_progress
from tspin_factory import factory_portfolio

# Installed once per Deep worker via ProcessPoolExecutor initializer.  The
# normal Fast solver never looks up the shared revision counter.
_DEEP_LATEST_REVISION = None


def set_deep_revision_counter(counter):
    global _DEEP_LATEST_REVISION
    _DEEP_LATEST_REVISION = counter


def _deep_obsolete(revision):
    return bool(revision and _DEEP_LATEST_REVISION is not None
                and _DEEP_LATEST_REVISION.value != revision)


BASE = {
    'I': ((0,0),(1,0),(2,0),(3,0)), 'O': ((0,0),(1,0),(0,1),(1,1)),
    'T': ((1,0),(0,1),(1,1),(2,1)), 'J': ((0,0),(0,1),(1,1),(2,1)),
    'L': ((2,0),(0,1),(1,1),(2,1)), 'S': ((1,0),(2,0),(0,1),(1,1)),
    'Z': ((0,0),(1,0),(1,1),(2,1))}


def norm(cells):
    a=min(x for x,y in cells); b=min(y for x,y in cells)
    return tuple(sorted((x-a,y-b) for x,y in cells))


def rotations(piece):
    s=norm(BASE[piece]); out=[]
    for r in range(4):
        if s not in [v for _,v in out]: out.append((r,s))
        s=norm([(-y,x) for x,y in s])
    return out


def observed_queue_prefix(queue, limit=3):
    """Only consecutive actually observed NEXT slots are usable.

    Never promote NEXT2 to NEXT1 just because NEXT1 was unreadable.
    """
    confirmed=[]
    for piece in queue:
        if piece not in BASE:break
        confirmed.append(piece)
        if len(confirmed)>=limit:break
    return tuple(confirmed)


SHAPES={p:rotations(p) for p in BASE}
BOTTOMS={p:tuple((r,s,tuple((x,max(y for a,y in s if a==x))
                           for x in range(max(a for a,b in s)+1)))
                 for r,s in rs) for p,rs in SHAPES.items()}


@lru_cache(maxsize=2048)
def columns(board):
    out=[0]*10
    for y,row in enumerate(board):
        while row:
            bit=row & -row;row-=bit
            out[bit.bit_length()-1]|=1<<y
    return tuple(out)


def identify(cells):
    shape=norm(cells)
    return next((p for p,rs in SHAPES.items() if any(s==shape for _,s in rs)), None)


@dataclass(slots=True)
class Move:
    piece: str
    rotation: int
    cells: tuple
    board: tuple
    lines: int
    score: float=0
    next_move: object=None
    spin: str=''
    tactical: bool=False


def placements(board,piece):
    # Return fresh mutable Move objects; cached geometry is immutable.
    for rotation,cells,after,lines in placement_data(board,piece):
        yield Move(piece,rotation,cells,after,lines)


@lru_cache(maxsize=512)
def placement_data(board,piece):
    out=[];cols=columns(board)
    for rotation,shape,bottoms in BOTTOMS[piece]:
        for col in range(11-len(bottoms)):
            if any(board[y]&(1<<(x+col)) for x,y in shape):continue
            row=20
            for x,bottom in bottoms:
                below=cols[x+col]>>(bottom+1)
                distance=(below & -below).bit_length()-1 if below else 19-bottom
                row=min(row,distance)
            cells=tuple((x+col,y+row) for x,y in shape)
            new=list(board)
            for x,y in cells: new[y]|=1<<x
            left=[r for r in new if r!=1023]; lines=20-len(left)
            out.append((rotation,cells,tuple([0]*lines+left),lines))
    return tuple(out)


@lru_cache(maxsize=8192)
def value(board,mode):
    heights=[21-(col & -col).bit_length() if col else 0 for col in columns(board)]
    holes=sum(heights)-sum(row.bit_count() for row in board)
    score=-sum(heights)*.48-holes*(10 if mode=='survival' else 7)-sum(abs(a-b) for a,b in zip(heights,heights[1:]))*.4-max(heights)*(1.4 if mode=='survival' else .55)
    # Preserve an accessible right-side I well only while the board is healthy.
    if mode=='attack' and max(heights)<10 and holes<=2:
        score-=heights[9]*1.8
    return score


def reward(lines,mode):
    if mode=='attack':return [0,1,5,10,28][lines]
    return [0,2,5,9,15][lines] if mode!='combo' else [0,7,9,11,13][lines]


def is_difficult(move):
    """Season-2-style B2B event the solver can infer from the local board."""
    return bool(move.lines and (move.lines==4 or move.spin or not any(move.board)))


def sustained_attack_value(estimate, b2b_before=0, combat='ATTACK', ruleset='standard'):
    """Score earned damage and *change* in stored Surge, not absolute bank.

    The earlier evaluation counted the whole stored bank on each simulated
    ply but its entire payout on release. That favored ending long B2Bs
    prematurely. Delta accounting telescopes across the lookahead horizon.
    Survival states value immediate release more than speculative savings.
    """
    bank_before=surge_lines(b2b_before,ruleset)
    bank_after=surge_lines(estimate.next_b2b,ruleset)
    preserve_weight={'ATTACK':6.2,'COUNTER':5.0,
                     'DEFENSE':3.0,'EMERGENCY':1.5}[combat]
    return estimate.expected*6.0+(bank_after-bank_before)*preserve_weight


def attack_utility(move,b2b=0,combo=-1,ruleset='standard',
                   combo_table='multiplier',rounding='auto'):
    """Deep-only B2B/Combo ranking with transition-based Surge value."""
    attack=attack_estimate(move,b2b,combo,ruleset,combo_table,rounding)
    utility=(sustained_attack_value(attack,b2b,'ATTACK',ruleset)
             +min(max(attack.next_combo,0),8)*.22)
    return utility,attack.next_b2b,attack.next_combo


def move_reward(move,mode,b2b=0,combo=-1):
    if mode=='attack':return attack_utility(move,b2b,combo)[0]
    score=reward(move.lines,mode)
    if move.spin and move.lines:
        score+=([0,10,25,38][min(move.lines,3)] if move.spin=='full' else 5)
    if move.lines and not any(move.board):score+=35
    return score


def fast_attack_reward(move):
    """v0.7-compatible hot-path reward; deep refinement owns richer PvP logic."""
    score=(0,1,5,10,28)[move.lines]
    if move.spin and move.lines:
        score+=((0,10,25,38)[min(move.lines,3)] if move.spin=='full' else 5)
    if move.lines and not any(move.board):score+=35
    return score


def fast_offensive_tempo(move, board, incoming=0, b2b=0):
    """Constant-time execution-tempo tie-breaker for low-risk ATTACK.

    Slow, complicated placements should not outrank nearly equal fast clears.
    Does not alter feasibility checks, tactical reachability or survival under
    incoming attacks. Scaled well below a true Quad/TSD attack reward.
    """
    if incoming or any(board[:8]):
        return 0.0
    col=min(x for x,y in move.cells)
    # A side tap / 90-degree input has a real wall-clock cost, unlike a plan
    # evaluation. Tucks require additional independently observed commands.
    turns=min(move.rotation%4,4-move.rotation%4)
    cost=abs(col-3)*.33+turns*.56
    if move.tactical:cost+=1.25
    # Reward reliable output over spammy ordinary Singles. This is small
    # enough not to override a clean significant attack or a safer stack.
    offense=(1.2 if move.lines>=2 else 0.0)
    if move.lines==4:
        # A Quad is a rule-unambiguous difficult clear: small B2B reward is
        # safer than relying on an unverified Mini server adjudication.
        offense+=.8+min(max(0,int(b2b)),6)*.45
    elif move.spin=='full' and move.lines>=2:offense+=.8
    return offense-cost


@lru_cache(maxsize=1024)
def _fast_reachable_poses(board, piece, limit, max_depth):
    """Immutable, exact-board memo for *completed* clock-free SRS BFS results.

    Fast may reevaluate the same locked board while CURRENT/preview/hold changes.
    Deadline-constrained Deep searches must NEVER use this cache, because a
    partial BFS result is not a complete legal-move set. The move generator
    always creates fresh mutable Move instances from the frozen poses.
    """
    from motion import reachable_states
    return tuple(reachable_states(board, piece, limit, max_depth))


def tactical_moves(board,piece,limit=850,max_depth=14,deadline=None,include_i=False):
    """Ordinary placements plus bounded reachable tuck/spin placements.

    Planning uses a path-free reachability BFS.  Exact input paths are generated
    only for the selected move, keeping the hot solver cheaper than v0.7's BFS.
    """
    from motion import reachable_states, cells_at
    moves={tuple(sorted(m.cells)):m for m in placements(board,piece)}
    if (piece in 'TJLSZ' or (include_i and piece=='I')) and any(board):
        # All-Mini+ rotation-labelled candidates are counted only if a legal
        # executable last-rotation state exists. I-kick search is Deep-only
        # and bounded separately to protect Fast's hot-loop latency.
        if piece=='I':
            limit=min(limit,320)
            max_depth=min(max_depth,11)
        states=(_fast_reachable_poses(board,piece,limit,max_depth)
                if deadline is None else reachable_states(
                    board,piece,limit,max_depth,deadline=deadline))
        for pose,spin in states:
            cells=cells_at(piece,pose)
            rows=list(board)
            for x,y in cells:rows[y]|=1<<x
            left=[r for r in rows if r!=1023];lines=20-len(left)
            move=Move(piece,pose[2],cells,tuple([0]*lines+left),lines,spin=spin,tactical=True)
            old=moves.get(cells)
            if old is None or fast_attack_reward(move)>fast_attack_reward(old):moves[cells]=move
    return list(moves.values())


@lru_cache(maxsize=4096)
def move_executable(board,piece,cells,spin=''):
    """Validate the selected target from the normal spawn before it is returned."""
    # With the spawn band clear, every ordinary straight-drop placement can be
    # rotated/translated above the stack. Avoid even a route predicate on the
    # common low-board fast path.
    if not spin and not any(board[:4]):return True
    from motion import can_route
    return can_route(board,piece,cells,spin)


def _best_executable(board,piece,moves,key):
    ordered=sorted(moves,key=key,reverse=True)
    if not ordered:return None
    if not any(board[:4]):return ordered[0]
    for move in ordered:
        if move.tactical or move_executable(board,piece,tuple(sorted(move.cells)),move.spin):return move
    return None


def replay_guarded_single_penalty(board, move, b2b=0, incoming=0):
    """Bounded, *non-binding* low-stack ordinary-Single preference (v1.1.9.3).

    Uploaded replay evidence showed 1,270 ordinary Singles but only three
    T-spin Doubles for the recurring player. This modest score adjustment is
    a testable hypothesis, NOT a learned policy or a ban on Single clears.
    It never applies to incoming pressure, an existing B2B chain, dirty/high
    boards, spins, doubles, triples or quads. The solver may still choose a
    Single if its board value outweighs the penalty. No extra search is added.
    """
    if incoming != 0 or b2b > 0 or move.lines != 1 or move.spin:
        return 0.0
    _, height, holes, *_ = board_metrics(board)
    if height >= 8 or holes > 0:
        return 0.0
    _, after_height, after_holes, *_ = board_metrics(move.board)
    if after_holes > holes or after_height >= 11:
        return 0.0
    return -4.0


def recommend_attack(board,piece,next_piece=None,b2b=0,combo=-1,deep=False,incoming=0,single_guard=True,tempo_bias=False,attack_urgency=0.0):
    # Preserve the original zero-chain Fast I path; run bounded I mini search
    # only if losing an established B2B would matter. Normal TJLSZ spins are
    # part of the same executable tactical candidate pool.
    i_mini=(piece=='I' and b2b>0 and any(board) and
            not any(m.lines==4 for m in placements(board,piece)))
    moves=tactical_moves(board,piece,include_i=i_mini)
    if not moves:return None

    if not deep:
        # Exact v0.7 hot-path scoring/depth. On the normal clear spawn band this
        # deliberately avoids any extra route work so planning latency cannot grow.
        # The target-APM bias is bounded and never applied while pressured.
        goal_bias = min(1.0,max(0.0,float(attack_urgency))) if not incoming and tempo_bias else 0.0
        # Potential on the terminal board only (not twice for a two-ply lookahead).
        def terminal_well_score(target):
            return goal_bias*1.30*safe_well_progress(target) if goal_bias else 0.0
        from attack_structure11113 import efficiency_reward
        def offense_reward(move, source):
            return efficiency_reward(move, goal_bias, source, incoming)
        for m in moves:m.score=(value(m.board,'attack')+offense_reward(m,board)
            +terminal_well_score(m.board)
            +fast_attack_goal_bias(m,board,b2b,combo,goal_bias)
            +(fast_offensive_tempo(m,board,incoming,b2b) if tempo_bias else 0.0)
            +(replay_guarded_single_penalty(board,m,b2b,incoming) if single_guard else 0.0))
        # A low, healthy board should not discard an already-confirmed B2B
        # for a weak Single that can be postponed without making new holes.
        if b2b>0 and board_metrics(board)[1]<12:
            from chain_director import fast_chain_penalty
            for m in moves:m.score+=fast_chain_penalty(m,b2b,board)
        if not next_piece:
            winner=max(moves,key=lambda m:m.score,default=None)
            if winner is None or not any(board[:4]) or winner.tactical:return winner
            if move_executable(board,piece,tuple(sorted(winner.cells)),winner.spin):return winner
            return _best_executable(board,piece,moves,lambda m:m.score)

        best=None;evaluated=[] if any(board[:4]) else None
        # A verified ATTACK shortfall justifies four more two-ply options.
        # Incoming pressure / disabled goal keeps the original six-candidate
        # low-latency route unchanged.
        width=10 if goal_bias>=.40 and not any(board[:9]) else 6
        shortlist=sorted(moves,key=lambda m:m.score,reverse=True)[:width]
        for m in shortlist:
            future=list(placements(m.board,next_piece))
            if not future:continue
            def rank(n):
                # Legacy +6 for any two clears overvalues ordinary low-attack
                # Double/Singles when pushing for a 100-model-APM objective.
                # Preserve it unchanged outside the safe, goal-directed mode.
                combo_bonus=(6.0 * (1.0-min(1.0,goal_bias/.75))
                             if m.lines and n.lines else 0.0)
                b2b_bonus=8 if (m.lines==4 or m.spin and m.lines) and n.lines==4 else 0
                next_b2b=0 if m.lines and not is_difficult(m) else b2b+int(is_difficult(m))
                score=(value(n.board,'attack')+offense_reward(n,m.board)+combo_bonus+b2b_bonus
                       +terminal_well_score(n.board)
                       +fast_attack_goal_bias(n,m.board,next_b2b,combo,goal_bias)
                       +(replay_guarded_single_penalty(m.board,n,b2b,incoming) if single_guard else 0.0))
                if b2b>0:
                    from chain_director import fast_chain_penalty
                    carried=0 if m.lines and not is_difficult(m) else b2b+int(is_difficult(m))
                    score+=fast_chain_penalty(n,carried,m.board)
                return score
            m.next_move=max(future,key=rank)
            m.score=(rank(m.next_move)+offense_reward(m,board)
                +fast_attack_goal_bias(m,board,b2b,combo,goal_bias)
                +(fast_offensive_tempo(m,board,incoming,b2b) if tempo_bias else 0.0)
                +(replay_guarded_single_penalty(board,m,b2b,incoming) if single_guard else 0.0))
            if b2b>0 and board_metrics(board)[1]<12:
                from chain_director import fast_chain_penalty
                m.score+=fast_chain_penalty(m,b2b,board)
            if evaluated is not None:evaluated.append(m)
            if best is None or m.score>best.score:best=m
        if best is None:return max(moves,key=lambda m:m.score,default=None)
        if evaluated is None:
            return best
        # High/blocked spawn only: reject impossible current/next targets before
        # they reach the controller. This path trades a little engine time for
        # avoiding a far larger failed-input/replan stall.
        for m in sorted(evaluated,key=lambda x:x.score,reverse=True):
            if not (m.tactical or move_executable(board,piece,tuple(sorted(m.cells)),m.spin)):continue
            n=m.next_move
            if n is None or not any(m.board[:4]) or n.tactical or move_executable(m.board,n.piece,tuple(sorted(n.cells)),n.spin):return m
        return _best_executable(board,piece,moves,lambda m:m.score)

    # Deep competitive refinement: Season-2 B2B/Surge utility and bounded
    # second-ply tactical reachability. This runs outside the latency path.
    for m in moves:
        immediate,b1,c1=attack_utility(m,b2b,combo)
        m.score=value(m.board,'attack')+immediate
    ordered=sorted(moves,key=lambda m:m.score,reverse=True)
    if not next_piece:return _best_executable(board,piece,ordered,lambda m:m.score)

    evaluated=[]
    for index,m in enumerate(ordered[:6]):
        current_reward,b1,c1=attack_utility(m,b2b,combo)
        if next_piece in 'TJLSZ' and any(m.board) and index<4:
            future=tactical_moves(m.board,next_piece,620,12)
        else:
            future=list(placements(m.board,next_piece))
        if not future:continue
        for n in future:
            second_reward,b2,c2=attack_utility(n,b1,c1)
            tempo=2.5 if m.lines and n.lines else 0
            n.score=value(n.board,'attack')+second_reward+tempo
        chosen=_best_executable(m.board,next_piece,future,lambda n:n.score)
        if chosen is None:continue
        m.next_move=chosen
        chosen_reward,_,_=attack_utility(chosen,b1,c1)
        m.score=(value(chosen.board,'attack')+current_reward+chosen_reward+
                 (2.5 if m.lines and chosen.lines else 0))
        evaluated.append(m)
    for m in sorted(evaluated,key=lambda x:x.score,reverse=True):
        if m.tactical or move_executable(board,piece,tuple(sorted(m.cells)),m.spin):return m
    return _best_executable(board,piece,ordered,lambda m:m.score)

def recommend(board,piece,next_piece=None,mode='balanced'):
    best=None
    for move in placements(board,piece):
        score=value(move.board,mode)+reward(move.lines,mode)
        if next_piece:
            future=list(placements(move.board,next_piece))
            if not future: continue
            def rank(n):
                return value(n.board,mode)+reward(n.lines,mode)+(12 if mode=='combo' and move.lines and n.lines else 0)
            move.next_move=max(future,key=rank)
            score=rank(move.next_move)+reward(move.lines,mode)
        move.score=score
        if best is None or score>best.score: best=move
    return best


def separate(grid,max_row=5):
    """Find a detached tetromino in the top six rows. Reject ambiguity."""
    board=tuple(sum((1<<x) for x in range(10) if grid[y][x]) for y in range(20))
    seen=set(); candidates=[]
    for y in range(20):
        for x in range(10):
            if not grid[y][x] or (x,y) in seen: continue
            stack=[(x,y)]; seen.add((x,y)); group=[]
            while stack:
                a,b=stack.pop(); group.append((a,b))
                for c,d in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if 0<=c<10 and 0<=d<20 and grid[d][c] and (c,d) not in seen:
                        stack.append((c,d)); seen.add((c,d))
            if len(group)==4 and max(b for a,b in group)<=max_row:
                p=identify(group)
                if p: candidates.append((p,group))
    if len(candidates)!=1: return None
    piece,cells=candidates[0]; locked=list(board)
    for x,y in cells: locked[y]&=~(1<<x)
    return tuple(locked),piece,cells


def recommend_fast(board,piece,next_piece=None,mode='balanced',beam=6):
    """Evaluate next piece only for the best immediate placements (approximate)."""
    moves=list(placements(board,piece))
    for move in moves: move.score=value(move.board,mode)+reward(move.lines,mode)
    if not next_piece: return max(moves,key=lambda m:m.score,default=None)
    best=None
    for move in sorted(moves,key=lambda m:m.score,reverse=True)[:beam]:
        future=list(placements(move.board,next_piece))
        if not future: continue
        def rank(n):
            return value(n.board,mode)+reward(n.lines,mode)+(12 if mode=='combo' and move.lines and n.lines else 0)
        move.next_move=max(future,key=rank)
        move.score=rank(move.next_move)+reward(move.lines,mode)
        if best is None or move.score>best.score: best=move
    return best if best is not None else recommend(board,piece,next_piece,mode)


@dataclass(slots=True)
class Plan:
    move: Move
    source_board: tuple
    source_piece: str
    use_hold: bool=False
    prepared_safe: bool=True
    observed_next: object=None
    observed_hold: object='?'
    b2b_before: int=0
    combo_before: int=-1
    continuation_source: str=field(default='fast',kw_only=True)
    # NEXT prediction provenance, distinct from THIS piece's immutable author.
    prefetch_source: str | None=field(default=None,kw_only=True)
    # Advisory provenance, never used as a key-input authorization.
    opening_profile: str=field(default='',kw_only=True)
    opening_verified_steps: int=field(default=0,kw_only=True)


@dataclass(slots=True)
class PvpPlan(Plan):
    deep_revision: int=0
    queue_revision: int=0
    incoming_before: int=0
    combat_mode: str='ATTACK'
    pieces_placed: int=0
    sent_upper_bound: int=0
    opening_active: bool=False
    opening_burst_gain: int=0
    opening_setup_gain: float=0.0


def _plan_move_uncached(board,piece,next_piece=None,hold='?',allow_hold=True,mode='balanced',fast=True,
              b2b=0,combo=-1,deep=False,incoming=0,single_guard=True,tempo_bias=False,attack_urgency=0.0):
    """Return the best bounded plan.

    '?' = unknown hold, None = confirmed empty, otherwise tetromino letter.
    Empty-hold comparison keeps equal horizons because using an empty hold consumes
    the only visible next piece.
    """
    can_hold=allow_hold and hold!='?' and (hold is not None or next_piece is not None)
    if mode=='attack':
        # The adjustment is symmetric across the known direct/HOLD candidates.
        # It does not guess a hidden HOLD piece or bypass HOLD confirmation.
        solver=lambda b,p,n,m:recommend_attack(b,p,n,b2b,combo,deep,incoming,single_guard,tempo_bias,attack_urgency)
    else:
        solver=recommend_fast if fast else recommend
    lookahead=None if can_hold and hold is None else next_piece
    direct=solver(board,piece,lookahead,mode)
    best=Plan(direct,board,piece,False,False,next_piece,hold,b2b,combo) if direct else None
    alternative=next_piece if hold is None else hold
    if can_hold and alternative!=piece:
        other=solver(board,alternative,lookahead,mode)
        if other is not None and (best is None or other.score>best.move.score+.15):
            best=Plan(other,board,piece,True,False,next_piece,hold,b2b,combo)
    if best is not None:
        following=best.move.next_move
        # Prepared execution is allowed only when the queue piece was actually
        # observed, HOLD was not uncertain, and the predicted next target has a
        # verified route.  Exact board/piece matching is checked again at runtime.
        # Fast trust gate: the queued piece must have been observed and must
        # match the predicted follow-up. Exact board/piece matching is checked
        # again at runtime, and the controller still validates the final route
        # before hard drop.
        best.prepared_safe=bool(next_piece is not None and following is not None and
                                following.piece==next_piece)
    return best



def _move_snapshot(move):
    """Freeze a two-ply result: never share mutable Move objects across frames."""
    if move is None:
        return None
    return (move.piece, move.rotation, move.cells, move.board, move.lines,
            move.score, _move_snapshot(move.next_move), move.spin, move.tactical)


def _move_from_snapshot(data):
    if data is None:
        return None
    piece,rotation,cells,board,lines,score,following,spin,tactical=data
    return Move(piece,rotation,cells,board,lines,score,
                _move_from_snapshot(following),spin,tactical)


@lru_cache(maxsize=768)
def _cached_attack_fast_plan(board,piece,next_piece,hold,allow_hold,
                             b2b,combo,single_guard,tempo_bias,attack_urgency):
    """Complete, immutable plan snapshot keyed by EVERY Fast scoring input.

    Only zero-incoming, ordinary Fast ATTACK goes here. Do not retain Plan or
    Move instances because the GUI/controller mutate them after selection.
    A changed board, CURRENT, observed NEXT, HOLD, chain, goal or user option
    therefore cannot retrieve an obsolete decision for a different state.
    """
    plan=_plan_move_uncached(board,piece,next_piece,hold,allow_hold,'attack',
        True,b2b,combo,False,0,single_guard,tempo_bias,attack_urgency)
    if plan is None:
        return None
    return (_move_snapshot(plan.move), plan.use_hold, plan.prepared_safe)


def plan_move(board,piece,next_piece=None,hold='?',allow_hold=True,mode='balanced',fast=True,
              b2b=0,combo=-1,deep=False,incoming=0,single_guard=True,tempo_bias=False,attack_urgency=0.0):
    """Return fresh executable advice; memoize only deterministic safe Fast input.

    Pressure, Deep, defense, survival, fallback and all other modes retain the
    unmodified solver. No physical input or spin adjudication is cached.
    """
    if (fast and not deep and mode=='attack' and incoming==0
            and isinstance(board,tuple) and len(board)==20
            and piece in BASE and (next_piece is None or next_piece in BASE)
            and (hold in BASE or hold in ('?',None))
            and isinstance(attack_urgency,(int,float))
            and math.isfinite(attack_urgency)):
        # Attack-rate urgency is a *heuristic hint* updated every UI tick.
        # Quantize to 1/1024 so tiny 8-ms clock drift does not force a full
        # BFS/HOLD search. The attack model itself is approximate; the bound
        # on this heuristic's change is <.000489, and board/pressure/input
        # safety gates remain exact. Non-finite hints bypass this optimization.
        stable_goal=round(max(0.0,min(1.0,float(attack_urgency)))*1024)/1024
        frozen=_cached_attack_fast_plan(board,piece,next_piece,hold,
            allow_hold,b2b,combo,single_guard,tempo_bias,stable_goal)
        if frozen is None:
            return None
        move,use_hold,prepared=frozen
        return Plan(_move_from_snapshot(move),board,piece,use_hold,prepared,
                    next_piece,hold,b2b,combo)
    return _plan_move_uncached(board,piece,next_piece,hold,allow_hold,mode,
                              fast,b2b,combo,deep,incoming,single_guard,
                              tempo_bias,attack_urgency)

def plan_move_deep(args):
    """Picklable worker entry point for out-of-process tactical refinement."""
    return plan_move(*args,deep=True)



@lru_cache(maxsize=8192)
def board_metrics(board):
    """Cheap structural features used only by the out-of-band PvP solver."""
    cols=columns(board)
    heights=tuple(21-(col & -col).bit_length() if col else 0 for col in cols)
    holes=0;covered=0;transitions=0
    for x,h in enumerate(heights):
        if not h:continue
        top=20-h
        seen=0
        last=True
        for y in range(top,20):
            filled=bool(board[y]&(1<<x))
            if last!=filled:transitions+=1
            last=filled
            if filled:seen+=1
            else:
                holes+=1
                covered+=seen
    wells=0
    for x,h in enumerate(heights):
        left=heights[x-1] if x else 20
        right=heights[x+1] if x<9 else 20
        wells+=max(0,min(left,right)-h)
    bump=sum(abs(a-b) for a,b in zip(heights,heights[1:]))
    return heights,max(heights,default=0),holes,covered,bump,wells,transitions


def attack_lines(move,b2b=0,combo=-1,ruleset='standard',
                 combo_table='multiplier',rounding='auto'):
    """Conservative generated lines; RNG rounding is a lower bound, not a roll."""
    return attack_estimate(move,b2b,combo,ruleset,combo_table,rounding).guaranteed


def combat_profile(board,incoming=0):
    """ATTACK/COUNTER/DEFENSE/EMERGENCY without touching the fast solver."""
    heights,maxh,holes,covered,bump,wells,transitions=board_metrics(board)
    incoming=max(0,int(incoming or 0))
    danger=incoming + maxh*.72 + holes*1.45 + covered*.08
    projected=maxh+incoming
    if projected>=19 or (maxh>=15 and incoming>=3) or danger>=22:return 'EMERGENCY',danger
    if projected>=16 or incoming>=8 or danger>=16:return 'DEFENSE',danger
    if incoming>=3 or danger>=10:return 'COUNTER',danger
    return 'ATTACK',danger


def pvp_board_value(board,combat='ATTACK',incoming=0):
    heights,maxh,holes,covered,bump,wells,transitions=board_metrics(board)
    if combat=='EMERGENCY':
        wh,wc,wb,wm,ww,wt=16.0,.22,.65,3.4,.10,.18
    elif combat=='DEFENSE':
        wh,wc,wb,wm,ww,wt=12.0,.17,.52,2.3,.08,.12
    elif combat=='COUNTER':
        wh,wc,wb,wm,ww,wt=8.5,.13,.43,1.25,.06,.08
    else:
        wh,wc,wb,wm,ww,wt=7.0,.10,.38,.60,.04,.05
    score=-(holes*wh + covered*wc + bump*wb + maxh*wm + wells*ww + transitions*wt)
    # Leave headroom proportional to still-uncancelled pressure.
    projected=maxh+max(0,int(incoming or 0))
    if projected>=20:score-=180+(projected-20)*40
    elif projected>=17:score-=(projected-16)*24
    return score


@lru_cache(maxsize=4096)
def execution_cost(board,piece,cells,spin=''):
    """Deep-only tie breaker: fewer inputs wins when strategic scores are close."""
    from motion import route_to, cells_at
    start=cells_at(piece,(3,0,0))
    route=route_to(board,piece,start,cells,spin)
    if route is None:return 999.0
    cost=0.0
    for action,_ in route:
        cost += 1.6 if action=='soft' else 1.0
    return cost


def _pvp_step(move,b2b,combo,incoming,combat,pieces_placed=20,
              sent_upper_bound=0,meter_confident=False,ruleset='standard',
              combo_table='multiplier',rounding='auto',
              garbage_mask=None,source_board=None):
    """Score a candidate including early cancellation and sustained B2B offense.

    Extra model work stays entirely inside the out-of-process Deep search.
    """
    # Compute once per candidate. Both aggression and cancellation share one
    # rules object, avoiding duplicate math for every Deep search node.
    estimate=attack_estimate(move,b2b,combo,ruleset,combo_table,rounding)
    # Use the incremental bank value once, rather than rewarding the same
    # stored Surge on every hypothetical follow-up.
    immediate=(sustained_attack_value(estimate,b2b,combat,ruleset)
               +min(max(estimate.next_combo,0),8)*.22)
    b2,c2=estimate.next_b2b,estimate.next_combo
    atk=estimate.guaranteed
    result=exchange(atk,incoming,pieces_placed,sent_upper_bound,meter_confident,
                    early_rule=True)
    remaining=result.remaining
    cancel=result.cancelled
    cancel_weight={'ATTACK':1.0,'COUNTER':7.0,'DEFENSE':11.0,'EMERGENCY':16.0}[combat]
    survival=pvp_board_value(move.board,combat,remaining)
    score=survival+immediate+cancel*cancel_weight
    # Genuine B2B continuity: quiet builds do not increment the counter;
    # non-difficult clears have a cost only when the stack remains safe.
    score+=chain_increment(move,b2b,combat,board_metrics(move.board)[1])
    score+=combat_tempo_value(move,result,incoming,combat,board_metrics(move.board)[1])
    if (garbage_mask is not None and any(garbage_mask)
            and source_board is not None and combat in ('COUNTER','DEFENSE','EMERGENCY')):
        score += garbage_progress(source_board,garbage_mask,move.cells)
    if combat in ('DEFENSE','EMERGENCY') and move.lines:score+=move.lines*2.0
    if combat=='EMERGENCY' and remaining==0:score+=18
    # Preserve a rolling combo only when survival remains credible. Its next
    # finisher will be scored with the *actual* multiplier at the next ply.
    # Do not force weak Single spam: the small continuity term is subordinate
    # to board safety and the actual attack estimate.
    if combo>=1 and move.lines==0 and combat in ('ATTACK','COUNTER'):
        score-=min(combo,9)*1.8
    if (combat=='ATTACK' and incoming==0 and move.lines==1 and not move.spin
            and estimate.expected<0.5 and board_metrics(move.board)[1]<=11):
        score-=3.0  # prefer real attack setup over gratuitous weak Singles
    # Reward actually starting pressure during the opening, not simply hoarding
    # a 9-0 well. Distinguish a chain extending difficult clear from a blind
    # Single that would break a charged B2B.
    if pieces_placed<14 and combat in ('ATTACK','COUNTER'):
        if atk>0:score+=min(atk,8)*1.5
        if is_difficult(move):score+=2.0+min(b2,5)*0.9
        if move.lines in (1,2) and not move.spin and b2b>0:score-=min(b2b,7)*1.7
        if move.lines==1 and not move.spin and not incoming:score-=4.0
        # A high, inaccessible opening stack is not worth marginal pressure.
        if board_metrics(move.board)[1]>=12:score-=18
    return score,b2,c2,remaining,atk,estimate.ceiling


def _estimated_finesse(plan):
    """Constant-time input-cost bound for the Qt callback (NO path BFS)."""
    m=plan.move
    col=min(x for x,y in m.cells)
    # Normal spawn at 3. Rotations near the ground or spins are more costly.
    turns=min(m.rotation % 4,4 - m.rotation % 4)
    return abs(col-3) + turns + (3 if m.tactical or m.spin else 0) + int(plan.use_hold)


def emergency_fast_plan(board,piece,next_piece=None,hold='?',allow_hold=True,
                        b2b=0,combo=-1,incoming=0,ruleset='standard',
                        combo_table='multiplier',rounding='auto'):
    """Bounded one-ply rescue for verified imminent top-out only.

    The normal ATTACK Fast path is untouched. Rank using cheap bitboard scores,
    and only run geometric path validation on the leading alternatives.
    """
    candidates=[]
    choices=[(piece,False)]
    if allow_hold and hold!='?':
        swap=next_piece if hold is None else hold
        if swap in BASE and swap!=piece:choices.append((swap,True))
    for shape,use_hold in choices:
        for m in placements(board,shape):
            atk=attack_lines(m,b2b,combo,ruleset,combo_table,rounding)
            residual=max(0,incoming-atk)
            cols=columns(m.board)
            maxheight=max((21 - (c & -c).bit_length() if c else 0) for c in cols)
            score=value(m.board,'survival')+min(incoming,atk)*18+min(atk,5)*2
            score-=max(0,maxheight+residual-16)*40
            score+=8 if m.lines==4 else 0
            candidates.append((score,m,use_hold))
    for score,m,use_hold in sorted(candidates,key=lambda x:x[0],reverse=True):
        if move_executable(board,m.piece,tuple(sorted(m.cells)),m.spin):
            m.score=score
            return Plan(m,board,piece,use_hold,False,next_piece,hold,b2b,combo)
    return None


def deep_adoption_allowed(current, proposed, pieces_placed=20, incoming=0,
                          ruleset='standard',combo_table='multiplier',rounding='auto'):
    """Use only cheap O(1) checks in the GUI thread after Deep finishes."""
    if current is None or proposed is None:return False
    if (proposed.source_board!=current.source_board or
            proposed.source_piece!=current.source_piece):return False
    # A Fast-thread canonical plan is already route-checked for THIS exact
    # board and known queue. A slower Deep search cannot silently break the
    # user's selected opening with a different target or speculative HOLD.
    # Both plans can still share SAME target/continuation metadata; actual
    # Garbage/vision recovery is handled by AutoPlayer, not this score rule.
    if (getattr(current,'continuation_source','')=='canonical_fast' and
            incoming==0 and getattr(current,'opening_profile','')):
        return bool(proposed.move.piece==current.move.piece and
                    proposed.use_hold==current.use_hold and
                    tuple(sorted(proposed.move.cells))==tuple(sorted(current.move.cells)) and
                    proposed.move.spin==current.move.spin)
    current_cost=_estimated_finesse(current)
    proposed_cost=_estimated_finesse(proposed)
    combat=proposed.combat_mode
    # A worker-only beam result is allowed to replace Fast ONLY before the
    # controller starts sending keys (enforced by App.accept_deep_result).
    # These cheap independent guards prevent the beam's large attack objective
    # from overriding a proven safe Fast move when risk has changed.
    if proposed.continuation_source in ('external_tbp','bundled_tbp'):
        # Third-party TBP moves are advisory, never instructions. Protect
        # currently-safe Fast placements against speculative high-risk targets;
        # no stale, UNKNOWN-HOLD or unrouteable external output reaches here.
        from motion import can_route
        if proposed.use_hold and proposed.observed_hold=='?':return False
        if not can_route(proposed.source_board,proposed.move.piece,
                         proposed.move.cells,proposed.move.spin):return False
        _,new_height,new_holes,*_=board_metrics(proposed.move.board)
        _,old_height,old_holes,*_=board_metrics(current.move.board)
        if (new_height>13 or new_height>old_height+3 or
                new_holes>old_holes+1 or proposed_cost>current_cost+4):
            return False
        if incoming:
            if incoming>(8 if proposed.continuation_source=='bundled_tbp' else 3) or proposed.use_hold!=current.use_hold:return False
            newer=attack_lines(proposed.move,proposed.b2b_before,proposed.combo_before,
                               ruleset,combo_table,rounding)
            older=attack_lines(current.move,current.b2b_before,current.combo_before,
                               ruleset,combo_table,rounding)
            return new_height+max(0,incoming-newer)<=13 and newer>older
        # A proven difficult clear is valuable; also admit vetted non-clear
        # B2B setups if stack/holes stay safe. The GUI checks that no input
        # has begun before replacing the Fast target.
        return True
    if (proposed.continuation_source=='counter_beam' and incoming>0):
        # Incoming is not a free attack: prefer a verified worker continuation
        # only if the current key can cancel MORE garbage without increasing
        # projected topout. Avoid new HOLD input in a high-pressure window.
        if proposed.use_hold!=current.use_hold or proposed_cost>current_cost+1:
            return False
        old_out=attack_lines(current.move,current.b2b_before,current.combo_before,
                             ruleset,combo_table,rounding)
        new_out=attack_lines(proposed.move,proposed.b2b_before,proposed.combo_before,
                             ruleset,combo_table,rounding)
        _,old_height,old_holes,*_=board_metrics(current.move.board)
        _,new_height,new_holes,*_=board_metrics(proposed.move.board)
        if (new_height+max(0,incoming-new_out)>old_height+max(0,incoming-old_out)
                or new_holes>old_holes or new_height>=14):return False
        return min(incoming,new_out)>=min(incoming,old_out)+1
    # A verified last-rotation Mini Single/Double is a *difficult* clear:
    # accepting it instead of an ordinary chain-breaking clear is often the
    # entire point of All-Mini+. This is a bounded, pressure-free Deep adoption
    # rule; the worker already had to validate the executable spin route.
    # No expensive route search or input change is made in the GUI callback.
    if (combat=='ATTACK' and incoming==0 and proposed.b2b_before>0
            and current.b2b_before==proposed.b2b_before
            and proposed.use_hold==current.use_hold
            and proposed.move.spin=='mini' and proposed.move.lines>0
            and proposed.move.piece in 'TIJLSZ'
            and proposed_cost<=current_cost+4):
        old=attack_estimate(current.move,current.b2b_before,current.combo_before,
                            ruleset,combo_table,rounding)
        new=attack_estimate(proposed.move,proposed.b2b_before,proposed.combo_before,
                            ruleset,combo_table,rounding)
        if old.next_b2b==0 and new.next_b2b>old.next_b2b:
            _,old_h,old_holes,*_=board_metrics(current.move.board)
            _,new_h,new_holes,*_=board_metrics(proposed.move.board)
            if (new_h<=min(12,old_h+1) and new_holes<=old_holes
                    and new_holes<=board_metrics(proposed.source_board)[2]+1):
                return True
    # Canonical proposals must be admitted by exact-book provenance, not the
    # generic immediate-attack/holes heuristic.  They are checked on the Deep
    # worker before reaching this O(1) UI gate, and only while no key has begun.
    if (incoming==0 and combat=='ATTACK' and not proposed.use_hold and
            not current.use_hold and proposed.pieces_placed==pieces_placed and
            getattr(proposed,'opening_profile','').startswith(
                ('TKI3_FLAT','DT_CANNON_FULL','PCO_TEN'))):
        from canonical_openers import matches_reference_target
        if matches_reference_target(proposed, pieces_placed):
            _,height,*_=board_metrics(proposed.move.board)
            return height<=10
        return False  # a tagged but NON-reference candidate must fail closed
    if (incoming==0 and combat=='ATTACK' and not proposed.use_hold and
            not current.use_hold and
            getattr(proposed,'opening_profile','')=='SIX_THREE'):
        from canonical_openers import score_63
        proposed_quality=score_63(proposed.source_board,proposed.move)
        current_quality=score_63(current.source_board,current.move)
        return (proposed_quality is not None and
                (current_quality is None or proposed_quality>current_quality+2.0)
                and proposed.move.lines<2)
    if (proposed.continuation_source=='canonical_dt_stage' and incoming==0
            and proposed.use_hold==current.use_hold
            and 7<=proposed.pieces_placed<15):
        # This path is exact-board, independently route-verified on worker.
        # A temporary cavity is part of the 15-piece reference, but cannot
        # justify a dangerous increase in height or input complexity.
        _,before_height,*_=board_metrics(proposed.source_board)
        _,after_height,*_=board_metrics(proposed.move.board)
        return (after_height<=10 and after_height<=max(10,before_height+2)
                and proposed_cost<=current_cost+4)
    if (proposed.continuation_source in ('offensive_beam','hold_sequence') and combat=='ATTACK'):
        if incoming:
            return False
        # A different HOLD decision is acceptable ONLY before input starts
        # (App gates this) and when the held piece identity is independently
        # observed; unknown HOLD must not be silently guessed.
        if proposed.use_hold != current.use_hold:
            known=(proposed.observed_hold in BASE or
                   proposed.observed_hold is None and proposed.observed_next in BASE)
            if not (proposed.use_hold and known and proposed.move.piece in BASE):
                return False
        _,height_now,holes_now,*_=board_metrics(current.move.board)
        _,height_new,holes_new,*_=board_metrics(proposed.move.board)
        # Worker-vetted NEXT1 TSD can justify a temporary cavity that the
        # next move actually consumes; ordinary buried holes remain forbidden.
        follow=proposed.move.next_move
        temporary_t_slot=bool(
            follow is not None and proposed.observed_next=='T'
            and follow.piece=='T' and follow.spin=='full' and follow.lines>=2
            and proposed.move.lines==0 and holes_new<=holes_now+2
            and board_metrics(follow.board)[2]<=holes_now
            and board_metrics(follow.board)[1]<=12)
        # The proven CURRENT -> observed NEXT1 -> NEXT2=T builder was already
        # route-checked for every step in the worker.  Prior versions permitted
        # a ONE-step T cavity but rejected every temporary two-step cavity as
        # a hole increase, nullifying the worker's better elite-style search.
        # Only permit an observed, two-quiet-placements, full TSD finish that
        # heals all temporary holes. The actual game still revalidates each
        # subsequent board/piece before issuing any input.
        third=follow.next_move if follow is not None else None
        three_step_slot=bool(
            not temporary_t_slot and proposed.move.lines==0
            and follow is not None and proposed.observed_next==follow.piece
            and follow.lines==0 and third is not None
            and third.piece=='T' and third.spin=='full' and third.lines==2
            and holes_new<=holes_now+1
            and board_metrics(follow.board)[1]<=12
            and board_metrics(follow.board)[2]<=holes_now+2
            and board_metrics(third.board)[1]<=12
            and board_metrics(third.board)[2]<=holes_now)
        safe_setup=temporary_t_slot or three_step_slot
        if (height_new>12 or height_new>height_now+2
                or (holes_new>holes_now and not safe_setup)
                or proposed_cost>current_cost+(4 if safe_setup else 3)):
            return False
        # A verified B2B attack may exceed the usual allowed height by one,
        # but never compensate for new holes or an unrecoverable stack.
        return True
    if combat in ('DEFENSE','EMERGENCY'):
        # Cheap headroom gate; detailed safety scoring already happened in Deep.
        current_height=max((21-(c & -c).bit_length() if c else 0)
                           for c in columns(current.move.board))
        proposed_height=max((21-(c & -c).bit_length() if c else 0)
                            for c in columns(proposed.move.board))
        before_attack=attack_lines(current.move,current.b2b_before,current.combo_before,ruleset,combo_table,rounding)
        after_attack=attack_lines(proposed.move,proposed.b2b_before,proposed.combo_before,ruleset,combo_table,rounding)
        before=current_height+max(0,incoming-before_attack)
        after=proposed_height+max(0,incoming-after_attack)
        return after<=before-2 and proposed_cost<=current_cost+4
    if combat=='COUNTER':
        before=attack_lines(current.move,current.b2b_before,current.combo_before,ruleset,combo_table,rounding)
        after=attack_lines(proposed.move,proposed.b2b_before,proposed.combo_before,ruleset,combo_table,rounding)
        return incoming>=3 and after>=before+2 and proposed_cost<=current_cost+1
    if combat=='ATTACK' and pieces_placed<14:
        before=attack_lines(current.move,current.b2b_before,current.combo_before,ruleset,combo_table,rounding)
        after=attack_lines(proposed.move,proposed.b2b_before,proposed.combo_before,ruleset,combo_table,rounding)
        if after>=before+2 and proposed_cost<=current_cost:
            return True
        # Opening-only: allow a known two-/three-piece burst setup even when
        # the immediate attack is smaller. Never add estimated inputs or accept
        # a higher/holier board to do so. The first key is still never delayed.
        if (getattr(proposed,'opening_active',False)
                and getattr(proposed,'opening_burst_gain',0)>=8
                and proposed.move.next_move is not None
                and proposed_cost<=current_cost):
            _,height_now,holes_now,*_=board_metrics(current.move.board)
            _,height_new,holes_new,*_=board_metrics(proposed.move.board)
            return (height_new<=min(10,height_now+1)
                    and holes_new<=holes_now and incoming<3)
        # A zero-attack setup is allowed only when the *observed* NEXT makes
        # a near-complete I-well executable. This is not speculative 7-bag
        # forecasting and never introduces buried holes or risky towers.
        if (getattr(proposed,'opening_active',False)
                and getattr(proposed,'opening_setup_gain',0)>=22
                and proposed_cost<=current_cost+1 and incoming<3):
            _,height_now,holes_now,*_=board_metrics(current.move.board)
            _,height_new,holes_new,*_=board_metrics(proposed.move.board)
            return (height_new<=min(10,height_now+2) and holes_new<=holes_now
                    and proposed.opening_setup_gain>quad_well_potential(current.move.board,(proposed.observed_next,))+6)
        return False
    # Before the first key of this turn, allow a genuinely reachable
    # NEXT=T T-spin setup to replace an inferior Fast attack placement.
    # This is strict: same confirmed NEXT, non-HOLD guess, no new buried
    # holes, no higher stack, and no sacrifice of a large immediate attack.
    if (combat=='ATTACK' and incoming==0 and proposed.observed_next=='T'
            and not proposed.use_hold and proposed.move.next_move is not None
            and proposed.move.next_move.piece=='T'
            and proposed.move.next_move.spin and proposed.move.next_move.lines
            and proposed_cost<=current_cost+1):
        # The Deep worker has already certified reachability; avoid searching
        # again in the GUI thread.  Require a direct, identical preview.
        old_attack=attack_lines(current.move,current.b2b_before,current.combo_before,
                                ruleset,combo_table,rounding)
        new_attack=attack_lines(proposed.move,proposed.b2b_before,proposed.combo_before,
                                ruleset,combo_table,rounding)
        _,old_h,old_holes,*_=board_metrics(current.move.board)
        _,new_h,new_holes,*_=board_metrics(proposed.move.board)
        if (new_attack>=old_attack-1 and new_h<=min(11,old_h+1)
                and new_holes<=old_holes and new_h<14):
            return True
    # Late-game chain-saving adoption only with *observed* NEXT difficult
    # follow-up and no extra buried holes, high stacks, or slower movement.
    if (combat=='ATTACK' and incoming==0 and pieces_placed>=14
            and current.b2b_before>=3 and proposed.move.next_move is not None
            and proposed.move.next_move.piece==proposed.observed_next
            and not (proposed.use_hold and proposed.observed_hold is None)
            and is_difficult(proposed.move.next_move)
            and proposed_cost<=current_cost+1):
        old=attack_estimate(current.move,current.b2b_before,current.combo_before,
                            ruleset,combo_table,rounding)
        new=attack_estimate(proposed.move,proposed.b2b_before,proposed.combo_before,
                            ruleset,combo_table,rounding)
        if old.next_b2b==0 and new.next_b2b>0:
            _,old_h,old_holes,*_=board_metrics(current.move.board)
            _,new_h,new_holes,*_=board_metrics(proposed.move.board)
            return new_h<=min(11,old_h+1) and new_holes<=old_holes
    return False

def continuation_beneficial(previous, candidate, source_board, b2b=0, combo=-1,
                            ruleset='standard', combo_table='multiplier', rounding='auto'):
    """O(1) admission gate for a precomputed move against the Fast fallback.

    A speculative Deep candidate is not automatically better. Keep the Fast
    continuation when the Deep choice trades away attack *and* board safety,
    or makes an equally strong move harder to execute without positional gain.
    Candidate reachability is checked in the worker before it gets here.
    """
    if candidate is None:return False
    if previous is None:return True
    if candidate.piece!=previous.piece:return False
    attack_old=attack_estimate(previous,b2b,combo,ruleset,combo_table,rounding).expected
    attack_new=attack_estimate(candidate,b2b,combo,ruleset,combo_table,rounding).expected
    _,old_height,old_holes,*_=board_metrics(previous.board)
    _,new_height,new_holes,*_=board_metrics(candidate.board)
    if new_height>=17 or new_height>old_height+3 or new_holes>old_holes+2:
        return False
    if attack_new>attack_old+.5:
        return True
    if attack_new<attack_old-.5:
        return False
    # Ties should have a clear structural benefit, not merely differ in score
    # scales between Fast and Deep search.
    return (new_holes<old_holes or new_height<old_height) and new_holes<=old_holes and new_height<=old_height


def _deep_candidates(board,piece,combat,limit=850,depth=14,deadline=None):
    # Tactical reachability matters most in attack/counter. In emergency, straight
    # legal placements are cheaper and less likely to waste the time budget.
    if combat in ('ATTACK','COUNTER'):
        return tactical_moves(board,piece,limit,depth,deadline=deadline,include_i=True)
    return list(placements(board,piece))


def recommend_attack_pvp(board,piece,queue=(),b2b=0,combo=-1,incoming=0,
                         budget_ms=18.0,beam=8,revision=0,pieces_placed=20,
                         sent_upper_bound=0,meter_confident=False,ruleset='standard',
                         combo_table='multiplier',rounding='auto',
                         opening_active=False,opening_remaining=12,garbage_mask=None):
    """Time-budgeted 2-3 ply PvP refinement. Never called by the fast path."""
    if _deep_obsolete(revision):return None
    deadline=time.perf_counter()+max(1.0,float(budget_ms))/1000.0
    queue=observed_queue_prefix(queue)
    combat,_=combat_profile(board,incoming)
    opening_active=bool(opening_active and combat=='ATTACK' and pieces_placed<OPENING_PIECES
                        and incoming<3 and opening_remaining>0)
    # The continuation planner reserves at most ~1/3 of the worker budget;
    # it searches confirmed current+NEXT pieces only, with a hard node cap.
    chain=None
    if ((b2b>=2 or (b2b==0 and board_metrics(board)[1]<9 and
         board_metrics(board)[2]<=2 and ('T' in queue[:2] or piece in ('T','I'))))
         and queue and combat in ('ATTACK','COUNTER') and incoming<5):
        chain_budget=min(6.0,max(0.0,float(budget_ms)*(.22 if b2b==0 else .30)))/1000.0
        chain=search_observed_chain(board,piece,queue,b2b,combo,
                                    min(deadline,time.perf_counter()+chain_budget),
                                    width=6,max_nodes=240)
    moves=_deep_candidates(board,piece,combat,850,14,deadline=deadline)
    if not moves or _deep_obsolete(revision):return None
    scored=[]
    for i,m in enumerate(moves):
        if (i & 15)==0 and time.perf_counter()>=deadline:
            break  # Return advisory fallback; never delay the Fast worker.
        s,b1,c1,inc1,atk1,cap1=_pvp_step(m,b2b,combo,incoming,combat,pieces_placed,
                                           sent_upper_bound,meter_confident,ruleset,combo_table,rounding,garbage_mask,board)
        if combat in ('ATTACK','COUNTER') and 'I' in queue:
            s+=verified_i_well_bonus(m.board,queue)*0.8
            s+=quad_well_potential(m.board,queue)*(1.8 if opening_active else .6)
        m.score=s;scored.append((s,m,b1,c1,inc1,atk1,cap1))
    if not scored:return None
    scored.sort(key=lambda z:z[0],reverse=True)
    if chain is not None and chain.next_move is not None:
        # Give the independently searched continuation one reserved option in
        # the first-ply beam. Never replace the emergency or top-score route.
        matched=next((z for z in scored if z[1].cells==chain.first.cells
                      and z[1].spin==chain.first.spin),None)
        if matched is not None:
            matched[1].next_move=chain.next_move
            chain_entry=matched
        else:
            chain_entry=None
    else:
        chain_entry=None
    # Factory candidates must enter the first-ply portfolio *before* the
    # beam prunes clean setup moves.  Probe only verified NEXT=T, in Deep,
    # with a small share of the existing deadline.  Never assume unseen T's.
    factory=()
    if queue and queue[0]=='T' and combat in ('ATTACK','COUNTER') and incoming<5:
        factory=factory_portfolio(board,scored[:max(12,beam*3)],queue[0],
                                  deadline,max_probes=3)
    if not queue:
        top=[z for z in scored[:max(beam,4)] if move_executable(board,piece,tuple(sorted(z[1].cells)),z[1].spin)]
        if not top:return None
        best=top[0][1]
        # Avoid expensive routes unless two moves are strategically almost tied.
        if len(top)>1 and top[0][0]-top[1][0]<1.5:
            best=min((z[1] for z in top[:3]),key=lambda m:execution_cost(board,piece,tuple(sorted(m.cells)),m.spin))
        return best

    best=None;bestscore=-1e30;best_burst_gain=0
    n1=queue[0]
    # The immediate top-N may drop all combo-extending lines before examining
    # NEXT. Keep tactical finishers and low-risk rolling clear candidates alive.
    # No additional move generation and no extra cost in the Fast path.
    reserve=(combo>=0 or opening_active or b2b>=2) and combat in ('ATTACK','COUNTER')
    portfolio=list(scored[:max(1,beam-4) if opening_active else max(1,beam-3) if reserve else beam])
    if opening_active:
        # Preserve tactical clear, clean setup and nonclear stack candidates.
        # This prevents one-ply beam pruning from excluding the entire opener.
        for criterion in (lambda z: z[1].lines==4 or (z[1].spin and z[1].lines),
                          lambda z: z[1].lines>0,
                          lambda z: z[1].lines==0 and board_metrics(z[1].board)[2]==0,
                          lambda z: z[1].lines==0):
            for candidate in scored:
                if criterion(candidate) and all(candidate[1] is not entry[1] for entry in portfolio):
                    portfolio.append(candidate);break
    if reserve:
        for criterion in (lambda z: z[1].lines==4 or bool(z[1].spin),
                          lambda z: z[1].lines==0 and b2b>=2,
                          lambda z: z[1].lines>0,
                          lambda z: z[1].lines>=2 or bool(z[1].spin),
                          lambda z: z[1].lines==0):
            for candidate in scored:
                if criterion(candidate) and candidate not in portfolio:
                    portfolio.append(candidate);break
    # A proved follow-up T-spin can rescue a builder from early beam pruning.
    # Its move is still rescored against the actual NEXT in the second ply.
    if factory:
        for record, tspin in factory:
            record[1].next_move=tspin
    portfolio=preserve_portfolio(scored,portfolio,b2b,combat,beam,factory)
    if chain_entry is not None and all(z[1] is not chain_entry[1] for z in portfolio):
        # Replace the weakest portfolio slot, but retain the highest-score
        # safety candidate.  Branch executability is checked below.
        portfolio[-1]=chain_entry
    # Cap as before; remove duplicates by source move identity, not dataclass
    # equality (Move carries mutable next_move fields).
    seen=set();chosen=[]
    for entry in (*portfolio,*scored):
        k=tuple(sorted(entry[1].cells))
        if k not in seen:seen.add(k);chosen.append(entry)
        if len(chosen)>=beam:break
    for rank,(s1,m,b1,c1,inc1,atk1,cap1) in enumerate(chosen):
        if _deep_obsolete(revision):return None
        if time.perf_counter()>=deadline:break
        if not (m.tactical or move_executable(board,piece,tuple(sorted(m.cells)),m.spin)):continue
        combat1,_=combat_profile(m.board,inc1)
        future=_deep_candidates(m.board,n1,combat1,620 if rank<4 else 320,12 if rank<4 else 9,deadline=deadline)
        if not future:continue
        fsc=[]
        # Cache this once per first-ply branch; a garbage rise is rare, and
        # most frames have no tracked garbage (zero overhead in the hot path).
        first_mask=(advance_mask(board,garbage_mask,m.cells)
                    if garbage_mask and any(garbage_mask) else None)
        for i,n in enumerate(future):
            if (i & 15)==0 and time.perf_counter()>=deadline:break
            s2,b2,c2,inc2,atk2,cap2=_pvp_step(n,b1,c1,inc1,combat1,pieces_placed+1,
                                               sent_upper_bound+cap1,meter_confident,ruleset,combo_table,rounding,
                                               first_mask,m.board)
            if combat1 in ('ATTACK','COUNTER') and 'I' in queue[1:]:
                s2+=verified_i_well_bonus(n.board,queue[1:])*0.8
                s2+=quad_well_potential(n.board,queue[1:])*(1.8 if opening_active else .6)
            fsc.append((s2,n,b2,c2,inc2,atk2,cap2))
        fsc.sort(key=lambda z:z[0],reverse=True)
        breadth=max(5,beam-2)
        shortlist=fsc[:breadth]
        if c1>=0 and combat1 in ('ATTACK','COUNTER'):
            for criterion in (lambda z: z[1].lines>0,
                              lambda z: z[1].lines==4 or bool(z[1].spin)):
                for candidate in fsc:
                    if criterion(candidate) and candidate not in shortlist:
                        shortlist.append(candidate);break
        chosen2=None;branch=-1e30;branch_burst=0
        for s2,n,b2,c2,inc2,atk2,cap2 in shortlist[:breadth+2]:
            if _deep_obsolete(revision):return None
            if time.perf_counter()>=deadline:break
            if not (n.tactical or move_executable(m.board,n1,tuple(sorted(n.cells)),n.spin)):continue
            # Potential-based lookahead: the second ply must not charge the
            # full absolute stack penalty a *second* time. Use its change in
            # board value, while retaining the independently earned attack.
            # This is Deep-only; the v0.7 Fast evaluator is unchanged.
            baseline1=pvp_board_value(m.board,combat1,inc1)
            total=s1+(s2-baseline1)*.92
            total+=verified_pair_bonus(m,n,b2b,combat1,board_metrics(n.board)[1])
            # Factory-vetted follow-up: this exact T spin has an SRS route.
            # No award for a general shape/ordinary T placement or a guessed
            # third preview.  F9's continuation still needs real verification.
            vetted=m.next_move
            if (vetted is not None and vetted.piece=='T' and vetted.spin and vetted.lines
                    and n.spin==vetted.spin and n.cells==vetted.cells):
                total+=7.0 if n.spin=='full' else 3.0
            burst_attack=atk1+atk2
            finisher=bool(n.lines==4 or (n.spin and n.lines))
            # Third ply only for the strongest surviving branches and only while
            # budget remains. This gives queue awareness without delaying Fast.
            if len(queue)>=2 and rank<4 and time.perf_counter()<deadline:
                n2=queue[1];combat2,_=combat_profile(n.board,inc2)
                # A visible T at NEXT2 needs a reachable spin/tuck search;
                # plain gravity-only placements silently erase T-slot setups.
                # Keep this optional and within the existing Deep deadline.
                thirds=(_deep_candidates(n.board,n2,combat2,240,10,deadline=deadline)
                        if n2=='T' and rank<2 else list(placements(n.board,n2)))
                third_scored=[]
                second_mask=(advance_mask(m.board,first_mask,n.cells)
                             if first_mask and any(first_mask) else None)
                for i,q in enumerate(thirds):
                    if (i & 15)==0 and time.perf_counter()>=deadline:break
                    s3,b3,c3,inc3,atk3,cap3=_pvp_step(q,b2,c2,inc2,combat2,pieces_placed+2,
                                                    sent_upper_bound+cap1+cap2,meter_confident,ruleset,combo_table,rounding,
                                                    second_mask,n.board)
                    if opening_active:s3+=quad_well_potential(q.board,queue[2:])*1.6
                    third_scored.append((s3,q,b3,c3,inc3,atk3,cap3))
                third_scored.sort(key=lambda z:z[0],reverse=True)
                if third_scored:
                    # Carry the observed third-piece continuation as an advisory
                    # for the next turn. It is revalidated at every real spawn.
                    n.next_move=third_scored[0][1]
                    best3=third_scored[0][0]
                    # NEXT3 is a low-cost terminal lookahead on only the two best
                    # third-ply branches. It improves HOLD/T-slot preservation
                    # without turning the Deep worker into an unbounded tree.
                    if len(queue)>=3 and rank<2 and time.perf_counter()<deadline:
                        n3=queue[2];terminal=-1e30
                        for s3,q,b3,c3,inc3,atk3,cap3 in third_scored[:2]:
                            combat3,_=combat_profile(q.board,inc3)
                            for r in placements(q.board,n3):
                                s4,_,_,_,_,_=_pvp_step(r,b3,c3,inc3,combat3,pieces_placed+3,
                                                     sent_upper_bound+cap1+cap2+cap3,meter_confident,ruleset,combo_table,rounding)
                                # Future stack quality contributes once;
                                # subsequent plies contribute their improvement.
                                terminal=max(terminal,s3+.42*(s4-pvp_board_value(q.board,combat3,inc3)))
                            if time.perf_counter()>=deadline:break
                        if terminal>-1e20:best3=terminal
                    total+=(best3-pvp_board_value(n.board,combat2,inc2))*.55
                    # The highest-scoring projected third-ply candidate is an
                    # estimate, not a committed attack. Penalize speculative
                    # tail attacks relative to the two observed nearer moves.
                    if opening_active and third_scored:
                        tail=third_scored[0]
                        burst_attack+=tail[5]
                        finisher=finisher or bool(tail[1].lines==4 or (tail[1].spin and tail[1].lines))
            if opening_active:
                total+=opening_burst_bonus((burst_attack,), opening_remaining, finisher)
                # Avoid speculative high towers during the bounded opener.
                height=board_metrics(n.board)[1]
                total-=max(0,height-9)*12
            if total>branch:
                branch=total;chosen2=n;branch_burst=burst_attack
        if _deep_obsolete(revision):return None
        if chosen2 is None:continue
        m.next_move=chosen2
        # Prefer a simpler execution if strategic value is effectively tied.
        effort=execution_cost(board,piece,tuple(sorted(m.cells)),m.spin)
        total=branch-effort*.12
        if total>bestscore:
            bestscore=total;m.score=total;best=m;best_burst_gain=branch_burst
    if _deep_obsolete(revision):return None
    if best is not None:return best
    return _best_executable(board,piece,[z[1] for z in scored],lambda m:m.score)


def plan_move_pvp(board,piece,queue=(),hold='?',allow_hold=True,b2b=0,combo=-1,
                  incoming=0,budget_ms=18.0,revision=0,queue_revision=0,
                  pieces_placed=20,sent_upper_bound=0,meter_confident=False,
                  ruleset='standard',combo_table='multiplier',rounding='auto',
                  opening_active=False,opening_remaining=12,garbage_mask=None,
                  adaptive_opening=False,canonical_book=False,stack63=False,pc_finish=False):
    """Deep competitive planner with HOLD and up to three observed NEXT pieces."""
    if _deep_obsolete(revision):return None
    # Bounded counter-offense: permit the worker's attack search to evaluate
    # cancellation at low, objectively safe pressure, rather than switching
    # off all tactical planning on the first incoming line. High/dirty boards
    # still use the conservative downstack policy. No GUI-side search.
    _, attack_height, attack_holes, *_ = board_metrics(board)
    confirmed=observed_queue_prefix(queue)
    battle_mode=combat_profile(board,incoming)[0]
    incoming=max(0,int(incoming or 0))
    pressure_eligible=(incoming<=3 and attack_height+incoming<=12
                       and attack_holes<=1)
    beam_eligible=(battle_mode in ('ATTACK','COUNTER') and pressure_eligible
                   and attack_height<11 and attack_holes<3
                   and len(confirmed)>=2
                   and (not garbage_mask or not any(garbage_mask))
                   and float(budget_ms)>=6.0)
    adaptive_active=bool(adaptive_opening and opening_active and incoming==0
                         and pieces_placed<10 and (not garbage_mask or not any(garbage_mask))
                         and battle_mode=='ATTACK' and len(confirmed)>=2)
    # A 10-lock PCO may need only CURRENT+NEXT1 (ninth lock) or CURRENT
    # alone (tenth lock).  Requiring >=2 previews would silently discard a
    # physically proven last step just as the preview horizon ends. This is
    # still Deep-only, strictly exact-board, positive-time-budget, and passed
    # through the same independent survival guard as any other internal plan.
    # DT Cannon is a 15-lock attack sequence, not merely a seven-lock base.
    # Later boards can have structural holes that disable normal beam search.
    # Permit a stage ONLY when the entire local board matches the preverified
    # SRS reference, and CURRENT/NEXT are observed. This never predicts the
    # remaining bag, skips a final drop check or runs under garbage pressure.
    if (canonical_book and adaptive_opening and opening_active and
            7<=pieces_placed<15 and incoming==0 and
            float(budget_ms)>=6.0 and
            (not garbage_mask or not any(garbage_mask)) and
            len(confirmed)>= (0 if pieces_placed==14 else 1)):
        from canonical_openers import choose_scaffold
        stage=choose_scaffold(board,piece,confirmed,placed=pieces_placed,
            deadline=time.perf_counter()+min(.004,float(budget_ms)/1000),
            enabled=True,pco_enabled=False)
        if (stage is not None and stage.name.startswith('DT_CANNON_FULL')
                and not _deep_obsolete(revision)):
            plan=PvpPlan(stage.move,board,piece,False,False,
                confirmed[0] if confirmed else None,hold,b2b,combo,
                revision,queue_revision,0,battle_mode,
                pieces_placed,sent_upper_bound)
            plan.opening_profile=stage.name
            plan.opening_verified_steps=1
            plan.continuation_source='canonical_dt_stage'
            return _guard_internal_stack(plan,board,piece,confirmed,hold,
                allow_hold,b2b,combo,incoming,revision,queue_revision,
                pieces_placed,sent_upper_bound,ruleset,combo_table,rounding)
    if (not beam_eligible and pc_finish and adaptive_opening and opening_active
            and 8<=pieces_placed<10 and incoming==0 and attack_height<11
            and attack_holes<=2 and float(budget_ms)>=6.0
            and (not garbage_mask or not any(garbage_mask))
            and len(confirmed)>=9-pieces_placed):
        from canonical_openers import choose_scaffold
        stage=choose_scaffold(board,piece,confirmed,placed=pieces_placed,
            deadline=time.perf_counter()+min(.005,float(budget_ms)/1000),
            enabled=False,pco_enabled=True)
        if stage is not None and stage.name.startswith('PCO_TEN') and not _deep_obsolete(revision):
            plan=PvpPlan(stage.move,board,piece,False,False,
                confirmed[0] if confirmed else None,hold,b2b,combo,
                revision,queue_revision,int(incoming),battle_mode,
                pieces_placed,sent_upper_bound)
            plan.opening_profile=stage.name
            plan.opening_verified_steps=1
            plan.continuation_source='canonical_pco_stage'
            return _guard_internal_stack(plan,board,piece,confirmed,hold,
                allow_hold,b2b,combo,incoming,revision,queue_revision,
                pieces_placed,sent_upper_bound,ruleset,combo_table,rounding)
    if beam_eligible:
        from offensive_beam import plan_attack_beam
        started=time.perf_counter()
        # Only a healthy, useful Deep worker may be assigned >30ms by the
        # GUI's feedback controller. Allow that spare ATTACK time to reach the
        # actual beam rather than silently capping it at the old 30ms.
        cap=(36.0 if battle_mode=='ATTACK' and incoming==0 and
             float(budget_ms)>=34.0 else 30.0)
        beam_end=started+min(cap,max(6.0,float(budget_ms)))/1000.0
        can_hold=(allow_hold and hold!='?' and (hold is not None or bool(confirmed)))
        other_piece=confirmed[0] if hold is None else hold
        other_possible=(can_hold and other_piece in BASE and other_piece!=piece)
        direct_limit=beam_end if not other_possible else min(beam_end,started+float(budget_ms)*.64/1000)
        # Reserve up to 3ms of the existing Deep deadline for validating the
        # observed opener route; never add work to Qt/Fast or input threads.
        if adaptive_active and direct_limit-started>.008:
            direct_limit-=.003
        # Reserve up to 6ms of the SAME deadline to find an explicit TSD
        # builder when NEXT1 is observed T. This has a real SRS route and
        # cannot be selected by a shape-only geometric heuristic.
        cradle=None
        if confirmed[0]=='T' and time.perf_counter()+.002<direct_limit:
            from tspin_initiator import find_verified_tsd_builder
            cradle=find_verified_tsd_builder(board,piece,confirmed[0],
                min(direct_limit,time.perf_counter()+.006),max_probes=3)
        elif (len(confirmed)>=2 and confirmed[1]=='T' and incoming==0
              and time.perf_counter()+.006<direct_limit):
            # An observed NEXT2=T is valid even with HOLD enabled.  The older
            # version accidentally disabled this verified TSD path whenever a
            # known HOLD alternative existed; live B2B stacks lost their
            # setup search as well.  Give the direct branch at most 9ms from
            # its existing total deadline, then separately compare HOLD.
            from tspin_initiator import find_verified_tsd_pipeline
            # Reserve more worker time only when the user's selected Deep
            # budget actually permits it. No GUI/input thread work is added.
            probe_seconds=min(.012, .006+max(0.0,float(budget_ms))*.00015)
            cradle=find_verified_tsd_pipeline(board,piece,confirmed[:2],
                min(direct_limit,time.perf_counter()+probe_seconds),
                max_first=14,max_probes=4)
        # Compute experimental proposals FIRST, under a single 4ms wall-clock
        # allowance and BEFORE the ordinary beam consumes its remaining budget.
        # Doing this after the beam frequently resulted in zero actual calls.
        from performance_budget import experimental_budget_ms
        experimental_end=min(beam_end,time.perf_counter()+experimental_budget_ms(budget_ms,incoming)/1000.0)
        book_step=None;pc_proof=None;stack_candidate=None
        if ((canonical_book or pc_finish) and adaptive_active and incoming==0
                and time.perf_counter()<experimental_end):
            from canonical_openers import choose_scaffold
            book_step=choose_scaffold(board,piece,confirmed,placed=pieces_placed,
                        deadline=experimental_end,enabled=canonical_book,
                        pco_enabled=pc_finish)
        if (pc_finish and adaptive_active and pieces_placed>=6 and incoming==0
                and time.perf_counter()<experimental_end):
            from perfect_clear_planner import find_pc_finish
            pc_proof=find_pc_finish(board,piece,confirmed,placed=pieces_placed,
                                    deadline=experimental_end,max_nodes=1600)
        if (stack63 and incoming==0 and pieces_placed<28
                and time.perf_counter()<experimental_end):
            from canonical_openers import choose_63
            stack_candidate=choose_63(board,piece,deadline=experimental_end,
                                      max_candidates=34)
        # Established direct/HOLD beam remains first; an optional new
        # sequence search may use otherwise idle worker budget LATER. Never
        # steal the existing beam's first-input scheduling allowance.
        attack=plan_attack_beam(board,piece,confirmed,b2b,combo,direct_limit,
                                ruleset=ruleset,combo_table=combo_table,rounding=rounding,
                                revision=revision,incoming=incoming)
        if cradle is not None:
            from survival_policy1195 import setup_is_safe
            chain=[cradle]
            while chain[-1].next_move is not None and len(chain)<3:
                chain.append(chain[-1].next_move)
            if (not chain[-1].spin=='full' or chain[-1].lines<2 or
                    not setup_is_safe(board,cradle.board,chain[-1].board,
                                      incoming=incoming,steps=len(chain),
                                      middle=tuple(m.board for m in chain[1:-1]))):
                cradle=None
        if cradle is not None:
            # A guaranteed NEXT TSD has higher priority than a non-difficult
            # low-attack setup, but never replaces an already difficult clear
            # or another proven TSD. Refuse unsafe board/hole growth.
            ordinary=(attack is None or (not is_difficult(attack[0]) and
                         (attack[0].next_move is None or
                          not (attack[0].next_move.spin=='full' and attack[0].next_move.lines>=2))))
            if ordinary and attack is not None:
                from survival_policy1195 import signature
                proposed_height=signature(cradle.board)[0]
                competitor_height=signature(attack[0].board)[0]
                # Do not sacrifice an immediately safe multi-line downstack
                # merely to collect a speculative finisher in two turns.
                if (attack[0].lines>=2 and competitor_height<=proposed_height
                        or attack[0].lines>0 and
                        proposed_height>=competitor_height+2):
                    ordinary=False
            if ordinary and not _deep_obsolete(revision):
                cradle.score=(attack[2] if attack else 0)+1.0
                branch=(cradle,cradle.next_move)
                # For NEXT2=T, preserve the actual third-ply verified finisher.
                if cradle.next_move is not None and cradle.next_move.next_move is not None:
                    branch=branch+(cradle.next_move.next_move,)
                attack=(cradle,branch,cradle.score)
        opener_profile=None
        if adaptive_active and not _deep_obsolete(revision):
            from adaptive_opener import select_adaptive_opening
            selected_open,opener_profile=select_adaptive_opening(
                board,piece,confirmed,attack,cradle,b2b=b2b,combo=combo,
                incoming=incoming,garbage_present=False,placed=pieces_placed,
                deadline=beam_end,ruleset=ruleset,combo_table=combo_table,
                rounding=rounding)
            # The opening selector never overrides a safer existing candidate
            # with an unverified or unobserved future placement.
            if selected_open is not None:
                attack=selected_open
        # Deterministic, route-proven canonical first-bag reference steps.  The
        # book is advisory, uses no unseen bag symbols and can be disabled
        # separately from the original adaptive selector.
        if book_step is not None and adaptive_active and not _deep_obsolete(revision):
            step=book_step
            if step is not None:
                reference=step.move
                incumbent=attack[0] if attack else None
                # Prioritize a verified local TSD over a template, and never
                # sacrifice an urgent real multi-line clear for the recipe.
                if (incumbent is None or incumbent.lines<2 and not incumbent.spin):
                    cur_holes=board_metrics(board)[2]
                    next_height=board_metrics(reference.board)[1]
                    next_holes=board_metrics(reference.board)[2]
                    if next_height<=9 and next_holes<=cur_holes+2:
                        reference.score=(incumbent.score if incumbent else 0)+1.0
                        attack=(reference,(reference,),reference.score)
                        opener_profile=step.name
        # A real PC finisher must be proven in all observed 2-4 pieces.  Do
        # NOT reward an unfinished 10-piece PC just because its area adds up.
        if pc_proof is not None and adaptive_active and not _deep_obsolete(revision):
            proof=pc_proof
            if proof.moves:
                first=proof.moves[0]
                incumbent=attack[0] if attack else None
                # A solved PC suffix must not silently relabel a matching
                # canonical stage. Preserve its exact pattern provenance for
                # subsequent per-turn verification and execution statistics.
                duplicate_book=bool(opener_profile is not None and
                    isinstance(opener_profile,str) and
                    opener_profile.startswith('PCO_TEN') and incumbent is not None and
                    tuple(sorted(incumbent.cells))==tuple(sorted(first.cells)) and
                    incumbent.spin==first.spin)
                if not duplicate_book and (incumbent is None or incumbent.lines<2 or not is_difficult(incumbent)):
                    first.score=(incumbent.score if incumbent else 0)+2.0
                    first.next_move=proof.moves[1] if len(proof.moves)>1 else None
                    for earlier,later in zip(proof.moves[1:-1],proof.moves[2:]):
                        earlier.next_move=later
                    attack=(first,proof.moves,first.score)
                    opener_profile='PCO_CONFIRMED'
        # 6-3 is a genuine per-spawn stacking policy, NOT a one-shot static
        # opener.  Its well stays at x=6 and it never overrides a difficult
        # attack, high stack, garbage, or an already selected book sequence.
        if stack63 and not incoming and opener_profile is None:
            if not _deep_obsolete(revision) and stack_candidate is not None:
                from canonical_openers import score_63
                candidate=stack_candidate
                if candidate is not None:
                    m,quality=candidate
                    incumbent=attack[0] if attack else None
                    current_quality=score_63(board,incumbent) if incumbent is not None else None
                    if (incumbent is None or
                            not incumbent.lines and not incumbent.spin and
                            current_quality is not None and quality>current_quality+2.0):
                        m.score=(incumbent.score if incumbent else 0)+.8
                        attack=(m,(m,),m.score)
                        opener_profile='SIX_THREE'
        selected=attack;swapped=False
        if other_possible and time.perf_counter()+.001<beam_end and not _deep_obsolete(revision):
            # Empty HOLD consumes NEXT1. A filled HOLD does not. Never invent
            # a preview piece after an unreadable queue slot.
            other_queue=confirmed[1:] if hold is None else confirmed
            alternate=plan_attack_beam(board,other_piece,other_queue,b2b,combo,
                                        beam_end,ruleset=ruleset,combo_table=combo_table,rounding=rounding,
                                        revision=revision,incoming=incoming)
            # Unequal horizon on empty HOLD should not encourage speculative
            # swaps just to use one fewer available future piece.
            horizon_penalty=5.0 if hold is None else 0.0
            if alternate and (not attack or alternate[2]-horizon_penalty>attack[2]+2.0):
                selected=alternate;swapped=True
        # A spare-budget, Deep-only observed-HOLD lookahead.  Baseline Deep
        # direct and first-HOLD alternatives have already been considered.
        # Only run when >=9ms is genuinely left; no new calculation lands on
        # the Fast/input/GUI thread and there is no increase of beam_end.
        hold_sequence=None
        if (not adaptive_active and pieces_placed>=16 and incoming==0
                and hold!='?' and allow_hold and len(confirmed)>=2
                and float(budget_ms)>=28.0 and 0<attack_height<=9
                and attack_holes<=1 and (b2b>=2 or combo>=2)
                and time.perf_counter()+.009<beam_end):
            from observed_attack_sequence import find_verified_sequence
            hold_sequence=find_verified_sequence(board,piece,confirmed,hold,True,
                b2b,combo,deadline=min(beam_end,time.perf_counter()+.007),
                width=6,max_nodes=360,revision=revision,ruleset=ruleset,
                combo_table=combo_table,rounding=rounding)
        used_sequence=False
        if hold_sequence is not None and not _deep_obsolete(revision):
            from observed_attack_sequence import adopt_sequence
            if adopt_sequence(hold_sequence,selected,board=board,b2b=b2b,
                              combo=combo,ruleset=ruleset,
                              combo_table=combo_table,rounding=rounding):
                selected=(hold_sequence.moves[0],hold_sequence.moves,
                          hold_sequence.score)
                swapped=hold_sequence.first_uses_hold
                opener_profile=None
                used_sequence=True
        if selected and not _deep_obsolete(revision):
            chosen=selected[0]
            if incoming:
                # A pressure-beam path may be excellent on an empty board but
                # unsafe when it postpones all cancellation. Accept it only
                # with a proven immediate cancel or a NEXT1 difficult finisher
                # and enough headroom for the still-queued garbage.
                immediate=attack_lines(chosen,b2b,combo,ruleset,combo_table,rounding)
                next_piece_after=(confirmed[1] if swapped and hold is None and len(confirmed)>1
                                  else confirmed[0] if not (swapped and hold is None) else None)
                next_valid=bool(chosen.next_move is not None and
                    chosen.next_move.piece==next_piece_after and
                    is_difficult(chosen.next_move))
                _,height_after,holes_after,*_=board_metrics(chosen.board)
                if (height_after+max(0,incoming-immediate)>=14 or
                    holes_after>attack_holes+1 or
                    not (immediate>=min(2,incoming) or next_valid)):
                    selected=None
        if selected and not _deep_obsolete(revision):
            chosen=selected[0]
            plan=PvpPlan(chosen,board,piece,swapped,False,
                         confirmed[0],hold,b2b,combo,
                         revision,queue_revision,int(incoming or 0),battle_mode,
                         pieces_placed,sent_upper_bound)
            plan.continuation_source=('counter_beam' if incoming else
                'hold_sequence' if used_sequence else 'offensive_beam')
            if not swapped and opener_profile is not None and (adaptive_active or opener_profile=='SIX_THREE'):
                if isinstance(opener_profile,str):
                    plan.opening_profile=opener_profile
                    # A named book stage validates only THIS observed move;
                    # never advertise seven future unseen placements as proven.
                    plan.opening_verified_steps=1
                else:
                    plan.opening_profile=opener_profile.family
                    plan.opening_verified_steps=opener_profile.steps
            plan.opening_active=bool(opening_active and pieces_placed<OPENING_PIECES
                                     and opening_remaining>0)
            if plan.opening_active:
                plan.opening_setup_gain=quad_well_potential(chosen.board,confirmed)
                plan.opening_burst_gain=attack_lines(chosen,b2b,combo,ruleset,combo_table,rounding)
            next_from_branch=(confirmed[1] if hold is None and swapped else confirmed[0])
            plan.prepared_safe=bool(chosen.next_move is not None
                                    and chosen.next_move.piece==next_from_branch
                                    and not swapped)
            if b2b>0:
                from chain_director import preserve_b2b
                plan=preserve_b2b(plan,board,piece,confirmed,b2b,incoming,
                                   deadline=beam_end,revision=revision)
            return _guard_internal_stack(plan, board, piece, confirmed, hold,
                                         allow_hold, b2b, combo, incoming,
                                         revision,queue_revision,pieces_placed,
                                         sent_upper_bound,ruleset,combo_table,rounding)
    opening_active=bool(opening_active and incoming<3 and pieces_placed<OPENING_PIECES
                        and combat_profile(board,incoming)[0]=='ATTACK')
    queue=observed_queue_prefix(queue)
    next_piece=queue[0] if queue else None
    can_hold=allow_hold and hold!='?' and (hold is not None or next_piece is not None)
    direct_queue=queue
    deep_started=time.perf_counter()
    total_budget_ms=max(1.0,float(budget_ms))
    # One wall-clock budget shared by direct and HOLD branches. Previously
    # each branch received its own budget and could nearly double CPU work.
    consider_hold=bool(can_hold and (next_piece if hold is None else hold)
                       and (next_piece if hold is None else hold)!=piece)
    direct_budget=total_budget_ms * (.65 if consider_hold else 1.0)
    direct=recommend_attack_pvp(board,piece,direct_queue,b2b,combo,incoming,direct_budget,revision=revision,pieces_placed=pieces_placed,
                                sent_upper_bound=sent_upper_bound,meter_confident=meter_confident,ruleset=ruleset,
                                combo_table=combo_table,rounding=rounding,opening_active=opening_active,opening_remaining=opening_remaining,garbage_mask=garbage_mask)
    if _deep_obsolete(revision):return None
    best=PvpPlan(direct,board,piece,False,False,next_piece,hold,b2b,combo,
                 revision,queue_revision,int(incoming or 0),combat_profile(board,incoming)[0],
                 pieces_placed,sent_upper_bound) if direct else None
    alternative=next_piece if hold is None else hold
    hold_ms=total_budget_ms-(time.perf_counter()-deep_started)*1000
    if consider_hold and hold_ms>=1.0 and not _deep_obsolete(revision):
        other_queue=queue[1:] if hold is None else queue
        other=recommend_attack_pvp(board,alternative,other_queue,b2b,combo,incoming,hold_ms,revision=revision,pieces_placed=pieces_placed,
                                     sent_upper_bound=sent_upper_bound,meter_confident=meter_confident,ruleset=ruleset,
                                combo_table=combo_table,rounding=rounding,opening_active=opening_active,opening_remaining=opening_remaining,garbage_mask=garbage_mask)
        if _deep_obsolete(revision):return None
        if other is not None and (best is None or other.score>best.move.score+.15):
            best=PvpPlan(other,board,piece,True,False,next_piece,hold,b2b,combo,
                         revision,queue_revision,int(incoming or 0),combat_profile(board,incoming)[0],
                         pieces_placed,sent_upper_bound)
    if best is not None:
        best.opening_active=bool(opening_active)
        if opening_active:
            best.opening_setup_gain=quad_well_potential(best.move.board,queue)
            # Only the current move and a *known* next move can justify an
            # immediate replacement. Do not infer a certain 10-15 line spike
            # from deeper branches that may later change.
            first=attack_estimate(best.move,b2b,combo,ruleset,combo_table,rounding)
            second=best.move.next_move
            second_attack=(attack_estimate(second,first.next_b2b,first.next_combo,
                                           ruleset,combo_table,rounding).guaranteed
                           if second is not None else 0)
            best.opening_burst_gain=first.guaranteed+second_attack
        following=best.move.next_move
        # Strict gate: observed queue + known HOLD state + exact predicted next.
        best.prepared_safe=bool(next_piece is not None and hold!='?' and following is not None and following.piece==next_piece)
    if b2b>0 and best is not None:
        from chain_director import preserve_b2b
        best=preserve_b2b(best,board,piece,queue,b2b,incoming,
                          deadline=None,revision=revision)
    return _guard_internal_stack(best, board, piece, queue, hold, allow_hold,
                                 b2b,combo,incoming,revision,queue_revision,
                                 pieces_placed,sent_upper_bound,ruleset,combo_table,rounding)


def guard_fast_stack(plan, board, piece, next_piece=None, hold='?',
                     allow_hold=True, b2b=0, combo=-1, incoming=0,
                     ruleset='standard',combo_table='multiplier',rounding='auto'):
    """Compare a risky native Fast placement against a legal emergency move.

    No external solver or GUI timing is touched. Used only in the Fast worker.
    """
    from survival_policy1195 import dangerous_stack, safer_replacement
    if (plan is None or plan.move is None or
            not dangerous_stack(board,plan.move.board,plan.move.lines,incoming)):
        return plan
    rescue=emergency_fast_plan(board,piece,next_piece,hold,allow_hold,
                               b2b,combo,incoming,ruleset,combo_table,rounding)
    if safer_replacement(plan,rescue,incoming):
        return rescue
    return plan


def _guard_internal_stack(plan, board, piece, queue, hold, allow_hold,
                          b2b, combo, incoming, revision, queue_revision,
                          pieces_placed, sent_upper_bound, ruleset,
                          combo_table, rounding):
    """Independent rescue comparison for dangerous speculative internal moves.

    Called in the Deep worker, not in the Qt hot path. Existing route proofs,
    CC2 exclusivity, and queue/HOLD facts are preserved.
    """
    if plan is None or _deep_obsolete(revision):
        return plan
    from survival_policy1195 import dangerous_stack, safer_replacement
    if not dangerous_stack(board,plan.move.board,plan.move.lines,incoming):
        return plan
    nxt=queue[0] if queue else None
    rescue=emergency_fast_plan(board,piece,nxt,hold,allow_hold,
                               b2b,combo,incoming,ruleset,combo_table,rounding)
    if not safer_replacement(plan,rescue,incoming) or _deep_obsolete(revision):
        return plan
    replacement=PvpPlan(rescue.move,board,piece,rescue.use_hold,False,
                        nxt,hold,b2b,combo,revision,queue_revision,
                        int(incoming or 0),combat_profile(board,incoming)[0],
                        pieces_placed,sent_upper_bound)
    replacement.continuation_source='survival_rescue'
    replacement.prepared_safe=False
    return replacement


@dataclass
class CC2WorkerResult:
    """Cross-process diagnostic, never a substitute for verified PvpPlan."""
    plan: object
    status: str
    details: dict=field(default_factory=dict)


def plan_move_cc2_diagnostic_worker(args):
    """CC2-only: bounded asynchronous probe, no internal planning fallback."""
    if _deep_obsolete(args[9]):return CC2WorkerResult(None,'STALE_REQUEST')
    try:
        from tbp_bridge import external_plan_diagnostic
        details={}
        plan,status=external_plan_diagnostic(args,timeout=.30,rejection_counts=details)
        if _deep_obsolete(args[9]):return CC2WorkerResult(None,'STALE_REQUEST')
        return CC2WorkerResult(plan,status,details)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        return CC2WorkerResult(None,'WORKER_ERROR:'+type(exc).__name__)


@dataclass
class HybridWorkerResult:
    """One CC2 observation and the independently chosen hybrid plan.

    The status does not imply adoption or execution. This wrapper is used only
    by GUI hybrid requests, leaving the original worker API backward compatible.
    """
    plan: object
    cc2_status: str
    cc2_details: dict=field(default_factory=dict)
    cc2_elapsed_ms: float=0.0
    native_fallback: bool=False


def plan_move_hybrid_diagnostic_worker(args):
    """Report *every* CC2 attempt before an ordinary native fallback.

    Never spend CPU on native fallback for a superseded revision. No GUI,
    filesystem export, command execution or input injection occurs here.
    """
    if _deep_obsolete(args[9]):
        return HybridWorkerResult(None,'STALE_REQUEST')
    begun=time.perf_counter()
    details={}
    try:
        from tbp_bridge import external_plan_diagnostic
        candidate,status=external_plan_diagnostic(args,timeout=.075,
                                                 rejection_counts=details)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        candidate=None
        status='WORKER_ERROR:'+type(exc).__name__
    elapsed=(time.perf_counter()-begun)*1000.0
    if _deep_obsolete(args[9]):
        return HybridWorkerResult(None,'STALE_REQUEST',details,elapsed)
    if candidate is not None:
        return HybridWorkerResult(candidate,status,details,elapsed,False)
    native=plan_move_pvp(*args)
    return HybridWorkerResult(native,status,details,elapsed,True)


def plan_move_pvp_worker(args, external_tbp=False):
    """Separate CC2-only / native-only / hybrid / legacy bundled strategies.

    CC2-only is FAIL CLOSED: missing/invalid CC2 response returns None. It may
    never silently become a native move. All external moves still pass TBP
    snapshot, route and board safety validation before reaching this worker.
    """
    from engine_policy import normalized_worker_mode
    strategy = normalized_worker_mode(external_tbp)
    if strategy == 'internal':
        return plan_move_pvp(*args)
    if _deep_obsolete(args[9]):
        return None
    if strategy in ('cc2', 'hybrid'):
        try:
            from tbp_bridge import maybe_external_plan
            candidate=maybe_external_plan(args, True)
            if candidate is not None and not _deep_obsolete(args[9]):
                return candidate
        except (OSError, ValueError, TypeError, RuntimeError):
            pass
        if strategy == 'cc2':
            return None
    if strategy == 'bundled':
        try:
            from tbp_bridge import maybe_bundled_plan
            candidate=maybe_bundled_plan(args, True)
            if candidate is not None and not _deep_obsolete(args[9]):
                return candidate
        except (OSError, ValueError, TypeError, RuntimeError):
            pass
    return plan_move_pvp(*args)


def projected_cells(board,cells):
    """Straight-down landing from the actual observed position."""
    if any(not (0<=x<10 and 0<=y<20) or board[y]&(1<<x) for x,y in cells): return None
    out=list(cells)
    while all(y+1<20 and not board[y+1]&(1<<x) for x,y in out):
        out=[(x,y+1) for x,y in out]
    return tuple(sorted(out))


def plan_nonsonic(board,piece,next_piece=None,hold='?',allow_hold=True,b2b=0,combo=-1):
    """Fail-safe attack fallback for empirically unreliable sonic drop.

    This runs in the Fast worker, not the GUI/input thread. Every target must
    have a route from spawn WITHOUT the soft key; no speculative T-spin path.
    The next-piece advisory is disabled to prevent a stale tactical plan from
    bypassing the next turn's explicit path validation.
    """
    from motion import route_to, cells_at
    can_hold=(allow_hold and hold!='?' and (hold is not None or next_piece is not None))
    options=[(piece,False)]
    alt=next_piece if hold is None else hold
    if can_hold and alt in BASE and alt!=piece:options.append((alt,True))
    ranked=[]
    for p,swapped in options:
        for move in placements(board,p):
            if move.spin or move.tactical:continue
            _,height,holes,*_=board_metrics(move.board)
            if height >= 19:continue
            score=value(move.board,'attack')+fast_attack_reward(move)
            if b2b>0:
                from chain_director import fast_chain_penalty
                score += fast_chain_penalty(move,b2b,board)
            ranked.append((score,swapped,move))
    ranked.sort(key=lambda entry: entry[0],reverse=True)
    for score,swapped,move in ranked:
        route=route_to(board,move.piece,cells_at(move.piece,(3,0,0)),
                       tuple(sorted(move.cells)),move.spin)
        if route is None or any(action=='soft' for action,_ in route):continue
        move.score=score
        return Plan(move,board,piece,swapped,False,next_piece,hold,b2b,combo,
                    continuation_source='safe_nonsonic')
    return None
