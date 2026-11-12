"""Bounded long-chain preference, strictly subordinate to survival.

These are *ranking* utilities, not claims that unseen pieces or server attacks
exist.  Only an observed, executable difficult clear can extend the counter.
"""
import math


def chain_increment(move, b2b, combat, height):
    """Value a genuine difficult clear and discourage breaking a live chain.

    Quiet placements retain B2B, but never earn a fictitious clear bonus.  At
    high stack or under urgent pressure these incentives collapse to zero.
    """
    if b2b <= 0 or combat not in ('ATTACK', 'COUNTER'):
        return 0.0
    if height >= 14:
        return 0.0
    from engine import is_difficult
    weight = max(0.0, 1.0 - max(0, height - 9) / 5.0)
    strength = min(19.0, 4.0 + 3.0 * math.log1p(b2b)) * weight
    if b2b < 3:
        strength *= .35  # don't freeze the opening chain before it can grow
    if move.lines and not is_difficult(move):
        return -strength
    if is_difficult(move):
        return min(7.0, strength * .47)
    return min(2.0, strength * .12)  # quiet placement preserves, never extends


def verified_pair_bonus(first, second, before_b2b, combat, height):
    """Extra value only when the exact observed following piece is reachable.

    Do not count quiet moves as attacks; do not celebrate a sequence that
    breaks the chain on a normal Single/Double/Triple.
    """
    if combat not in ('ATTACK', 'COUNTER') or before_b2b <= 0 or height >= 13:
        return 0.0
    from engine import is_difficult
    if first.lines and not is_difficult(first):
        return 0.0
    if second.lines and not is_difficult(second):
        return -min(11.0, 3.0 + math.log1p(before_b2b)*2.0)
    if is_difficult(second):
        return min(8.0, 2.0 + math.log1p(before_b2b)*1.7)
    return 0.0


def preserve_portfolio(scored, portfolio, b2b, combat, beam, factory=()):
    """Bounded beam keeping independent B2B, quiet and verified T setups.

    Factory entries receive a reserved slot only when an actual next-piece
    T-spin is reachable.  Never evict *every* highest-score candidate.
    """
    from engine import is_difficult, board_metrics
    results = []
    def append_once(record):
        if all(record[1] is not entry[1] for entry in results):
            results.append(record)
    reserve = 1 if factory else 0
    chain_active=(b2b>0 and combat in ('ATTACK', 'COUNTER'))
    for record in portfolio[:max(1,beam-2 if chain_active else beam-reserve)]:
        append_once(record)
    if factory:
        # A real T-slot beats an unverified geometric guess, but cannot
        # commandeer the whole beam or the emergency survival branch.
        append_once(factory[0][0])
    if chain_active:
        for predicate in (lambda m: is_difficult(m), lambda m: not m.lines):
            for record in scored:
                move=record[1]
                if predicate(move) and all(move is not entry[1] for entry in results):
                    _,height,holes,*_=board_metrics(move.board)
                    if height<=12 and holes<=2:
                        append_once(record)
                        break
    for record in (*portfolio,*scored):
        if len(results)>=beam:break
        append_once(record)
    return results[:beam]

from dataclasses import dataclass
import time

@dataclass(frozen=True, slots=True)
class ChainCandidate:
    first: object
    next_move: object
    verified_depth: int
    projected_b2b: int
    difficult_clears: int
    explored_nodes: int


def search_observed_chain(board, piece, queue, b2b, combo, deadline, width=6,
                          max_nodes=280):
    """Explicit rolling B2B continuation search over ONLY observed pieces.

    This is an advisory worker-side search, never a replacement for scene
    confirmation or input route checks.  A no-line placement preserves B2B,
    but DOES NOT count as a difficult clear.  State dedup keeps it bounded.
    """
    from engine import BASE, placements, tactical_moves, board_metrics, observed_queue_prefix
    from combo_rules import attack_estimate
    if b2b<0 or piece not in BASE or time.perf_counter()>=deadline:
        return None
    sequence=(piece,)+observed_queue_prefix(queue,2)
    if len(sequence)<2:return None
    width=max(2,min(12,int(width)))
    max_nodes=max(32,min(500,int(max_nodes)))
    # state (board, b2b, combo, accumulated score, moves, difficult count)
    frontier=[(board,b2b,combo,0.0,(),0)]
    explored=0;last=()
    for ply,p in enumerate(sequence):
        candidates=[]
        for current_board, bb, co, score, path, clears in frontier:
            if time.perf_counter()>=deadline or explored>=max_nodes:break
            if p=='T':
                moves=tactical_moves(current_board,p,limit=245,max_depth=10,
                                     deadline=deadline)
            else:
                moves=placements(current_board,p)
            for move in moves:
                if explored>=max_nodes or time.perf_counter()>=deadline:break
                explored+=1
                _,height,holes,*_=board_metrics(move.board)
                if height>=17 or holes>=7:continue
                estimate=attack_estimate(move,bb,co)
                difficult=int(bool(move.lines and (move.lines==4 or move.spin or not any(move.board))))
                broken=bool(bb>0 and move.lines and not difficult)
                continuation=(bb>0 and not broken)
                # Incremental terminal-quality penalty, rather than awarding
                # favorable absolute board scores on every hypothetical ply.
                risk=(holes*2.6+max(0,height-10)*2.8)
                credit=estimate.guaranteed*3.2 + (7.0 if difficult else 0.0)
                credit+=4.0 if difficult and bb>0 else 0.0
                if broken:credit-=min(28.0, 9.0+4.0*math.log1p(bb))
                elif continuation and not move.lines:credit+=1.2
                if height>=14:credit-=12.0
                # The first two plies matter much more than uncertain tails.
                new_score=score+(credit-risk*(.7 if ply==0 else .32))
                candidates.append((move.board,estimate.next_b2b,estimate.next_combo,
                                   new_score,path+(move,),clears+difficult))
        if not candidates:break
        # Every candidate has been computed for known sequence only.  A board
        # and chain state may have multiple routes; retain the best score.
        unique={}
        for entry in candidates:
            key=(entry[0],entry[1],entry[2])
            if key not in unique or entry[3]>unique[key][3]:unique[key]=entry
        frontier=sorted(unique.values(),key=lambda s:s[3],reverse=True)[:width]
        last=frontier
        if time.perf_counter()>=deadline or explored>=max_nodes:break
    if not last:return None
    # Prefer projected B2B extension among competitive scores, never choose
    # fatal stacks merely to inflate a counter.
    eligible=([state for state in last if state[1]>0 and state[5]>0]
              if b2b==0 else last)
    if not eligible:return None  # Never pretend that a quiet opener initiated B2B.
    candidate=max(eligible,key=lambda e:e[3]+min(14,(e[1]-b2b)*4))
    path=candidate[4]
    if not path:return None
    # Selected continuation is path-checked once in the Deep worker; an
    # optimistic tactile spin label without a route must not enter GUI state.
    from motion import can_route
    previous=board
    for move in path[:2]:
        if time.perf_counter()>=deadline:
            return None
        if not can_route(previous,move.piece,move.cells,move.spin):
            return None
        previous=move.board
    # A future move is an advisory only; the controller must observe matching
    # next piece and verify the actual after-board before reusing it.
    return ChainCandidate(path[0],path[1] if len(path)>1 else None,
                          len(path),candidate[1],candidate[5],explored)
