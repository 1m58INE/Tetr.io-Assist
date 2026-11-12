"""Season 2 Multiplier-combo estimates for TETR.IO.

Reference: https://tetris.wiki/TETR.IO#Combo_System:_%22Multiplier%22
This does *not* claim full server/room custom-rule parity: spin recognition,
all-clear stacking, surge packet timing, and other rule modifiers can differ.

The Mini reward uses the Season-2 base table (Mini Single=0, Double=1).
The return type separates guaranteed attack (safe for garbage cancellation)
from expected attack (useful for strategic ordering in RNG-rounding rooms).
All calculations in this module are called by the Deep solver, not by the
ordinary latency-sensitive Fast planner.
"""
from dataclasses import dataclass
import math

SUPPORTED_COMBO_TABLES=('multiplier','none')
SUPPORTED_ROUNDING=('down','rng')


@dataclass(frozen=True, slots=True)
class AttackEstimate:
    base: float
    combo: int
    b2b_bonus: int
    multiplier: float
    expected: float
    guaranteed: int
    surge: int
    next_b2b: int
    next_combo: int
    # Worst-case total generation used for conservative sent-attack upper bounds.
    ceiling: int


def surge_lines(b2b, ruleset='standard'):
    # Internal counter: first difficult clear => 1; B2Bx4 => counter 5.
    b2b=max(0,int(b2b))
    if b2b<5:return 0
    return b2b-4 if ruleset=='quick_play' else b2b-1


def attack_estimate(move,b2b=0,combo=-1,ruleset='standard',
                    combo_table='multiplier',rounding='auto'):
    """Estimate generated attack, not received garbage or attack distribution.

    combo holds the value *before* this placement: -1 before the first clear,
    then 0 after the first clear, 1 after the second, etc.
    """
    if combo_table not in SUPPORTED_COMBO_TABLES:
        raise ValueError('Unsupported combo table: '+str(combo_table))
    if rounding=='auto':rounding='rng' if ruleset=='quick_play' else 'down'
    if rounding not in SUPPORTED_ROUNDING:
        raise ValueError('Unsupported rounding mode: '+str(rounding))
    lines=int(move.lines)
    b2b=max(0,int(b2b))
    if lines==0:
        return AttackEstimate(0.0,-1,0,1.0,0.0,0,0,b2b,-1,0)

    if move.spin=='full':base=(0,2,4,6)[min(lines,3)]
    elif move.spin=='mini':
        # Season-2 All-Mini+: base Mini Single=0, Double=1, Triple=2.
        # A Mini Single only produces a line at zero combo via the existing
        # B2B bonus. Previous v1.1.9.8 over-counted every Mini by one.
        base=(0,0,1,2)[min(lines,3)]
    else:base=(0,0,1,2,4)[min(lines,4)]
    perfect=not any(move.board)
    # Season 2: All Clear attack is 5 (standard) or 3 (Quick Play).
    if perfect:base=max(base,3 if ruleset=='quick_play' else 5)
    if ruleset=='quick_play' and lines==1 and not move.spin and combo<0:
        base=max(base,1)

    difficult=lines==4 or bool(move.spin) or perfect
    b2b_bonus=1 if difficult and b2b>0 else 0
    base+=b2b_bonus
    next_b2b=b2b+(2 if perfect and ruleset=='quick_play' else 1) if difficult else 0
    surge=surge_lines(b2b,ruleset) if not difficult else 0
    next_combo=max(-1,int(combo))+1
    multiplier=1+0.25*next_combo if combo_table=='multiplier' else 1.0

    # Wiki's separate formula when ordinary base attack = 0. Trigger it only
    # from 2-combo, and NEVER treat accumulated Surge as multiplicative base.
    if combo_table=='multiplier' and base==0 and next_combo>=2:
        raw=math.log1p(1.25*next_combo)
    else:
        raw=base*multiplier
    guaranteed=int(math.floor(raw+1e-10))
    expected=raw if rounding=='rng' else float(guaranteed)
    return AttackEstimate(float(base),next_combo,b2b_bonus,multiplier,
                          expected+surge,guaranteed+surge,surge,
                          next_b2b,next_combo,math.ceil(raw-1e-10)+surge)
