"""Verified-board survival vetoes for speculative multi-turn offensive moves.

This is a *risk selector*, not an alternative route generator. The engine only
replaces an unsafe plan with a separately route-checked legal emergency plan.
Scores are never treated as server-authoritative garbage measurements.
"""

def signature(board):
    from engine import board_metrics
    _, height, holes, covered, *_ = board_metrics(board)
    return height, holes, covered


def setup_is_safe(source, first, final, *, incoming=0, steps=2, middle=()):
    """Allow a verified TSD setup only with bounded peak and recovery cost.

    A three-ply all-quiet buildup must retain more headroom than a 2-ply
    finisher. This is intentionally strict after a stack-up regression.
    """
    if incoming:
        return False
    src_h,src_holes,_=signature(source)
    a_h,a_holes,_=signature(first)
    end_h,end_holes,_=signature(final)
    if steps>=3 and src_h>7 or steps<3 and src_h>9:
        return False
    if a_h>min(11,src_h+2) or a_holes>src_holes+1:
        return False
    if end_h>min(10,src_h+1) or end_holes>src_holes:
        return False
    for intermediate in middle:
        mid_h,mid_holes,_=signature(intermediate)
        if mid_h>min(12,src_h+3) or mid_holes>src_holes+2:
            return False
    return True


def dangerous_stack(source, after, lines=0, incoming=0):
    """Flag a *candidate* for independent route-checked survival comparison.

    Do not ban every non-clear on a high board: sometimes no clear exists.
    """
    sh,holes,_=signature(source)
    h,new_holes,_=signature(after)
    residual=max(0,int(incoming))
    if lines<=0 and (h>=17 or h+residual>=18):
        return True
    if lines<=0 and sh>=10 and h>sh:
        return True
    if lines<=0 and sh>=8 and h>sh+2:
        return True
    if new_holes>holes+2 and h>=12:
        return True
    return False


def safer_replacement(original, rescue, incoming=0):
    """Return True only for a demonstrably better physical board outcome."""
    if rescue is None or original is None or rescue.move is None or original.move is None:
        return False
    old_h,old_holes,old_covered=signature(original.move.board)
    new_h,new_holes,new_covered=signature(rescue.move.board)
    old_risk=max(0,old_h+max(0,int(incoming))-16)
    new_risk=max(0,new_h+max(0,int(incoming))-16)
    if (new_h>=old_h and new_holes>=old_holes and new_risk>=old_risk):
        return False
    # An improved height must not bury substantially more holes.
    if new_holes>old_holes+1 or new_covered>old_covered+6:
        return False
    return (new_h<=old_h-2 or new_risk<old_risk or
            (new_h<=old_h and new_holes<old_holes) or
            (rescue.move.lines>original.move.lines and new_h<=old_h))
