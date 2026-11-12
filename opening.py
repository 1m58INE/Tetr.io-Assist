"""Opening-only burst policy. Deep-only scoring; no capture or fast-path work.

These estimates are generated garbage, NOT confirmed lines delivered to opponents.
The opener stops irreversibly after the configured pieces/seconds/goal or a threat.
"""

from dataclasses import dataclass

OPENING_TARGET = 12
OPENING_PIECES = 24
OPENING_SECONDS = 12.0


@dataclass(frozen=True, slots=True)
class OpeningDecision:
    active: bool
    reason: str


def opening_decision(eligible, finished, placed, elapsed, generated,
                     incoming=0, target=OPENING_TARGET,
                     max_pieces=OPENING_PIECES, max_seconds=OPENING_SECONDS):
    if not eligible or finished:
        return OpeningDecision(False, 'off')
    if generated >= target:
        return OpeningDecision(False, 'target')
    if placed >= max_pieces or elapsed >= max_seconds:
        return OpeningDecision(False, 'timeout')
    if incoming >= 3:
        return OpeningDecision(False, 'pressure')
    return OpeningDecision(True, 'burst')


def opening_burst_bonus(attacks, remaining, finisher=False):
    """Rank near-term known attacks only. No reward for unsupported future pieces."""
    if remaining <= 0 or not attacks:
        return 0.0
    total = sum(max(0, int(attack)) for attack in attacks)
    count = min(total, remaining)
    # Cap all benefits once the target is reached; huge speculative banks do
    # not justify dangerous stacks. A nearby Quad/T-spin finisher is preferred.
    finish = 7.0 if finisher and total >= max(4, remaining - 3) else 0.0
    reached = 12.0 if total >= remaining else 0.0
    return count * 3.4 + reached + finish



def verified_i_well_bonus(board, future=()):
    """Bounded, queue-verified setup preference for the limited opener.

    Only reward an I well when the I is actually visible among the known next
    pieces. Do not invent unseen future bags, force a high 9-0 tower, or turn
    a blocked well into a 'promising' opener. Called only in Deep.
    """
    if 'I' not in future or len(board)!=20:
        return 0.0
    heights=[]
    for x in range(10):
        mask=1<<x
        top=next((r for r,row in enumerate(board) if row&mask),20)
        h=20-top
        if any(not (board[y]&mask) for y in range(top,20)):
            return 0.0  # buried hole makes the imagined well inaccessible
        heights.append(h)
    if max(heights)>=10:
        return 0.0
    best=0.0
    # A verified I-well may be in any of the 10 columns. It needs adjacent
    # support and unobstructed access from above; no imaginary bag forecast.
    for x in range(10):
        adjacent=([heights[1]] if x==0 else [heights[8]] if x==9
                  else [heights[x-1],heights[x+1]])
        depth=min(adjacent)-heights[x]
        if (max(adjacent)-min(adjacent)<=2 and 1<=depth<=5
                and not any(row&(1<<x) for row in board)):
            best=max(best,min(6.0,depth*1.25))
    return best


def quad_well_potential(board, future=()):
    """Observed-I-only near-complete four-row edge well potential, Deep-only.

    Never reward speculative stacking when the observed queue cannot finish a
    Tetris. A hole beneath the well or an unobserved I yields zero.
    """
    if len(board) != 20 or 'I' not in future or any(board[:10]):
        return 0.0
    i_distance=future.index('I')
    best=0.0
    for x in range(10):
        bit=1<<x
        lower=board[-4:]
        if any(row & bit for row in board):continue
        cells=sum((r & (1023 ^ bit)).bit_count() for r in lower)
        # At most four stack cells can arrive from each *observed* pre-I piece;
        # insist there is a plausible four-high finishing opportunity.
        if 36-cells > 4*i_distance:continue
        # A one-block wide hole above a four-high platform is not a Tetris.
        if any(row & bit for row in board[-8:-4]):continue
        full=sum((r & (1023 ^ bit)) == (1023 ^ bit) for r in lower)
        best=max(best,12+cells*.45+full*3)
    return best
