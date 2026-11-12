"""Evidence-only recovery for a settled board shifted by incoming garbage.

No guessed pixels, no historical piece reuse: the entire *current* 20x10 grid
must contain (1) all translated old settled cells, (2) 1-hole filled rows
at the bottom, and (3) exactly four surplus cells forming a known active
piece. Callers must still require consecutive frames before key input.
"""
from garbage_transition import verified_rise
from engine import identify


def recover_risen_piece(grid, locked_board, expected_piece=None, max_rise=18):
    """Return (settled board, active piece, four cells, rise) or None.

    Verify all fixed-row pixels using bitmasks before materializing any
    tetromino cells. This changes neither the exact-garbage requirement nor
    the four-mino shape proof. Support all 1..18 shifts allowed by the exact
    transition classifier, rather than silently failing for 13..18 rows.
    """
    if (len(locked_board)!=20 or len(grid)!=20 or
            expected_piece is not None and expected_piece not in 'IOTJLSZ'):
        return None
    try:
        if any(type(r) is not int or not 0<=r<=1023 for r in locked_board):
            return None
        observed=tuple(sum(1<<x for x,occupied in enumerate(row) if occupied)
                       for row in grid if len(row)==10)
    except (TypeError,ValueError):
        return None
    if len(observed)!=20 or observed[-1].bit_count()!=9:
        # Every exact garbage insertion ends in a nine-mino row. Most normal
        # frames fail this cheap check; avoid probing 18 impossible shifts.
        return None
    max_rise=max(0,min(18,int(max_rise)))
    for rise in range(1,max_rise+1):
        # Every newly inserted row must be 9/10 occupied, and all existing
        # fixed rows must be preserved exactly by the physical rise.
        garbage=observed[-rise:]
        if any(r.bit_count()!=9 for r in garbage):
            continue
        settled=tuple(locked_board[rise:])+garbage
        if verified_rise(locked_board,settled)!=rise:
            continue
        delta=[]
        count=0
        for y,(fixed,seen) in enumerate(zip(settled,observed)):
            if fixed & ~seen:
                break
            extra=seen & ~fixed
            if extra:
                count+=extra.bit_count()
                if count>4:
                    break
                delta.append((y,extra))
        else:
            if count!=4:
                continue
            cells=[]
            for y,mask in delta:
                while mask:
                    bit=mask & -mask
                    cells.append((bit.bit_length()-1,y))
                    mask^=bit
            piece=identify(cells)
            if piece is not None and (expected_piece is None or piece==expected_piece):
                return settled,piece,tuple(cells),rise
    return None


def proven_active_reference(grid, references):
    """Require a fresh four-mino active piece against an independent board.

    Used to authorize a lowered (but nonzero) minimum background sample
    ONLY on very high stacks. No arbitrary unknown cell is filled in.
    """
    from vision import against_board
    for board,expected_piece in references:
        if board is None or len(board)!=20:continue
        direct=against_board(grid,board,expected_piece)
        if direct is not None:return direct
        shifted=recover_risen_piece(grid,board,expected_piece)
        if shifted is not None:return shifted[:3]
    return None


class ConfirmedLegacyFallback:
    """Two fresh grayscale frames + independently verified four-mino state.

    This NEVER infers unknown pixels and never reuses an old frame as input.
    Called only after the calibrated color classifier reported UNKNOWN.
    Both snapshots must agree, match a physically provable current piece on
    an independently observed settled board, and arrive within 250 ms.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_grid=None
        self.last_proof=None
        self.last_at=None

    def consider(self, grid, references, now):
        if not references or grid is None:
            self.reset()
            return None
        proof=proven_active_reference(grid,references)
        if proof is None:
            self.reset()
            return None
        serial=tuple(tuple(bool(c) for c in row) for row in grid)
        identity=(proof[0],proof[1],tuple(sorted(proof[2])))
        valid_pair=(serial==self.last_grid and identity==self.last_proof and
                    self.last_at is not None and 0 < now-self.last_at<=.25)
        if valid_pair:
            self.reset()
            return grid
        self.last_grid=serial
        self.last_proof=identity
        self.last_at=float(now)
        return None
