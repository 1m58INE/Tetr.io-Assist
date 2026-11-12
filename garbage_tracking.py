"""Track only garbage cells confirmed by an observable post-lock garbage rise.

Unknown cells (e.g., when enabling F9 midgame) stay unknown rather than being
misclassified. 20 row bitmasks mirror the settled playfield.
"""

EMPTY_MASK = (0,) * 20


def advance_mask(board, mask, cells, inserted=0, observed=None):
    """Apply a known placement's full-row clears, then confirmed garbage rise."""
    if len(board) != 20 or len(mask) != 20:
        raise ValueError('20 board/mask rows required')
    settled = list(board)
    for x,y in cells:
        if not 0 <= x < 10 or not 0 <= y < 20:
            raise ValueError('invalid move coordinates')
        settled[y] |= 1 << x
    remaining = [int(mask[y]) for y, row in enumerate(settled) if row != 1023]
    remaining = [0] * (20 - len(remaining)) + remaining
    if inserted:
        if observed is None or not 0 < inserted < 20 or len(observed) != 20:
            raise ValueError('observed board required for garbage rise')
        # Confirmed 9-cell garbage rows; board tracking has already validated
        # the upward shift and count. The mask never fabricates missing rows.
        remaining = remaining[inserted:] + [int(row) for row in observed[-inserted:]]
    return tuple(remaining)


def garbage_access_cost(board, mask):
    """Lower is easier to downstack: cells above observed garbage holes."""
    if not any(mask):
        return 0
    cost = 0
    for y, garb in enumerate(mask):
        if not garb:
            continue
        holes = (~garb) & 1023
        for x in range(10):
            if holes & (1 << x):
                cost += sum(bool(board[row] & (1 << x)) for row in range(y))
    return cost


def garbage_progress(board, mask, cells):
    """Return deep-only score for actually removed garbage and opened holes."""
    if not any(mask):
        return 0.0
    after_mask = advance_mask(board, mask, cells)
    before = sum(int(row).bit_count() for row in mask)
    after = sum(int(row).bit_count() for row in after_mask)
    # Every nine garbage minos cleared is roughly one actual garbage row.
    cleared = max(0, before - after) / 9.0
    # Positive only if the candidate reduces obstruction of a known hole.
    # Unknown/uncaptured garbage never gets fake progress credit.
    before_access = garbage_access_cost(board, mask)
    new_board=list(board)
    for x,y in cells:new_board[y] |= 1 << x
    left=[r for r in new_board if r != 1023]
    new_board=tuple([0]*(20-len(left))+left)
    after_access=garbage_access_cost(new_board, after_mask)
    return cleared*13.0 + (before_access-after_access)*1.8
