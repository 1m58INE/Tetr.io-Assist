"""Strict, observable garbage insertion; never infer it from a meter alone.

Rows have 10 bits, top-to-bottom.  This module is safe to import in the GUI
thread: all operations are tiny, deterministic, and allocation bounded.
"""

def verified_rise(before, after):
    """0=same board, positive=exact shifted board, None=unexplained change.

    A rise is certified only if every pre-existing visible row survives the
    shift AND each new bottom row is a 9-cell garbage row.  Simultaneous
    non-garbage board modifications intentionally fail closed.
    """
    if len(before) != 20 or len(after) != 20:
        return None
    if any(not isinstance(row, int) or row < 0 or row > 1023 for row in (*before, *after)):
        return None
    if tuple(before) == tuple(after):
        return 0
    for inserted in range(1, 19):
        if tuple(after[:-inserted]) != tuple(before[inserted:]):
            continue
        if all(row.bit_count() == 9 for row in after[-inserted:]):
            return inserted
    return None


def mask_after_rise(before, mask, after, inserted):
    """Promote only visually proven new garbage; preserve old provenance."""
    if (len(mask) != 20 or verified_rise(before, after) != inserted or inserted <= 0
            or any(not isinstance(m, int) or m < 0 or m > 1023
                   or m & ~int(b) for m,b in zip(mask,before))):
        raise ValueError('unverified garbage rise or stale garbage provenance')
    return tuple(mask[inserted:]) + tuple(after[-inserted:])
