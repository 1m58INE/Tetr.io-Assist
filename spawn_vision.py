"""Optional 1-4 row spawn buffer, aligned above a real 10x20 PLAYFIELD.

Only positive, color-supported 4-mino observations are accepted. This module
never makes keyboard input and never invents occupied cells from stale frames.
When enabled, a single MSS rectangle covers both user-selected areas; there
is no extra capture call per frame compared with legacy BOARD mode.
"""
from __future__ import annotations
import numpy as np
from engine import identify


def spawn_pair_rows(board, spawn):
    """Return aligned spawn row count (1..4), or None. Dicts are untrusted."""
    if not isinstance(board, dict) or not isinstance(spawn, dict):
        return None
    try:
        bw,bh=board['width'],board['height']
        sw,sh=spawn['width'],spawn['height']
        if any(type(x) is not int for x in (
                bw,bh,sw,sh,board['left'],board['top'],spawn['left'],spawn['top'])):
            return None
        if min(bw,bh,sw,sh)<10:
            return None
        cellw=bw/10.0; cellh=bh/20.0
        if abs(cellw-cellh)>max(1.5,cellw*.07):
            return None
        if abs(spawn['left']-board['left'])>2 or abs(sw-bw)>2:
            return None
        if abs(spawn['top']+sh-board['top'])>2:
            return None
        rows=round(sh/cellh)
        if not 1 <= rows <=4 or abs(sh-rows*cellh)>max(2,cellh*.12):
            return None
        return rows
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        return None


def spawn_from_board(board, rows=2):
    """Derive an exact aligned SPAWN strip from an existing 10x20 BOARD.

    UI may offer this as a one-click option instead of requiring a precise
    second drag. Caller must still validate the on-screen region and ensure
    the game window contains the generated coordinates before arming input.
    """
    if not isinstance(board,dict) or type(rows) is not int or not 1<=rows<=4:
        return None
    try:
        h=round(board['height']*rows/20)
        spawn=dict(left=board['left'],top=board['top']-h,
                   width=board['width'],height=h)
        return spawn if spawn_pair_rows(board,spawn)==rows else None
    except (KeyError,ValueError,TypeError):
        return None


def split_legacy_22(region):
    """Turn old 10x22-ish capture into 10x2 spawn + true 10x20 board."""
    if not isinstance(region,dict):return None
    try:
        l,t,w,h=(region[k] for k in ('left','top','width','height'))
        if any(type(v) is not int for v in (l,t,w,h)) or w<100 or h<200:
            return None
        # Require a credible two-extra-row legacy capture; never silently
        # mangle an already-correct 10x20 setting.
        if not (2.08 < h/w < 2.33):return None
        upper=round(h*2/22)
        spawn=dict(left=l,top=t,width=w,height=upper)
        board=dict(left=l,top=t+upper,width=w,height=h-upper)
        return (board,spawn) if spawn_pair_rows(board,spawn)==2 else None
    except (KeyError,TypeError):return None


def capture_board_and_spawn(capture, board, spawn=None):
    """Capture once, slice two ROIs in-memory. Capture accepts an MSS dict."""
    if spawn is None:
        return np.asarray(capture.grab(board))[:,:,:3],None
    if spawn_pair_rows(board,spawn) is None:
        raise ValueError('SPAWN and BOARD grid alignment invalid')
    left=min(board['left'],spawn['left'])
    top=min(board['top'],spawn['top'])
    right=max(board['left']+board['width'],spawn['left']+spawn['width'])
    bottom=max(board['top']+board['height'],spawn['top']+spawn['height'])
    combined=np.asarray(capture.grab(dict(left=left,top=top,
                                          width=right-left,height=bottom-top)))[:,:,:3]
    def crop(region):
        y=region['top']-top;x=region['left']-left
        return combined[y:y+region['height'],x:x+region['width'],:]
    return crop(board),crop(spawn)


def _colored_spawn_cells(bgr, rows, background):
    """Read full-color minos only; reject white guides, noise, or UI decals."""
    a=np.asarray(bgr)
    if (a.ndim!=3 or a.dtype!=np.uint8 or a.shape[2]<3
            or a.shape[0]<rows*9 or a.shape[1]<80):
        return ()
    bg=np.asarray(background if background is not None else (0,0,0),dtype=np.float32)
    if bg.shape!=(3,):return ()
    off=np.array((.30,.50,.70))
    yy=np.minimum(((np.arange(rows)[:,None]+off)*a.shape[0]/rows).astype(int),a.shape[0]-1)
    xx=np.minimum(((np.arange(10)[:,None]+off)*a.shape[1]/10).astype(int),a.shape[1]-1)
    px=a[yy[:,None,:,None],xx[None,:,None,:],:3].reshape(rows,10,9,3).astype(np.float32)
    diff=np.max(np.abs(px-bg),axis=-1)
    chroma=px.max(axis=-1)-px.min(axis=-1)
    bright=px.max(axis=-1)
    # White projection is high-contrast, but not chromatic; never accept it.
    good=(diff>=46)&(chroma>=34)&(bright>=85)
    hits=np.count_nonzero(good,axis=-1)
    occupied=hits>=7
    # Ambiguous partly visible colored chunks must not be treated as a solid
    # piece. Fail closed rather than returning an invented 4-cell identity.
    if np.any((hits>=2)&(hits<7)):
        return ()
    cells=tuple((int(x),int(y)-rows) for y,x in np.argwhere(occupied))
    return cells if 1<=len(cells)<=4 else ()


def _legacy_spawn_cells(bgr, rows, threshold):
    """Conservative monochrome 9-sample SPAWN proof, no chroma dependency.

    All accepted cells require nearly complete bright interiors. Partial
    sampling artifacts reject the entire candidate, not just the cell.
    Unlike COLOR SPAWN this cannot tell white guides from real white minos;
    callers must still cross-check the fixed BOARD and require independent
    successive frames before any input is committed.
    """
    a=np.asarray(bgr)
    if (a.ndim!=3 or a.dtype!=np.uint8 or a.shape[2]<3
            or type(rows) is not int or not 1<=rows<=4
            or a.shape[0]<rows*9 or a.shape[1]<80
            or type(threshold) is not int or not 20<=threshold<=250):
        return ()
    offsets=np.array((.35,.5,.65))
    ys=np.minimum(((np.arange(rows)[:,None]+offsets)*a.shape[0]/rows).astype(int),a.shape[0]-1)
    xs=np.minimum(((np.arange(10)[:,None]+offsets)*a.shape[1]/10).astype(int),a.shape[1]-1)
    bright=a[ys[:,None,:,None],xs[None,:,None,:],:3].max(axis=-1)
    hits=np.count_nonzero(bright>=threshold,axis=(-1,-2))
    if np.any((hits>=2)&(hits<8)):
        return ()
    pts=np.argwhere(hits>=8)
    if not 1<=len(pts)<=4:
        return ()
    return tuple((int(x),int(y)-rows) for y,x in pts)


def fuse_spawn_active(spawn_bgr, board_grid, background, rows, previous=None, expected=None, *, reading="color", threshold=105):
    """Verify current tetromino from SPAWN plus visible BOARD fragments.

    Returns (locked_board, piece, canonical_cells) or None. Returns None
    when a partial piece cannot be separated from previously locked cells.
    A canonical y>=0 pose is supplied because the existing movement solver
    uses a 20-row coordinate system whose spawn position starts at row zero.
    """
    if board_grid is None or len(board_grid)!=20 or any(len(r)!=10 for r in board_grid):
        return None
    if reading=='legacy':
        upper=_legacy_spawn_cells(spawn_bgr,rows,threshold)
    elif reading=='color':
        upper=_colored_spawn_cells(spawn_bgr,rows,background)
    else:
        return None
    if not upper:return None
    observed=tuple(sum((1<<x) for x in range(10) if board_grid[y][x]) for y in range(20))
    if len(upper)==4:
        # A current piece may be fully above BOARD. Do not accidentally
        # promote a pure-white landing guide or a stale/changed board to
        # locked occupancy when a trusted previous/post-lock state exists.
        references=[r for r in (expected,previous)
                    if isinstance(r,tuple) and len(r)==20]
        if references and observed not in references:
            return None
        candidates=[(observed, ())]
    else:
        candidates=[]
        for prior in (expected,previous):
            if isinstance(prior,tuple) and len(prior)==20 and all(type(v) is int for v in prior):
                if all((v & obs)==v for v,obs in zip(prior,observed)):
                    delta=tuple((x,y) for y in range(20) for x in range(10)
                                if (observed[y]&~prior[y])&(1<<x))
                    if len(delta)+len(upper)==4:
                        candidates.append((prior,delta))
    for locked, extra in candidates:
        world=upper+extra
        piece=identify(world)
        if not piece:continue
        # Preserve SRS pose identity by shifting all coordinates together.
        canonical=tuple(sorted((x,y+rows) for x,y in world))
        if (any(y<0 or y>=20 or locked[y]&(1<<x) for x,y in canonical)
                or len(set(canonical))!=4):
            continue
        return locked,piece,canonical
    return None


def mask_confirmed_spawn_white_guide(board_bgr,spawn_bgr,background,uncertain,prior,rows):
    """Narrow fallback for a 4-row white projection, with CURRENT I in SPAWN.

    Preserve all non-white pixels. The caller MUST re-run COLOR recognition
    and verify the resulting locked board against the independently confirmed
    last board or expected post-lock board before using the frame for input.
    """
    if background is None or uncertain is None or prior is None or rows is None:
        return None
    a=np.asarray(board_bgr)
    bg=np.asarray(background,dtype=np.float32)
    if (a.ndim!=3 or a.dtype!=np.uint8 or a.shape[2]<3 or a.shape[0]<400
            or a.shape[1]<200 or bg.shape!=(3,) or np.max(bg)>20):
        return None
    sp=_colored_spawn_cells(spawn_bgr,rows,bg)
    if len(sp)!=4 or identify(sp)!='I' or len({y for x,y in sp})!=1:
        return None
    if not isinstance(prior,tuple) or len(prior)!=20:return None
    pts=np.argwhere(np.asarray(uncertain,dtype=bool))
    if len(pts)!=1:return None
    y,x=map(int,pts[0]);h,w=a.shape[:2]
    if y<1 or y>15 or any(prior[yy]&(1<<x) for yy in range(y,y+4)):
        return None
    def fraction(row,col):
        yl,yh=int((row+.22)*h/20),int((row+.78)*h/20)
        xl,xh=int((col+.22)*w/10),int((col+.78)*w/10)
        patch=a[yl:yh,xl:xh,:3].astype(np.int16)
        if not patch.size:return 0.0
        hi=patch.max(axis=-1);lo=patch.min(axis=-1)
        return float(np.mean((lo>=249)&(hi-lo<=5)))
    if not .20<=fraction(y,x)<=.78:return None
    if any(fraction(yy,x)<.88 for yy in range(y+1,y+4)):return None
    if fraction(y-1,x)>.03:return None
    for neighbor in (x-1,x+1):
        if 0<=neighbor<10 and any(fraction(yy,neighbor)>.06 for yy in range(y,y+4)):
            return None
    out=a.copy()
    yl,yh=int(y*h/20),min(h,int((y+5)*h/20))
    xl,xh=int(x*w/10),min(w,int((x+1)*w/10))
    patch=out[yl:yh,xl:xh,:3]
    lo=patch.min(axis=-1).astype(np.int16)
    hi=patch.max(axis=-1).astype(np.int16)
    mask=(lo>=249)&(hi-lo<=5)
    patch[mask]=np.clip(bg,0,255).astype(np.uint8)
    return out


def mask_solid_spawn_projection(board_bgr, spawn_bgr, background, rows, locked):
    """Suppress only a proven 4-cell, featureless pure-white I projection.

    This differs from the older partially-filled top-cell fallback: once the
    BOARD/2-row SPAWN areas are separated, the white landing guide may fill
    four *complete* BOARD cells. Callers must verify a fresh post-mask grid
    against `locked`, and accept it only when CURRENT I is visible in SPAWN.
    """
    a=np.asarray(board_bgr)
    if background is None or not isinstance(locked,tuple) or len(locked)!=20:
        return None
    bg=np.asarray(background,dtype=np.float32)
    if (a.ndim!=3 or a.dtype!=np.uint8 or a.shape[2]<3 or a.shape[0]<400
            or a.shape[1]<200 or bg.shape!=(3,) or np.max(bg)>20):
        return None
    sp=_colored_spawn_cells(spawn_bgr,rows,bg)
    if len(sp)!=4 or identify(sp)!='I' or len(set(y for x,y in sp))!=1:
        return None
    h,w=a.shape[:2]
    # Constant-cost vectorized prefilter. On 99% of frames there is no
    # full-white vertical column and the slower patch checks are skipped.
    yc=np.minimum(((np.arange(20)+.5)*h/20).astype(int),h-1)
    xc=np.minimum(((np.arange(10)+.5)*w/10).astype(int),w-1)
    centers=a[yc[:,None],xc[None,:],:3]
    center_hi=centers.max(axis=-1).astype(np.int16)
    center_lo=centers.min(axis=-1).astype(np.int16)
    center_white=(center_lo>=249)&(center_hi-center_lo<=5)
    # Four consecutive pure-white centers; the fast rejection uses exactly
    # one tiny vectorized array, no Python nested loop in the hot path.
    stacks=(center_white[1:16] & center_white[2:17]
            & center_white[3:18] & center_white[4:19])
    if not np.any(stacks):
        return None
    def frac(y,x):
        y0,y1=int((y+.22)*h/20),int((y+.78)*h/20)
        x0,x1=int((x+.22)*w/10),int((x+.78)*w/10)
        patch=a[y0:y1,x0:x1,:3].astype(np.int16)
        if patch.size==0:return 0.0
        lo=patch.min(axis=-1);hi=patch.max(axis=-1)
        return float(np.mean((lo>=249)&(hi-lo<=5)))
    for y0,x0 in np.argwhere(stacks):
        y=int(y0)+1; x=int(x0)
        if any(locked[yy]&(1<<x) for yy in range(y,y+4)):
            continue
        if frac(y-1,x)>.04 or (y+4<20 and frac(y+4,x)>.04):
            continue
        if any(frac(yy,x)<.90 for yy in range(y,y+4)):
            continue
        if any(0<=xx<10 and frac(yy,xx)>.07
               for xx in (x-1,x+1) for yy in range(y,y+4)):
            continue
        out=a.copy()
        y0,y1=int(y*h/20),int((y+4)*h/20)
        x0,x1=int(x*w/10),int((x+1)*w/10)
        patch=out[y0:y1,x0:x1,:3]
        lo=patch.min(axis=-1).astype(np.int16)
        hi=patch.max(axis=-1).astype(np.int16)
        white=(lo>=249)&(hi-lo<=5)
        patch[white]=np.clip(bg,0,255).astype(np.uint8)
        return out
    return None


def shifted_queue_after_verified_lock(previous_key, queue_slots, observed_piece, lock_confirmed):
    """Reuse already observed NEXT2/3 after a positively verified lock.

    This is not a 7-bag guess. A real capture of NEXT1 and the newly observed
    ACTIVE piece must agree with slot zero; only then can the previously read
    NEXT2 be the new NEXT1. HOLD swaps and unsynchronized queue observations
    fall back to a fresh capture. No state is ever predicted from stale time.

    Returns (new_next1, remaining_previews) or (None, ()).
    """
    valid = frozenset('IOTJLSZ')
    if not lock_confirmed or not isinstance(previous_key, tuple) or len(previous_key) < 3:
        return None, ()
    if (not isinstance(queue_slots, (tuple, list)) or len(queue_slots) < 2
            or type(observed_piece) is not str or observed_piece not in valid
            or previous_key[2] != observed_piece or queue_slots[0] != observed_piece
            or type(queue_slots[1]) is not str or queue_slots[1] not in valid):
        return None, ()
    result = [queue_slots[1]]
    for candidate in queue_slots[2:3]:
        if type(candidate) is not str or candidate not in valid:
            break
        result.append(candidate)
    return result[0], tuple(result)
