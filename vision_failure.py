"""Bounded, read-only evidence for COLOR board failures.

Never guesses a game state or changes input targets. Only the already selected
BOARD region is sampled, and this runs once per recognition-failure episode.
"""
from __future__ import annotations
import numpy as np
from vision import sample_indices

MAX_CELLS = 8
_OFFSETS_25 = np.array((.30, .40, .50, .60, .70))
_OFFSETS_49 = np.array((.22, .31, .40, .50, .60, .69, .78))


def _sample_cell(a, x, y, offsets, dx=0, dy=0):
    h, w = a.shape[:2]
    ys = np.clip(((y+offsets)*h/20).astype(np.int32)+dy, 0, h-1)
    xs = np.clip(((x+offsets)*w/10).astype(np.int32)+dx, 0, w-1)
    return a[ys[:, None], xs[None, :], :3].reshape(-1, 3)


def failure_evidence(bgr, background, unknown, *, max_cells=MAX_CELLS):
    """Freeze pixel/background evidence at the *first* failed frame.

    Values are deliberately limited to at most eight failing cells and 9/25
    raw samples per cell.  Pixel-shift probes are diagnostic only: a shift
    that seems better on one image must NEVER auto-adjust the capture region.
    """
    a = np.asarray(bgr)
    if (a.ndim != 3 or a.shape[2] < 3 or a.dtype != np.uint8
            or not 80 <= a.shape[0] <= 8192 or not 40 <= a.shape[1] <= 8192
            or a.shape[0]*a.shape[1] > 5_000_000
            or not .42 < a.shape[1]/a.shape[0] < .58):
        raise ValueError('Invalid board evidence frame')
    bg = None
    if background is not None:
        bg_array = np.asarray(background, dtype=np.float32)
        if bg_array.shape != (3,) or not np.isfinite(bg_array).all():
            raise ValueError('Invalid calibrated background')
        bg = bg_array
    result = {'board_size_px': [int(a.shape[1]), int(a.shape[0])],
              'background_bgr_at_failure': None if bg is None else [round(float(v), 2) for v in bg],
              'coordinate_order': 'y,x (top-left origin, 20 rows x 10 columns)',
              'interpretation': 'ROI shift probes are diagnostic; the software does not alter the ROI.',
              'cells': []}
    if unknown is None:
        return result
    mask = np.asarray(unknown, dtype=bool)
    if mask.shape != (20,10):
        return result
    ys, xs = sample_indices(a.shape[0],a.shape[1])
    sparse = a[ys,xs,:3].reshape(20,10,9,3)
    positions = np.argwhere(mask)[:max(0,min(int(max_cells),MAX_CELLS))]
    for y,x in positions:
        x=int(x); y=int(y)
        samples=sparse[y,x].astype(np.int16)
        entry={'cell_yx':[y,x], 'sparse9_bgr':samples.astype(int).tolist()}
        if bg is not None:
            diff=np.max(np.abs(samples-bg),axis=-1)
            dense=_sample_cell(a,x,y,_OFFSETS_25).astype(np.int16)
            diff25=np.max(np.abs(dense-bg),axis=-1)
            surface=_sample_cell(a,x,y,_OFFSETS_49).astype(np.int16)
            diff49=np.max(np.abs(surface-bg),axis=-1)
            entry.update(sparse_strong=int(np.count_nonzero(diff>=43)),
                         sparse_moderate=int(np.count_nonzero(diff>=22)),
                         dense25_strong=int(np.count_nonzero(diff25>=43)),
                         dense25_moderate=int(np.count_nonzero(diff25>=22)),
                         surface49_strong=int(np.count_nonzero(diff49>=43)),
                         surface49_moderate=int(np.count_nonzero(diff49>=22)),
                         shift_probe_strong={
                             f'dx{dx:+d}_dy{dy:+d}':int(np.count_nonzero(
                                 np.max(np.abs(_sample_cell(a,x,y,_OFFSETS_25,dx,dy).astype(np.int16)-bg),axis=-1)>=43))
                             for dx,dy in ((0,0),(-2,0),(2,0),(0,-2),(0,2))})
        result['cells'].append(entry)
    return result
