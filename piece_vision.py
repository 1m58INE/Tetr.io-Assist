"""Conservative, bounded recognition for cropped NEXT/HOLD tetromino previews.

Shape is the primary identity.  Hue is only auxiliary evidence: player skins
are customizable, and gray/ghost/decorative pixels must never become a piece
by guessing the nearest default RGB value.  A truncated or multi-piece ROI is
UNKNOWN.  Nothing in this module touches input devices or external services.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import numpy as np

from engine import BASE, norm, rotations

# The 7 actual tetromino shapes, all four 90-degree orientations.  No texture
# or fixed palette is needed to distinguish S from O (or any other pair).
_TEMPLATES = {
    norm(points): name
    for name in BASE
    for _rotation, points in rotations(name)
}

# Only use these fallback colors when geometric evidence is genuinely absent.
# BGR references approximate common default skins, NOT a universal truth.
_PALETTE = {
    'I': (45, 205, 220), 'O': (225, 195, 35), 'T': (160, 70, 205),
    'J': (60, 85, 215), 'L': (225, 135, 35), 'S': (80, 195, 70),
    'Z': (220, 65, 65),
}
_PALETTE_UNIT = np.array([_PALETTE[p] for p in 'IOTJLSZ'], dtype=np.float32)
_PALETTE_UNIT /= np.maximum(_PALETTE_UNIT.max(axis=1, keepdims=True), 1)


@dataclass(frozen=True)
class PieceReading:
    piece: str | None
    method: str  # SHAPE / UNKNOWN
    confidence: float
    reason: str
    color_hint: str | None = None


def _validate_pixels(pixels):
    pixels = np.asarray(pixels)
    if (pixels.ndim != 3 or pixels.shape[2] not in (3, 4)
            or pixels.dtype != np.uint8):
        raise ValueError('NEXT/HOLD image must contain uint8 BGR or BGRA pixels')
    height, width = pixels.shape[:2]
    if (not 12 <= height <= 1024 or not 12 <= width <= 1024
            or height * width > 500_000):
        raise ValueError('NEXT/HOLD image outside bounded capture dimensions')
    return pixels[:, :, :3]


def _color_hint(pixels):
    """Diagnostic only. A skin may deliberately swap all seven colors."""
    if pixels.shape[0] == 0:
        return None
    rgb = np.median(pixels[:, ::-1].astype(np.float32), axis=0)
    if np.max(rgb) < 45:
        return None
    unit = rgb / np.max(rgb)
    distances = np.linalg.norm(_PALETTE_UNIT - unit, axis=1)
    best = np.argsort(distances)
    if distances[best[0]] < .24 and distances[best[1]] - distances[best[0]] >= .07:
        return 'IOTJLSZ'[int(best[0])]
    return None



def _has_square_boundaries(a, left, top, tile_w, tile_h, columns, rows):
    """Require tile seams for shapes that fill their entire rectangle.

    Without seams, a rounded/gradient O-sized UI panel would be identical to
    four colored tiles.  Compare *bands*, not single pixels, so gradient or
    random per-pixel noise cannot manufacture a boundary.
    """
    evidence = 0
    width = a.shape[1]
    height = a.shape[0]
    half_strip = max(1, int(min(tile_h, tile_w) * .18))

    def strip(x0, x1, y0, y1):
        x0 = max(0, int(round(x0))); x1 = min(width, int(round(x1)))
        y0 = max(0, int(round(y0))); y1 = min(height, int(round(y1)))
        if x1 <= x0 or y1 <= y0:
            return None
        return np.median(a[y0:y1, x0:x1, :3], axis=(0, 1))

    for x in range(1, columns):
        bx = left + x*tile_w
        for y in range(rows):
            cy = top + (y+.5)*tile_h
            boundary=strip(bx-2,bx+2,cy-half_strip,cy+half_strip)
            core_a=strip(bx-.6*tile_w,bx-.4*tile_w,cy-half_strip,cy+half_strip)
            core_b=strip(bx+.4*tile_w,bx+.6*tile_w,cy-half_strip,cy+half_strip)
            if (boundary is not None and core_a is not None and core_b is not None
                    and min(np.max(abs(boundary-core_a)),np.max(abs(boundary-core_b))) >= 12):
                evidence += 1
    for y in range(1, rows):
        by = top + y*tile_h
        for x in range(columns):
            cx = left + (x+.5)*tile_w
            boundary=strip(cx-half_strip,cx+half_strip,by-2,by+2)
            core_a=strip(cx-half_strip,cx+half_strip,by-.6*tile_h,by-.4*tile_h)
            core_b=strip(cx-half_strip,cx+half_strip,by+.4*tile_h,by+.6*tile_h)
            if (boundary is not None and core_a is not None and core_b is not None
                    and min(np.max(abs(boundary-core_a)),np.max(abs(boundary-core_b))) >= 12):
                evidence += 1
    return evidence >= 2

def read_piece(pixels) -> PieceReading:
    """Read a single, tightly cropped tetromino without unsafe guesses.

    Uses the chromatic *interiors* of four square tiles, not exact theme RGB.
    Respects rotation and rejects cropped/multiple/ambiguous shapes.  A
    correctly colored but non-tetromino patch is never silently accepted.
    """
    a = _validate_pixels(pixels)
    # A piece identity is determined by four squares, not by full-resolution
    # textures. Sparse isotropic sampling bounds cost for 500k-pixel NEXT ROIs
    # without resizing/interpolation or adding a dependency.  Normal small
    # thumbnails take the unmodified path.
    stride = max(1, ceil(max(a.shape[:2]) / 112))
    if stride > 1:
        a = a[::stride, ::stride]
    p = a.astype(np.int16)
    hi = p.max(axis=2)
    lo = p.min(axis=2)
    # Reject monochrome labels/lines and distinguish tinted tile faces from
    # black or desaturated UI.  This intentionally fails closed on monochrome
    # custom skins until a separate approved calibration is available.
    chroma = hi - lo
    foreground = (chroma >= 30) & (hi >= 70)
    count = int(np.count_nonzero(foreground))
    if count < 32:
        return PieceReading(None, 'UNKNOWN', 0.0, '색상 블록 근거 부족')
    ys, xs = np.nonzero(foreground)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    width, height = right - left, bottom - top
    if min(width, height) < 8:
        return PieceReading(None, 'UNKNOWN', 0.0, '너무 작은 블록 형태')
    # A featureless saturated rectangle can be a UI panel, not four tiles.
    # Refuse to invent a shape when no per-tile texture/seam exists.
    if np.max(np.ptp(a[foreground], axis=0)) < 10:
        return PieceReading(None, 'UNKNOWN', 0.0, '타일 경계가 없는 단색 영역')
    color_hint = _color_hint(a[foreground][::5])

    found = []
    # A tetromino bounding rectangle is 1x4 / 4x1 / 2x2 / 2x3 / 3x2.
    # Tile pitches must be square even when the ROI includes black padding.
    for columns, rows in ((1, 4), (4, 1), (2, 2), (2, 3), (3, 2)):
        tile_w, tile_h = width / columns, height / rows
        if min(tile_w, tile_h) < 6 or abs(tile_w / tile_h - 1) > .19:
            continue
        occupied = []
        ambiguous = False
        inside = []
        for y in range(rows):
            for x in range(columns):
                # Evaluate the core of each tile: contours and grid lines
                # are unreliable in textured/animated TETR.IO skins.
                x0 = max(0, left + int((x + .24) * tile_w))
                x1 = min(a.shape[1], left + int((x + .76) * tile_w))
                y0 = max(0, top + int((y + .24) * tile_h))
                y1 = min(a.shape[0], top + int((y + .76) * tile_h))
                if x1 <= x0 or y1 <= y0:
                    ambiguous = True
                    break
                fraction = float(np.mean(foreground[y0:y1, x0:x1]))
                if fraction >= .62:
                    occupied.append((x, y))
                    inside.append(fraction)
                elif fraction > .16:
                    ambiguous = True
                    break
            if ambiguous:
                break
        if ambiguous or len(occupied) != 4:
            continue
        # The occupied cells must exactly span the inferred bounds, ruling
        # out arbitrary big panels whose centers happen to resemble a piece.
        if (min(x for x, y in occupied) != 0 or max(x for x, y in occupied) != columns - 1
                or min(y for x, y in occupied) != 0 or max(y for x, y in occupied) != rows - 1):
            continue
        shape = norm(occupied)
        piece = _TEMPLATES.get(shape)
        if piece and (piece not in ('I', 'O') or
                      _has_square_boundaries(a, left, top, tile_w, tile_h, columns, rows)):
            found.append((piece, float(min(inside))))
    if len(found) != 1:
        return PieceReading(None, 'UNKNOWN', 0.0,
                            '4칸 형태 불일치 또는 ROI 중복·잘림', color_hint)
    piece, fill = found[0]
    # Reject confident-looking garbage colors only if the crop is suspicious;
    # the geometric identity itself is stable across player-selected skins.
    return PieceReading(piece, 'SHAPE', min(.99, .84 + .15 * fill),
                        '4칸 회전 형태 검증', color_hint)


def classify_piece_pixels(pixels):
    """Compatibility API used by GUI, NEXT/HOLD and bounded ROI worker."""
    return read_piece(pixels).piece
