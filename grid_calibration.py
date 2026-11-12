"""Physical-pixel 10x20 PLAYFIELD and separate SPAWN calibration.

Only geometry is computed here; this module does not capture, press keys,
change settings, or attempt to interpret pixels as trustworthy blocks.
The GUI must explicitly confirm the final overlay before saving it.
"""
from __future__ import annotations
import math


def precise_board_grid(board):
    """True for a near-square 10x20 crop; reject old 10x22 selections.

    The total height must match twice the width within a few physical pixels,
    rather than the old permissive 0.42..0.58 aspect window. Previous settings
    are intentionally still LOADABLE so the one-click 22-row splitter works.
    """
    if not isinstance(board, dict) or set(board) != {'left','top','width','height'}:
        return False
    if any(type(board[k]) is not int for k in ('left','top','width','height')):
        return False
    w,h=board['width'],board['height']
    return w>=100 and h>=200 and abs(h-2*w)<=max(3,round(w*.01))


def _inside(region,screen):
    return (region['left'] >= screen['left'] and
            region['top'] >= screen['top'] and
            region['left']+region['width'] <= screen['left']+screen['width'] and
            region['top']+region['height'] <= screen['top']+screen['height'])


def calibrated_pair(cell_width,cell_height,board_left,board_bottom,screen=None,spawn_rows=2):
    """Construct (board,spawn) from a visible tile and playfield bottom-left.

    Pixel coordinates refer to the captured monitor, NOT scaled Qt logical
    pixels. The user's tile drag must approximate a complete physical square;
    the horizontal pitch is used for both axes to avoid drift from mismatched
    10/20 subdivisions. No stale or guessed geometry is silently accepted.
    """
    vals=(cell_width,cell_height,board_left,board_bottom)
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in vals):
        return None
    if type(spawn_rows) is not int or not (1<=spawn_rows<=4):
        return None
    if not (12<=cell_width<=240 and 12<=cell_height<=240):
        return None
    if abs(cell_width-cell_height)>max(2,cell_width*.08):
        return None
    pitch=round(cell_width)
    if not 12<=pitch<=240:
        return None
    x,y=round(board_left),round(board_bottom)
    board=dict(left=x,top=y-20*pitch,width=10*pitch,height=20*pitch)
    spawn=dict(left=x,top=board['top']-spawn_rows*pitch,
               width=board['width'],height=spawn_rows*pitch)
    if not precise_board_grid(board):
        return None
    if screen is not None:
        if not isinstance(screen,dict) or set(screen)!= {'left','top','width','height'}:
            return None
        if any(type(screen[k]) is not int for k in screen) or min(screen['width'],screen['height'])<1:
            return None
        if not _inside(board,screen) or not _inside(spawn,screen):
            return None
    return board,spawn


def diagnostic_geometry(board, spawn=None):
    """Human-readable warning for existing geometry; no automatic mutation."""
    if precise_board_grid(board):
        from spawn_vision import spawn_pair_rows
        if spawn is None:
            return 'BOARD 10×20 정사각 격자 · SPAWN 미설정'
        if spawn_pair_rows(board,spawn) is not None:
            return 'BOARD + SPAWN 격자 정렬 통과'
        return 'SPAWN 위치가 BOARD와 일치하지 않습니다.'
    if not isinstance(board,dict):
        return 'BOARD 영역을 먼저 설정하세요.'
    from spawn_vision import split_legacy_22
    if split_legacy_22(board) is not None:
        return '기존 22행 캡처 감지 · 위의 22행 분리 버튼 또는 1칸 정밀 설정을 사용하세요.'
    return 'BOARD 격자 비율 불일치 · 1칸 정밀 설정으로 실제 10×20을 지정하세요.'


def initial_guide(board, screen):
    """Suggest a visible 10x20 guide without modifying saved settings.

    Recognize legacy 22-row geometry, or reuse a precise BOARD, otherwise
    place a conservative movable draft on the selected monitor. The preview
    is NOT considered calibrated until the user explicitly confirms it.
    """
    from spawn_vision import split_legacy_22
    if not isinstance(screen,dict) or not all(k in screen for k in ('left','top','width','height')):
        return None
    if any(type(screen[k]) is not int for k in ('left','top','width','height')):
        return None
    if min(screen['width'],screen['height'])<400:
        return None
    candidate = board if precise_board_grid(board) else None
    if candidate is None:
        pair=split_legacy_22(board)
        if pair is not None:
            candidate=pair[0]
    if candidate is not None:
        pitch=round(candidate['width']/10)
        anchor=(candidate['left'],candidate['top']+candidate['height'])
        pair=calibrated_pair(pitch,pitch,*anchor,screen=screen)
        if pair is not None:
            return pitch,anchor,'기존 BOARD에서 10×20 + SPAWN 초안을 자동 구성했습니다.'
    pitch=max(12,min(240,round(screen['height']/30)))
    x=screen['left']+round((screen['width']-pitch*10)/2)
    bottom=screen['top']+screen['height']-round(pitch*.7)
    max_bottom=screen['top']+screen['height']
    min_bottom=screen['top']+22*pitch
    bottom=max(min_bottom,min(max_bottom,bottom))
    pair=calibrated_pair(pitch,pitch,x,bottom,screen=screen)
    if pair is None:
        return None
    return pitch,(x,bottom),'BOARD 저장값이 없어 화면 중앙에 가이드 초안을 표시했습니다.'
