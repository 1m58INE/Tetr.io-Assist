"""Pure, inspectable bounds checks for screenshot regions.

All region coordinates use physical desktop pixels. Nothing outside the
verified foreground game's window is authorized for automatic capture.
"""
from __future__ import annotations

REGION_NAMES = ('spawn', 'next', 'next2', 'next3', 'hold', 'garbage')


def valid_rect(rect):
    try:
        left, top, right, bottom = (rect[k] for k in ('left','top','right','bottom'))
    except (KeyError,TypeError):
        return False
    return (all(type(x) is int for x in (left,top,right,bottom))
            and -32768 <= left < right <= 32768
            and -32768 <= top < bottom <= 32768)


def region_issues(rect, regions):
    """Return [(name, reason)] without sampling any outside-window pixels.

    A dict of optional ROIs is preferred; legacy sequences remain supported.
    Five or six extra ROIs are both valid (SPAWN adds a sixth one).
    """
    if isinstance(regions, dict):
        items = tuple(regions.items())
    else:
        try:
            items = tuple(zip(REGION_NAMES, regions))
        except TypeError:
            return [('CAPTURE', '영역 목록이 올바르지 않습니다.')]
    if len(items) > len(REGION_NAMES):
        return [('CAPTURE', f'최대 {len(REGION_NAMES)}개 영역을 지원합니다.')]
    if not valid_rect(rect):
        return [('GAME', '게임 창의 화면 좌표를 확인할 수 없습니다.')]
    problems=[]
    for name,region in items:
        if name not in REGION_NAMES:
            problems.append((str(name), '알 수 없는 영역입니다.'))
            continue
        if not isinstance(region,dict) or set(region)!= {'left','top','width','height'}:
            problems.append((name,'영역 형식 오류'))
            continue
        try:
            x,y,w,h=(region[k] for k in ('left','top','width','height'))
        except (KeyError,TypeError):
            problems.append((name,'영역 형식 오류'))
            continue
        if (any(type(v) is not int for v in (x,y,w,h)) or w<=0 or h<=0):
            problems.append((name,'영역 크기 오류'))
        elif not (rect['left']<=x and rect['top']<=y and
                  x+w<=rect['right'] and y+h<=rect['bottom']):
            problems.append((name,f'게임 창 바깥: ({x},{y}) {w}×{h}px'))
    return problems


def format_region_issues(issues):
    if not issues:
        return '캡처 영역이 게임 창 안에 있습니다.'
    return ' · '.join(f'{name.upper()} {message}' for name,message in issues)
