"""Conservative JSON settings boundary. No command execution or dynamic objects.

Settings are local and never transmitted. Reject oversized/invalid rectangles
before mss capture and save atomically in the same directory.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile

from windows_input import DEFAULT, valid_binding

MAX_SETTINGS_BYTES = 128 * 1024
REGIONS = ('board', 'spawn', 'next', 'next2', 'next3', 'hold', 'garbage')
NUMBERS = {'pulse_ms': (4, 24), 'threshold': (20, 250)}
ENUMS = {
    'ruleset': ('quick_play', 'standard'),
    'combo_table': ('multiplier', 'none'),
    'rounding': ('auto', 'down', 'rng'),
    'human_intensity': ('off', 'low', 'medium', 'high'),
    'vision_mode': ('auto', 'strict', 'legacy'),
    'engine_mode': ('internal', 'cc2', 'hybrid', 'bundled'),
}
BOOLEANS = ('opening_enabled', 'blank_hold_on_new_game', 'external_tbp', 'bundled_tbp', 'safe_single_guard')


def valid_region(region):
    if not isinstance(region, dict) or set(region) != {'left', 'top', 'width', 'height'}:
        return False
    if any(type(region.get(k)) is not int for k in ('left','top','width','height')):
        return False
    left, top, w, h = (region[k] for k in ('left','top','width','height'))
    return (-32768 <= left <= 32768 and -32768 <= top <= 32768 and
            10 <= w <= 8192 and 10 <= h <= 8192 and w*h <= 33_554_432)


# Per-region pixel budgets prevent a poisoned settings.json or accidental
# selection from forcing tens of megapixels of capture at 125Hz.  High-DPI
# board areas remain possible; queue/hold regions should be much smaller.
PIXEL_LIMITS = {
    'board': 5_000_000, 'spawn': 900_000,
    'next': 500_000, 'next2': 500_000, 'next3': 500_000,
    'hold': 500_000, 'garbage': 850_000,
}


def valid_region_for(name, region):
    if name not in PIXEL_LIMITS or not valid_region(region):
        return False
    if region['width'] * region['height'] > PIXEL_LIMITS[name]:
        return False
    if name=='board' and not .42 < region['width']/region['height'] < .58:
        return False
    if name=='spawn' and not 2.1 < region['width']/region['height'] < 11.5:
        return False
    return True


def sanitize_settings(raw):
    """Never propagate nested arbitrary objects to Qt/mss/SendInput."""
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for name in REGIONS:
        if valid_region_for(name,raw.get(name)):
            clean[name] = {k: raw[name][k] for k in ('left','top','width','height')}
    # A SPAWN ROI must be physically adjacent to the real 20-row board.
    # Never accept a poisoned or stale unaligned rectangle.
    if 'spawn' in clean and 'board' in clean:
        from spawn_vision import spawn_pair_rows
        if spawn_pair_rows(clean['board'], clean['spawn']) is None:
            clean.pop('spawn',None)
    for name, (lo,hi) in NUMBERS.items():
        value=raw.get(name)
        if type(value) is int and lo <= value <= hi:
            clean[name] = value
    for name, choices in ENUMS.items():
        value=raw.get(name)
        if isinstance(value,str) and value in choices:
            clean[name] = value
    for name in BOOLEANS:
        if type(raw.get(name)) is bool:
            clean[name] = raw[name]
    raw_bindings=raw.get('bindings')
    if isinstance(raw_bindings,dict):
        clean_bindings={}
        for action in DEFAULT:
            val=raw_bindings.get(action)
            if isinstance(val,str) and len(val) <= 12 and valid_binding(val):
                clean_bindings[action] = val.upper().strip()
        if clean_bindings:
            clean['bindings']=clean_bindings
    return clean


def load_settings(path):
    path=Path(path)
    try:
        if path.stat().st_size > MAX_SETTINGS_BYTES:
            return {}, '설정 파일 크기가 제한을 초과하여 무시했습니다.'
        raw=json.loads(path.read_text(encoding='utf-8'))
        return sanitize_settings(raw), None
    except FileNotFoundError:
        return {}, None
    except (OSError,UnicodeError,ValueError,TypeError):
        return {}, '설정 파일이 손상되어 안전한 기본값을 적용했습니다.'


def save_settings(path, settings):
    path=Path(path)
    clean=sanitize_settings(settings)
    content=json.dumps(clean,ensure_ascii=False,separators=(',',':'))
    if len(content.encode('utf-8')) > MAX_SETTINGS_BYTES:
        raise ValueError('settings size limit exceeded')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='.settings-',
                                         suffix='.tmp',dir=str(path.parent),delete=False) as handle:
            tmp=Path(handle.name)
            try: os.chmod(tmp,0o600)
            except OSError: pass
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:
        if tmp is not None:
            try: tmp.unlink(missing_ok=True)
            except OSError: pass
    return clean
