"""Optional bounded input pulse rhythm for permitted offline testing.

No added idle time, fake mistakes, replayed human traces, or attempt to evade
platform bot/automation detection. Only a zero-mean pattern of small key-down
pulse length variations after successful *observed* inputs. Timing stays under
existing controller/foreground checks. This is not a guarantee of equal PPS.
"""
from __future__ import annotations

LEVELS={
    'off': (0,),
    'low': (0,0,1,0,0,-1),
    'medium': (0,1,0,-1,1,-1),
    'high': (0,2,1,-1,-2,0),
}

class InputRhythm:
    def __init__(self,level='off'):
        self.level=level if level in LEVELS else 'off'
        self.counter=0

    def reset(self,level=None):
        self.counter=0
        if level is not None:
            self.level=level if level in LEVELS else 'off'

    def pulse(self,base_ms,verified_streak=0,recovery_ms=0,actions=()):
        base=max(4,min(24,int(base_ms)))
        if (self.level=='off' or verified_streak<8 or recovery_ms>0 or
                base<6 or base>22 or any(a in ('drop','soft','hold') for a in actions)):
            return base
        offsets=LEVELS[self.level]
        offset=offsets[self.counter % len(offsets)]
        self.counter+=1
        return max(4,min(24,base+offset))
