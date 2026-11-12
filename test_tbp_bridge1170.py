"""Deterministic protocol harness; NOT a Cold Clear 2 binary or Windows test."""
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from engine import BASE, placements, plan_move_pvp_worker
from motion import cells_at, SHAPES
from tbp_bridge import (ANCHORS, ORIENTATIONS, TBPClient, as_pvp_plan,
                        parse_tbp_move, snapshot_to_tbp)

EMPTY = (0,)*20


def tbp_coordinates(piece, rotation, cells):
    """Convert known motion pose into a TBP rotation-center fixture."""
    ax,ay=ANCHORS[piece][rotation]
    shape=SHAPES[piece][rotation]
    ox=min(x for x,y in cells)-min(x for x,y in shape)
    oy=min(y for x,y in cells)-min(y for x,y in shape)
    return {'location':{'type':piece,'orientation':ORIENTATIONS[rotation],
                        'x':ox+ax,'y':19-(oy+ay)},'spin':'none'}


class TBPAdapterTests(unittest.TestCase):
    def test_board_conversion_and_unknown_hold(self):
        b=list(EMPTY);b[-1]=3;b[-3]=8
        v=snapshot_to_tbp(tuple(b),'T',('I','S'),None,0,-1)
        self.assertEqual(v['type'],'start')
        self.assertEqual(v['queue'],['T','I','S'])
        self.assertEqual((len(v['board']),len(v['board'][0])),(40,10))
        self.assertEqual(v['board'][0][:4],['G','G',None,None])
        self.assertEqual(v['board'][2][3],'G')
        self.assertEqual(v['board'][20],[None]*10)
        with self.assertRaises(ValueError):
            snapshot_to_tbp(EMPTY,'T',('I',),'?',0,-1)
        with self.assertRaises(ValueError):
            snapshot_to_tbp(EMPTY,'T',('I','?', 'O'),None,0,-1)

    def test_every_orientation_geometry_match(self):
        # Compare official CC2 rotation-center conventions on all 7 pieces,
        # including the special I and O center definitions.
        import tbp_bridge
        for p in BASE:
            for r in range(4):
                shape=SHAPES[p][r]
                ox=4-min(x for x,y in shape);oy=16-min(y for x,y in shape)
                expected=tuple(sorted((ox+x,oy+y) for x,y in shape))
                raw=tbp_coordinates(p,r,expected)
                ax,ay=ANCHORS[p][r]
                loc=raw['location']
                result=cells_at(p,(loc['x']-ax,19-loc['y']-ay,r))
                self.assertEqual(result,expected,(p,r))

    def test_external_valid_placement(self):
        for p in BASE:
            m=next(m for m in placements(EMPTY,p) if m.rotation==0 and min(x for x,y in m.cells)>=2)
            raw=tbp_coordinates(p,0,m.cells)
            decoded=parse_tbp_move(EMPTY,p,(),None,True,raw)
            self.assertIsNotNone(decoded,p)
            move,swapped=decoded
            self.assertEqual(move.cells,tuple(sorted(m.cells)))
            self.assertFalse(swapped)

    def test_hold_reject_unknown_and_mismatch(self):
        m=next(m for m in placements(EMPTY,'I') if m.rotation==0)
        raw=tbp_coordinates('I',0,m.cells)
        self.assertIsNone(parse_tbp_move(EMPTY,'T',('O',),'?',True,raw))
        self.assertIsNone(parse_tbp_move(EMPTY,'T',('O',),None,True,raw))
        self.assertIsNone(parse_tbp_move(EMPTY,'T',('I',),None,False,raw))
        self.assertIsNotNone(parse_tbp_move(EMPTY,'T',('I',),None,True,raw))

    def test_reject_floating_overlap_and_invalid_spin(self):
        m=next(m for m in placements(EMPTY,'T') if m.rotation==0)
        raw=tbp_coordinates('T',0,m.cells)
        self.assertIsNotNone(parse_tbp_move(EMPTY,'T',(),None,True,raw))
        floating={**raw,'location':{**raw['location'],'y':raw['location']['y']+3}}
        self.assertIsNone(parse_tbp_move(EMPTY,'T',(),None,True,floating))
        covered=list(EMPTY);covered[19]=1023
        self.assertIsNone(parse_tbp_move(tuple(covered),'T',(),None,True,raw))
        self.assertIsNone(parse_tbp_move(EMPTY,'T',(),None,True,{**raw,'spin':'full'}))
        self.assertIsNone(parse_tbp_move(EMPTY,'T',(),None,True,{**raw,'location':{**raw['location'],'x':True}}))

    def test_plan_source_and_fallback(self):
        m=next(iter(placements(EMPTY,'T')))
        raw=tbp_coordinates('T',m.rotation,m.cells)
        p=as_pvp_plan(EMPTY,'T',('I','O'),None,True,0,-1,0,[raw],3,1)
        self.assertEqual(p.continuation_source,'external_tbp')
        self.assertFalse(p.prepared_safe)
        self.assertEqual((p.deep_revision,p.queue_revision),(3,1))
        # No external executable on this runner: opt-in must still return an
        # original native plan, not an error or phantom external result.
        args=(EMPTY,'T',('I','O'),None,True,0,-1,0,9,0,0,20,0,False,'standard','multiplier','auto',False,12,None)
        result=plan_move_pvp_worker(args,external_tbp=True)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.continuation_source,'external_tbp')

    def test_snapshot_attack_pressure_guard(self):
        m=next(iter(placements(EMPTY,'T')))
        raw=tbp_coordinates('T',m.rotation,m.cells)
        self.assertIsNone(as_pvp_plan(EMPTY,'T',('I','O'),None,True,0,-1,4,[raw]))

    def test_external_next_prefetch_happens_only_with_confirmed_hold(self):
        from concurrent.futures import Future
        from unittest.mock import MagicMock
        from test_offensive098 import Offensive098Tests
        class RecordingPool:
            def __init__(self):self.calls=[]
            def submit(self,fn,*args):
                self.calls.append((fn,args))
                f=Future();f.set_result(plan_move_pvp_worker(*args));return f
        app=Offensive098Tests().app()
        app.deep_pool=RecordingPool()
        app.external_tbp=MagicMock()
        app.external_tbp.isChecked.return_value=True
        app.hold_enabled=MagicMock()
        app.hold_enabled.isChecked.return_value=True
        app.auto.held=None  # confirmed empty, not UNKNOWN
        app._schedule_committed_continuation(('T','I'),12,9)
        self.assertEqual(len(app.deep_pool.calls),1)
        _,arguments=app.deep_pool.calls[0]
        self.assertEqual(arguments[0][3:5],(None,True))
        self.assertIs(arguments[0][3],None)
        self.assertTrue(arguments[1])
        # Unobserved HOLD cannot invoke the external adviser speculatively.
        other=Offensive098Tests().app()
        other.deep_pool=RecordingPool()
        other.external_tbp=app.external_tbp
        other.hold_enabled=app.hold_enabled
        other.auto.held='?'
        other._schedule_committed_continuation(('T',),12,9)
        _,arguments=other.deep_pool.calls[0]
        self.assertEqual(len(arguments),1)
        self.assertEqual(arguments[0][3:5],('?',False))

    def test_solver_identity_is_rejected_without_manual_trust_override(self):
        from tbp_bridge import TBPError
        with tempfile.TemporaryDirectory() as d:
            script=Path(d)/'fake.py'
            script.write_text("import json,time; print(json.dumps({'type':'info','name':'Imposter'}),flush=True); time.sleep(1)\n")
            bot=TBPClient([sys.executable,str(script)],handshake_timeout=.25)
            try:
                self.assertIsNone(bot.suggest(snapshot_to_tbp(EMPTY,'O',(),None,0,-1),timeout=.05))
            finally:bot.close()

    def test_protocol_process_handshake_and_timeout(self):
        mock = '''
import json,sys,time
print(json.dumps({'type':'info','name':'mock','features':[]}),flush=True)
for line in sys.stdin:
 m=json.loads(line)
 if m['type']=='rules': print(json.dumps({'type':'ready'}),flush=True)
 if m['type']=='start':
  assert len(m['board'])==40 and m['queue'][0]=='O'
 if m['type']=='suggest':
  print(json.dumps({'type':'suggestion','moves':[{'location':{'type':'O','orientation':'north','x':4,'y':0},'spin':'none'}]}),flush=True)
 if m['type']=='stop':pass
'''
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'tbp_mock.py';f.write_text(textwrap.dedent(mock))
            client=TBPClient([sys.executable,str(f)],handshake_timeout=1.5,require_identity=False)
            try:
                snap=snapshot_to_tbp(EMPTY,'O',('I',),None,0,-1)
                one=client.suggest(snap,timeout=.3)
                two=client.suggest(snap,timeout=.3)
                self.assertEqual(one,two)
                self.assertEqual(one[0]['location']['type'],'O')
            finally:client.close()
            hung=Path(d)/'hung.py'
            hung.write_text("import time;time.sleep(2)\n")
            stop=TBPClient([sys.executable,str(hung)],handshake_timeout=.055)
            try:
                before=time.monotonic()
                self.assertIsNone(stop.suggest(snap,timeout=.05))
                self.assertLess(time.monotonic()-before,.5)
            finally:stop.close()

if __name__=='__main__':unittest.main()
