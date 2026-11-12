"""Exercise the actual App.tick/worker-result bridge with UI/OS adapters faked.
This is not a Qt, Windows capture, or real keyboard integration test.
"""
import importlib.util
import sys
import types
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from autoplay import AutoPlayer
from engine import BASE, norm, projected_cells
from vision import BoardTracker


def load_app():
    replacements={'mss':types.ModuleType('mss'),'PySide6':types.ModuleType('PySide6')}
    for suffix,names in {
        'QtCore':'Qt QRect QRectF QTimer Signal',
        'QtGui':'QColor QPainter QPen QImage QPixmap',
        'QtWidgets':'QApplication QWidget QVBoxLayout QHBoxLayout QPushButton QLabel QComboBox QSpinBox QCheckBox QRadioButton QButtonGroup QLineEdit QFormLayout QScrollArea QTabWidget',
    }.items():
        module=types.ModuleType('PySide6.'+suffix)
        for name in names.split():setattr(module,name,object if name=='QWidget' else MagicMock())
        replacements[module.__name__]=module
    spec=importlib.util.spec_from_file_location('assist_flow_test',Path(__file__).with_name('assist.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,replacements):spec.loader.exec_module(module)
    return module


class Clock:
    now=0
    def perf_counter(self):return self.now


class DirectPool:
    def __init__(self):self.calls=[]
    def submit(self,fn,*args):
        self.calls.append(args[0])
        f=Future()
        try:f.set_result(fn(*args))
        except Exception as e:f.set_exception(e)
        return f


class ScreenSimulation:
    down=None
    def __init__(self,clock):
        self.clock=clock;self.foreground=True;self.blank_until=0
        self.board=(0,)*16+(1023^(1<<4),)*4
        self.piece='O';self.cells=tuple((x+3,y+1) for x,y in BASE['O'])
        self.actions=[];self.drops=0;self.holds=0
    def arm(self,region):return True,''
    def focused(self):return self.foreground
    def poll(self):pass
    def release(self):pass
    def disarm(self):pass
    def press_sequence(self,keys):
        from motion import pose_from,step,cells_at
        pose=pose_from(self.piece,self.cells)
        for key in keys:
            self.actions.append(key)
            action=dict(LEFT='left',RIGHT='right',X='cw',Z='ccw',DOWN='soft')[key]
            pose,_=step(self.board,self.piece,pose,action)
        self.cells=cells_at(self.piece,pose)
    def grab(self,region):
        a=np.zeros((800,400,4),dtype=np.uint8)
        if self.clock.now<self.blank_until:return a
        cells=set(self.cells)|{(x,y) for y in range(20) for x in range(10) if self.board[y]&(1<<x)}
        for x,y in cells:a[y*40:(y+1)*40,x*40:(x+1)*40,:3]=180
        return a
    def press(self,key):
        assert self.foreground
        self.actions.append(key)
        if key=='C':
            self.holds+=1;self.piece='I';self.cells=tuple((x+3,y+1) for x,y in BASE['I'])
            self.blank_until=self.clock.now+1.2
        elif key=='X':
            col=min(x for x,y in self.cells);row=min(y for x,y in self.cells)
            self.cells=tuple((x+col,y+row) for x,y in norm([(-y,x) for x,y in norm(self.cells)]))
        elif key in ('LEFT','RIGHT'):
            dx=1 if key=='RIGHT' else -1;self.cells=tuple((x+dx,y) for x,y in self.cells)
        elif key=='SPACE':
            rows=list(self.board)
            for x,y in projected_cells(self.board,self.cells):rows[y]|=1<<x
            rows=[r for r in rows if r!=1023];self.board=tuple([0]*(20-len(rows))+rows)
            self.drops+=1;self.piece='T';self.cells=tuple((x+3,y+1) for x,y in BASE['T'])


class AppFlowTests(unittest.TestCase):
    def test_actual_tick_long_hold_gap_replanning_and_focus_resume(self):
        self.run_flow(False)
    def test_turbo_attack_tick_hold_recovery_and_focus(self):
        self.run_flow(True)
    def test_color_mode_full_tick_hold_and_focus(self):
        self.run_flow(False, color=True)
    def run_flow(self,turbo,color=False):
        module=load_app();clock=Clock();module.time=clock
        app=object.__new__(module.App);sim=ScreenSimulation(clock)
        app.inputs=sim;app.cap=sim;app.auto=AutoPlayer();app.tracker=BoardTracker()
        app.cfg={'board':dict(left=0,top=0,width=400,height=800)}
        app.running=False;app.future=None;app.key=None;app.generation=0;app.current_plan=None
        app.submitted_key=None;app.retry_after=0;app.pool=DirectPool();app.capture_ms=0
        app.failure_frame=None;app.failure_details=None;app.was_recognized=False
        app.preview=MagicMock();app.overlay=MagicMock();app.status=MagicMock();app.auto_status=MagicMock();app.detail=MagicMock()
        if color:
            from color_vision import ColorBoardVision
            app.color_reader=ColorBoardVision()
            app.vision_mode=MagicMock()
            app.vision_mode.currentData.return_value='auto'
            app.vision_status=MagicMock()
            app.vision_status_next_at=0
        app.result_ready=types.SimpleNamespace(emit=app.accept_result)
        app.mode=MagicMock();app.mode.currentIndex.return_value=3 if turbo else 0
        app.turbo=MagicMock();app.turbo.isChecked.return_value=turbo
        app.instant=MagicMock();app.instant.isChecked.return_value=turbo
        app.piece_box=lambda which:'T' if which=='next' else None
        app.fast=MagicMock();app.fast.isChecked.return_value=True
        app.hold_enabled=MagicMock();app.hold_enabled.isChecked.return_value=True
        app.hold_source=MagicMock();app.hold_source.currentIndex.return_value=2;app.hold_source.currentText.return_value='I'
        app.threshold=MagicMock();app.threshold.value.return_value=105
        app.show_overlay=MagicMock();app.show_overlay.isChecked.return_value=False
        app.screen=MagicMock();g=app.screen.geometry.return_value
        g.width.return_value=800;g.height.return_value=800;g.x.return_value=0;g.y.return_value=0
        app.monitor=dict(left=0,top=0,width=800,height=800)
        app.bind_fields={}
        for k,v in dict(left='LEFT',right='RIGHT',cw='X',ccw='Z',soft='DOWN',drop='SPACE',hold='C').items():
            e=MagicMock();e.text.return_value=v;app.bind_fields[k]=e
        app.toggle_auto()
        for _ in range(220):
            clock.now+=.016;app.tick()
            self.assertTrue(app.auto.enabled,app.auto.message)
            if sim.drops:break
        self.assertEqual(sim.holds,1)
        self.assertEqual(sim.drops,1,(app.auto.message,app.status.setText.call_args))
        self.assertEqual(sim.board,(0,)*20)
        # The swapped-in I must never be planned before the hold is acked,
        # with the stale allow_hold=True flag (the old double-compute path).
        for key in app.pool.calls:
            if key[1]=='I':self.assertFalse(key[6])
        sim.foreground=False;n=len(sim.actions)
        for _ in range(80):clock.now+=.016;app.tick()
        self.assertTrue(app.auto.enabled);self.assertEqual(len(sim.actions),n)
        sim.foreground=True
        for _ in range(150):
            clock.now+=.016;app.tick()
            if len(sim.actions)>n:break
        self.assertGreater(len(sim.actions),n,app.auto.message)
        if turbo:
            self.assertFalse(any(k[0]==(0,)*20 and k[1]=='T' for k in app.pool.calls),
                             'Expected the confirmed precomputed T placement to skip new search')
        app.stop_auto('F9');n=len(sim.actions)
        for _ in range(10):clock.now+=.016;app.tick()
        self.assertFalse(app.auto.enabled);self.assertEqual(len(sim.actions),n)


if __name__=='__main__':unittest.main()
