import os, sys, json, ctypes, traceback, time, zipfile, threading, multiprocessing as mp
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import mss
from PySide6.QtCore import Qt, QRect, QRectF, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QImage, QPixmap
from PySide6.QtWidgets import (QApplication,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QComboBox,QSpinBox,QCheckBox,QRadioButton,QButtonGroup,QLineEdit,QFormLayout,QScrollArea,QTabWidget)
from engine import (plan_move, plan_move_pvp_worker, plan_move_cc2_diagnostic_worker,
                    plan_move_hybrid_diagnostic_worker, CC2WorkerResult, HybridWorkerResult,
                    combat_profile, separate,
                    deep_adoption_allowed, emergency_fast_plan)
from autoplay import AutoPlayer
from windows_input import WindowsInput, DEFAULT, valid_binding
from window_roi import format_region_issues
from vision import read_grid, BoardTracker
from color_vision import ColorBoardVision, suppress_confirmed_white_i_projection, CalibrationProgress
from vision_failure import failure_evidence
from legacy_integrity import LegacyTransitionGate
from legacy_quarantine import allow_provisional_hold
from pressure_refresh import pressure_change, may_refresh_idle, permit_prepared_commit
from spawn_vision import (spawn_pair_rows, split_legacy_22, spawn_from_board,
                          capture_board_and_spawn,
                          fuse_spawn_active, shifted_queue_after_verified_lock,
                          mask_confirmed_spawn_white_guide,
                          mask_solid_spawn_projection)
from telemetry import GarbageFilter
from offense_runtime import conservative_incoming, defer_for_attack
from attack_rate import AttackRateWindow
from performance_budget import deep_budget_ms
from engine_policy import configured_mode, DISPLAY_NAMES, exclusive_cc2_safe, hybrid_wait_for_cc2
from stall_monitor import StallMonitor
from vision_recovery import proven_active_reference, ConfirmedLegacyFallback
from config_security import load_settings, save_settings, valid_region_for
from input_rhythm import InputRhythm
from grid_calibration import precise_board_grid, calibrated_pair, diagnostic_geometry, initial_guide

CONFIG=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'TetrioAssistPrototype'/'settings.json'
# NEXT/HOLD: four-square shape evidence takes priority over custom skin colors.
# The compatibility API is also used by the bounded ROI worker.
from piece_vision import classify_piece_pixels, read_piece

_ROI_LOCAL = threading.local()

def _roi_capture(region):
    if region is None:return None
    cap=getattr(_ROI_LOCAL,'cap',None)
    if cap is None:
        cap=mss.mss();_ROI_LOCAL.cap=cap
    return classify_piece_pixels(np.asarray(cap.grab(region))[:,:,:3])

def estimate_garbage_pixels(a):
    """Conservative bottom-filled meter estimate; no false certainty from blanks.

    A constant patch is not evidence of an empty or a full meter. A visibly
    partial fill must be bottom-anchored and separated from the upper track.
    With no such evidence, return UNKNOWN rather than fabricate 0/20 lines.
    This stays on the 80ms ROI worker, never on the 8ms board/input loop.
    """
    a=np.asarray(a)
    if a.ndim!=3 or a.shape[2]<3 or a.shape[0]<40 or a.shape[1]<3:
        return None,0.0
    a=a[:,:,:3].astype(np.int16,copy=False)
    h,w=a.shape[:2]
    if h < w*1.25:
        return None,0.0  # full-height *vertical* meter required
    hi=a.max(axis=2);lo=a.min(axis=2);chroma=hi-lo
    # Reject flat ROI selections: black/grey/white/color are all ambiguous.
    if float(hi.max()-hi.min())<12 and float(chroma.max()-chroma.min())<12:
        return None,0.0
    active=(hi>=85)&((chroma>=18)|(hi>=165))
    # Discard a one-pixel bright edge; a filled segment must occupy substantial
    # horizontal width rather than just the rail or a random adjacent UI line.
    rows=[]
    for i in range(20):
        y0=round(i*h/20);y1=max(y0+1,round((i+1)*h/20))
        rows.append(float(active[y0:y1].mean()))
    count=0;weak=0;strength=[]
    for ratio in reversed(rows):
        if ratio>=.23:
            count+=1;strength.append(ratio)
        elif ratio>=.12 and weak<1 and count:
            count+=1;weak+=1;strength.append(ratio)
        else:break
    if count==0:
        # An unlit rail cannot establish its own validity from a screenshot.
        # Only subsequent valid fill evidence should identify this ROI as a
        # meter; callers must not equate UNKNOWN with a confirmed zero.
        return None,0.0
    if count==20:
        # Full-height uniform UI regions can also be white/colorful. Require
        # a visible lateral meter boundary before accepting a fully lit bar.
        middle=active[:,max(0,w//4):max(1,(3*w)//4)].mean()
        edge=np.concatenate((active[:,:max(1,w//8)],active[:,-max(1,w//8):]),axis=1).mean()
        if middle-edge<.25:return None,0.0
        return count,min(.85,float(middle-edge)+.35)
    inside=sum(strength)/len(strength)
    above=rows[:20-count]
    separation=inside-(sum(above)/len(above) if above else 0.0)
    if inside<.23 or separation<.18 or max(above,default=0)>.65:
        return None,0.0
    confidence=max(0.0,min(.95,separation*1.45+inside*.22))
    return count,confidence

def estimate_garbage_lines(region):
    """Capture a selected vertical incoming-garbage meter as 0..20 lines."""
    if region is None:return None,0.0
    cap=getattr(_ROI_LOCAL,'cap',None)
    if cap is None:
        cap=mss.mss();_ROI_LOCAL.cap=cap
    return estimate_garbage_pixels(np.asarray(cap.grab(region))[:,:,:3])

def retry_piece_regions(next_region,hold_region,next2_region=None,next3_region=None,garbage_region=None,attempts=3,gap=.008):
    """Read optional combat telemetry off the GUI/Fast path.

    NEXT2/NEXT3 and garbage are snapshots, not retry targets. Only a missing
    NEXT1/HOLD is retried. This keeps the hot capture path from repeatedly
    grabbing static queue boxes and avoids the old three-pass ROI burst.
    """
    nxt=hold=nxt2=nxt3=None;incoming=None;confidence=0.0
    # Static/deep-only telemetry: exactly one capture for this scheduled job.
    if next2_region is not None:nxt2=_roi_capture(next2_region)
    if next3_region is not None:nxt3=_roi_capture(next3_region)
    if garbage_region is not None:incoming,confidence=estimate_garbage_lines(garbage_region)
    for index in range(max(1,attempts)):
        if nxt is None and next_region is not None:nxt=_roi_capture(next_region)
        if hold is None and hold_region is not None:hold=_roi_capture(hold_region)
        ready=(next_region is None or nxt is not None) and (hold_region is None or hold is not None)
        if ready:break
        if index+1<attempts:time.sleep(gap)
    return dict(next=nxt,hold=hold,next2=nxt2,next3=nxt3,incoming=incoming,garbage_confidence=confidence,
                captured_at=time.perf_counter())



def _deep_worker_init(revision_counter=None):
    """Keep Deep refinement low priority and make stale searches cooperative."""
    if revision_counter is not None:
        from engine import set_deep_revision_counter
        set_deep_revision_counter(revision_counter)
    try:
        if sys.platform=='win32':
            kernel=C=ctypes.windll.kernel32
            kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:
        pass

def rgb_image(a):
    a=np.ascontiguousarray(a)
    return QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QImage.Format.Format_RGB888).copy()

class Selector(QWidget):
    def __init__(self,screen,pixels,callback,message='Drag region. ESC = cancel.'):
        super().__init__(None,Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(screen.geometry()); self.pix=QPixmap.fromImage(rgb_image(pixels)); self.callback=callback; self.message=message
        self.start=None; self.end=None; self.setCursor(Qt.CursorShape.CrossCursor); self.show()
    def paintEvent(self,e):
        p=QPainter(self); p.drawPixmap(self.rect(),self.pix); p.fillRect(self.rect(),QColor(0,0,0,75))
        p.setPen(QPen(QColor('#56ffd3'),2))
        if self.start and self.end: p.drawRect(QRect(self.start,self.end).normalized())
        p.drawText(20,30,self.message)
    def mousePressEvent(self,e): self.start=e.position().toPoint(); self.end=self.start
    def mouseMoveEvent(self,e): self.end=e.position().toPoint(); self.update()
    def mouseReleaseEvent(self,e):
        if self.start is None: return
        r=QRect(self.start,e.position().toPoint()).normalized(); self.close()
        if r.width()>8 and r.height()>8: self.callback(r)
    def keyPressEvent(self,e):
        if e.key()==Qt.Key.Key_Escape: self.close()

class GridCalibrator(QWidget):
    """Immediately-visible grid guide, never a blind two-step selection.

    The initial 10x20/2-row geometry is inferred from the user's saved
    BOARD (including a legacy 22-row region) or from a movable draft.
    Drag the whole guide, click to relocate its bottom-left anchor, wheel to
    change cell size, arrows for 1px, Enter to save; ESC always cancels.
    """
    def __init__(self,screen,pixels,monitor,callback,board=None):
        super().__init__(None,Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(screen.geometry())
        self.pix=QPixmap.fromImage(rgb_image(pixels))
        self.monitor=dict(monitor);self.callback=callback
        candidate=initial_guide(board,self.monitor)
        self.tile_size,self.anchor,self.hint=(candidate if candidate is not None else (None,None,'격자 초안을 만들 수 없습니다.'))
        self.drag_origin=None;self.drag_anchor=None
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show();self.activateWindow();self.setFocus()

    def _to_pixel(self,pos):
        return (self.monitor['left']+round(pos.x()*self.monitor['width']/max(1,self.width())),
                self.monitor['top']+round(pos.y()*self.monitor['height']/max(1,self.height())))

    def _to_view(self,x,y):
        return ((x-self.monitor['left'])*self.width()/self.monitor['width'],
                (y-self.monitor['top'])*self.height()/self.monitor['height'])

    def _pair(self):
        if self.tile_size is None or self.anchor is None:return None
        return calibrated_pair(self.tile_size,self.tile_size,*self.anchor,
                               screen=self.monitor,spawn_rows=2)

    def paintEvent(self,event):
        p=QPainter(self)
        p.drawPixmap(self.rect(),self.pix)
        p.fillRect(self.rect(),QColor(0,0,0,25))
        pair=self._pair()
        if pair:
            board,spawn=pair
            for region,rows,color in ((spawn,2,QColor('#ffbd55')),(board,20,QColor('#43ffe1'))):
                left,top=self._to_view(region['left'],region['top'])
                right,bottom=self._to_view(region['left']+region['width'],region['top']+region['height'])
                p.setPen(QPen(color,2))
                p.drawRect(QRectF(left,top,right-left,bottom-top))
                p.setPen(QPen(color,1))
                for col in range(1,10):
                    x=left+(right-left)*col/10
                    p.drawLine(round(x),round(top),round(x),round(bottom))
                for row in range(1,rows):
                    y=top+(bottom-top)*row/rows
                    p.drawLine(round(left),round(y),round(right),round(y))
            lx,ly=self._to_view(board['left'],board['top']+board['height'])
            p.setPen(QPen(QColor('#ffe36f'),3))
            p.drawLine(round(lx)-10,round(ly),round(lx)+10,round(ly))
            p.drawLine(round(lx),round(ly)-10,round(lx),round(ly)+10)
        panel_height=100
        p.fillRect(8,8,max(230,self.width()-16),panel_height,QColor(10,18,30,238))
        p.setPen(QPen(QColor('#ffffff')))
        p.drawText(18,29,'초록 10×20 = 실제 보드 · 주황 2줄 = SPAWN (격자를 바로 움직일 수 있습니다)')
        p.drawText(18,52,'마우스 드래그: 격자 이동 | 휠 또는 +/-: 칸 크기 | 방향키: 1px 이동 (Shift: 5px)')
        p.drawText(18,76,'Enter: 적용 및 저장 | ESC: 취소 · 기존 설정 그대로 유지')
        if pair:
            p.drawText(18,97,f'칸 {self.tile_size}px · BOARD ({pair[0]["left"]}, {pair[0]["top"]}) '
                             f'{pair[0]["width"]}×{pair[0]["height"]} · {self.hint[:26]}')
        else:
            p.setPen(QPen(QColor('#ff956e')))
            p.drawText(18,97,'격자가 화면 밖입니다. 드래그하여 전체 BOARD + SPAWN을 화면 안으로 옮기세요.')
        p.end()

    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton:return
        self.drag_origin=self._to_pixel(event.position())
        self.drag_anchor=self.anchor
        self.setFocus()

    def mouseMoveEvent(self,event):
        if self.drag_origin is not None and self.drag_anchor is not None:
            px,py=self._to_pixel(event.position())
            self.anchor=(self.drag_anchor[0]+px-self.drag_origin[0],
                         self.drag_anchor[1]+py-self.drag_origin[1])
            self.update()

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:
            self.drag_origin=None;self.drag_anchor=None

    def wheelEvent(self,event):
        if self.tile_size is None:return
        delta=event.angleDelta().y()
        if delta:
            self.tile_size=max(12,min(240,self.tile_size+(1 if delta>0 else -1)))
            self.update()

    def keyPressEvent(self,event):
        key=event.key()
        if key==Qt.Key.Key_Escape:self.close();return
        if key in (Qt.Key.Key_Plus,Qt.Key.Key_Equal):
            self.tile_size=min(240,self.tile_size+1)
        elif key==Qt.Key.Key_Minus:
            self.tile_size=max(12,self.tile_size-1)
        elif key in (Qt.Key.Key_Left,Qt.Key.Key_Right,Qt.Key.Key_Up,Qt.Key.Key_Down):
            amount=5 if bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1
            dx=(-amount if key==Qt.Key.Key_Left else amount if key==Qt.Key.Key_Right else 0)
            dy=(-amount if key==Qt.Key.Key_Up else amount if key==Qt.Key.Key_Down else 0)
            self.anchor=(self.anchor[0]+dx,self.anchor[1]+dy)
        elif key in (Qt.Key.Key_Return,Qt.Key.Key_Enter):
            pair=self._pair()
            if pair:
                self.callback(*pair)
                self.close()
            return
        self.update()

class Overlay(QWidget):
    def __init__(self):
        super().__init__(None,Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint|Qt.WindowType.Tool|Qt.WindowType.WindowTransparentForInput|Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.cells=[]
    def paintEvent(self,e):
        p=QPainter(self); w=self.width()/10; h=self.height()/20
        p.setPen(QPen(QColor('#56ffd3'),3)); p.setBrush(QColor(86,255,211,65))
        for x,y in self.cells: p.drawRect(QRectF(x*w+2,y*h+2,w-4,h-4))
    def exclude_capture(self):
        if sys.platform!='win32': return False
        fn=ctypes.windll.user32.SetWindowDisplayAffinity
        fn.argtypes=[ctypes.c_void_p,ctypes.c_uint]; fn.restype=ctypes.c_int
        return bool(fn(int(self.winId()),0x11))

class Preview(QWidget):
    def __init__(self):
        super().__init__(); self.grid=None; self.move=None; self.active=[]; self.colors=None; self.ghosts=None; self.unknown=None; self.setFixedSize(200,400)
    def paintEvent(self,e):
        p=QPainter(self); p.fillRect(self.rect(),QColor('#0d1524'))
        if self.grid is not None:
            for y,row in enumerate(self.grid):
                for x,v in enumerate(row):
                    if v:
                        label=(self.colors[y][x] if self.colors else '')
                        palette={'I':'#2dcddc','O':'#e1c323','T':'#a046cd',
                                 'J':'#3c55d7','L':'#e18723','S':'#50c346','Z':'#dc4141'}
                        p.fillRect(x*20+1,y*20+1,18,18,QColor(palette.get(label,'#61738c')))
                    elif self.ghosts and self.ghosts[y][x]:
                        p.setPen(QPen(QColor('#54789c'),1)); p.drawRect(x*20+3,y*20+3,14,14)
                    if self.unknown and self.unknown[y][x]:
                        p.fillRect(x*20+1,y*20+1,18,18,QColor('#aa4255'))
        for x,y in self.active: p.fillRect(x*20+1,y*20+1,18,18,QColor('#efb84e'))
        p.setPen(QPen(QColor('#56ffd3'),2))
        if self.move:
            for x,y in self.move.cells: p.drawRect(x*20+1,y*20+1,18,18)

class App(QWidget):
    result_ready=Signal(int, object, object, float, object)
    deep_ready=Signal(int, object, object, float, object)
    continuation_ready=Signal(int, object, object, object)
    cc2_warm_ready=Signal(str)
    def __init__(self):
        super().__init__(); self.setWindowTitle('TETR.IO Assist · 1.1.10.17 VERIFIED ROUTE GUARD')
        self.screen=QApplication.primaryScreen(); self.cap=mss.mss(); self.monitor=self.cap.monitors[1]
        # Match the Windows primary monitor (physical origin 0,0).
        self.monitor=next((m for m in self.cap.monitors[1:] if m['left']==0 and m['top']==0),self.monitor)
        self.cfg={}; self.running=False; self.future=None; self.key=None; self.generation=0; self.pool=ThreadPoolExecutor(max_workers=1)
        self.roi_pool=ThreadPoolExecutor(max_workers=1); self.roi_future=None; self.roi_context=None
        self.roi_pending=None; self.next_queue_retry_at=0; self.next_queue_retry_done=False; self.next_queue_missing=(); self.next_queue_trusted=False
        self.deep_mp=mp.get_context('spawn')
        self.deep_cancel_revision=self.deep_mp.Value('i',0,lock=False)
        self.deep_pool=ProcessPoolExecutor(max_workers=1,mp_context=self.deep_mp,
                                           initializer=_deep_worker_init,initargs=(self.deep_cancel_revision,))
        self.deep_future=None; self.deep_pending=None; self.deep_cancelled_count=0
        self.deep_failures=0; self.deep_disabled=False
        self.continuation_future=None; self.continuation_token=None; self.continuation_requested=0; self.continuation_applied=0; self.continuation_stale=0
        self.events=deque(maxlen=256); self.session_at=time.perf_counter()
        self.deep_applied=0;self.deep_received=0;self.deep_stale=0
        self.deep_budget_last_ms=0.0;self.deep_budget_low_count=0;self.deep_budget_boost_count=0
        self.hold_unknown_observations=0;self.hold_eligible_observations=0;self.hold_blocked_observations=0
        self.external_received=0;self.external_selected=0;self.external_unadopted=0
        self.bundled_received=0;self.bundled_selected=0;self.bundled_unadopted=0;self.bundled_verified_locks=0
        self.external_prefetch_received=0;self.external_prefetch_applied=0
        self.bundled_prefetch_received=0;self.bundled_prefetch_applied=0
        self.cc2_verified_locks=0;self.cc2_no_plan=0;self.cc2_strict_waits=0
        self.cc2_last_reason='NOT_REQUESTED';self.cc2_last_elapsed_ms=None
        self.cc2_hybrid_attempts=0;self.cc2_hybrid_fallbacks=0
        self.cc2_hybrid_passes=0;self.cc2_hybrid_stale=0
        self.cc2_hybrid_reasons={};self.cc2_hybrid_last_probe_ms=None
        self.input_pressure_invalidations=0
        self.meter_only_updates=0
        self.cc2_retry_count=0;self.cc2_retry_at=0.0;self.cc2_warm_future=None
        self.deep_wait_frames=0;self.deep_wait_turns=0;self.deep_wait_seen=None
        self.beam_verified_locks=0;self.counter_beam_verified_locks=0;self.vision_rise_recovered=0
        self.hold_sequence_applied=0;self.hold_sequence_verified_locks=0
        self.deep_revision=0; self.queue_revision=0; self.deep_queue=(); self.incoming_garbage=0; self.garbage_confidence=0.0; self.garbage_filter=GarbageFilter(); self.telemetry_next_at=0; self.ui_next_at=0; self.preview_next_at=0; self.diagnostic_next_at=0; self._auto_text=None
        self.result_ready.connect(self.accept_result, Qt.ConnectionType.QueuedConnection)
        self.deep_ready.connect(self.accept_deep_result, Qt.ConnectionType.QueuedConnection)
        self.continuation_ready.connect(self.accept_continuation_result, Qt.ConnectionType.QueuedConnection)
        self.cc2_warm_ready.connect(self._accept_cc2_warm, Qt.ConnectionType.QueuedConnection)
        self.detected_at=0; self.capture_ms=0; self.submitted_key=None; self.current_plan=None
        self.tracker=BoardTracker(); self.legacy_gate=LegacyTransitionGate(); self.color_reader=ColorBoardVision(); self.legacy_recovery=ConfirmedLegacyFallback(); self.last_frame=None; self.failure_frame=None; self.last_grid=None; self.was_recognized=False; self.vision_failure_open=False; self.vision_failure_at=None; self.failure_vision=None
        self.retry_after=0; self.failure_details=None
        self.legacy_provisional_since=None;self.legacy_provisional_frames=0
        self.legacy_provisional_resolved=0;self.legacy_provisional_escalated=0
        self.pressure_fast_recompute_pending=False;self.pressure_fast_refreshes=0
        self.pressure_fast_refresh_at=None
        self.auto=AutoPlayer(); self.stall_monitor=StallMonitor(); self.inputs=WindowsInput(); self.f9=False; self.bindings={}
        # Invalidate outstanding batched keys when observed fixed board changes.
        self.input_board_ticket=None
        self.input_ticket_epoch=0
        self.input_abort_seen=0
        self.session_roi_excluded=set() 
        self.overlay=Overlay(); self.overlay.show(); self.capture_safe=self.overlay.exclude_capture(); self.overlay.hide()
        self.cfg,self.settings_warning=load_settings(CONFIG)
        self.rhythm=InputRhythm(self.cfg.get('human_intensity','off'))
        self.last_fast_ms=0.0; self.last_delivery_ms=0.0; self.last_deep_ms=None
        self.current_first_input_ms=None; self.current_drop_ms=None; self.last_first_input_ms=None; self.last_drop_ms=None; self.last_lock_ms=None
        self.region_status={}; self._combat_text='ATTACK'; self._run_state='OFF'
        self.spawn_fast_hits=0; self.spawn_fallbacks=0; self.next_cache_hits=0; self.next_cache_misses=0
        self.resize(1020,730); self.setMinimumSize(790,600)

        outer=QVBoxLayout(self); outer.setContentsMargins(14,14,14,12); outer.setSpacing(10)
        header=QHBoxLayout(); outer.addLayout(header)
        brand=QVBoxLayout(); header.addLayout(brand,1)
        hero=QLabel('TETR.IO ASSIST'); hero.setObjectName('heroTitle'); brand.addWidget(hero)
        sub=QLabel('1.1.10.17 · BOUNDED ROUTE PROOF / HUMAN RHYTHM'); sub.setObjectName('heroSub'); brand.addWidget(sub)
        self.run_chip=QLabel('AUTO OFF'); self.run_chip.setObjectName('chipOff'); self.run_chip.setAlignment(Qt.AlignmentFlag.AlignCenter); self.run_chip.setMinimumWidth(112); header.addWidget(self.run_chip)

        top=QHBoxLayout(); top.setSpacing(12); outer.addLayout(top,1)

        left=QWidget(); left.setObjectName('sidePanel'); left_l=QVBoxLayout(left); left_l.setContentsMargins(12,12,12,12); left_l.setSpacing(9)
        pv_title=QLabel('BOARD PREVIEW'); pv_title.setObjectName('sectionTitle'); left_l.addWidget(pv_title)
        pv_sub=QLabel('LEGACY 밝기 / COLOR 선택 인식 · 노랑 현재 · 민트 추천'); pv_sub.setObjectName('muted'); left_l.addWidget(pv_sub)
        self.preview=Preview(); left_l.addWidget(self.preview,0,Qt.AlignmentFlag.AlignHCenter)
        self.vision_status=QLabel('VISION · 배경 보정 대기'); self.vision_status.setObjectName('muted'); self.vision_status.setWordWrap(True); left_l.addWidget(self.vision_status)
        self.queue_status=QLabel('CURRENT —   NEXT —   HOLD —'); self.queue_status.setObjectName('liveValue'); self.queue_status.setWordWrap(True); left_l.addWidget(self.queue_status)
        self.status=QLabel(self.settings_warning or '보드 영역을 지정하세요.'); self.status.setObjectName('messageBox'); self.status.setWordWrap(True); self.status.setMinimumWidth(220); left_l.addWidget(self.status)
        self.detail=QLabel('추천 결과 대기'); self.detail.setObjectName('muted'); self.detail.setWordWrap(True); left_l.addWidget(self.detail)
        hotkeys=QLabel('F7  PREVIEW   ·   F9  AUTO   ·   ESC  STOP'); hotkeys.setObjectName('hotkeyBar'); hotkeys.setAlignment(Qt.AlignmentFlag.AlignCenter); left_l.addWidget(hotkeys)
        left_l.addStretch(); top.addWidget(left,0)

        panel=QWidget(); controls=QVBoxLayout(panel); controls.setContentsMargins(4,4,4,4); controls.setSpacing(10)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); scroll.setMinimumWidth(475); top.addWidget(scroll,1)

        live,live_l=self._section('LIVE TELEMETRY','게임 판단과 속도를 한곳에서 확인')
        self.auto_status=QLabel('자동 OFF · 게임 창에서 F9로 시작'); self.auto_status.setObjectName('primaryStatus'); self.auto_status.setWordWrap(True); live_l.addWidget(self.auto_status)
        self.pvp_status=QLabel('PvP  ATTACK · incoming 미설정 · Deep NEXT1'); self.pvp_status.setObjectName('combatAttack'); self.pvp_status.setWordWrap(True); live_l.addWidget(self.pvp_status)
        self.pace_status=QLabel('착지 PPS 측정 대기 · 모델 공격 목표 ≥ 100 APM'); self.pace_status.setWordWrap(True); live_l.addWidget(self.pace_status)
        self.latency_status=QLabel('CAP — ms   FAST — ms   DELIVERY — ms   DEEP — ms'); self.latency_status.setObjectName('monoStatus'); self.latency_status.setWordWrap(True); live_l.addWidget(self.latency_status)
        self.deep_stats=QLabel('ENGINE INTERNAL · CC2착지 0 · LOCAL착지 0'); self.deep_stats.setObjectName('muted'); self.deep_stats.setWordWrap(True); live_l.addWidget(self.deep_stats)
        controls.addWidget(live)
        # Only live data belongs on the always-visible dashboard. Settings are
        # split into independent, scrollable pages, not a 1,500px control list.
        self.settings_tabs=QTabWidget(); self.settings_tabs.setObjectName('settingsTabs')
        region_page,region_page_l=self._settings_page()
        combat_page,combat_page_l=self._settings_page()
        system_page,system_page_l=self._settings_page()
        self.settings_tabs.addTab(region_page,'① 영역 인식')
        self.settings_tabs.addTab(combat_page,'② 전투 전략')
        self.settings_tabs.addTab(system_page,'③ 입력 · 도구')
        controls.addWidget(self.settings_tabs,1)

        regions,regions_l=self._section('CAPTURE REGIONS','BOARD는 실제 10×20 · SPAWN은 그 위 1~4줄 · 둘은 같은 격자 폭')
        guide=QPushButton('① 격자 가이드 띄우기  ▶  게임 화면 위에 10×20선 표시')
        guide.setMinimumHeight(38)
        guide.setStyleSheet('QPushButton{background:#14685f;color:#ffffff;font-weight:bold;border:1px solid #57e8d2;border-radius:6px;padding:7px;} QPushButton:hover{background:#198176;}')
        guide.setToolTip('버튼 누르면 기존 위치에 격자가 바로 표시됩니다. 드래그 이동 / 휠 크기 / ENTER 저장 / ESC 취소')
        guide.clicked.connect(self.precise_grid_setup)
        regions_l.addWidget(guide)
        guide_help=QLabel('클릭 → 초록 BOARD / 주황 SPAWN 가이드 즉시 표시 → 마우스로 옮기기 → Enter 저장')
        guide_help.setObjectName('muted');guide_help.setWordWrap(True)
        regions_l.addWidget(guide_help)
        for label,which,hint in [
            ('BOARD','board','10×20 실제 판'),('SPAWN','spawn','상단 등장 1~4줄'),('NEXT 1','next','Fast'),('HOLD','hold','자동 인식'),
            ('NEXT 2','next2','Deep'),('NEXT 3','next3','Deep'),('GARBAGE','garbage','PvP incoming')]:
            self._region_row(regions_l,label,which,hint)
        split=QPushButton('기존 22줄 BOARD를 상단 2줄 + 실제 20줄로 자동 분리')
        split.clicked.connect(self.split_existing_board)
        regions_l.addWidget(split)
        auto_spawn=QPushButton('현재 20줄 BOARD 위에 SPAWN 2줄 자동 정렬')
        auto_spawn.clicked.connect(self.create_aligned_spawn)
        regions_l.addWidget(auto_spawn)
        region_page_l.addWidget(regions)
        vision_box,vision_l=self._section('COLOR VISION','컬러는 점유 판단에 사용 · 블록 종류 표시는 참고용')
        row=QHBoxLayout(); row.addWidget(QLabel('보드 인식'))
        self.vision_mode=QComboBox()
        for title, code in [('AUTO · 컬러 우선 / 미보정 시 자동입력 대기','auto'),
                            ('STRICT · 불확실하면 입력 보류','strict'),
                            ('LEGACY · 기존 밝기 인식','legacy')]:
            self.vision_mode.addItem(title,code)
        index=['auto','strict','legacy'].index(self.cfg.get('vision_mode','auto'))
        self.vision_mode.setCurrentIndex(index);row.addWidget(self.vision_mode,1)
        vision_l.addLayout(row)
        recalibrate=QPushButton('컬러 배경 재보정 · 4프레임 확인');recalibrate.clicked.connect(self.reset_color_calibration)
        vision_l.addWidget(recalibrate)
        note=QLabel('재보정 시 자동입력 중단 · AUTO 실패 시 독립 보드와 2프레임 일치한 LEGACY 후보만 복구 · 추정 입력 금지')
        note.setObjectName('muted');note.setWordWrap(True);vision_l.addWidget(note)
        region_page_l.addWidget(vision_box)

        strategy,strategy_l=self._section('STRATEGY','공격 판단은 Deep, 즉시 배치는 Fast 우선')
        row=QHBoxLayout(); row.addWidget(QLabel('모드')); self.mode=QComboBox(); self.mode.addItems(['균형','콤보 우선','생존 우선','공격 · All-Mini+ / B2B']); self.mode.setCurrentIndex(3); row.addWidget(self.mode,1); strategy_l.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('전투 규칙'))
        self.ruleset=QComboBox();self.ruleset.addItem('QUICK PLAY · 다인전','quick_play')
        self.ruleset.addItem('TETRA LEAGUE / 일반','standard')
        selected=self.cfg.get('ruleset','standard')
        self.ruleset.setCurrentIndex(0 if selected=='quick_play' else 1)
        row.addWidget(self.ruleset,1);strategy_l.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('콤보 테이블'))
        self.combo_table=QComboBox()
        self.combo_table.addItem('MULTIPLIER · 기본', 'multiplier')
        self.combo_table.addItem('NONE · 콤보 보너스 없음', 'none')
        self.combo_table.setCurrentIndex(1 if self.cfg.get('combo_table','multiplier')=='none' else 0)
        row.addWidget(self.combo_table,1);strategy_l.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('공격 반올림'))
        self.rounding=QComboBox()
        self.rounding.addItem('AUTO · 모드 기본값','auto')
        self.rounding.addItem('DOWN · 버림','down')
        self.rounding.addItem('RNG · 확률 반올림','rng')
        saved_rounding=self.cfg.get('rounding','auto')
        self.rounding.setCurrentIndex({'auto':0,'down':1,'rng':2}.get(saved_rounding,0))
        row.addWidget(self.rounding,1);strategy_l.addLayout(row)
        self.opening_enabled=QCheckBox('초반 전용 Opening Burst · 목표 12줄(추정) / 12초 또는 24수')
        self.opening_enabled.setChecked(bool(self.cfg.get('opening_enabled',True)))
        strategy_l.addWidget(self.opening_enabled)
        self.adaptive_opener=QCheckBox('검증형 Adaptive Opener · NEXT 기반 TSD / Quad (Deep 전용)')
        self.adaptive_opener.setChecked(bool(self.cfg.get('adaptive_opener',True)))
        strategy_l.addWidget(self.adaptive_opener)
        # All four strategies are experimental and independently opt-in;
        # they may never force an unobserved future piece or bypass drop guards.
        self.canonical_book=QCheckBox('TKI-3 7수 / DT Cannon 15수 검증 예제 · 실험')
        self.canonical_book.setChecked(bool(self.cfg.get('canonical_book',False)))
        strategy_l.addWidget(self.canonical_book)
        self.pco_finish=QCheckBox('PCO 10수 예제 경로 + 실제 NEXT 종결 검증 (실험)')
        self.pco_finish.setChecked(bool(self.cfg.get('pco_finish',False)))
        strategy_l.addWidget(self.pco_finish)
        self.stacker63=QCheckBox('6-3 전용 스태커 · 7열 웰 우선 (실험)')
        self.stacker63.setChecked(bool(self.cfg.get('stack63',False)))
        strategy_l.addWidget(self.stacker63)
        self.blank_hold=QCheckBox('빈 보드 새 판일 때 HOLD 비어 있음 가정 (직접 확인 후 켜기)')
        self.blank_hold.setChecked(bool(self.cfg.get('blank_hold_on_new_game',False)))
        strategy_l.addWidget(self.blank_hold)
        from tbp_bridge import executable_is_ready, binary_location
        self.solver_group=QButtonGroup(self)
        self.solver_group.setExclusive(True)
        self.solver_buttons={}
        solver_options=(
            ('internal','① INTERNAL ONLY — 내장 엔진만'),
            ('cc2','② COLD CLEAR 2 ONLY — CC2만 사용 · 실패 시 안전 중지'),
            ('hybrid','③ AUTO HYBRID — CC2 우선 / 실패 시 내장 엔진'),
            ('bundled','④ LEGACY BUNDLED — 자체 Python 전략 별도 프로세스'))
        cc2_found=executable_is_ready()
        for code,label in solver_options:
            radio=QRadioButton(label)
            radio.setToolTip(('필요한 CC2 실행파일: '+str(binary_location()) if code in ('cc2','hybrid')
                              else '로컬 자체 엔진으로 실행 · Cold Clear 2 아님' if code=='bundled'
                              else 'CC2 프로세스를 호출하지 않음'))
            if code in ('cc2','hybrid') and not cc2_found:
                radio.setEnabled(False)
                radio.setToolTip('CC2 실행파일을 찾지 못했습니다: '+str(binary_location())+'\n설치한 뒤 앱을 재시작하세요.')
            self.solver_group.addButton(radio)
            self.solver_buttons[code]=radio
            strategy_l.addWidget(radio)
        requested_mode=configured_mode(self.cfg)
        if requested_mode in ('cc2','hybrid') and not cc2_found:
            requested_mode='internal'
        self.solver_buttons[requested_mode].setChecked(True)
        self.engine_explain=QLabel('내장 / CC2 단독 / 자동 혼합은 동시에 사용되지 않습니다. CC2 단독 실패 시 내장 엔진으로 전환하지 않습니다.')
        self.engine_explain.setObjectName('muted'); self.engine_explain.setWordWrap(True)
        strategy_l.addWidget(self.engine_explain)
        self.hold_enabled=QCheckBox('HOLD 포함 추천'); self.hold_enabled.setChecked(True); strategy_l.addWidget(self.hold_enabled)
        row=QHBoxLayout(); row.addWidget(QLabel('HOLD 상태')); self.hold_source=QComboBox(); self.hold_source.addItems(['HOLD 영역 자동 인식','HOLD 비어 있음','I','O','T','J','L','S','Z']); row.addWidget(self.hold_source,1); strategy_l.addLayout(row)
        self.fast=QCheckBox('Fast 저지연 판단 · 공격 목표 활성 시 최대 10후보'); self.fast.setChecked(True); strategy_l.addWidget(self.fast)
        self.safe_single=QCheckBox('내장 Fast: 안전할 때만 Single 억제 (실험 · Garbage 측정 필요)')
        self.safe_single.setChecked(bool(self.cfg.get('safe_single_guard',False)))
        self.safe_single.setToolTip('Garbage 계기판이 구성되어 incoming=0으로 확인되며 B2B가 없고 보드가 낮고 깨끗할 때만 일반 Single -4점. 미측정/압박에는 적용하지 않습니다.')
        strategy_l.addWidget(self.safe_single)
        self.turbo=QCheckBox('고속 묶음 입력'); self.turbo.setChecked(True); strategy_l.addWidget(self.turbo)
        self.instant=QCheckBox('신뢰된 prepared 즉시 사용'); self.instant.setChecked(True); strategy_l.addWidget(self.instant)
        combat_page_l.addWidget(strategy)

        inputbox,input_l=self._section('INPUT & VISION','기본 pulse 8ms · 밝기 저장값 유지 (LEGACY 70은 화면에 따라 조정)')
        row=QHBoxLayout(); row.addWidget(QLabel('키 누름')); self.pulse=QSpinBox(); self.pulse.setRange(4,24); self.pulse.setValue(self.cfg.get('pulse_ms',8)); self.pulse.setSuffix(' ms'); row.addWidget(self.pulse,1); input_l.addLayout(row)
        row=QHBoxLayout(); row.addWidget(QLabel('블록 밝기')); self.threshold=QSpinBox(); self.threshold.setRange(20,250); self.threshold.setValue(self.cfg.get('threshold',105)); row.addWidget(self.threshold,1); input_l.addLayout(row)
        row=QHBoxLayout(); row.addWidget(QLabel('휴먼 강도'))
        self.human_intensity=QComboBox()
        for title,code in [('OFF · 기존 고정 입력','off'),('LOW · 미세 리듬','low'),
                           ('MEDIUM · 자연 리듬','medium'),('HIGH · 강한 리듬','high')]:
            self.human_intensity.addItem(title,code)
        chosen=self.cfg.get('human_intensity','off')
        self.human_intensity.setCurrentIndex(['off','low','medium','high'].index(chosen))
        row.addWidget(self.human_intensity,1); input_l.addLayout(row)
        rhythm_note=QLabel('안정적인 입력 8회 확인 후 이동·회전에만 적용 · 추가 대기/실수 없음 · 기본 OFF')
        rhythm_note.setObjectName('muted'); rhythm_note.setWordWrap(True); input_l.addWidget(rhythm_note)
        self.show_overlay=QCheckBox('게임 위 추천 착지 오버레이'); self.show_overlay.setChecked(self.capture_safe); self.show_overlay.setEnabled(self.capture_safe); input_l.addWidget(self.show_overlay)
        system_page_l.addWidget(inputbox)

        keys,keys_l=self._section('KEY BINDINGS','게임 CONFIG와 동일하게 맞춤')
        self.bind_fields={}; form=QFormLayout(); keys_l.addLayout(form)
        for action,label in [('left','왼쪽'),('right','오른쪽'),('cw','시계 회전'),('ccw','반시계 회전'),('rot180','180° 회전'),('soft','소닉드롭'),('drop','하드드롭'),('hold','홀드')]:
            edit=QLineEdit(self.cfg.get('bindings',{}).get(action,DEFAULT[action])); edit.setMaximumWidth(110); self.bind_fields[action]=edit; form.addRow(label,edit)
        system_page_l.addWidget(keys)

        tools,tools_l=self._section('TOOLS','인식 확인과 문제 진단')
        actions=QHBoxLayout(); self.button=QPushButton('PREVIEW · F7'); self.button.setObjectName('secondaryButton'); self.button.clicked.connect(self.toggle); actions.addWidget(self.button)
        diagnostic=QPushButton('진단 ZIP 저장'); diagnostic.clicked.connect(self.save_diagnostic); actions.addWidget(diagnostic); tools_l.addLayout(actions)
        helptext=QLabel('AUTO는 TETR.IO 창이 포커스된 상태에서 F9 · 긴급 중지는 ESC'); helptext.setObjectName('muted'); helptext.setWordWrap(True); tools_l.addWidget(helptext)
        system_page_l.addWidget(tools)
        for page_l in (region_page_l,combat_page_l,system_page_l):page_l.addStretch()

        footer=QLabel('SDF ∞ 권장  ·  설정은 %LOCALAPPDATA%\\TetrioAssistPrototype\\settings.json 에 자동 저장')
        footer.setObjectName('footer'); footer.setAlignment(Qt.AlignmentFlag.AlignCenter); outer.addWidget(footer)

        self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.setTimerType(Qt.TimerType.PreciseTimer); self.timer.start(8); self.f7=False
        self.turbo.toggled.connect(self.settings_changed); self.pulse.valueChanged.connect(self.settings_changed)
        self.instant.toggled.connect(self.settings_changed); self.fast.toggled.connect(self.settings_changed)
        self.safe_single.toggled.connect(self.settings_changed)
        self.hold_enabled.toggled.connect(self.settings_changed); self.hold_source.currentIndexChanged.connect(self.settings_changed)
        for edit in self.bind_fields.values(): edit.textEdited.connect(self.settings_changed)
        self.mode.currentIndexChanged.connect(self.settings_changed); self.threshold.valueChanged.connect(self.settings_changed)
        self.vision_mode.currentIndexChanged.connect(self.settings_changed)
        self.ruleset.currentIndexChanged.connect(self.settings_changed)
        self.combo_table.currentIndexChanged.connect(self.settings_changed)
        self.rounding.currentIndexChanged.connect(self.settings_changed)
        self.opening_enabled.toggled.connect(self.settings_changed)
        self.adaptive_opener.toggled.connect(self.settings_changed)
        self.canonical_book.toggled.connect(self.settings_changed)
        self.pco_finish.toggled.connect(self.settings_changed)
        self.stacker63.toggled.connect(self.settings_changed)
        self.blank_hold.toggled.connect(self.settings_changed)
        for button in self.solver_buttons.values():
            button.toggled.connect(self.settings_changed)
        for code in ('cc2','hybrid'):
            self.solver_buttons[code].toggled.connect(
                lambda checked: self._prewarm_cc2() if checked else None)
        if requested_mode in ('cc2','hybrid'):
            QTimer.singleShot(0,self._prewarm_cc2)
        self.human_intensity.currentIndexChanged.connect(self.settings_changed)
        self.show_overlay.toggled.connect(lambda checked:self.overlay.hide() if not checked else None)
        if not self.capture_safe:self.status.setText('캡처 제외 기능 미지원 · 별도 미리보기 사용')
        self._refresh_region_status(); self._update_run_chip('OFF')
        self.setStyleSheet('''
            QWidget{background:#101723;color:#e8eef7;font-size:13px;font-family:"Segoe UI","Malgun Gothic";}
            QWidget#sidePanel,QWidget#card{background:#172235;border:1px solid #26364c;border-radius:10px;}
            QLabel#heroTitle{font-size:21px;font-weight:800;color:#ffffff;letter-spacing:1px;}
            QLabel#heroSub,QLabel#muted,QLabel#footer{color:#8fa2b8;}
            QLabel#sectionTitle{font-size:12px;font-weight:800;color:#9fb2c8;letter-spacing:1px;}
            QLabel#primaryStatus{font-size:14px;font-weight:700;color:#f4f8ff;padding:3px 0;}
            QLabel#liveValue{font-size:14px;font-weight:750;color:#ffffff;background:#101923;border-radius:7px;padding:9px;}
            QLabel#monoStatus{font-family:"Consolas","D2Coding";background:#101923;border-radius:7px;padding:8px;color:#c9d7e7;}
            QLabel#messageBox{background:#101923;border-radius:7px;padding:9px;color:#c9d7e7;}
            QLabel#hotkeyBar{background:#0c121c;border:1px solid #26364c;border-radius:7px;padding:8px;color:#8fa2b8;font-weight:700;}
            QLabel#chipOff,QLabel#chipPause,QLabel#chipOn{border-radius:9px;padding:8px 12px;font-weight:800;}
            QLabel#chipOff{background:#2b3442;color:#aebbc9;} QLabel#chipPause{background:#594a20;color:#ffd973;} QLabel#chipOn{background:#143f35;color:#56ffd3;}
            QLabel#combatAttack,QLabel#combatCounter,QLabel#combatDefense,QLabel#combatEmergency{border-radius:7px;padding:8px;font-weight:800;}
            QLabel#combatAttack{background:#123b32;color:#56ffd3;} QLabel#combatCounter{background:#3f3818;color:#ffe57a;} QLabel#combatDefense{background:#432c19;color:#ffb067;} QLabel#combatEmergency{background:#491f29;color:#ff7a95;}
            QLabel#regionOk,QLabel#regionMissing{border-radius:6px;padding:5px 8px;font-size:11px;font-weight:700;}
            QLabel#regionOk{background:#143f35;color:#56ffd3;} QLabel#regionMissing{background:#2a313c;color:#8796a8;}
            QPushButton{padding:8px 11px;background:#24344a;border:1px solid #3b526e;border-radius:7px;font-weight:650;}
            QPushButton:hover{background:#304966;border-color:#56a8c9;} QPushButton:pressed{background:#1c2b3e;}
            QPushButton#secondaryButton{background:#1b293b;}
            QComboBox,QSpinBox,QLineEdit{padding:7px;background:#101923;border:1px solid #33475f;border-radius:6px;selection-background-color:#365271;}
            QCheckBox{spacing:8px;padding:3px 0;} QScrollArea{border:0;background:transparent;}
            QTabWidget::pane{border:1px solid #30435c;border-radius:8px;background:#111a29;top:-1px;}
            QTabBar::tab{background:#172235;color:#9fb2c8;min-width:88px;padding:9px 10px;margin-right:3px;border-radius:6px;}
            QTabBar::tab:selected{background:#32506e;color:#ffffff;font-weight:bold;}
            QTabBar::tab:hover{background:#243d59;}

        ''')
    @staticmethod
    def _settings_page():
        page=QWidget(); outer=QVBoxLayout(page); outer.setContentsMargins(0,4,0,4)
        inner=QWidget(); layout=QVBoxLayout(inner)
        layout.setContentsMargins(8,8,8,8); layout.setSpacing(10)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(inner)
        outer.addWidget(scroll)
        return page,layout

    def _section(self,title,subtitle=''):
        box=QWidget(); box.setObjectName('card'); layout=QVBoxLayout(box); layout.setContentsMargins(12,11,12,11); layout.setSpacing(7)
        head=QLabel(title); head.setObjectName('sectionTitle'); layout.addWidget(head)
        if subtitle:
            sub=QLabel(subtitle); sub.setObjectName('muted'); sub.setWordWrap(True); layout.addWidget(sub)
        return box,layout

    def _region_row(self,parent,label,which,hint):
        row=QHBoxLayout(); name=QLabel(label); name.setMinimumWidth(72); row.addWidget(name)
        note=QLabel(hint); note.setObjectName('muted'); note.setMinimumWidth(92); row.addWidget(note,1)
        state=QLabel('미설정'); state.setAlignment(Qt.AlignmentFlag.AlignCenter); state.setMinimumWidth(58); self.region_status[which]=state; row.addWidget(state)
        button=QPushButton('설정'); button.setMaximumWidth(58); button.clicked.connect(lambda checked=False,k=which:self.select(k)); row.addWidget(button)
        clear=QPushButton('×'); clear.setMaximumWidth(30); clear.setToolTip('이 영역 설정 지우기'); clear.clicked.connect(lambda checked=False,k=which:self.clear_region(k)); row.addWidget(clear)
        parent.addLayout(row)

    def precise_grid_setup(self):
        # No settings mutation until the user presses ENTER on the overlay.
        self.stop_auto('정밀 격자 설정');self.running=False;self.invalidate();self.overlay.hide();self.hide()
        def shot():
            try:
                pixels=np.array(self.cap.grab(self.monitor))[:,:,:3][:,:,::-1]
                def done(board,spawn):
                    if (not precise_board_grid(board) or
                        not valid_region_for('board',board) or
                        not valid_region_for('spawn',spawn) or
                        spawn_pair_rows(board,spawn)!=2):
                        self.status.setText('정밀 격자 범위 오류 · 영역을 저장하지 않았습니다.');return
                    # New geometry invalidates any previous position/plan.
                    self.cfg['board']=board;self.cfg['spawn']=spawn
                    self.tracker.reset();
                    if hasattr(self,'legacy_gate'):self.legacy_gate.reset()
                    self.color_reader.reset();self.failure_frame=None
                    self.save();self._refresh_region_status()
                    self.status.setText('격자 저장 완료 · 초록 BOARD와 주황 SPAWN을 F7에서 확인하세요.')
                self.grid_selector=GridCalibrator(self.screen,pixels,self.monitor,done,self.cfg.get('board'))
                self.grid_selector.destroyed.connect(self.show)
                self.grid_selector.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            except Exception as exc:
                self.show();self.status.setText('정밀 설정 실패: '+str(exc))
        QTimer.singleShot(250,shot)

    def create_aligned_spawn(self):
        board=self.cfg.get('board')
        spawn=spawn_from_board(board,rows=2)
        if spawn is None or not valid_region_for('spawn',spawn):
            self.status.setText('실제 10×20 BOARD를 먼저 정확하게 지정해야 합니다.')
            return
        self.stop_auto('등장 영역 자동 생성');self.running=False;self.invalidate()
        self.cfg['spawn']=spawn
        self.tracker.reset();
        if hasattr(self,'legacy_gate'):self.legacy_gate.reset()
        self.color_reader.reset();self.failure_frame=None
        self.save();self._refresh_region_status()
        self.status.setText('BOARD 위 SPAWN 2줄 생성 · F7에서 실제 위치 확인하세요.')

    def split_existing_board(self):
        # Explicit one-click migration only. The old BOARD remains untouched
        # if its height does not match an approximately 22-row selection.
        pair=split_legacy_22(self.cfg.get('board'))
        if pair is None or not all(valid_region_for(k,r) for k,r in zip(('board','spawn'),pair)):
            self.status.setText('22줄 형식 BOARD가 아닙니다. BOARD 20줄과 SPAWN을 각각 설정하세요.')
            return
        self.stop_auto('격자 자동 분리');self.running=False;self.invalidate()
        self.cfg['board'],self.cfg['spawn']=pair
        self.tracker.reset();
        if hasattr(self,'legacy_gate'):self.legacy_gate.reset()
        self.color_reader.reset();self.failure_frame=None
        self.save();self._refresh_region_status()
        self.status.setText('SPAWN 2줄 + BOARD 20줄 분리 완료 · F7에서 위치 확인')

    def clear_region(self,which):
        if which not in self.cfg:return
        self.stop_auto('영역 초기화'); self.running=False; self.tracker.reset(); self.failure_frame=None
        self.cfg.pop(which,None)
        if which=='board':
            self.cfg.pop('spawn',None)  # never leave an old, unaligned spawn
            self.color_reader.reset()
        if which=='spawn':self.tracker.reset()
        if which=='garbage':
            self.garbage_filter=GarbageFilter();self.incoming_garbage=0;self.garbage_confidence=0.0
        self.save(); self.invalidate(); self._refresh_region_status()
        self.status.setText(f'{which.upper()} 영역 설정을 지웠습니다.')

    def _refresh_region_status(self):
        for which,label in getattr(self,'region_status',{}).items():
            ready=which in self.cfg
            geometry_ok=not (which=='board' and ready and not precise_board_grid(self.cfg[which]))
            label.setText('설정됨' if ready and geometry_ok else '확인필요' if ready else '미설정')
            label.setObjectName('regionOk' if ready and geometry_ok else 'regionMissing')
            try: label.style().unpolish(label); label.style().polish(label)
            except Exception: pass
        if hasattr(self,'region_status'):
            ready=sum(1 for k in self.region_status if k in self.cfg)
            core=all(k in self.cfg for k in ('board','next'))
            if hasattr(self,'status') and not self.running:
                if 'board' in self.cfg and not precise_board_grid(self.cfg['board']):
                    self.status.setText(diagnostic_geometry(self.cfg['board'],self.cfg.get('spawn')))
                elif core:
                    self.status.setText(f'영역 {ready}/7 설정 · F7 미리보기 또는 게임에서 F9 자동')

    def _update_run_chip(self,state):
        self._run_state=state
        if not hasattr(self,'run_chip'):return
        mapping={'ON':('AUTO ON','chipOn'),'PAUSE':('AUTO WAIT','chipPause'),'OFF':('AUTO OFF','chipOff')}
        text,name=mapping.get(state,mapping['OFF']); self.run_chip.setText(text); self.run_chip.setObjectName(name)
        try: self.run_chip.style().unpolish(self.run_chip); self.run_chip.style().polish(self.run_chip)
        except Exception: pass

    def _set_combat_status(self,combat,text):
        changed=combat!=getattr(self,'_combat_text',None); self._combat_text=combat
        if not hasattr(self,'pvp_status'):return
        names={'ATTACK':'combatAttack','COUNTER':'combatCounter','DEFENSE':'combatDefense','EMERGENCY':'combatEmergency'}
        self.pvp_status.setText(text)
        if changed:
            self.pvp_status.setObjectName(names.get(combat,'combatAttack'))
            try: self.pvp_status.style().unpolish(self.pvp_status); self.pvp_status.style().polish(self.pvp_status)
            except Exception: pass

    def _update_latency_text(self):
        if not hasattr(self,'latency_status'):return
        deep='—' if self.last_deep_ms is None else f'{self.last_deep_ms:.1f}'
        first='—' if self.last_first_input_ms is None else f'{self.last_first_input_ms:.0f}'
        drop='—' if self.last_drop_ms is None else f'{self.last_drop_ms:.0f}'
        lock='—' if self.last_lock_ms is None else f'{self.last_lock_ms:.0f}'
        os_ack=getattr(self.auto,'last_os_to_lock_ms',None)
        os_ack_text='—' if os_ack is None else f'{os_ack:.1f}'
        self.latency_status.setText(f'CAP {self.capture_ms:.1f}   FAST {self.last_fast_ms:.1f}   PLAN {self.last_delivery_ms:.1f}   DEEP {deep} ms\nFIRST {first}   DROP {drop}   LOCK {lock} ms   OS→ACK {os_ack_text} ms · PULSE {self.inputs.pulse_ms} / GAP {self.inputs.gap_ms} ms')

    def _update_queue_text(self,piece=None,nxt=None,hold=None):
        if not hasattr(self,'queue_status'):return
        if piece is None and self.key is not None:piece=self.key[1]
        if nxt is None and self.key is not None:nxt=self.key[2]
        if hold is None:
            hold=(self.auto.held if self.auto.enabled else
                  (self.key[5] if self.key is not None else '?'))
        queue='→'.join(self.deep_queue) if self.deep_queue else (nxt or '—')
        incoming=(self.incoming_garbage if self.garbage_filter.is_known() else '?') if 'garbage' in self.cfg else '—'
        b2b=self.auto.b2b_count if self.auto.enabled else 0
        b2b_actionable=self._planning_b2b()
        display_b2b=max(0,b2b-1)
        opening=''
        if (self.auto.enabled and getattr(self.auto,'opening_eligible',False)
                and getattr(self.auto,'opening_enabled',False)):
            opening=f' · BURST≈{self.auto.opening_generated}/{self.auto.opening_target}'
            if self.auto.opening_finished:
                opening+=(' GOAL' if self.auto.opening_generated>=self.auto.opening_target else ' END')
            active_plan=getattr(self,'current_plan',None)
            profile=getattr(active_plan,'opening_profile','') if active_plan is not None else ''
            if profile and getattr(self,'adaptive_opener',None) is not None and self.adaptive_opener.isChecked():
                opening+=' · OPENER '+profile.replace('VERIFIED_','')
            if getattr(self.auto,'opener_confirmed',0):
                opening+=f' · VERIFIED {self.auto.opener_confirmed}'
        surge=f' · SURGE≈{(b2b-4 if self.auto.ruleset=="quick_play" else b2b-1)}' if b2b>=5 else ''
        combo=max(0,self.auto.combo) if self.auto.enabled else 0
        combo_live=self.auto.combo if self.auto.enabled else -1
        combo_text=f' · COMBO x{combo}' if combo_live>=0 else ' · COMBO —'
        hold_label=('EMPTY' if hold is None else 'UNKNOWN' if hold=='?' else hold)
        self.queue_status.setText(f'CURRENT {piece or "—"}   NEXT {queue}   HOLD {hold_label}   IN {incoming}\nB2B 예측 x{display_b2b} · 로컬 확실 범위 ≥{max(0,self.auto.b2b_local_floor-1) if self.auto.enabled else 0} (서버 미검증) · 전략 B2B x{max(0,b2b_actionable-1)}{surge}{combo_text}{opening}')

    def _event(self,kind,**fields):
        """Bounded, in-memory event trace; disk I/O only on diagnostic export."""
        if not hasattr(self,'events'):return
        self.events.append(dict(t=round(time.perf_counter()-self.session_at,3),kind=kind,**fields))

    def _planning_b2b(self):
        """Evidence-conservative chain hypothesis for internal search.

        Spin adjudication is not observable through LEGACY. An apparent Mini
        clear cannot create a verified B2B chain. The independently observed
        Quad/all-clear lower bound is the only actionable local chain value.
        """
        if not getattr(self.auto,'enabled',False):return 0
        prediction=getattr(self.auto,'b2b_count',0)
        floor=getattr(self.auto,'b2b_local_floor',0)
        if type(prediction) is not int or type(floor) is not int:return 0
        return max(0,min(prediction,floor))

    def _refresh_idle_fast_pressure(self,board,piece,old_lines,old_conf,
                                    old_known,new_lines,new_conf,new_known):
        """Replan current Fast only when the meter changes executable risk.

        IMPORTANT: this function never cancels a Windows input burst, revokes
        an input ticket, or changes an AutoPlayer-committed plan. The pure
        meter-risk policy is shared by accepted ROI readings and expiry.
        """
        change=pressure_change(old_lines,old_conf,old_known,
            new_lines,new_conf,new_known,meter_configured='garbage' in self.cfg)
        driver=getattr(self,'inputs',None)
        now=time.perf_counter()
        if not may_refresh_idle(
                enabled=self.auto.enabled,phase=self.auto.phase,
                input_busy=(getattr(driver,'down',None) is not None or
                            bool(getattr(driver,'busy',False))),
                key_matches=(self.key is not None and self.key[:2]==(board,piece)),
                engine_mode=self._engine_mode(),changed_risk=change,
                now=now,last_refresh=getattr(self,'pressure_fast_refresh_at',None)):
            return False
        self.pressure_fast_refresh_at=now
        self.pressure_fast_refreshes=getattr(self,'pressure_fast_refreshes',0)+1
        # Never let an obsolete Fast target be issued while an old computation
        # is pending. The current frame must be read again before execution.
        self.current_plan=None
        if self.future is None:
            self.submitted_key=None
        else:
            self.pressure_fast_recompute_pending=True
        if hasattr(self,'_event'):
            self._event('pressure_idle_recompute',old=change[0],new=change[1])
        return True

    def _safe_single_active(self):
        """The checkbox alone is NOT proof of a safe zero-pressure turn."""
        return bool(hasattr(self,'safe_single') and self.safe_single.isChecked()
                    and self._engine_mode() in ('internal','hybrid')
                    and hasattr(self,'mode') and self.mode.currentIndex()==3
                    and 'garbage' in self.cfg
                    and getattr(self,'garbage_confidence',0)>=.55
                    and getattr(self,'incoming_garbage',0)==0)

    def _update_deep_stats(self):
        if not hasattr(self,'deep_stats'):return
        a=self.auto
        single_state=('ACTIVE' if self._safe_single_active() else
                      'WAIT' if hasattr(self,'safe_single') and self.safe_single.isChecked() else 'OFF')
        self.deep_stats.setText(
            f'ENGINE {DISPLAY_NAMES.get(self._engine_mode(),"INTERNAL")} · DEEP 선택 {self.deep_applied}/{self.deep_received} · NEXT 연결 {self.continuation_applied}/{self.continuation_requested}'
            f' · NEXT실행 {getattr(a,"prepared_locked",0)}'
            f' (DEEP {getattr(a,"deep_prepared_locked",0)})'
            f' · 전체 선계산착지 {getattr(a,"prepared_locked",0)}/{getattr(a,"prepared_submitted",0)}'
            f' · BEAM착지 {getattr(self,"beam_verified_locks",0)}'
            f' · CC2채택 {getattr(self,"external_selected",0)}/{getattr(self,"external_received",0)} · CC2착지 {getattr(self,"cc2_verified_locks",0)} · CC2응답없음 {getattr(self,"cc2_no_plan",0)}'
            f' · LOCAL채택 {getattr(self,"bundled_selected",0)}/{getattr(self,"bundled_received",0)} · LOCAL착지 {getattr(self,"bundled_verified_locks",0)}'
            f' · CC2 상태 {getattr(self,"cc2_last_reason","NOT_REQUESTED")}'
            f' · HYBRID시도 {getattr(self,"cc2_hybrid_attempts",0)}'
            f' PASS {getattr(self,"cc2_hybrid_passes",0)}'
            f' 내부계산시도 {getattr(self,"cc2_hybrid_fallbacks",0)}'
            f' 프로브 {getattr(self,"cc2_hybrid_last_probe_ms",None)}ms'
            f' · CC2 NEXT {getattr(self,"external_prefetch_applied",0)}/{getattr(self,"external_prefetch_received",0)} · LOCAL NEXT {getattr(self,"bundled_prefetch_applied",0)}/{getattr(self,"bundled_prefetch_received",0)}'
            f' · HOLD 관측기회 {getattr(self,"hold_eligible_observations",0)} 미인식 {getattr(self,"hold_unknown_observations",0)} 확인 {getattr(a,"verified_holds",0)}'
            f' · SINGLE보호 {single_state}'
            f' · B2B보존착지 {getattr(a,"b2b_guard_verified_locks",0)}'
            f' · Mini 목표착지 {getattr(a,"all_mini_planned_locks",0)}(서버판정 미확인)'
            f' · 최대B2B추정 {getattr(a,"b2b_max_observed",0)}'
            f' · 공격대기턴 {getattr(self,"deep_wait_turns",0)}'
            f' · Garbage복구 {getattr(self,"vision_rise_recovered",0)}'
            f' · stale {self.deep_stale+getattr(self,"continuation_stale",0)}')

    def invalidate(self,*args):
        self.generation+=1; self.deep_revision=getattr(self,'deep_revision',0)+1; self.key=None; self.submitted_key=None; self.current_plan=None
        self.roi_context=None; self.roi_pending=None; self.deep_queue_slots=()
        self.next_queue_trusted=False
        self.next_queue_retry_at=0; self.next_queue_retry_done=False; self.next_queue_missing=()
        if hasattr(self,'deep_cancel_revision'):self.deep_cancel_revision.value=self.deep_revision
        if hasattr(self,'deep_pending'):self.deep_pending=None
        self.overlay.hide(); self.preview.move=None; self.preview.update()
    def save(self):
        self.cfg['bindings']={k:e.text().upper().strip() for k,e in self.bind_fields.items()}
        self.cfg['ruleset']=self.ruleset.currentData()
        self.cfg['combo_table']=self.combo_table.currentData()
        self.cfg['rounding']=self.rounding.currentData()
        self.cfg['opening_enabled']=self.opening_enabled.isChecked()
        self.cfg['adaptive_opener']=self.adaptive_opener.isChecked()
        self.cfg['canonical_book']=self.canonical_book.isChecked()
        self.cfg['pco_finish']=self.pco_finish.isChecked()
        self.cfg['stack63']=self.stacker63.isChecked()
        self.cfg['blank_hold_on_new_game']=self.blank_hold.isChecked()
        self.cfg['safe_single_guard']=self.safe_single.isChecked()
        self.cfg['engine_mode']=self._engine_mode()
        # Legacy checkbox settings must not override the explicit engine selector.
        self.cfg.pop('external_tbp',None);self.cfg.pop('bundled_tbp',None)
        self.cfg['pulse_ms']=self.pulse.value()
        self.cfg['human_intensity']=self.human_intensity.currentData()
        self.cfg['threshold']=self.threshold.value()
        self.cfg['vision_mode']=self.vision_mode.currentData()
        self.cfg=save_settings(CONFIG,self.cfg)
    def select(self,which):
        self.tracker.reset(); self.failure_frame=None; self.stop_auto('영역 변경'); self.running=False; self.invalidate(); self.hide()
        def shot():
            try:
                pixels=np.array(self.cap.grab(self.monitor))[:,:,:3][:,:,::-1]
                def done(r):
                    sx=self.monitor['width']/self.screen.geometry().width(); sy=self.monitor['height']/self.screen.geometry().height()
                    region={'left':self.monitor['left']+round(r.x()*sx),'top':self.monitor['top']+round(r.y()*sy),'width':round(r.width()*sx),'height':round(r.height()*sy)}
                    if not valid_region_for(which,region):
                        self.status.setText('영역 크기·좌표·픽셀 예산이 안전 범위를 벗어났습니다.'); return
                    if which=='board' and not precise_board_grid(region) and split_legacy_22(region) is None:
                        self.status.setText('10×20 정사각 격자 불일치 · 1칸 정밀 설정을 사용하세요.'); return
                    # Verify alignment before accepting a new dual-ROI setup.
                    candidate_board=region if which=='board' else self.cfg.get('board')
                    candidate_spawn=region if which=='spawn' else self.cfg.get('spawn')
                    if which=='board' and split_legacy_22(region) is not None:
                        candidate_spawn=None  # 22-line board must be split explicitly
                    if candidate_board is not None and candidate_spawn is not None:
                        if spawn_pair_rows(candidate_board,candidate_spawn) is None:
                            self.status.setText('SPAWN은 BOARD 바로 위, 같은 10열 폭/셀 크기로 맞추세요.')
                            return
                    self.cfg[which]=region
                    if which=='board' and not precise_board_grid(region):
                        self.cfg.pop('spawn',None)
                    if which in ('board','spawn'):self.color_reader.reset()
                    self.save(); self._refresh_region_status()
                    self.status.setText(diagnostic_geometry(region,self.cfg.get('spawn')) if which=='board'
                                        else '영역 저장됨 · F7에서 격자를 확인하세요.')
                messages={
                    'board':'BOARD = actual 10x20 playfield ONLY, exclude upper spawn buffer. ESC cancels.',
                    'spawn':'SPAWN = 1-4 rows immediately ABOVE BOARD, same 10-column width. ESC cancels.',
                    'next':'NEXT 1: tightly select only the first visible piece. ESC = cancel.',
                    'hold':'HOLD: tightly select the held piece. ESC = cancel.',
                    'next2':'NEXT 2: tightly select only the second visible queue piece. ESC = cancel.',
                    'next3':'NEXT 3: tightly select only the third visible queue piece. ESC = cancel.',
                    'garbage':'Incoming garbage: select only the full vertical garbage meter, bottom to top. ESC = cancel.',
                }
                self.selector=Selector(self.screen,pixels,done,messages.get(which,'Drag region. ESC = cancel.'))
                self.selector.destroyed.connect(self.show)
                self.selector.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            except Exception as e: self.show(); self.status.setText(str(e))
        QTimer.singleShot(250,shot)
    def save_diagnostic(self):
        auto_details=self.auto.snapshot()
        self.stop_auto('진단 저장')
        if self.last_frame is None:
            self.status.setText('F7로 화면을 먼저 인식한 뒤 저장하세요.');return
        try:
            frame,grid,prior=self.failure_frame or (self.last_frame,self.last_grid,self.tracker.previous)
            directory=CONFIG.parent/'diagnostics'/str(time.time_ns());directory.mkdir(parents=True)
            png=directory/'board.png'
            board_image=rgb_image(frame[:,:,::-1])
            if not board_image.save(str(png)):raise OSError('PNG 저장 실패')
            # Alignment preview for the actual FIRST failed BOARD frame.
            # This is diagnostic drawing only: ROI coordinates and the input
            # controller are NEVER adjusted using this image.
            alignment=directory/'grid_overlay.png'
            highlighted=board_image.copy()
            painter=QPainter(highlighted)
            painter.setPen(QPen(QColor(0,218,255,155),1))
            w=highlighted.width();h=highlighted.height()
            for col in range(11):
                x=round(col*w/10)
                painter.drawLine(x,0,x,h-1)
            for row in range(21):
                y=round(row*h/20)
                painter.drawLine(0,y,w-1,y)
            coords=(getattr(self,'failure_vision',None) or {}).get('uncertain_cells_yx',[])
            painter.setPen(QPen(QColor(255,50,65),2))
            for y,x in coords[:32]:
                if 0<=int(x)<10 and 0<=int(y)<20:
                    x0=round(int(x)*w/10);y0=round(int(y)*h/20)
                    x1=round((int(x)+1)*w/10);y1=round((int(y)+1)*h/20)
                    painter.drawRect(x0,y0,max(1,x1-x0-1),max(1,y1-y0-1))
            painter.end()
            overlay_saved=highlighted.save(str(alignment))
            info={'version':'1.1.10.17-deadline-route' ,'threshold':self.threshold.value(),'vision_mode':self.vision_mode.currentData(),'vision_background_at_export':(self.color_reader.background.tolist() if self.color_reader.background is not None else None),'board_region':self.cfg.get('board'),'spawn_region':self.cfg.get('spawn'),'grid':grid,'previous':prior,'auto_at_save':auto_details,'auto_at_failure':self.failure_details,'vision_at_failure':getattr(self,'failure_vision',None),'spawn_first':{'accepted':getattr(self,'spawn_fast_hits',0),'fallbacks':getattr(self,'spawn_fallbacks',0),'next_cache_hits':getattr(self,'next_cache_hits',0),'next_cache_misses':getattr(self,'next_cache_misses',0)},'pace':self.auto.pace(time.perf_counter()),'attack_pace':(self.auto.attack_window.snapshot(time.perf_counter()).as_dict() if isinstance(getattr(self.auto,'attack_window',None),AttackRateWindow) else {}),'pvp':{'deep_revision':self.deep_revision,'queue_revision':self.queue_revision,'queue':self.deep_queue,'incoming':self.incoming_garbage,'garbage_confidence':self.garbage_confidence,'continuation_requested':self.continuation_requested,'continuation_applied':self.continuation_applied,'continuation_stale':getattr(self,'continuation_stale',0),'beam_verified_locks':getattr(self,'beam_verified_locks',0),'counter_beam_verified_locks':getattr(self,'counter_beam_verified_locks',0),'hold_sequence_applied':getattr(self,'hold_sequence_applied',0),'hold_sequence_verified_locks':getattr(self,'hold_sequence_verified_locks',0),'b2b_guard_verified_locks':auto_details.get('b2b_guard_verified_locks',0),'all_mini_planned_locks':auto_details.get('all_mini_planned_locks',0),'b2b_max_internal':auto_details.get('b2b_max_observed',0),'b2b_local_floor':auto_details.get('b2b_local_floor',0),'b2b_unverified_spins':auto_details.get('b2b_unverified_spins',0),'rotation_ack_rejects':auto_details.get('rotation_ack_rejects',0),'rotation_180_suppressed':auto_details.get('rotation_180_suppressed',0),'post_drop_rise_resyncs':auto_details.get('post_drop_rise_resyncs',0),'post_drop_fresh_resyncs':auto_details.get('post_drop_fresh_resyncs',0),'deep_wait_turns':getattr(self,'deep_wait_turns',0),'deep_budget_last_ms':getattr(self,'deep_budget_last_ms',0),'deep_budget_low_count':getattr(self,'deep_budget_low_count',0),'deep_budget_boost_count':getattr(self,'deep_budget_boost_count',0),'bundled_received':getattr(self,'bundled_received',0),'bundled_verified_locks':getattr(self,'bundled_verified_locks',0),'bundled_selected':getattr(self,'bundled_selected',0),'bundled_unadopted':getattr(self,'bundled_unadopted',0),'external_received':getattr(self,'external_received',0),'external_selected':getattr(self,'external_selected',0),'external_unadopted':getattr(self,'external_unadopted',0),'external_prefetch_received':getattr(self,'external_prefetch_received',0),'external_prefetch_applied':getattr(self,'external_prefetch_applied',0),'engine_mode':self._engine_mode(),'adaptive_opener':(self.adaptive_opener.isChecked() if hasattr(self,'adaptive_opener') else False),'canonical_book':(self.canonical_book.isChecked() if hasattr(self,'canonical_book') else False),'pco_finish':(self.pco_finish.isChecked() if hasattr(self,'pco_finish') else False),'stack63':(self.stacker63.isChecked() if hasattr(self,'stacker63') else False),'opener_fast_selected':getattr(self,'opener_fast_selected',0),'opener_deep_selected':getattr(self,'opener_deep_selected',0),'opener_first_wait_frames':getattr(self,'opener_first_wait_frames',0),'opener_confirmed':auto_details.get('opener_confirmed',0),'opener_last':auto_details.get('opener_last',''),
                                'safe_single_guard':(self.safe_single.isChecked() if hasattr(self,'safe_single') else True),'safe_single_active':self._safe_single_active(),'hold_observations':{'eligible':getattr(self,'hold_eligible_observations',0),'unknown':getattr(self,'hold_unknown_observations',0),'blocked':getattr(self,'hold_blocked_observations',0)},'cc2_verified_locks':getattr(self,'cc2_verified_locks',0),'cc2_no_plan':getattr(self,'cc2_no_plan',0),'cc2_strict_waits':getattr(self,'cc2_strict_waits',0),'cc2_last_reason':getattr(self,'cc2_last_reason','NOT_REQUESTED'),'cc2_hybrid_attempts':getattr(self,'cc2_hybrid_attempts',0),'cc2_hybrid_passes':getattr(self,'cc2_hybrid_passes',0),'cc2_hybrid_fallbacks':getattr(self,'cc2_hybrid_fallbacks',0),'cc2_hybrid_stale':getattr(self,'cc2_hybrid_stale',0),'cc2_hybrid_reasons':dict(getattr(self,'cc2_hybrid_reasons',{})),'input_pressure_invalidations':getattr(self,'input_pressure_invalidations',0),'meter_only_updates':getattr(self,'meter_only_updates',0),'legacy_transition_rejected':getattr(getattr(self,'legacy_gate',None),'rejections',0),'legacy_transition_confirmed':getattr(getattr(self,'legacy_gate',None),'confirmed_changes',0),'legacy_pose_jump_rejected':getattr(getattr(self,'legacy_gate',None),'pose_jump_rejections',0),'legacy_provisional_frames':getattr(self,'legacy_provisional_frames',0),'legacy_provisional_resolved':getattr(self,'legacy_provisional_resolved',0),'legacy_provisional_escalated':getattr(self,'legacy_provisional_escalated',0),'pressure_fast_refreshes':getattr(self,'pressure_fast_refreshes',0),'cc2_last_elapsed_ms':getattr(self,'cc2_last_elapsed_ms',None),'bundled_prefetch_received':getattr(self,'bundled_prefetch_received',0),'bundled_prefetch_applied':getattr(self,'bundled_prefetch_applied',0),'vision_rise_recovered':getattr(self,'vision_rise_recovered',0),'event_count':len(getattr(self,'events',()))},'events':list(getattr(self,'events',())),'control_gaps':[x.__dict__ for x in getattr(getattr(self,'stall_monitor',None),'samples',[])],
                  'control_transitions':(self.stall_monitor.trace_snapshot() if hasattr(self,'stall_monitor') else []),'latency':{'capture_ms':self.capture_ms,'fast_ms':self.last_fast_ms,'plan_ms':self.last_delivery_ms,'deep_ms':self.last_deep_ms,'first_input_ms':self.last_first_input_ms,'drop_ms':self.last_drop_ms,'lock_ms':self.last_lock_ms,'os_to_lock_ms':(getattr(self.auto,'last_os_to_lock_ms',None) if
                    type(getattr(self.auto,'last_os_to_lock_ms',None)) in (int,float) else None)},'note':'First-failure BOARD crop only; spawn_region geometry is listed but full desktop/SPAWN pixels are not exported. Pixel evidence freezes background at failure time.'}
            data=directory/'recognition.json';data.write_text(json.dumps(info,ensure_ascii=False,indent=2),'utf-8')
            bundle=directory/'recognition.zip'
            with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
                z.write(png,'board.png');z.write(data,'recognition.json')
                if overlay_saved:z.write(alignment,'grid_overlay.png')
            self.status.setText('진단 ZIP 저장됨: '+str(bundle))
            if sys.platform=='win32':os.startfile(str(directory))
        except Exception as e:self.status.setText('진단 저장 오류: '+str(e))
    def reset_color_calibration(self):
        """Explicit transactional recalibration; never trust one UI-covered frame."""
        if 'board' not in self.cfg:
            self.status.setText('BOARD 영역을 먼저 지정하세요.'); return
        if hasattr(self,'vision_mode') and self.vision_mode.currentData()=='legacy':
            self.status.setText('LEGACY는 배경 보정을 사용하지 않습니다. AUTO 또는 STRICT를 선택하세요.'); return
        self.stop_auto('컬러 배경 재보정'); self.tracker.reset()
        self._reset_vision_episode()
        self.failure_frame=None; self.invalidate()
        self.color_reader.begin_recalibration(time.perf_counter())
        self.running=True
        self._calibration_hidden=True
        self.overlay.hide()
        self.vision_status.setText('VISION CALIBRATING · 0/4')
        self.status.setText('안전 재보정 중: 게임 화면 4프레임을 확인한 뒤 복귀합니다. 자동입력은 중지됩니다.')
        # The GUI can cover the selected ROI, causing the former one-frame
        # recalibration to learn UI pixels. Hide only for this bounded capture.
        self.hide()

    def _finish_color_calibration(self, result):
        if getattr(self,'_calibration_hidden',False):
            self._calibration_hidden=False
            self.show()
        self.tracker.reset(); self.invalidate()
        self.vision_status.setText('VISION COLOR · '+result.reason if result.status=='success'
                                   else 'VISION UNKNOWN · '+result.reason)
        self.running=False  # Showing this GUI can occlude BOARD. Restart preview explicitly.
        self.status.setText(result.reason + ' · 게임 화면이 보이는 상태에서 F7로 미리보기 확인')


    def _reset_vision_episode(self):
        self.vision_failure_open=False
        self.vision_failure_at=None

    def settings_changed(self,*args):
        self._reset_vision_episode()
        self.tracker.reset(); self.color_reader.reset();
        if hasattr(self,'legacy_gate'):self.legacy_gate.reset()
        if hasattr(self,'legacy_recovery'):self.legacy_recovery.reset()
        self.failure_frame=None; self.stop_auto('설정 변경'); self.invalidate()
    def pause_auto(self,reason):
        # Focus loss can persist for thousands of 8-ms ticks. Once the worker
        # is fully released, do not repeatedly revoke tickets and rejoin it.
        if (self.auto.paused and self.inputs.down is None and
                not getattr(self.inputs,'busy',False)):
            self.auto.pause(reason,time.perf_counter())
            return
        self.input_ticket_epoch=getattr(self,'input_ticket_epoch',0)+1
        self.auto.pause(reason,time.perf_counter())
        try:self.inputs.release()
        except OSError:pass
        self.auto_status.setText(self.auto.message); self._update_run_chip('PAUSE')
    def stop_auto(self,reason='사용자 중지'):
        self.input_delivery_active=None
        self.input_ticket_epoch=getattr(self,'input_ticket_epoch',0)+1
        self.input_board_ticket=None
        self.auto.stop(reason)
        try: self.inputs.disarm()
        except OSError as e: self.auto.message+=' / '+str(e)
        self.auto_status.setText(self.auto.message); self._update_run_chip('OFF')
        self.session_roi_excluded=set()
        if (hasattr(self,'deep_pool') and self._engine_mode() in ('cc2','hybrid')):
            # Never leave a timeout-preserved CC2 search consuming CPU after
            # F9/ESC stop. A queued close is confined to the single worker.
            try:
                from tbp_bridge import close_external_worker
                self.deep_pool.submit(close_external_worker)
            except (RuntimeError,OSError,ValueError):pass

    def _runtime_roi(self,name):
        # A stale or outside-window region is NEVER read after F9 excludes it.
        if name in getattr(self,'session_roi_excluded',()):return None
        return self.cfg.get(name)

    def toggle_auto(self):
        if getattr(getattr(self,'color_reader',None),'recalibrating',False):
            self.auto_status.setText('컬러 배경 재보정이 끝난 뒤 자동입력을 시작하세요.'); return
        if self.auto.enabled:
            self.stop_auto(); self.invalidate(); return
        if 'board' not in self.cfg:
            self.auto_status.setText('먼저 보드 영역을 지정하세요.'); return
        if not precise_board_grid(self.cfg['board']):
            self.auto_status.setText(diagnostic_geometry(self.cfg['board'],self.cfg.get('spawn')))
            return
        bindings={k:e.text().upper().strip() for k,e in self.bind_fields.items()}
        if not all(valid_binding(v) for v in bindings.values()) or len(set(bindings.values()))!=len(bindings):
            self.auto_status.setText('서로 다른 키를 지정하세요: A–Z, LEFT, RIGHT, UP, DOWN, SPACE, SHIFT, CTRL'); return
        if 'spawn' in self.cfg and spawn_pair_rows(self.cfg['board'],self.cfg['spawn']) is None:
            self.auto_status.setText('SPAWN/BOARD 격자가 맞지 않습니다. ① 영역 인식에서 재설정하세요.'); return
        if self._engine_mode() in ('cc2','hybrid'):
            from tbp_bridge import executable_is_ready, binary_location
            if not executable_is_ready():
                self.auto_status.setText('Cold Clear 2 실행파일 없음: '+str(binary_location()))
                return
            hold_unknown=self.read_hold()=='?'
            opted_blank=(hasattr(self,'blank_hold') and self.blank_hold.isChecked()
                         and 'hold' in self.cfg)
            if self._engine_mode()=='cc2' and hold_unknown and not opted_blank:
                self.auto_status.setText('CC2 단독 시작 불가: HOLD 상태 미인식 · HOLD 영역을 설정하거나 실제 비었음을 확인한 후 HOLD 비어 있음 선택')
                self.status.setText('CC2 ONLY: HOLD UNKNOWN · TBP 안전 제약으로 시작 보류')
                return
            # If the user explicitly opted into blank-HOLD on an empty new
            # board, the existing observed-board/first-piece guard can still
            # prove it later. Do not make an early unobserved HOLD claim.
        self.session_roi_excluded=set()
        ok,reason=self.inputs.arm(self.cfg['board'])
        if not ok:
            self.auto_status.setText(reason)
            self.status.setText(reason + ' · 게임 창을 맨 앞으로 띄운 뒤 F9 또는 BOARD 격자를 확인하세요.')
            return
        excluded_message=''
        if hasattr(self.inputs,'set_capture_regions'):
            extras={k:self.cfg[k] for k in ('spawn','next','next2','next3','hold','garbage') if k in self.cfg}
            if not self.inputs.set_capture_regions(extras):
                issues=getattr(self.inputs,'capture_problems',[])
                if isinstance(issues,list) and issues and all(
                        isinstance(item,tuple) and len(item)==2
                        and item[0] in extras and '게임 창 바깥' in item[1] for item in issues):
                    # Optional screenshot regions outside the verified game
                    # window are dropped for this session, never read. Keep
                    # the user's saved settings unchanged so they can fix ROI.
                    self.session_roi_excluded={name for name,_ in issues}
                    safe={k:r for k,r in extras.items() if k not in self.session_roi_excluded}
                    if self.inputs.set_capture_regions(safe):
                        excluded_message=('창 바깥 ROI 자동 비활성화: '+
                            ', '.join(sorted(self.session_roi_excluded)).upper()+
                            ' · ① 영역 인식에서 해당 항목을 다시 설정하세요.')
                    else:
                        issues=getattr(self.inputs,'capture_problems',issues)
                if not excluded_message:
                    self.inputs.disarm()
                    detail=format_region_issues(issues) if isinstance(issues,list) else '게임 캡처 범위 확인 실패'
                    self.auto_status.setText('F9 실행 불가: '+detail)
                    self.status.setText('F9: '+detail+' · 해당 영역을 재지정하세요.')
                    return
        if hasattr(self,'stall_monitor'):self.stall_monitor.reset()
        self.input_abort_seen=getattr(self.inputs,'abort_count',0)
        self.input_board_ticket=None
        self.input_ticket_epoch=getattr(self,'input_ticket_epoch',0)+1
        self.auto.turbo=self.turbo.isChecked() if hasattr(self,'turbo') else False
        if hasattr(self,'pulse'):self.inputs.pulse_ms=self.pulse.value()
        # A new F9 session can be a different match. Never carry stale incoming
        # meter pressure or its confidence into the next match.
        self.garbage_filter=GarbageFilter();self.incoming_garbage=0;self.garbage_confidence=0.0
        self.cc2_last_reason=('WAITING_FIRST_BOARD' if self._engine_mode()=='cc2'
                              else 'NOT_REQUESTED')
        self.cc2_last_elapsed_ms=None
        self.cc2_hybrid_attempts=0;self.cc2_hybrid_fallbacks=0
        self.cc2_hybrid_passes=0;self.cc2_hybrid_stale=0
        self.cc2_hybrid_reasons={};self.cc2_hybrid_last_probe_ms=None
        self.input_pressure_invalidations=0
        self.meter_only_updates=0
        self.cc2_retry_count=0;self.cc2_retry_at=0.0
        if hasattr(self,'rhythm') and hasattr(self,'human_intensity'):
            self.rhythm.reset(self.human_intensity.currentData())
        if hasattr(self,'events'):self.events.clear();self.session_at=time.perf_counter()
        self.tracker.reset();
        if hasattr(self,'legacy_recovery'):self.legacy_recovery.reset()
        if hasattr(self,'legacy_gate'):self.legacy_gate.reset()
        self.failure_frame=None; self._reset_vision_episode(); self.legacy_provisional_since=None;self.legacy_provisional_frames=0;self.legacy_provisional_resolved=0;self.legacy_provisional_escalated=0;self.pressure_fast_recompute_pending=False;self.pressure_fast_refresh_at=None;self.pressure_fast_refreshes=0; self.bindings=bindings; self.auto.start(self.read_hold()); self.auto.ruleset=self.ruleset.currentData() if hasattr(self,'ruleset') else 'standard'; self.auto.combo_table=self.combo_table.currentData() if hasattr(self,'combo_table') else 'multiplier'; self.auto.rounding=self.rounding.currentData() if hasattr(self,'rounding') else 'auto'; self.auto.opening_enabled=self.opening_enabled.isChecked() if hasattr(self,'opening_enabled') else True; self.auto.opening_adaptive_enabled=self.adaptive_opener.isChecked() if hasattr(self,'adaptive_opener') else False; self.running=True; self.invalidate(); self.auto_status.setText(self.auto.message); self._update_run_chip('ON')
        if excluded_message:
            self.status.setText(excluded_message)
            self.auto_status.setText(self.auto.message + ' · ' + excluded_message)
    def _prewarm_cc2(self):
        """Non-blocking, opt-in CC2 protocol handshake on the Deep worker."""
        if self.auto.enabled or self._engine_mode() not in ('cc2','hybrid'):
            return
        prior=getattr(self,'cc2_warm_future',None)
        if prior is not None and not prior.done():
            return
        try:
            from tbp_bridge import prewarm_external_worker
            self.cc2_last_reason='STARTING'
            future=self.deep_pool.submit(prewarm_external_worker)
            self.cc2_warm_future=future
            def done(f):
                try: status=f.result()
                except (OSError,ValueError,RuntimeError):status='WORKER_ERROR'
                try:self.cc2_warm_ready.emit(status)
                except RuntimeError:pass
            future.add_done_callback(done)
        except (RuntimeError,OSError,ValueError):
            self.cc2_last_reason='WORKER_ERROR'

    def _accept_cc2_warm(self,status):
        if self._engine_mode() not in ('cc2','hybrid'):
            return
        # A pre-warm result must never override the active turn's more useful
        # CALCULATING/PASS/NO_SUGGESTION status if F9 already started.
        if not self.auto.enabled:
            self.cc2_last_reason=status
            self._update_deep_stats()
            if status!='READY':
                self.status.setText('CC2 실행 준비 실패: '+status+' · 단독 모드에서 F9 전에 경로/프로토콜 확인')

    def _record_control_gap(self,now,valid):
        monitor=getattr(self,'stall_monitor',None)
        if monitor is None:return
        checkpoint=monitor.observe(now,enabled=self.auto.enabled,
             phase=self.auto.phase,paused=self.auto.paused,
             reason=self.auto.message,capture_valid=valid,
             input_busy=bool(getattr(self.inputs,'busy',False)),
             deep_pending=bool(getattr(self,'deep_future',None)),
             has_plan=self.current_plan is not None)
        if checkpoint is not None and hasattr(self,'_event'):
            self._event('control_gap',**checkpoint.__dict__)

    def _drain_input_delivery(self,now):
        """Consume OS receipts once; a KEYDOWN is not a game acknowledgement.

        No extra screen capture, Qt access from the worker, or blind resend.
        A cancelled *undispatched* drop is replanned from fresh pixels. An
        already-dispatched drop stays in ACK_WAIT until actual board evidence.
        """
        driver=getattr(self,'inputs',None)
        if driver is None or not isinstance(driver,WindowsInput):return
        events=driver.drain_receipts()
        active=getattr(self,'input_delivery_active',None)
        for token,stage,sent,total,at in events:
            if active is None or token!=active[0]:continue
            _,kind,serial,_submitted_at=active
            if self.auto.serial!=serial:
                if stage=='finished':self.input_delivery_active=None
                continue
            if stage=='key_down' and kind=='drop':
                if self.auto.note_drop_dispatched(token,at):
                    self._event('drop_os_keydown',delivery_ms=round((at-active[3])*1000,2))
            elif stage=='finished':
                self.input_delivery_active=None
                if sent<total:
                    if kind=='drop':
                        if self.auto.note_drop_unsent(token,now,self.current_plan):
                            self._event('drop_unsent_resync')
                    elif (self.auto.enabled and self.auto.plan is not None
                          and self.auto.phase not in ('idle','hold_wait','drop_wait')):
                        self.auto.input_partial_resyncs+=1
                        self.auto._replan(self.current_plan,
                                          '입력 묶음 부분 실행 · 실제 위치 재판단',now,
                                          self.auto.hold_used)
                        self._event('input_partial_resync',sent=sent,total=total)
        # The old aggregate abort counter double-counted already dispatched
        # drops and could replan a turn for the wrong (later) burst.
        self.input_abort_seen=getattr(driver,'abort_count',0)

    def pump_auto(self,state,now):
        if not self.auto.enabled:return
        self._drain_input_delivery(now)
        self._record_control_gap(now,state is not None)
        if self.inputs.down is not None or getattr(self.inputs,'busy',False):return
        # Bounded first-turn admission for user-enabled recipes.  An exact
        # NEXT2 arrives asynchronously, so avoid committing a speculative
        # Fast target a few milliseconds before a genuine book proposal.
        # This is ONLY the first piece, pressure-free, <=65 ms, never a per-
        # placement delay or wait for an unobserved future piece.
        if (state is not None and self.current_plan is not None and
                self._engine_mode()=='internal' and self.auto.phase=='idle' and
                self.auto.confirmed_locks==0 and
                getattr(self.auto,'opening_eligible',None) is True and
                not self.auto.garbage_rise_events and
                not self.auto.unverified_board_resyncs and
                not getattr(self.auto,'soft_disabled',False) and
                getattr(self.auto,'recovery_blocked_target',None) is None and
                getattr(self,'opening_enabled',None) is not None and self.opening_enabled.isChecked() and
                getattr(self,'adaptive_opener',None) is not None and self.adaptive_opener.isChecked() and
                (('garbage' not in self.cfg) or self.garbage_filter.is_known()) and
                self.incoming_garbage==0):
            from opener_commit import opener_start_possible, confirmed_reference_start
            canonical=bool(getattr(self,'canonical_book',None) is not None and self.canonical_book.isChecked())
            pco=bool(getattr(self,'pco_finish',None) is not None and self.pco_finish.isChecked())
            nxt=self.key[2] if self.key else None
            known=tuple(getattr(self,'deep_queue',()) or ())
            queue_still_possible=(len(known)<2 or
                confirmed_reference_start(state[0],state[1],known,
                    canonical=canonical,pco=pco))
            if (queue_still_possible and
                    opener_start_possible(state[0],state[1],nxt,canonical=canonical,pco=pco) and
                    not getattr(self.current_plan,'opening_profile','') and
                    (now-getattr(self,'detected_at',now))<.065):
                self.opener_first_wait_frames=getattr(self,'opener_first_wait_frames',0)+1
                return
        # The Fast plan is available, but v1.1.1 typically started executing it
        # before the worker delivered a stronger offensive beam. Allow at most
        # 32ms from fresh detection for a safe, observed ATTACK decision.
        # No held input, pressure/UNKNOWN meter, recovery, or transition may
        # enter this gate. The Fast plan proceeds at the deadline regardless.
        if self._engine_mode()=='cc2' and state is not None:
            # Exclusive mode cannot execute the Fast preview or a bundled/native
            # fallback. Wait only for a verified CC2 decision from this turn.
            if (self.current_plan is None or
                    getattr(self.current_plan,'continuation_source',None)!='external_tbp'):
                self.cc2_strict_waits=getattr(self,'cc2_strict_waits',0)+1
                detected=getattr(self,'detected_at',0)
                if detected and now-detected>2.0:
                    self.cc2_last_reason='CC2_EXCLUSIVE_TIMEOUT'
                    self._event('cc2_exclusive_timeout',elapsed_ms=round((now-detected)*1000))
                    self.stop_auto('CC2 단독: 유효한 판단을 받지 못했습니다 · 내부 엔진으로 전환하지 않음')
                return
        if state is not None and self.current_plan is not None and getattr(self,'key',None) is not None:
            known=(('garbage' not in self.cfg) or self.garbage_filter.is_known())
            incoming=(self.incoming_garbage if known else 3)
            mode=self._engine_mode()
            if mode=='cc2':
                waiting=False
            elif mode=='hybrid':
                waiting=hybrid_wait_for_cc2(
                    now,getattr(self,'detected_at',0),
                    pending=getattr(self,'deep_future',None) is not None,
                    phase=self.auto.phase,focused=self.inputs.focused(),
                    vision_ok=not getattr(self,'vision_failure_open',False)
                              and not self.auto.paused,
                    incoming=incoming,incoming_known=known,
                    hold=self.key[5],allow_hold=self.key[6])
            else:
                waiting=defer_for_attack(
                    state[0],self.current_plan,self.key[3],self.auto.phase,
                    now,getattr(self,'detected_at',0),
                    getattr(self,'deep_future',None) is not None,
                    foreground=self.inputs.focused(),
                    vision_ok=not getattr(self,'vision_failure_open',False)
                              and not self.auto.paused,
                    incoming=incoming,incoming_known=known)
            if waiting:
                self.deep_wait_frames=getattr(self,'deep_wait_frames',0)+1
                if getattr(self,'deep_wait_seen',None)!=self.generation:
                    self.deep_wait_seen=self.generation
                    self.deep_wait_turns=getattr(self,'deep_wait_turns',0)+1
                    if hasattr(self,'_event'):self._event('deep_wait',budget_ms=110 if mode=='hybrid' else 32)
                return
        # Invalidate an aborted asynchronous command against fresh evidence.
        # In particular, a hard-drop key that was CANCELLED before key-down
        # cannot be treated as a pending server lock indefinitely.
        aborted=getattr(self.inputs,'abort_count',0)
        if type(aborted) is not int:aborted=0
        if aborted!=getattr(self,'input_abort_seen',0) and not isinstance(self.inputs,WindowsInput):
            self.input_abort_seen=aborted
            if (state is not None and self.auto.enabled
                    and self.auto.phase not in ('idle','hold_wait')
                    and self.auto.plan is not None):
                self._event('input_guard_resync',
                            phase=self.auto.phase,reason='QUEUED_COMMAND_ABORTED')
                self.auto._replan(self.current_plan,
                                  '취소된 입력·현재 보드 재확인',now,
                                  self.auto.hold_used,verified_transition=False)
                # Next tick must recompute against independently recognized
                # current board and serial; never repeat the old drop.
                return
        locks_before=self.auto.confirmed_locks
        meter_known=('garbage' not in getattr(self,'cfg',{}) or self.garbage_filter.is_known())
        pressure=bool('garbage' in getattr(self,'cfg',{}) and
                      (not meter_known or self.incoming_garbage>0))
        action=self.auto.observe(state,self.current_plan,now,self.inputs.focused(),pressure=pressure)
        # The committed controller target is now known, even if the first Deep
        # result arrived before the controller left idle. Pipeline NEXT into
        # spare process time; never wait for it in the 8-ms input loop.
        if (self.auto.enabled and self.auto.phase in ('align','rotate_wait','drop_wait')
                and getattr(self,'deep_queue',()) and self.key is not None
                and hasattr(self,'instant') and self.instant.isChecked()):
            self._schedule_committed_continuation(self.deep_queue,self.deep_revision,self.queue_revision)
        if self.auto.confirmed_locks>locks_before:
            self._event('lock',piece=getattr(self.auto,'last_locked_piece','?'),locks=self.auto.confirmed_locks,
                        decision_source=getattr(self.auto,'last_lock_source','fast'),
                        prefetch_source=getattr(self.auto,'last_lock_prefetch_source',None),
                        prepared_locked=getattr(self.auto,'prepared_locked',0),generated=self.auto.sent_attack_upper_bound,
                        os_to_lock_ms=getattr(self.auto,'last_os_to_lock_ms',None))
            if getattr(self.auto,'last_lock_source','')=='bundled_tbp':
                self.bundled_verified_locks+=1
            if getattr(self.auto,'last_lock_source','')=='external_tbp':
                self.cc2_verified_locks+=1
            if getattr(self.auto,'last_lock_source','') in ('offensive_beam','counter_beam','hold_sequence'):
                self.beam_verified_locks=getattr(self,'beam_verified_locks',0)+1
                if self.auto.last_lock_source=='counter_beam':
                    self.counter_beam_verified_locks=getattr(self,'counter_beam_verified_locks',0)+1
                elif self.auto.last_lock_source=='hold_sequence':
                    self.hold_sequence_verified_locks=getattr(self,'hold_sequence_verified_locks',0)+1
            self._update_deep_stats()
            self.last_first_input_ms=getattr(self,'current_first_input_ms',None); self.last_drop_ms=getattr(self,'current_drop_ms',None)
            detected=getattr(self,'detected_at',0); self.last_lock_ms=(now-detected)*1000 if detected else None
        chip='OFF' if not self.auto.enabled else ('PAUSE' if self.auto.paused else 'ON')
        if chip!=getattr(self,'_run_state',None):self._update_run_chip(chip)
        if not self.auto.enabled:
            self.inputs.disarm()
        elif action:
            detected=getattr(self,'detected_at',0); since_detect=(now-detected)*1000 if detected else None
            if getattr(self,'current_first_input_ms',None) is None:self.current_first_input_ms=since_detect
            if action=='drop' or (isinstance(action,tuple) and 'drop' in action):self.current_drop_ms=since_detect
            if hasattr(self,'pulse'):
                base=min(24,self.pulse.value()+self.auto.extra_pulse_ms)
                actions=action if isinstance(action,tuple) else (action,)
                rhythm=getattr(self,'rhythm',None)
                self.inputs.pulse_ms=(rhythm.pulse(base,
                    getattr(self.auto,'verified_input_streak',0),
                    self.auto.extra_pulse_ms,actions)
                    if rhythm is not None else base)
                # No extra gap, no reordering or extra keys for rhythm mode.
                self.inputs.gap_ms=4
            # InputWorker is asynchronous: ensure an obsolete plan cannot
            # inject another key after a newly observed Garbage rise, pause,
            # F9 stop, or different turn. The predicate is O(1) and MUST NOT
            # access Qt widgets or perform any screen capture.
            epoch=getattr(self,'input_ticket_epoch',0)
            serial=self.auto.serial
            ticket_board=getattr(self,'input_board_ticket',None)
            auto=self.auto
            hard_drop=(action=='drop' or (isinstance(action,tuple) and 'drop' in action))
            dispatch_deadline=(now+.090 if hard_drop else None)
            def still_valid():
                return (auto.enabled and not auto.paused and auto.serial==serial
                        and self.input_ticket_epoch==epoch
                        and self.input_board_ticket==ticket_board
                        and (dispatch_deadline is None or time.perf_counter()<=dispatch_deadline))
            if isinstance(self.inputs,WindowsInput):
                try:
                    if isinstance(action,tuple):
                        token=self.inputs.press_sequence(
                            tuple(self.bindings[a] for a in action),valid=still_valid)
                    else:
                        token=self.inputs.press(self.bindings[action],valid=still_valid)
                except Exception:
                    # No key sequence was accepted for dispatch. Do not leave
                    # the controller waiting for a lock that cannot arrive.
                    if hard_drop and self.auto.phase=='drop_wait':
                        self.auto._replan(self.current_plan,
                            'DROP 요청 실패 · 새 보드 확인',now,self.auto.hold_used)
                    raise
                if token is not None:
                    kind='drop' if hard_drop else 'path'
                    self.input_delivery_active=(token,kind,serial,now)
                    if hard_drop:self.auto.note_drop_queued(token,now)
            elif isinstance(action,tuple):
                self.inputs.press_sequence(tuple(self.bindings[a] for a in action))
            else:
                self.inputs.press(self.bindings[action])
            if hasattr(self,'stall_monitor'):self.stall_monitor.sent(now)
        message=self.auto.message
        if message!=getattr(self,'_auto_text',None):
            self.auto_status.setText(message);self._auto_text=message
        # PPS text is diagnostic UI, not control feedback. Updating it at 125 Hz
        # wastes GUI time and can contend with capture/input on Windows.
        if hasattr(self,'pace_status') and now>=getattr(self,'ui_next_at',0):
            pace=self.auto.pace(now)
            rate='측정 중' if pace['pps'] is None else f'{pace["pps"]:.2f} PPS'
            first='대기' if pace['first_lock_ms'] is None else f'{pace["first_lock_ms"]:.0f}ms'
            behind=pace['pps'] is not None and pace['pps']<pace['target_pps']
            # Generated attack is a local rules-model estimate only: we cannot
            # observe the opponent's net incoming, canceled garbage or server spin adjudication.
            attack=self.auto.attack_pace(now)
            if not attack.sample_ready:
                attack_line=f'모델 공격 APM 측정 중 ({attack.observed_seconds:.0f}/10초) / 목표 ≥ 100'
            else:
                attack_line=(f'모델 공격 {attack.modeled_apm:.1f} APM'
                             f' (예상 상한 {attack.modeled_ceiling_apm:.1f}) / 목표 ≥ 100'
                             + (' · 목표 미달' if attack.modeled_apm<attack.target_apm else ' · 모델 목표 달성')
                             + (f' · 모델상 필요 {attack.required_pps_for_target:.1f} PPS'
                                if attack.required_pps_for_target is not None else ''))
            self.pace_status.setText(
                f'관측 착지 {rate} / 착지 참고 목표 ≥ 1.20\n'
                f'{attack_line}\n첫 착지 {first} · 실제 상대 수신 공격량 미확인'
                + ('\n착지 속도 목표 미달 · '+message if behind else ''))
            self._update_latency_text(); self.ui_next_at=now+.10
    def read_hold(self):
        if self.hold_source.currentIndex()==0:return self.piece_box('hold') or '?'
        if self.hold_source.currentIndex()==1:return None
        return self.hold_source.currentText()
    def toggle(self):
        if hasattr(self,'color_reader') and self.color_reader.recalibrating:
            self.color_reader.cancel_recalibration()
            if getattr(self,'_calibration_hidden',False):
                self._calibration_hidden=False; self.show()
            self.running=False
            self.tracker.reset(); self.failure_frame=None
            self.stop_auto('재보정 취소'); self.invalidate()
            self.status.setText('컬러 배경 재보정 취소 · 기존 배경 유지')
            return
        self.tracker.reset(); self.failure_frame=None; self._reset_vision_episode(); self.stop_auto('F7 미리보기 전환')
        if hasattr(self,'legacy_gate'):self.legacy_gate.reset()
        if 'board' not in self.cfg: self.status.setText('먼저 보드 영역을 지정하세요.'); return
        if not precise_board_grid(self.cfg['board']):
            self.running=False
            self.status.setText(diagnostic_geometry(self.cfg['board'],self.cfg.get('spawn')))
            return
        self.running=not self.running; self.invalidate(); self.status.setText('상단에서 새 블록을 기다리는 중' if self.running else '중지됨')
    def _verified_next_after_lock(self, observed_piece, lock_confirmed):
        """Reuse *previously captured* NEXT2 only across a confirmed hard drop.

        Falling back to the existing ROI capture is mandatory when the previous
        queue or transition cannot be verified; no NEXT identity is guessed.
        Pure read: callers must consume the result before invalidate() clears
        the old turn's queue slots.
        """
        nxt, remaining = shifted_queue_after_verified_lock(
            getattr(self,'key',None),getattr(self,'deep_queue_slots',()),
            observed_piece,bool(lock_confirmed and getattr(self,'next_queue_trusted',False)))
        if nxt is not None:
            self.next_cache_hits=getattr(self,'next_cache_hits',0)+1
            return nxt,remaining
        if lock_confirmed:
            self.next_cache_misses=getattr(self,'next_cache_misses',0)+1
        return None,()

    def piece_box(self,which):
        region=self._runtime_roi(which)
        if region is None:return None
        return classify_piece_pixels(np.asarray(self.cap.grab(region))[:,:,:3])

    def schedule_roi_retry(self,board,piece,nxt,hold,periodic=False,queue_retry=False):
        if not hasattr(self,'roi_pool'):return
        if self.roi_future is not None:
            # A previous turn's ROI job must never make the new turn miss its
            # one-shot NEXT2/3 sampling. Periodic garbage polling is disposable.
            if not periodic:
                self.roi_pending=(self.generation,board,piece,nxt,hold,queue_retry)
            return
        need_next=(nxt is None and self._runtime_roi('next') is not None and not periodic and not queue_retry)
        auto_hold=self.hold_source.currentIndex()==0 and self._runtime_roi('hold') is not None
        need_hold=(hold=='?' and auto_hold and not periodic and not queue_retry)
        if periodic:
            next2=next3=None
        elif queue_retry:
            next2=self._runtime_roi('next2') if 'next2' in self.next_queue_missing else None
            next3=self._runtime_roi('next3') if 'next3' in self.next_queue_missing else None
        else:
            next2=self._runtime_roi('next2');next3=self._runtime_roi('next3')
        garbage=None if queue_retry else self._runtime_roi('garbage')
        if not (need_next or need_hold or next2 or next3 or garbage):return
        kind='queue_retry' if queue_retry else ('periodic' if periodic else 'initial')
        self.roi_context=(self.generation,board,piece,self.auto.serial,kind)
        self.roi_future=self.roi_pool.submit(retry_piece_regions,
            self._runtime_roi('next') if need_next else None,
            self._runtime_roi('hold') if need_hold else None,
            next2,next3,garbage,1 if periodic or queue_retry else 3,.008)
        if queue_retry:
            self.next_queue_retry_done=True
            self.next_queue_retry_at=0
        else:
            self.telemetry_next_at=time.perf_counter()+.080

    def _resume_roi_pending(self,board,piece):
        pending=getattr(self,'roi_pending',None)
        self.roi_pending=None
        if not pending or not self.running or self.key is None:return
        generation,pb,pp,nxt,hold,queue_retry=pending
        if generation!=self.generation or (pb,pp)!=(board,piece):return
        self.schedule_roi_retry(pb,pp,nxt,hold,False,queue_retry)

    def poll_roi_retry(self,board,piece):
        if not hasattr(self,'roi_future') or self.roi_future is None or not self.roi_future.done():return
        future=self.roi_future;context=self.roi_context
        self.roi_future=None;self.roi_context=None
        try:
            try:data=future.result()
            except Exception:return
            if (context is None or self.key is None or context[0]!=self.generation or
                    context[1:3]!=(board,piece) or context[3]!=self.auto.serial):return
            kind=context[4] if len(context)>4 else 'initial'
            current_nxt=self.key[2];current_hold=self.key[5]
            recovered_nxt=current_nxt or data.get('next')
            recovered_hold=current_hold if current_hold!='?' else (data.get('hold') or '?')
            if data.get('hold') is not None and self.auto.enabled and self.auto.held=='?':
                self.auto.held=data['hold']

            # Preserve non-contiguous observed slots: NEXT3 may be visible even
            # when NEXT2 failed. Otherwise a successful NEXT2 retry loses NEXT3.
            prior=list(getattr(self,'deep_queue_slots',()))
            if not prior or prior[0]!=recovered_nxt:prior=[recovered_nxt]
            while len(prior)<3:prior.append(None)
            # A queued ROI worker may actually read NEXT2 after the user
            # hard-dropped the current piece, but before observe() noticed
            # the new one. Such a late snapshot belongs to the *new* queue;
            # never poison a previous-turn NEXT2 cache with it.
            capture_time=data.get('captured_at')
            phase=self.auto.phase if self.auto.enabled else 'idle'
            within_turn=(type(capture_time) in (float,int) and
                         (phase not in ('drop_wait','hold_wait') or
                          capture_time<=getattr(self.auto,'sent_at',-1)))
            queue_consistent=(data.get('next') is None or current_nxt is None or
                              data['next']==current_nxt)
            # Legacy synthetic tests supply results without timing. Preserve
            # their Deep-only queue behavior, but do not use such an unstamped
            # result for the latency-sensitive NEXT cache.
            safe_snapshot=queue_consistent and (capture_time is None or within_turn)
            if safe_snapshot:
                if data.get('next2') is not None:prior[1]=data['next2']
                if data.get('next3') is not None:prior[2]=data['next3']
                if (within_turn and len(prior)>1 and prior[0]==recovered_nxt
                        and type(prior[1]) is str and prior[1] in 'IOTJLSZ'):
                    self.next_queue_trusted=True
            elif ((data.get('next2') is not None or data.get('next3') is not None)
                  and hasattr(self,'_event')):
                self._event('queue_snapshot_discarded',phase=phase)
            prior[0]=recovered_nxt
            self.deep_queue_slots=tuple(prior)
            contiguous=[]
            for q in prior:
                if q is None:break
                contiguous.append(q)
            new_queue=tuple(contiguous)

            if kind=='initial' and not getattr(self,'next_queue_retry_done',False):
                missing=tuple(k for k in ('next2','next3') if k in self.cfg and data.get(k) is None)
                self.next_queue_missing=missing
                if missing:self.next_queue_retry_at=time.perf_counter()+.024

            if not hasattr(self,'garbage_filter'):self.garbage_filter=GarbageFilter()
            was_known=self.garbage_filter.is_known()
            prior_incoming=getattr(self,'incoming_garbage',0)
            prior_confidence=getattr(self,'garbage_confidence',0.0)
            incoming,confidence,_=self.garbage_filter.update(data.get('incoming'),data.get('garbage_confidence',0))
            meter_known=self.garbage_filter.is_known()
            previous_meter_known=getattr(self,'meter_was_known',False)
            pressure_risk_change=pressure_change(prior_incoming,prior_confidence,was_known,
                incoming,confidence,meter_known,meter_configured='garbage' in self.cfg)
            # A meter-only change is a prediction, NOT evidence of the fixed
            # stack changing. Revoking the asynchronous input ticket here used
            # to strand partly moved pieces and amplify damage under pressure.
            # Physical board changes/vision loss still revoke input below.
            if (incoming!=prior_incoming or meter_known!=was_known):
                self.meter_only_updates=getattr(self,'meter_only_updates',0)+1
                if hasattr(self,'_event'):
                    self._event('meter_only_change',
                                incoming=incoming if meter_known else None)
            self.meter_was_known=meter_known
            changed=(new_queue!=self.deep_queue or incoming!=self.incoming_garbage or
                     meter_known!=previous_meter_known or pressure_risk_change is not None or
                     recovered_nxt!=current_nxt or recovered_hold!=current_hold)
            if new_queue!=self.deep_queue:
                self.deep_queue=new_queue;self.queue_revision+=1
                # If NEXT2 arrived after the initial Fast fallback but
                # BEFORE any input, recompute Fast once with the real queue.
                # The prior version relied on a late Deep proposal and often
                # executed the wrong first recipe move. Never cancel input.
                if (len(new_queue)>=2 and self.key is not None and
                        self.auto.enabled and self.auto.phase=='idle' and
                        self.auto.confirmed_locks==0 and
                        self.inputs.down is None and not getattr(self.inputs,'busy',False)):
                    options,_,_,eligible=self._native_opener_snapshot()
                    if eligible and (options[0] or options[1]):
                        from opener_commit import confirmed_reference_start
                        if confirmed_reference_start(board,piece,new_queue,
                                canonical=options[0],pco=options[1]):
                            if self.future is None:
                                self.submitted_key=None
                                self.current_plan=None
                            else:
                                self.opener_fast_recompute_pending=True
                            self._event('opener_queue_ready',piece=piece,queue=list(new_queue))
            self.incoming_garbage=incoming;self.garbage_confidence=confidence
            # The previous Fast plan may have been computed against the old
            # garbage risk: Deep receives updates, but Fast normally did not.
            # Refresh ONLY if the exact current turn is still idle, not armed
            # to send any key. Never touch input_ticket_epoch or a worker.
            self._refresh_idle_fast_pressure(board,piece,
                prior_incoming,prior_confidence,was_known,
                incoming,confidence,meter_known)
            combat,_=combat_profile(board,incoming)
            if hasattr(self,'pvp_status'):
                queue_text='→'.join(new_queue) if new_queue else 'NEXT 미인식'
                garbage_label = str(incoming) if self.garbage_filter.is_known() else '?'
                gtext = (f'incoming {garbage_label} · conf {confidence:.2f}'
                         if 'garbage' in self.cfg else 'incoming 미설정')
                self._set_combat_status(combat,f'PvP  {combat} · {gtext} · Deep {queue_text}')
                self._update_queue_text(piece,recovered_nxt,recovered_hold if recovered_hold!='?' else None)
            if changed:
                self._event('observation',queue=list(new_queue),meter_state='KNOWN' if meter_known else 'UNKNOWN',incoming=incoming)
                refined=list(self.key);refined[2]=recovered_nxt;refined[5]=recovered_hold
                self.schedule_deep(tuple(refined))
        finally:
            # Process even after a stale result or a failed capture. New turns
            # must not silently lose their one-shot queue telemetry.
            self._resume_roi_pending(board,piece)

    def schedule_deep(self,key):
        solver_mode=self._engine_mode()
        # An explicitly selected external engine is independent of the
        # internal offensive strategy dropdown. Previously BALANCED/SURVIVAL
        # made CC2 ONLY wait forever without ever submitting a CC2 request.
        if (not hasattr(self,'deep_pool') or getattr(self,'deep_disabled',False)
                or (solver_mode=='internal' and key[3] not in ('attack','combo'))
                or (key[2] is None and solver_mode=='internal')):return
        # Once an input target is committed, refining *that same turn* cannot
        # improve the command; preserve CPU for next-turn continuation.
        if (self.auto.enabled and self.auto.plan is not None and
                self.auto.phase in ('align','rotate_wait','drop_wait','hold_wait')):return
        # Every richer observation supersedes all older Deep results. The process
        # may finish old work, but stale results are discarded by revision.
        self.deep_revision+=1
        revision=self.deep_revision
        if hasattr(self,'deep_cancel_revision'):self.deep_cancel_revision.value=revision
        board,piece,nxt,mode,fast,hold,allow_hold,b2b,combo,serial=key
        queue=self.deep_queue if self.deep_queue and self.deep_queue[0]==nxt else (nxt,)
        incoming=conservative_incoming(self.incoming_garbage,
                 'garbage' in self.cfg,self.garbage_filter.is_known())
        request=(self.generation,key,revision,self.queue_revision,queue,incoming)
        if self.deep_future is not None:
            self.deep_pending=request
            return
        self._start_deep_request(request)

    def _native_opener_snapshot(self):
        """Capture immutable GUI settings before scheduling the Fast thread.

        Never read QWidgets in a background worker. No game process memory.
        Empty NEXT2 is *unknown*, not an invented next bag member.
        """
        auto=self.auto
        internal=self._engine_mode()=='internal'
        enabled=bool(internal and auto.enabled and auto.opening_enabled and
                     getattr(auto,'opening_eligible',None) is True and
                     not auto.garbage_rise_events and
                     not auto.unverified_board_resyncs and
                     not getattr(auto,'soft_disabled',False) and
                     getattr(auto,'recovery_blocked_target',None) is None and
                     auto.opening_reason not in ('garbage','resync','danger') and
                     getattr(self,'opening_enabled',None) is not None and
                     self.opening_enabled.isChecked() and
                     getattr(self,'adaptive_opener',None) is not None and
                     self.adaptive_opener.isChecked() and
                     not ('garbage' in self.cfg and not self.garbage_filter.is_known()))
        # Opening Burst is a *scoring campaign*, not an instruction to kill
        # a route-proven TKI/DT/PCO halfway through its physical sequence.
        # However a new template MUST NOT spontaneously start after the burst
        # cap. Only the exact, previously locked canonical family may finish.
        can_book=bool(enabled and getattr(self,'canonical_book',None) is not None and self.canonical_book.isChecked())
        can_pco=bool(enabled and getattr(self,'pco_finish',None) is not None and self.pco_finish.isChecked())
        if getattr(auto,'opening_finished',False):
            continued=False
            if getattr(auto,'opening_reason','') in ('target','timeout') and (can_book or can_pco):
                from opener_commit import confirmed_opener_continuation
                continued=confirmed_opener_continuation(
                    self.key[0] if self.key is not None else (),
                    self.key[1] if self.key is not None else None,
                    int(auto.confirmed_locks),int(getattr(auto,'opener_confirmed',0)),
                    getattr(auto,'opener_last',''),canonical=can_book,pco=can_pco)
            can_book=bool(can_book and continued)
            can_pco=bool(can_pco and continued)
        flags=(can_book,can_pco,
               bool(enabled and getattr(self,'stacker63',None) is not None and self.stacker63.isChecked()))
        q=tuple(self.deep_queue) if self.deep_queue else ()
        if self.key is not None and self.key[2] is not None and (not q or q[0]!=self.key[2]):
            q=(self.key[2],)
        return flags,q,int(auto.confirmed_locks),enabled

    def _engine_mode(self):
        """The only authoritative user-selected engine for the current session."""
        buttons=getattr(self,'solver_buttons',None)
        if isinstance(buttons,dict):
            for code,radio in buttons.items():
                if radio.isChecked():return code
            return 'internal'
        # Backward-compatible fake App instances in older tests; real GUI uses
        # QButtonGroup radios above, never these obsolete checkboxes.
        if getattr(self,'external_tbp',None) and self.external_tbp.isChecked():return 'hybrid'
        if getattr(self,'bundled_tbp',None) and self.bundled_tbp.isChecked():return 'bundled'
        return 'internal'

    def _tbp_mode(self):
        """Legacy external=True requests were hybrid, not fail-closed."""
        mode=self._engine_mode()
        if not hasattr(self,'solver_buttons') and mode=='hybrid':return True
        return mode

    def _start_deep_request(self,request):
        generation,key,revision,queue_revision,queue,incoming=request
        board,piece,nxt,mode,fast,hold,allow_hold,b2b,combo,serial=key
        combat,_=combat_profile(board,incoming)
        # Offensive setups need a longer *background* planning horizon than
        # the legacy 8-ms ATTACK slice.  This job lives in ProcessPoolExecutor
        # and never holds the GUI/input thread. Under fast PPS or heavy capture
        # cost, reduce its CPU footprint instead of starving vision/input.
        observed_pps=self.auto.pace(time.perf_counter()).get('pps') if self.auto.enabled else None
        budget=deep_budget_ms(combat, getattr(self,'capture_ms',0), observed_pps,
                mode=self._engine_mode(), received=getattr(self,'deep_received',0),
                adopted=getattr(self,'deep_applied',0),stale=getattr(self,'deep_stale',0))
        self.deep_budget_last_ms=budget
        if budget<{'ATTACK':28,'COUNTER':22,'DEFENSE':14,'EMERGENCY':14}[combat]:
            self.deep_budget_low_count=getattr(self,'deep_budget_low_count',0)+1
        elif budget>28:
            self.deep_budget_boost_count=getattr(self,'deep_budget_boost_count',0)+1
        # The opening campaign is a one-shot policy. Query only when scheduling
        # the asynchronous Deep job, never on the 8-ms fast calculation path.
        # Unknown is not 0. Still allow the user-enabled opener on a low,
        # visibly safe board: real empty meters can be visually featureless.
        # Never claim this is a verified zero or apply early cancellation on it.
        meter_unknown='garbage' in self.cfg and not self.garbage_filter.is_known()
        low_risk_unknown=(not any(board[:12]) and self.auto.garbage_rise_events==0
                          and incoming==0) if self.auto.enabled else False
        burst=((not meter_unknown or low_risk_unknown) and
               self.auto.opening_status(time.perf_counter(),incoming,board)
               if self.auto.enabled and hasattr(self.auto,'opening_status') else False)
        opening_remaining=max(0,getattr(self.auto,'opening_target',12)-
                                getattr(self.auto,'opening_generated',0))
        pieces_placed=self.auto.confirmed_locks if self.auto.enabled else 20
        sent_upper_bound=getattr(self.auto,'sent_attack_upper_bound',0) if self.auto.enabled else 0
        meter_confident=('garbage' in self.cfg and self.garbage_confidence>=.55)
        ruleset=self.ruleset.currentData() if hasattr(self,'ruleset') else 'standard'
        table=self.combo_table.currentData() if hasattr(self,'combo_table') else 'multiplier'
        rounding=self.rounding.currentData() if hasattr(self,'rounding') else 'auto'
        args=(board,piece,queue,hold,allow_hold,b2b,combo,incoming,budget,revision,
              queue_revision,pieces_placed,sent_upper_bound,meter_confident,ruleset,
              table,rounding,burst,opening_remaining,
              self.auto.garbage_mask if self.auto.enabled else None,
              bool(self.auto.enabled and getattr(self.auto,'opening_adaptive_enabled',False)
                   and self._engine_mode()=='internal' and not meter_unknown))
        # Extra strategy args go to the INTERNAL Deep planner only. CC2, HYBRID,
        # BUNDLED and their external protocol snapshots retain their old shape.
        experimental=(
            bool(self.auto.enabled and getattr(self,'adaptive_opener',None) is not None and self.adaptive_opener.isChecked() and getattr(self,'opening_enabled',None) is not None and self.opening_enabled.isChecked()
                 and getattr(self,'canonical_book',None) is not None and self.canonical_book.isChecked() and not meter_unknown
                 and not getattr(self.auto,'soft_disabled',False) and getattr(self.auto,'recovery_blocked_target',None) is None),
            bool(self.auto.enabled and getattr(self,'stacker63',None) is not None and self.stacker63.isChecked() and not meter_unknown
                 and not getattr(self.auto,'soft_disabled',False) and getattr(self.auto,'recovery_blocked_target',None) is None),
            bool(self.auto.enabled and getattr(self,'adaptive_opener',None) is not None and self.adaptive_opener.isChecked() and getattr(self,'opening_enabled',None) is not None and self.opening_enabled.isChecked()
                 and getattr(self,'pco_finish',None) is not None and self.pco_finish.isChecked() and not meter_unknown
                 and not getattr(self.auto,'soft_disabled',False) and getattr(self.auto,'recovery_blocked_target',None) is None),
        )
        context=(key,revision,queue_revision,queue,incoming)
        started=time.perf_counter()
        try:
            if self._engine_mode()=='cc2':
                self.deep_future=self.deep_pool.submit(plan_move_cc2_diagnostic_worker,args)
            elif self._engine_mode()=='hybrid':
                self.deep_future=self.deep_pool.submit(plan_move_hybrid_diagnostic_worker,args)
            else:
                native_args=args+experimental if self._engine_mode()=='internal' else args
                self.deep_future=self.deep_pool.submit(plan_move_pvp_worker,native_args,self._tbp_mode())
        except (RuntimeError, OSError, ValueError) as exc:
            self.deep_future=None
            self.deep_pending=None
            self._deep_worker_failure(str(exc))
            return
        def done(f,g=generation,k=context,t=started):
            try:self.deep_ready.emit(g,k,f.result(),(time.perf_counter()-t)*1000,None)
            except Exception as e:
                try:self.deep_ready.emit(g,k,None,0,str(e))
                except RuntimeError:pass
        self.deep_future.add_done_callback(done)

    def _deep_worker_failure(self,error):
        """Keep Fast/autoplay alive if the optional ProcessPool becomes broken."""
        self.deep_failures=getattr(self,'deep_failures',0)+1
        self._event('deep_failure',reason=str(error)[:150])
        if self.deep_failures>=3:
            self.deep_disabled=True
            if hasattr(self,'pvp_status'):
                self.pvp_status.setText('DEEP 오류로 비활성화 · 엔진 상태 확인 필요')

    @staticmethod
    def _same_target(a,b):
        return bool(a and b and a.use_hold==b.use_hold and a.move.piece==b.move.piece and
                    tuple(sorted(a.move.cells))==tuple(sorted(b.move.cells)) and a.move.spin==b.move.spin)

    def accept_deep_result(self,g,context,plan,elapsed,error):
        # Distinguish no engine reply, unsafe route, and missing observations;
        # a failed CC2 decision must never silently invoke an internal move.
        cc2_reason=None
        cc2_details={}
        if isinstance(plan,HybridWorkerResult):
            hybrid=plan
            cc2_reason=hybrid.cc2_status
            cc2_details=hybrid.cc2_details
            plan=hybrid.plan
            if self.running and g==self.generation:
                if cc2_reason=='STALE_REQUEST' or context[1]!=self.deep_revision:
                    self.cc2_hybrid_stale=getattr(self,'cc2_hybrid_stale',0)+1
                else:
                    self.cc2_hybrid_attempts=getattr(self,'cc2_hybrid_attempts',0)+1
                    self.cc2_hybrid_passes=getattr(self,'cc2_hybrid_passes',0)+(cc2_reason=='PASS')
                    self.cc2_hybrid_fallbacks=getattr(self,'cc2_hybrid_fallbacks',0)+bool(hybrid.native_fallback)
                    reasons=getattr(self,'cc2_hybrid_reasons',{})
                    reasons[cc2_reason]=reasons.get(cc2_reason,0)+1
                    self.cc2_hybrid_reasons=reasons
                    self.cc2_last_reason=cc2_reason
                    self.cc2_last_elapsed_ms=round(hybrid.cc2_elapsed_ms,1)
                    self.cc2_hybrid_last_probe_ms=round(hybrid.cc2_elapsed_ms,1)
                    self._event('hybrid_probe',status=cc2_reason,
                                native_fallback=hybrid.native_fallback,
                                probe_ms=round(hybrid.cc2_elapsed_ms,1),
                                rejected=cc2_details)
        if isinstance(plan,CC2WorkerResult):
            cc2_reason=plan.status
            cc2_details=plan.details
            plan=plan.plan
            if self.running and g==self.generation:
                self.cc2_last_reason=cc2_reason
                self.cc2_last_elapsed_ms=round(elapsed,1)
                if hasattr(self,'_event'):
                    self._event('cc2_attempt',result=cc2_reason,elapsed_ms=round(elapsed,1),
                                rejection_counts=cc2_details)
        self.deep_future=None
        pending=self.deep_pending if hasattr(self,'deep_pending') else None
        if hasattr(self,'deep_pending'):self.deep_pending=None
        key,revision,queue_revision,queue,incoming=context
        if error:
            self._deep_worker_failure(error)
            if (self._engine_mode()=='cc2' and self.running and g==self.generation
                    and revision==self.deep_revision and queue_revision==self.queue_revision):
                self.cc2_last_reason='WORKER_ERROR'
                self.stop_auto('CC2 단독: 프로세스 오류 · 내부 엔진으로 전환하지 않음')
        elif plan is not None:self.deep_failures=0
        if (not error and plan is not None and revision==self.deep_revision and self.running and g==self.generation):
            self.deep_received=getattr(self,'deep_received',0)+1
        elif revision!=self.deep_revision:
            self.deep_stale=getattr(self,'deep_stale',0)+1
        if (not error and plan is None and revision!=self.deep_revision and
                hasattr(self,'deep_cancelled_count')):
            self.deep_cancelled_count+=1
        valid=(not error and plan is not None and self.running and g==self.generation and
               revision==self.deep_revision and queue_revision==self.queue_revision and
               self.key is not None and
               (key[0],key[1],key[-1])==(self.key[0],self.key[1],self.key[-1]))
        exclusive=self._engine_mode()=='cc2'
        if exclusive and valid and 'garbage' in self.cfg and not self.garbage_filter.is_known():
            valid=False  # fail closed: do not adopt an unknown-pressure plan
            self.cc2_last_reason='GARBAGE_UNKNOWN'
            self._event('cc2_rejected',reason='GARBAGE_UNKNOWN')
        if exclusive and valid and plan.continuation_source!='external_tbp':
            valid=False  # no foreign/native plan may enter CC2-only
        is_external=bool(valid and plan.continuation_source=='external_tbp')
        is_bundled=bool(valid and plan.continuation_source=='bundled_tbp')
        selected_external=False
        selected_internal=False
        if (exclusive and not error and plan is None and self.running and
                g==self.generation and revision==self.deep_revision and
                queue_revision==self.queue_revision and self.key is not None and
                (key[0],key[1],key[-1])==(self.key[0],self.key[1],self.key[-1])):
            self.cc2_no_plan+=1
            self._event('cc2_no_valid_plan',piece=key[1],reason=cc2_reason or 'UNKNOWN',
                        rejection_counts=cc2_details)
            # A first empty/slow response does not prove this board cannot be
            # played. Retry a maximum of two times on the SAME verified turn;
            # hard protocol/route/observation errors remain fail-closed.
            recoverable=cc2_reason in ('NO_REPLY','NO_SUGGESTION')
            attempt=getattr(self,'cc2_retry_count',0)
            age=time.perf_counter()-getattr(self,'detected_at',0.0)
            if (recoverable and attempt<2 and 0<=age<1.5 and
                    self.auto.phase=='idle' and not self.auto.paused):
                self.cc2_retry_count=attempt+1
                self.cc2_retry_at=time.perf_counter()+.08
                self.cc2_last_reason='RETRY_PENDING:'+cc2_reason
                self._event('cc2_retry_pending',attempt=self.cc2_retry_count,reason=cc2_reason)
            else:
                self.cc2_retry_at=0.0
                reason_text={
                'HOLD_UNKNOWN':'HOLD 상태 미인식 · HOLD 영역을 인식하거나 실제 비어 있다면 HOLD 비어 있음 선택',
                'INCOMING_TOO_HIGH':'Garbage 4줄 이상 · CC2 안전 경로 사용 제한',
                'EXECUTABLE_MISSING':'Cold Clear 2 실행파일이 현재 실행 위치에 없음',
                'INVALID_OBSERVATION':'CURRENT/NEXT/BOARD 관측 정보 불완전',
                'NO_REPLY':'프로토콜 응답 없음 · coldclear_check.py 검증 필요',
                'NO_SUGGESTION':'검색시간 내 추천 없음 · coldclear_check.py 검증 필요',
                'NO_VALID_ROUTE':'추천은 수신했으나 SRS 이동경로 또는 보드 안전성 검증에 실패',
                }.get((cc2_reason or ''),'CC2 계획 생성 실패 · 진단 ZIP에서 cc2_last_reason 확인')
                self.stop_auto('CC2 단독 안전 중단: '+reason_text+' ['+(cc2_reason or 'UNKNOWN')+'] · 내부 엔진 미사용')
        if is_external:
            self.external_received+=1
            if exclusive:self.cc2_retry_at=0.0
        if is_bundled:self.bundled_received+=1
        if valid:
            current=self.current_plan
            if current is None and exclusive and is_external:
                known_now=(('garbage' not in self.cfg) or self.garbage_filter.is_known())
                pressure_now=self.incoming_garbage if known_now else 3
                idle=(not self.auto.enabled or (self.auto.phase=='idle' and
                      self.inputs.down is None and not getattr(self.inputs,'busy',False)
                      and not self.auto.paused))
                if (idle and known_now and pressure_now==incoming and
                        exclusive_cc2_safe(None,plan,pressure_now)):
                    self.current_plan=plan
                    self.preview.move=plan.move;self.preview.update()
                    self.overlay.cells=plan.move.cells;self.overlay.update()
                    if hasattr(self,'status'):self.status.setText('CC2 실제 선택 · 독립 계획 승인')
                    if hasattr(self,'show_overlay') and self.show_overlay.isChecked():self.overlay.show()
                    self.external_selected+=1;self.deep_applied+=1
                    selected_external=True
                    self._event('deep_selected',piece=plan.move.piece,lines=plan.move.lines,
                                spin=plan.move.spin,source='external_tbp')
            if current is not None:
                idle=(not self.auto.enabled or (self.auto.phase=='idle' and self.inputs.down is None and not getattr(self.inputs,'busy',False)))
                # In autoplay, ATTACK/COUNTER Deep is advisory only: it may enrich
                # the prepared next move but must not replace the current Fast
                # target with a longer route and reduce PPS. Survival states may
                # replace the current target before any input starts.
                known_now=(('garbage' not in self.cfg) or self.garbage_filter.is_known())
                pressure_now=(self.incoming_garbage if known_now else 3)
                can_replace=(not self.auto.enabled or (known_now and not self.auto.paused and
                    pressure_now==incoming and
                    (exclusive_cc2_safe(current,plan,pressure_now) if exclusive else
                     (is_external and self._engine_mode()=='hybrid' and pressure_now==0
                      and exclusive_cc2_safe(current,plan,pressure_now)) or
                     deep_adoption_allowed(current,plan,self.auto.confirmed_locks,pressure_now,
                     self.auto.ruleset,getattr(self.auto,'combo_table','multiplier'),
                     getattr(self.auto,'rounding','auto')))))
                if idle and can_replace and (exclusive or not self._same_target(current,plan)):
                    self.deep_applied=getattr(self,'deep_applied',0)+1
                    selected_internal=True
                    if plan.continuation_source=='hold_sequence':
                        self.hold_sequence_applied=getattr(self,'hold_sequence_applied',0)+1
                    if getattr(plan,'opening_profile',''):
                        self.opener_deep_selected=getattr(self,'opener_deep_selected',0)+1
                    if is_external:
                        self.external_selected+=1
                        selected_external=True
                    if is_bundled:
                        self.bundled_selected+=1
                        selected_external=True
                    self._event('deep_selected',piece=plan.move.piece,lines=plan.move.lines,spin=plan.move.spin,source=plan.continuation_source)
                    self.current_plan=plan;self.preview.move=plan.move;self.preview.update()
                    self.overlay.cells=plan.move.cells;self.overlay.update();current=plan
                    if exclusive and hasattr(self,'status'):
                        self.status.setText(f'CC2 실제 선택 · {plan.move.piece} · {plan.move.lines}줄 삭제 예상 · 계산 {elapsed:.1f}ms')
                        if hasattr(self,'show_overlay') and self.show_overlay.isChecked():self.overlay.show()
                active=self.auto.plan if self.auto.enabled else current
                if self._same_target(active,plan):
                    active.move.next_move=plan.move.next_move
                    active.prepared_safe=plan.prepared_safe
                    active.observed_next=plan.observed_next;active.observed_hold=plan.observed_hold
                    active.move.score=plan.move.score
                    # A matching Fast target can inherit a verified opener
                    # only BEFORE execution and under identical pressure.
                    # Never rewrite in-flight plan provenance after keys start.
                    if (idle and can_replace and known_now and pressure_now==incoming
                            and not getattr(self.auto,'paused',False)
                            and getattr(plan,'opening_profile','')):
                        active.opening_profile=plan.opening_profile
                        active.opening_verified_steps=plan.opening_verified_steps
            combat=plan.combat_mode; self.last_deep_ms=elapsed; self._update_latency_text()
            if hasattr(self,'pvp_status'):
                q='→'.join(queue) if queue else 'NEXT1'
                source=('COUNTER-BEAM' if plan.continuation_source=='counter_beam'
                        else 'HOLD-SEQ' if plan.continuation_source=='hold_sequence'
                        else 'BEAM' if plan.continuation_source=='offensive_beam' else 'COLD CLEAR 2' if plan.continuation_source=='external_tbp' else 'BUNDLED TBP' if plan.continuation_source=='bundled_tbp' else 'DEEP')
                profile=getattr(plan,'opening_profile','')
                if profile and not getattr(plan,'use_hold',False):
                    source+=' / OPENER '+profile.replace('VERIFIED_','')
                self._set_combat_status(combat,f'PvP  {combat} · incoming {incoming} · {source} {q} · {elapsed:.1f}ms')
        if valid and not selected_external and not selected_internal and plan is not None and not exclusive:
            # A calculated result and a physical lock are separate outcomes.
            self._event('deep_decision_not_selected',
                        source=getattr(plan,'continuation_source',None),
                        phase=getattr(self.auto,'phase','none'),
                        idle=bool(not self.auto.enabled or self.auto.phase=='idle'))
        if is_bundled and not selected_external:self.bundled_unadopted+=1
        if is_external and not selected_external:
            # A calculated suggestion is NOT proof that it was executed.
            self.external_unadopted+=1
            self._event('external_tbp_not_adopted',phase=getattr(self.auto,'phase','none'),
                        reason=('GARBAGE_UNKNOWN' if exclusive and 'garbage' in self.cfg and
                                not self.garbage_filter.is_known() else
                                'NOT_IDLE_OR_GUARD_OR_SAME_TARGET'))
        # Never delay the current key: once the Fast move has been committed,
        # calculate the *observed next piece* against its predicted landing.
        # The result is admitted only after an exact post-lock board/piece match.
        if valid and self.auto.enabled and self.auto.plan is not None and self.current_plan is not None:
            self._schedule_committed_continuation(queue,revision,queue_revision)
        self._update_deep_stats()
        if pending is not None:
            pg,pk,pr,pqr,pq,pinc=pending
            if (pg==self.generation and self.running and self.key is not None and
                    (pk[0],pk[1],pk[-1])==(self.key[0],self.key[1],self.key[-1])):
                # pending already owns the newest revision; don't increment again
                self._start_deep_request(pending)
    def _schedule_committed_continuation(self,queue,revision,queue_revision):
        """Prepare next spawn on spare CPU, never on the GUI or Fast path.

        The current controller's locked target is authoritative, not an
        unexecuted Deep alternative. No speculative HOLD/unknown NEXT branch.
        """
        if (getattr(self,'deep_disabled',False) or not hasattr(self,'continuation_future') or self.continuation_future is not None
                or not queue or self.key is None or not self.auto.enabled
                or self.auto.phase=='idle' or self.auto.plan is None):
            return
        current=self.auto.plan
        if (current.source_board,current.source_piece)!=(self.key[0],self.key[1]):return
        if current.use_hold:return  # HOLD swaps can consume NEXT1 or require acknowledgement
        token=(self.generation,self.auto.serial,self.key,queue_revision)
        if token==getattr(self,'continuation_token',None):return
        if current.move.next_move is not None and current.move.next_move.spin:
            return  # never override a verified tactical continuation blindly
        # A committed block's command cannot be changed by the current Deep
        # job. Bump the cooperative revision so the shared worker can yield
        # quickly to the deadline-sensitive NEXT continuation.
        if getattr(self,'deep_future',None) is not None:
            self.deep_revision+=1
            revision=self.deep_revision
            if hasattr(self,'deep_cancel_revision'):
                self.deep_cancel_revision.value=revision
            if hasattr(self,'deep_pending'):self.deep_pending=None
        from combo_rules import attack_estimate
        rule=self.auto.ruleset;table=self.auto.combo_table;rounding=self.auto.rounding
        estimate=attack_estimate(current.move,self._planning_b2b(),self.auto.combo,rule,table,rounding)
        # NEXT1 was observed on the committed source turn. Rejected results
        # are not automatically requeued; the next actual state rechecks all.
        next_piece=queue[0]
        predicted=current.move.board
        # The NEXT prediction is only actionable if the confirmed post-lock
        # board/piece matches it. A known current HOLD survives a no-HOLD lock;
        # unknown HOLD never enters external TBP (the protocol cannot encode it).
        observed_held=self.auto.held
        # Bundled local adviser can precompute a NO-HOLD continuation even
        # when HOLD pixels are UNKNOWN. A speculative HOLD is still forbidden;
        # Cold Clear 2 retains the original explicit HOLD evidence gates.
        mode=self._tbp_mode()
        confirmed_hold=bool(observed_held!='?' and
                            getattr(self,'hold_enabled',None) and
                            self.hold_enabled.isChecked())
        external_prefetch=bool(mode=='bundled' or (mode in (True,'cc2','hybrid') and confirmed_hold))
        future_can_hold=bool(mode in (True,'cc2','hybrid') and confirmed_hold)
        args=(predicted,next_piece,tuple(queue[1:]),
              observed_held if future_can_hold else '?',future_can_hold,
              estimate.next_b2b,estimate.next_combo,0,8.0,revision,queue_revision,
              self.auto.confirmed_locks+1,getattr(self.auto,'sent_attack_upper_bound',0),
              False,rule,table,rounding,False,0,None)
        context=(self.generation,self.auto.serial,revision,queue_revision,
                 predicted,next_piece,current,self.key)
        # Optional speculation must never pause or abort live autoplay if a
        # worker crashes, is shutting down, or cannot accept more work.
        try:
            future=(self.deep_pool.submit(plan_move_pvp_worker,args,self._tbp_mode()) if external_prefetch
                    else self.deep_pool.submit(plan_move_pvp_worker,args))
        except (RuntimeError, OSError, ValueError) as exc:
            self._deep_worker_failure(str(exc))
            return
        self.continuation_token=token
        self.continuation_requested+=1
        self._event('next_requested',piece=next_piece,revision=revision)
        self.continuation_future=future
        def done(f,c=context):
            try:self.continuation_ready.emit(c[0],c,f.result(),None)
            except Exception as exc:
                try:self.continuation_ready.emit(c[0],c,None,str(exc))
                except RuntimeError:pass
        self.continuation_future.add_done_callback(done)

    def accept_continuation_result(self,g,context,result,error):
        self.continuation_future=None
        if error or result is None or not self.running or not self.auto.enabled:
            if error:self._deep_worker_failure(error)
            self.continuation_stale=getattr(self,'continuation_stale',0)+1
            self._update_deep_stats()
            return
        generation,serial,revision,queue_revision,board,piece,source_plan,key=context
        external_next=result.continuation_source=='external_tbp'
        bundled_next=result.continuation_source=='bundled_tbp'
        if external_next:self.external_prefetch_received+=1
        if bundled_next:self.bundled_prefetch_received+=1
        if self._engine_mode()=='cc2' and not external_next:
            self.continuation_stale+=1;self._update_deep_stats();return
        if (g!=self.generation or serial!=self.auto.serial
                or revision!=self.deep_revision or queue_revision!=self.queue_revision
                or self.key!=key):
            self.continuation_stale=getattr(self,'continuation_stale',0)+1
            self._update_deep_stats()
            return
        committed=self.auto.plan
        if (committed is None or committed is not source_plan
                or committed.move.board!=board or committed.move.piece not in ('I','O','T','J','L','S','Z')
                or (result.source_board,result.source_piece)!=(board,piece)
                or result.use_hold or result.move is None):
            self.continuation_stale=getattr(self,'continuation_stale',0)+1
            self._update_deep_stats()
            return
        # Verify the proposed route in the background worker, not here.
        # The live controller will recheck the final projected landing.
        # The Deep proposal must not replace a safer/faster proven Fast next
        # move just because it arrived later. Also reject under known pressure.
        if ('garbage' in getattr(self,'cfg',{}) and
                self.garbage_filter.is_known() and self.incoming_garbage>=3):
            self.continuation_stale=getattr(self,'continuation_stale',0)+1
            self._event('next_rejected',reason='incoming')
            self._update_deep_stats()
            return
        from engine import continuation_beneficial
        improves=continuation_beneficial(committed.move.next_move,result.move,
                                         board,self._planning_b2b(),self.auto.combo,
                                         self.auto.ruleset,self.auto.combo_table,self.auto.rounding)
        if not improves and result.continuation_source in ('offensive_beam','hold_sequence'):
            # Allow a route-verified, known NEXT offensive builder into the
            # existing exact-state prefetch only when it improves clean stack
            # structure without sacrificing generated attack. Never override
            # a current move in progress or relax pressure/vision guards.
            from offensive_beam import beam_continuation_beneficial
            improves=beam_continuation_beneficial(
                committed.move.next_move,result.move,board,
                self._planning_b2b(),self.auto.combo,self.incoming_garbage,
                self.auto.ruleset,self.auto.combo_table,self.auto.rounding)
        # CC2-only does not compare its validated next target to Fast's score.
        # The predicted board and true next piece are still matched above.
        if self._engine_mode()=='cc2' and external_next:
            improves=True
        if not improves:
            self.continuation_stale=getattr(self,'continuation_stale',0)+1
            self._event('next_rejected',reason='risk_or_no_gain')
            self._update_deep_stats()
            return
        committed.move.next_move=result.move
        # Never relabel this in-flight piece as if CC2/Bundled drove it. The
        # speculative NEXT source is assigned only to the next piece after
        # its independently confirmed post-lock board/piece match.
        committed.prefetch_source=result.continuation_source
        committed.prepared_safe=True
        committed.observed_next=piece
        self.continuation_applied+=1
        if external_next:self.external_prefetch_applied+=1
        if bundled_next:self.bundled_prefetch_applied+=1
        self._event('next_attached',piece=piece,lines=result.move.lines,spin=result.move.spin,
                    source=result.continuation_source)
        self._update_deep_stats()

    def accept_result(self,g,key,plan,elapsed,error):
        # Runs on the Qt GUI thread as soon as the worker completes.
        self.future=None
        if not self.running or g!=self.generation or key!=self.key: return
        if getattr(self,'pressure_fast_recompute_pending',False):
            if (self.auto.enabled and self.auto.phase=='idle' and
                    self.inputs.down is None and not getattr(self.inputs,'busy',False)):
                self.pressure_fast_recompute_pending=False
                self.current_plan=None
                self.submitted_key=None
                if hasattr(self,'_event'):
                    self._event('pressure_fast_result_discarded')
                return  # next current-frame tick plans with new meter risk
        if getattr(self,'opener_fast_recompute_pending',False):
            self.opener_fast_recompute_pending=False
            if (self.auto.enabled and self.auto.phase=='idle' and
                    self.inputs.down is None and not getattr(self.inputs,'busy',False)):
                self.current_plan=None
                self.submitted_key=None
                self._event('opener_fast_recomputed',piece=key[1])
                return  # next tick computes from confirmed NEXT2
        if error:
            if self._engine_mode()=='cc2':
                # Fast is ONLY a visual preview in exclusive mode. A Fast
                # failure must not veto a separately verifiable CC2 action.
                self.current_plan=None
                self._event('cc2_fast_preview_error',reason=str(error)[:100])
                self.status.setText('내장 미리보기 오류 · CC2 별도 계산 확인 중')
                self.schedule_deep(key)
                return
            self.pause_auto('계산 재시도'); self.invalidate(); self.retry_after=time.perf_counter()+.5; self.status.setText('계산 오류: '+error); return
        self.current_plan=plan; move=plan.move if plan else None
        if plan is not None and getattr(plan,'opening_profile',''):
            self.opener_fast_selected=getattr(self,'opener_fast_selected',0)+1
            self._event('opener_fast_selected',profile=plan.opening_profile,
                        stage=getattr(self.auto,'confirmed_locks',None),
                        source=getattr(plan,'continuation_source','fast'))
        if self._engine_mode()=='cc2' and plan is not None:
            self._event('cc2_fast_preview_only',piece=key[1])
        exclusive_preview=self._engine_mode()=='cc2'
        self.preview.move=(None if exclusive_preview else move); self.preview.update()
        if move is None:
            if self.auto.enabled and self._engine_mode()!='cc2':self.pause_auto('유효한 배치/보드 대기')
            self.overlay.hide(); self.status.setText('내장 빠른 탐색 배치 없음 · CC2 확인 중' if self._engine_mode()=='cc2' else '탐색 범위 안에 가능한 배치가 없습니다.')
            if self._engine_mode()=='cc2':self.schedule_deep(key)
            return
        board,piece,nxt,mode,fast,hold,allow_hold,b2b,combo,serial=key
        delivery=(time.perf_counter()-self.detected_at)*1000
        self.last_fast_ms=elapsed; self.last_delivery_ms=delivery; self._update_latency_text(); self._update_queue_text(piece,nxt,hold if hold!='?' else None)
        if exclusive_preview:
            self.status.setText(f'CC2 단독 모드 · {piece} · Cold Clear 2 계획 검증 중 (내장 Fast 결과 실행 안 함)')
            self.detail.setText('CC2 전용 · 내부 엔진은 미리 계산만 하며 입력에 사용하지 않습니다.')
        else:
            spin_label=(f'{move.piece}-SPIN {move.spin.upper()} · B2B 후보' if move.spin and move.lines
                        else f'{move.piece}-SPIN {move.spin.upper()} (무삭제)' if move.spin else '')
            self.status.setText(f'현재 {piece} / NEXT {nxt or "미인식"} · {move.lines}줄 삭제 예상 {spin_label}\nFast {elapsed:.1f}ms · 감지→결과 {delivery:.1f}ms')
            self.detail.setText(('HOLD → '+move.piece+' 배치\n' if plan.use_hold else '현재 블록 배치\n')+(f'다음 배치: {move.next_move.lines}줄 삭제 예상' if move.next_move else '추가 다음 배치 없음 · 현재 배치 사용'))
        self.overlay.cells=(() if exclusive_preview else move.cells)
        region=self.cfg['board']
        sx=self.screen.geometry().width()/self.monitor['width']; sy=self.screen.geometry().height()/self.monitor['height']
        self.overlay.setGeometry(round((region['left']-self.monitor['left'])*sx)+self.screen.geometry().x(),round((region['top']-self.monitor['top'])*sy)+self.screen.geometry().y(),round(region['width']*sx),round(region['height']*sy))
        self.overlay.update()
        if exclusive_preview:self.overlay.hide()
        elif self.show_overlay.isChecked():self.overlay.show()
        if not getattr(self.auto,'soft_disabled',False) or self._engine_mode()=='cc2':self.schedule_deep(key)
    def tick(self):
        try:
            self.inputs.poll()
            if self.auto.enabled and not self.inputs.focused(): self.pause_auto('게임 창으로 돌아오면 재개')
        except Exception as e:
            self.pause_auto(str(e)); self.retry_after=time.perf_counter()+.5
        if sys.platform=='win32':
            down=bool(ctypes.windll.user32.GetAsyncKeyState(0x76)&0x8000)
            if down and not self.f7: self.toggle()
            self.f7=down
            if (hasattr(self,'color_reader') and self.color_reader.recalibrating
                    and ctypes.windll.user32.GetAsyncKeyState(0x1B)&0x8000):
                self.toggle()  # ESC cancels staged capture and restores GUI.
                return
            down9=bool(ctypes.windll.user32.GetAsyncKeyState(0x78)&0x8000)
            if down9 and not self.f9:
                try: self.toggle_auto()
                except Exception as e: self.stop_auto(str(e))
            self.f9=down9
            if self.auto.enabled and ctypes.windll.user32.GetAsyncKeyState(0x1B)&0x8000: self.stop_auto('ESC')
        if not self.running: return
        if self.auto.enabled and not self.inputs.focused():
            self._record_control_gap(time.perf_counter(),False)
            return
        if time.perf_counter()<self.retry_after:return
        try:
            started=time.perf_counter()
            region=self.cfg['board']
            a,spawn_frame=capture_board_and_spawn(self.cap,region,self._runtime_roi('spawn'))
            if hasattr(self,'color_reader') and self.color_reader.recalibrating:
                progress=self.color_reader.collect_recalibration(a,started)
                self.preview.active=[]
                if progress.status in ('success','failed'):
                    if progress.status=='success':
                        # Preserve one actual game-frame snapshot for diagnosis,
                        # without continually capturing our own newly shown GUI.
                        proof=self.color_reader.read(a,mode='strict',color_hints=True)
                        if proof.grid is not None:
                            self.preview.grid=proof.grid
                            self.preview.colors=proof.colors
                            self.preview.ghosts=proof.ghosts
                            self.preview.unknown=proof.unknown
                            self.last_grid=proof.grid
                            self.last_frame=a.copy()
                            self.preview.update()
                    self._finish_color_calibration(progress)
                elif started >= getattr(self,'vision_status_next_at',0):
                    self.vision_status.setText('VISION CALIBRATING · '+progress.reason)
                    self.vision_status_next_at=started+.2
                self.capture_ms=(time.perf_counter()-started)*1000
                return
            # Legacy test harnesses construct App without invoking __init__.
            # Keep the historical recognition behavior in that narrow case.
            if not hasattr(self,'color_reader'):self.color_reader=ColorBoardVision()
            mode=self.vision_mode.currentData() if hasattr(self,'vision_mode') else 'legacy'
            spawn_rows=spawn_pair_rows(region,self.cfg['spawn']) if spawn_frame is not None else None
            spawn_bg=self.color_reader.background
            spawn_proof=False
            if spawn_rows is not None and spawn_bg is None and mode!='legacy':
                # A fully visible colored tetromino in the separate SPAWN
                # validates an otherwise blank playfield. This lets initial
                # color calibration complete BEFORE the first block locks.
                try:
                    candidate=self.color_reader._candidate(a)
                    if candidate is not None and candidate[1]>=.90:
                        spawn_bg=candidate[0]
                        proof=fuse_spawn_active(spawn_frame,[[False]*10 for _ in range(20)],
                                                spawn_bg,spawn_rows)
                        spawn_proof=(proof is not None)
                except (ValueError,TypeError,IndexError):
                    spawn_proof=False
            # With a true BOARD+SPAWN split, the game's vertical white I
            # landing projection is no longer clipped by the old 22-row ROI.
            # It can falsely look like four FULL solid minos in BOARD. Mask it
            # only when we have a separately observed colored spawn-I, a
            # precise white 4-cell guide, and an independently confirmed
            # locked-board reference; then re-check the entire resulting grid.
            read_frame=a
            if (spawn_rows is not None and mode!='legacy' and
                    self.color_reader.background is not None):
                references=[]
                if self.tracker.previous is not None:
                    references.append(self.tracker.previous[0])
                if self.auto.enabled and self.auto.plan is not None and self.auto.phase=='drop_wait':
                    references.append(self.auto.plan.move.board)
                if (self.auto.enabled and self.auto.confirmed_locks==0 and
                        self.tracker.previous is None):
                    references.append((0,)*20)
                for locked in references:
                    masked=mask_solid_spawn_projection(
                        a,spawn_frame,self.color_reader.background,spawn_rows,locked)
                    if masked is None:continue
                    check=self.color_reader.read(masked,mode='strict',
                        threshold=self.threshold.value(),color_hints=False,
                        allow_legacy_fallback=False,now=started)
                    if check.grid is None:continue
                    found=fuse_spawn_active(spawn_frame,check.grid,
                        self.color_reader.background,spawn_rows,locked)
                    if found is not None and found[0]==locked and found[1]=='I':
                        read_frame=masked
                        if hasattr(self,'_event') and started-getattr(self,'white_guide_log_at',-1e12)>=.5:
                            self._event('spawn_white_guide_solid',rows=spawn_rows)
                            self.white_guide_log_at=started
                        break
            vision=self.color_reader.read(
                read_frame, mode=mode, threshold=self.threshold.value(),
                color_hints=started>=getattr(self,'preview_next_at',0),
                allow_legacy_fallback=not self.auto.enabled,
                initial_confirmations=3, spawn_evidence=spawn_proof, now=started)
            grid=vision.grid
            # During a verified, nearly-full stack the usual 12-cell empty
            # background requirement can be physically impossible. Do NOT
            # globally relax it. Retry with >=4 proven empty interiors only
            # when the whole newly classified grid independently contains a
            # four-cell active piece against an existing exact board (or a
            # strictly verified garbage-shifted board). Otherwise fail closed.
            if (grid is None and vision.method=='UNKNOWN'
                    and vision.reason.startswith('배경 검증 부족')
                    and self.color_reader.background is not None):
                references=[]
                if self.auto.enabled and self.auto.plan is not None:
                    if self.auto.phase=='drop_wait':
                        references.append((self.auto.plan.move.board,None))
                    references.append((self.auto.plan.source_board,
                                       self.auto.plan.move.piece if self.auto.phase!='drop_wait' else None))
                if self.tracker.previous is not None:
                    references.append(self.tracker.previous)
                if references:
                    trial=self.color_reader.read(read_frame,mode='strict',
                        threshold=self.threshold.value(),color_hints=False,
                        allow_legacy_fallback=False,reference_floor=4,now=started)
                    if trial.grid is not None and proven_active_reference(trial.grid,references):
                        vision=trial;grid=trial.grid
                        if hasattr(self,'_event'):
                            self._event('high_stack_vision_verified')
            # A live failure capture revealed a pure-white 4-row placement
            # guide in the same column as an I landing, not ordinary colored
            # minos. Recover ONLY that exact, independently verified pattern.
            # Require complete consistency with the LAST CONFIRMED locked
            # board and the independently visible current I identity. The
            # original image remains untouched for any failure diagnostics.
            if (grid is None and mode!='legacy' and vision.unknown is not None
                    and self.auto.enabled and self.auto.phase=='drop_wait'
                    and hasattr(self.tracker,'previous') and self.tracker.previous is not None):
                candidate=suppress_confirmed_white_i_projection(
                    a,self.color_reader.background,vision.unknown,self.tracker.previous)
                if candidate is not None:
                    proposed=self.color_reader.read(
                        candidate,mode='strict',threshold=self.threshold.value(),
                        color_hints=started>=getattr(self,'preview_next_at',0),
                        allow_legacy_fallback=False,now=started)
                    if proposed.grid is not None:
                        from vision import locate_piece
                        verified=locate_piece(proposed.grid,self.tracker.previous)
                        if (verified is not None and verified[1]=='I' and
                                verified[0]==self.tracker.previous[0]):
                            vision=proposed
                            grid=vision.grid
                            if (hasattr(self,'_event') and
                                    started-getattr(self,'white_guide_log_at',-1e12) >= .5):
                                self._event('white_guide_suppressed',confirmed='I')
                                self.white_guide_log_at=started
            # With a separately calibrated SPAWN, the CURRENT I may be above
            # the logical 20-row BOARD. Preserve the same guarded white-guide
            # recovery without assuming that I lies inside the board crop.
            if (grid is None and spawn_rows is not None and mode!='legacy'
                    and vision.unknown is not None and self.auto.enabled
                    and self.auto.phase=='drop_wait'):
                refs=[]
                if self.auto.plan is not None:
                    refs.append(self.auto.plan.move.board)
                if self.tracker.previous is not None:
                    refs.append(self.tracker.previous[0])
                for locked in refs:
                    masked=mask_confirmed_spawn_white_guide(
                        a,spawn_frame,self.color_reader.background,vision.unknown,
                        locked,spawn_rows)
                    if masked is None:continue
                    proposed=self.color_reader.read(
                        masked,mode='strict',threshold=self.threshold.value(),
                        color_hints=started>=getattr(self,'preview_next_at',0),
                        allow_legacy_fallback=False,now=started)
                    if proposed.grid is not None:
                        # Real settled cells must be fully consistent with the
                        # locked board; never modify them based on the overlay.
                        evidence=fuse_spawn_active(
                            spawn_frame,proposed.grid,self.color_reader.background,
                            spawn_rows,locked)
                        if evidence is not None and evidence[0]==locked and evidence[1]=='I':
                            vision=proposed;grid=vision.grid
                            if hasattr(self,'_event'):
                                self._event('spawn_white_guide_verified',rows=spawn_rows)
                            break
            if (grid is None and mode=='auto' and self.auto.enabled and
                    vision.method=='UNKNOWN' and self.color_reader.background is not None):
                # v0.9.8 brightness path, but with a strong visual proof that
                # the entire fresh frame matches an independently confirmed
                # locked board and exactly four active minos. Repeated twice.
                references=[]
                if self.auto.plan is not None:
                    if self.auto.phase=='drop_wait':
                        references.append((self.auto.plan.move.board,None))
                    else:
                        references.append((self.auto.plan.source_board,
                                           self.auto.plan.move.piece))
                if self.tracker.previous is not None:
                    references.append(self.tracker.previous)
                if not hasattr(self,'legacy_recovery'):self.legacy_recovery=ConfirmedLegacyFallback()
                legacy=read_grid(read_frame,self.threshold.value())
                candidate=self.legacy_recovery.consider(legacy,references,started)
                if candidate is not None:
                    from color_vision import VisionReading
                    grid=candidate
                    vision=VisionReading(grid,None,None,None,'LEGACY',1.0,
                        '2개 새 프레임 · 기존 보드와 실제 4칸 블록 교차검증')
                    self._event('legacy_recovery_verified',source='dual_frame_proof')
            else:
                if hasattr(self,'legacy_recovery'):self.legacy_recovery.reset()
            if grid is None:
                # Never feed stale or partly guessed occupancy to the planner.
                # But do not discard a perfectly good *locked board hypothesis*
                # after a one-frame animation. A later fresh, complete image
                # must verify every old cell and exactly four active cells.
                if hasattr(self.tracker,'mark_missing'):
                    self.tracker.mark_missing(started)
                if not getattr(self,'vision_failure_open',False):
                    self.vision_failure_open=True
                    self.vision_failure_at=started
                    # One bounded in-memory snapshot of the ACTUAL failed frame.
                    # Captures only the selected BOARD, never the desktop.
                    self.failure_frame=(a.copy(),None,self.tracker.previous)
                    self.failure_details=self.auto.snapshot()
                    coords=(np.argwhere(vision.unknown).tolist()[:32]
                            if vision.unknown is not None else [])
                    self.failure_vision=dict(method=vision.method,
                                             reason=vision.reason[:180],
                                             uncertain_count=(int(np.count_nonzero(vision.unknown))
                                                              if vision.unknown is not None else 0),
                                             uncertain_cells_yx=coords)
                    # Freeze the exact background reference and pixel samples
                    # together with the first failed frame. The background may
                    # be different by the time the user exports diagnostics.
                    # Evidence is capped at 8 cells and is never used to plan.
                    try:
                        self.failure_vision['pixel_evidence']=failure_evidence(
                            a,self.color_reader.background,vision.unknown)
                    except (ValueError,TypeError) as error:
                        self.failure_vision['audit_error']=str(error)[:100]
                    if hasattr(self,'_event'):
                        self._event('vision_lost',reason=vision.reason[:80])
                transient=('불확실한 칸' in vision.reason)
                since=started-(getattr(self,'vision_failure_at',None) or started)
                recovering=transient and since<.20
                self.preview.grid=[[False]*10 for _ in range(20)]
                self.preview.colors=None;self.preview.ghosts=None;self.preview.unknown=vision.unknown
                self.last_grid=None
                self.was_recognized=False
                if hasattr(self,'vision_status') and started>=getattr(self,'vision_status_next_at',0):
                    phase='RECOVERING' if recovering else vision.method
                    self.vision_status.setText(f'VISION {phase} · {vision.reason}')
                    self.vision_status_next_at=started+.20
                # Immediately release all held keys, even during transient
                # recovery. Three consecutive fresh observations are required
                # before AutoPlayer resumes; no stale frame is actionable.
                if self.auto.enabled and not self.auto.paused:
                    self.pause_auto('일시적 화면 효과 · 안전 재확인' if recovering
                                    else ('LEGACY 밝기 인식 불확실 · BOARD/임계값 재확인'
                                          if mode=='legacy' else '보드 색상 인식 불확실 · 보정/ROI 재확인'))
                if self.auto.enabled:
                    self._record_control_gap(started,False)
                    # Tell the controller about the visual gap. This is vital
                    # to distinguish an acknowledged lock from a duplicate
                    # identical-looking active piece after line-clear effects.
                    # observe(None) never emits an input action.
                    self.auto.observe(None,None,started)
                if self.key is not None and not recovering:
                    self.invalidate()
                # The old CURRENT/NEXT values are historical after a visual
                # dropout. Never present them as newly recognized information.
                if hasattr(self,'queue_status'):
                    self.queue_status.setText('CURRENT ?   NEXT ?   HOLD ?   IN ?\n보드 인식 중단 · 직전 조각 정보는 현재 상태가 아닙니다')
                if not recovering and hasattr(self,'status') and since >= .3:
                    where=(self.failure_vision or {}).get('uncertain_cells_yx',[])
                    self.status.setText('BOARD 인식 중단 · F7에서 ROI 격자 확인 · 진단 ZIP 저장'
                                        +(' · '+str(where[:4]) if where else ''))
                self.preview.active=[];self.preview.update()
                self.capture_ms=(time.perf_counter()-started)*1000
                return
            self.preview.grid=grid
            if vision.colors is not None:
                self.preview.colors=vision.colors;self.preview.ghosts=vision.ghosts
                self.preview.unknown=vision.unknown
            elif vision.method=='LEGACY':
                self.preview.colors=None;self.preview.ghosts=None;self.preview.unknown=None
            if hasattr(self,'vision_status') and started>=getattr(self,'vision_status_next_at',0):
                self.vision_status.setText(f'VISION {vision.method} · {vision.reason}' + (' · SPAWN '+str(spawn_rows)+'줄' if spawn_rows is not None else ''))
                self.vision_status_next_at=started+.25
            hints=[]
            if self.auto.enabled and self.auto.plan:
                if self.auto.phase=='drop_wait':hints.append((self.auto.plan.move.board,None))
                hints.append((self.auto.plan.source_board,None))
            # SPAWN-FIRST: the freshly observed SPAWN tetromino is normally
            # cheaper to identify than reconstructing a connected component
            # in a growing BOARD. Only accept it when four colored minos are
            # visible and the settled BOARD agrees with a verified reference;
            # otherwise the old BoardTracker is authoritative. This reorders
            # existing checks but does not speculate about the next piece.
            state=None
            # The tracker is speculative until a LEGACY board transition is
            # confirmed. Save its exact reference so a one-frame brightness
            # artifact cannot poison subsequent opponent-Garbage recovery.
            tracker_checkpoint=(self.tracker.previous,self.tracker.seen_at,
                                self.tracker.dropout_at,self.tracker.last_rise)
            spawn_tried=False
            # During ordinary align/move frames the BOARD tracker already
            # knows the current piece. Avoid paying for SPAWN-first there;
            # prioritize it only at initial detection and lock/HOLD transitions.
            spawn_priority=(spawn_rows is not None and
                (self.tracker.previous is None or
                 (self.auto.enabled and self.auto.phase in ('drop_wait','hold_wait'))))
            if spawn_priority:
                spawn_tried=True
                prior=self.tracker.previous[0] if self.tracker.previous else None
                expected=(self.auto.plan.move.board if self.auto.enabled
                          and self.auto.plan is not None and self.auto.phase=='drop_wait'
                          else None)
                state=fuse_spawn_active(spawn_frame,grid,self.color_reader.background,
                                        spawn_rows,prior,expected,
                                        reading='legacy' if mode=='legacy' else 'color',
                                        threshold=self.threshold.value())
                if state is not None:
                    self.tracker.previous=state[:2]
                    self.tracker.seen_at=started
                    self.tracker.dropout_at=None
                    self.spawn_fast_hits=getattr(self,'spawn_fast_hits',0)+1
                    if hasattr(self,'_event') and started-getattr(self,'spawn_seen_log_at',-1e12)>.5:
                        self._event('spawn_first',piece=state[1],rows=spawn_rows)
                        self.spawn_seen_log_at=started
            if state is None:
                state=self.tracker.update(grid,started,hints)
                if state is not None and getattr(self.tracker,'last_rise',None):
                    rise=self.tracker.last_rise[2]
                    self.vision_rise_recovered=getattr(self,'vision_rise_recovered',0)+1
                    if hasattr(self,'_event'):
                        self._event('vision_rise_verified',rows=rise,piece=state[1])
                # A themed piece can have a dark center yet colored edges.
                # In that case preserve the old full SPAWN fallback instead
                # of rejecting an otherwise readable new tetromino.
                if state is None and spawn_rows is not None and not spawn_tried:
                    prior=self.tracker.previous[0] if self.tracker.previous else None
                    expected=(self.auto.plan.move.board if self.auto.enabled
                              and self.auto.plan is not None and self.auto.phase=='drop_wait'
                              else None)
                    state=fuse_spawn_active(spawn_frame,grid,self.color_reader.background,
                                            spawn_rows,prior,expected,
                                            reading='legacy' if mode=='legacy' else 'color',
                                            threshold=self.threshold.value())
                    if state is not None:
                        self.tracker.previous=state[:2]
                        self.tracker.seen_at=started
                        self.tracker.dropout_at=None
                        self.spawn_fast_hits=getattr(self,'spawn_fast_hits',0)+1
                if state is None and spawn_rows is not None:
                    self.spawn_fallbacks=getattr(self,'spawn_fallbacks',0)+1
            legacy_provisional=False
            if mode=='legacy' and self.auto.enabled:
                if not hasattr(self,'legacy_gate'):
                    self.legacy_gate=LegacyTransitionGate()
                current_plan=self.auto.plan
                independently_verified=False
                if state is not None and current_plan is not None:
                    previous_board=current_plan.source_board
                    candidate_board=state[0]
                    if (self.auto.phase=='drop_wait' and
                            candidate_board==current_plan.move.board):
                        # The post-drop transition classifier independently
                        # requires two complete frames and a physical lock.
                        independently_verified=True
                    elif (self.auto.phase=='hold_wait' and
                          candidate_board==previous_board):
                        # HOLD acknowledgement also requires stable evidence.
                        independently_verified=True
                    elif state[1]==current_plan.move.piece:
                        from garbage_transition import verified_rise
                        shift=verified_rise(previous_board,candidate_board)
                        # Exact 9/10-row rise is independently verified a
                        # second time by AutoPlayer before a replan executes.
                        if shift is not None and shift>0:
                            independently_verified=True
                caution=(self.auto.phase not in ('idle','drop_wait','hold_wait')
                         and current_plan is not None and state is not None
                         and state[0]!=current_plan.source_board)
                if not self.legacy_gate.accept(state,started,extra_caution=caution,
                                               independently_verified=independently_verified):
                    if state is not None:
                        (self.tracker.previous,self.tracker.seen_at,
                         self.tracker.dropout_at,self.tracker.last_rise)=tracker_checkpoint
                        if hasattr(self,'_event') and self.legacy_gate.count==1:
                            self._event('legacy_transition_held',
                                        reason=self.legacy_gate.last_reason)
                        # Do not allow a transient *otherwise valid* legacy
                        # candidate to tear down an in-flight MOVEMENT burst.
                        # It is NOT approved as a current board: this entire
                        # tick skips planning and all hard-drop authorization.
                        # Real unreadable frames and DROP/HOLD transitions
                        # still follow the original fail-closed cancellation.
                        active=getattr(self,'input_delivery_active',None)
                        busy=bool(getattr(self.inputs,'busy',False))
                        from_plan=(current_plan.move.piece if current_plan is not None else None)
                        entered=getattr(self,'legacy_provisional_since',None)
                        legacy_provisional=allow_provisional_hold(
                            gate_reason=self.legacy_gate.last_reason,
                            phase=self.auto.phase,observed_piece=state[1],
                            planned_piece=from_plan,paused=self.auto.paused,
                            input_busy=busy,
                            drop_inflight=(self.auto.phase=='drop_wait' or
                                           (active is not None and active[1]=='drop')),
                            now=started,entered_at=entered)
                        # Unknown or unverified boards must not enter planning,
                        # input acknowledgement or hard-drop authorization.
                        state=None
                # The gate only approves LIVE captures; it never returns a
                # previous frame as an input target.
            # Diagnostic snapshots do not participate in control. Copying the
            # whole board image every 8 ms wastes memory bandwidth, so retain a
            # recent snapshot at 10 Hz and always capture an actual failure frame.
            self.last_grid=grid
            if started>=getattr(self,'diagnostic_next_at',0):
                self.last_frame=a.copy();self.diagnostic_next_at=started+.10
            if state is None and (self.was_recognized or self.failure_frame is None):
                self.failure_frame=(a.copy(),grid,self.tracker.previous)
                self.last_frame=self.failure_frame[0]
                self.failure_details=self.auto.snapshot()
            self.was_recognized=state is not None
            self.capture_ms=(time.perf_counter()-started)*1000
            if 'garbage' in self.cfg and hasattr(self,'garbage_filter'):
                incoming,confidence,expired=self.garbage_filter.expire(started)
                if expired:
                    previously_known=getattr(self,'meter_was_known',False)
                    old_incoming=getattr(self,'incoming_garbage',0)
                    old_confidence=getattr(self,'garbage_confidence',0.0)
                    self.incoming_garbage=incoming;self.garbage_confidence=confidence
                    self.meter_was_known=False
                    self._refresh_idle_fast_pressure(board,piece,
                        old_incoming,old_confidence,previously_known,
                        incoming,confidence,False)
                    # Stale telemetry never revokes a valid board/turn input
                    # ticket. The planner still treats the meter as UNKNOWN.
                    if previously_known:
                        self.meter_only_updates=getattr(self,'meter_only_updates',0)+1
                    self._event('meter_expired')
                    if self.key is not None and self.running:
                        self.schedule_deep(self.key)
            if state is None:
                if legacy_provisional:
                    if getattr(self,'legacy_provisional_since',None) is None:
                        self.legacy_provisional_since=started
                        if hasattr(self,'_event'):self._event('legacy_provisional_enter')
                    self.legacy_provisional_frames=getattr(self,'legacy_provisional_frames',0)+1
                    # The real candidate is untrusted and has been discarded.
                    # No new input or old-state hard drop is issued here. An
                    # already running path-only burst may complete and will be
                    # acknowledged ONLY on a subsequent freshly verified frame.
                    return
                if getattr(self,'legacy_provisional_since',None) is not None:
                    self.legacy_provisional_escalated=getattr(self,'legacy_provisional_escalated',0)+1
                    self.legacy_provisional_since=None
                    if hasattr(self,'_event'):self._event('legacy_provisional_escalated')
                if self.auto.enabled and getattr(self,'input_board_ticket',None) is not None:
                    self.input_board_ticket=None
                    self.input_ticket_epoch+=1
                    if getattr(self.inputs,'busy',False):
                        self.inputs.cancel_pending()
                        self._event('input_guard_cancel',reason='VISION_LOST')
                if hasattr(self,'queue_status'):
                    self.queue_status.setText('CURRENT ?   NEXT ?   HOLD ?   IN ?\nACTIVE 블록 미확인 · 현재 수 판단 보류')
                if self.key is not None: self.invalidate()
                self.preview.active=[]; self.preview.update(); self.preview_next_at=started+.033
                self.status.setText('현재 블록 식별 대기'); self.pump_auto(None,started); return
            if getattr(self,'legacy_provisional_since',None) is not None:
                self.legacy_provisional_resolved=getattr(self,'legacy_provisional_resolved',0)+1
                self.legacy_provisional_since=None
                if hasattr(self,'_event'):self._event('legacy_provisional_resolved')
            board,piece,cells=state; self.preview.active=cells
            if self.auto.enabled:
                fresh_ticket=(board,piece)
                if fresh_ticket!=getattr(self,'input_board_ticket',None):
                    was_present=getattr(self,'input_board_ticket',None) is not None
                    self.input_board_ticket=fresh_ticket
                    self.input_ticket_epoch=getattr(self,'input_ticket_epoch',0)+1
                    if was_present and getattr(self.inputs,'busy',False):
                        self.inputs.cancel_pending()
                        self._event('input_guard_cancel',reason='BOARD_OR_PIECE_CHANGED')
            # After a dropout, first mark only the freshly identified CURRENT.
            # NEXT/HOLD will be refreshed by the ordinary replan/ROI path.
            if getattr(self,'vision_failure_open',False) and hasattr(self,'queue_status'):
                self.queue_status.setText(f'CURRENT {piece}   NEXT 재확인   HOLD 재확인\n보드 인식 복구 · 입력 전 상태 검증 중')
            # Clear episode guard only after a newly VERIFIED active state.
            # Preserve the original failure frame for diagnostic export.
            self.vision_failure_open=False
            self.vision_failure_at=None
            # Paint the diagnostic preview at ~30 Hz while the recognition and
            # controller continue at the original 8 ms cadence.
            if started>=getattr(self,'preview_next_at',0):
                self.preview.update();self.preview_next_at=started+.033
            # Acknowledge hold/lock BEFORE computing the next key. Previously
            # we computed it, rejected it during acknowledgement, then computed
            # it a second time with a new revision on the following tick.
            handled_transition=False
            verified_lock=False
            if self.auto.enabled and self.auto.phase in ('drop_wait','hold_wait'):
                locks_before=self.auto.confirmed_locks
                self.pump_auto(state,started);handled_transition=True
                verified_lock=self.auto.confirmed_locks>locks_before
                if self.auto.phase in ('drop_wait','hold_wait'):return
            self.poll_roi_retry(board,piece)
            # Establish new-match eligibility *before* reading initial HOLD and
            # submitting the very first Fast plan. Do not wait for observe().
            if self.auto.enabled and getattr(self.auto,'first_seen',None) is None:
                self.auto.prime_opening(board,started)
            mode=['balanced','combo','survival','attack'][self.mode.currentIndex()]
            if self.key is None or (board,piece)!=self.key[:2] or self.key[-1]!=self.auto.serial:
                # A successful controller lock plus the physically observed
                # new ACTIVE piece validates the shift of previous NEXT2 to
                # NEXT1. Avoid a synchronous MSS grab on the critical input
                # path; any missing/mismatched observation uses the old route.
                cached_queue=()
                cached_next=None
                if (self.auto.enabled and verified_lock and
                        not getattr(self.auto,'hold_uncertain',False)):
                    cached_next,cached_queue=self._verified_next_after_lock(piece,True)
                nxt=cached_next if cached_next is not None else self.piece_box('next')
                hold=self.auto.held if self.auto.enabled else self.read_hold()
                if self.auto.enabled and hold=='?':
                    hold=(self.piece_box('hold') or '?') if self.auto.hold_uncertain else self.read_hold()
                    self.auto.held=hold
                # Only a user-enabled assumption, only for an empty *new-game*
                # board; never silently infer blank HOLD from a dark ROI.
                if (self.auto.enabled and hold=='?' and not any(board)
                        and getattr(self.auto,'opening_eligible',False)
                        and self.auto.confirmed_locks==0 and 'hold' in self.cfg
                        and hasattr(self,'blank_hold') and self.blank_hold.isChecked()):
                    hold=None;self.auto.held=None
                allow_hold=self.hold_enabled.isChecked() and not (self.auto.enabled and self.auto.hold_used)
                if self.auto.enabled and self.hold_enabled.isChecked():
                    if hold=='?':
                        self.hold_unknown_observations=getattr(self,'hold_unknown_observations',0)+1
                    elif not allow_hold or (hold is None and nxt is None):
                        self.hold_blocked_observations=getattr(self,'hold_blocked_observations',0)+1
                    else:
                        self.hold_eligible_observations=getattr(self,'hold_eligible_observations',0)+1
                b2b=self._planning_b2b()
                combo=self.auto.combo if self.auto.enabled else -1
                self.invalidate(); self.key=(board,piece,nxt,mode,self.fast.isChecked(),hold,allow_hold,b2b,combo,self.auto.serial); self.detected_at=started
                self.current_first_input_ms=None; self.current_drop_ms=None
                self.opener_fast_recompute_pending=False
                self.pressure_fast_recompute_pending=False
                self.cc2_retry_count=0;self.cc2_retry_at=0.0
                # Preserve only queue entries already observed for the old
                # turn. Fresh NEXT2/NEXT3 snapshots still run asynchronously.
                self.deep_queue=cached_queue if cached_queue else ((nxt,) if nxt is not None else ())
                self.deep_queue_slots=tuple(self.deep_queue)+(None,)*(3-len(self.deep_queue))
                # Old NEXT3 remains the next NEXT2 only when the prior queue
                # was already verified and the real active piece agreed.
                self.next_queue_trusted=len(cached_queue)>1
                self.queue_revision=getattr(self,'queue_revision',0)+1
                self._update_queue_text(piece,nxt,hold if hold!='?' else None)
                self.schedule_roi_retry(board,piece,nxt,hold)
                if self._engine_mode()=='cc2':
                    # On this turn, the sole author of the actual placement
                    # must be CC2. Launch independently of the native Fast
                    # preview and preserve the controller's idle state.
                    self.submitted_key=self.key
                    self.cc2_last_reason='CALCULATING'
                    self.schedule_deep(self.key)
                    if self.auto.enabled and hasattr(self,'auto_status'):
                        self.auto_status.setText('CC2 ONLY · 계산 중 · 유효한 외부 판단만 실행')
            elif (getattr(self,'next_queue_retry_at',0) and not getattr(self,'next_queue_retry_done',False)
                  and time.perf_counter()>=self.next_queue_retry_at):
                # One bounded second sample ONLY when NEXT2/3 failed at spawn.
                if self.roi_future is None and self.roi_pending is None:
                    self.schedule_roi_retry(board,piece,self.key[2],self.key[5],queue_retry=True)
            elif 'garbage' in self.cfg and time.perf_counter()>=self.telemetry_next_at:
                # 80ms garbage-only telemetry: never re-read NEXT2/NEXT3 here.
                self.schedule_roi_retry(board,piece,self.key[2],self.key[5],True)
            prepared=self.auto.prepared
            if (prepared is not None and self.auto.enabled and self._engine_mode()=='internal'
                    and self.key is not None):
                options,known,placed,eligible=self._native_opener_snapshot()
                if eligible and (options[0] or options[1]):
                    from opener_commit import observable_reference_stage
                    if observable_reference_stage(board,piece,known,placed,
                            canonical=options[0],pco=options[1]):
                        # Preserve the previous plan for diagnostics, but
                        # don't execute its speculative NEXT continuation
                        # before the exact per-turn reference is checked.
                        prepared=None
                        self.auto.prepared=None
                        self._event('opener_prefetch_veto',piece=piece,stage=placed)
            if (self.auto.enabled and self._engine_mode()!='cc2'
                    and not getattr(self.auto,'soft_disabled',False)
                    and self.auto.turbo and prepared is not None
                    and prepared.prepared_safe
                    # A stale NEXT continuation must never leap over a newly
                    # observed attack or an unreadable selected garbage meter.
                    and permit_prepared_commit(
                        meter_configured='garbage' in self.cfg,
                        meter_known=(self.garbage_filter.is_known()
                                     if 'garbage' in self.cfg else True),
                        incoming=getattr(self,'incoming_garbage',0),
                        confidence=getattr(self,'garbage_confidence',0.0))
                    and hasattr(self,'instant') and self.instant.isChecked()
                    and self.auto.phase=='idle' and self.current_plan is None and self.future is None
                    and (prepared.source_board,prepared.source_piece)==(board,piece)):
                self.auto.prepared=None;self.submitted_key=self.key
                self.auto.prepared_submitted+=1
                self.auto.expected_prepared=prepared
                self._event('next_submitted',piece=piece,lines=prepared.move.lines,spin=prepared.move.spin)
                self._update_deep_stats()
                self.accept_result(self.generation,self.key,prepared,0,None)
            if (self.auto.enabled and self._engine_mode()=='cc2' and self.key is not None
                    and getattr(self,'cc2_retry_at',0)
                    and started>=self.cc2_retry_at and self.deep_future is None
                    and self.auto.phase=='idle' and self.current_plan is None
                    and started-getattr(self,'detected_at',started)<1.7):
                self.cc2_retry_at=0.0
                self.cc2_last_reason='RETRYING'
                self.schedule_deep(self.key)
            if self._engine_mode()=='cc2' and self.key is not None:
                # Prevent a legacy pending Fast task from setting an internal
                # controller target during an exclusive CC2 session.
                self.submitted_key=self.key
            if self.future is None and self.submitted_key!=self.key:
                key=self.key; generation=self.generation; self.submitted_key=key
                self.future=self.pool.submit(self.compute,key,
                    conservative_incoming(self.incoming_garbage,'garbage' in self.cfg,
                                          self.garbage_confidence>=.55),
                    self.ruleset.currentData() if hasattr(self,'ruleset') else 'standard',
                    self.combo_table.currentData() if hasattr(self,'combo_table') else 'multiplier',
                    self.rounding.currentData() if hasattr(self,'rounding') else 'auto',
                    bool(getattr(self.auto,'soft_disabled',False)),
                    self._safe_single_active(),
                    self._native_opener_snapshot(),
                    # Bounded goal urgency from visually confirmed locks only.
                    self.auto.attack_pace(started,
                        incoming=conservative_incoming(
                            self.incoming_garbage,'garbage' in self.cfg,
                            self.garbage_confidence>=.55),
                        safe=(not any(key[0][:9]))).urgency)
                def done(f,g=generation,k=key):
                    try:
                        move,elapsed=f.result(); self.result_ready.emit(g,k,move,elapsed,None)
                    except Exception as e:
                        try: self.result_ready.emit(g,k,None,0,str(e))
                        except RuntimeError: pass
                self.future.add_done_callback(done)
            if not handled_transition:self.pump_auto(state,started)
        except Exception as e:
            if (hasattr(self,'color_reader') and self.color_reader.recalibrating):
                self.color_reader.cancel_recalibration()
                self._finish_color_calibration(CalibrationProgress(
                    'failed', 0, 4, '재보정 중 화면 캡처 오류 · 기존 배경 유지: '+str(e)[:100]))
            else:
                self.pause_auto('화면/입력 재확인'); self.invalidate()
                self.retry_after=time.perf_counter()+.5
                self.status.setText('재시도 대기: '+str(e))
    @staticmethod
    def compute(key,incoming=0,ruleset='standard',combo_table='multiplier',rounding='auto',soft_disabled=False,single_guard=True,opener_snapshot=None,attack_urgency=0.0):
        board,piece,nxt,mode,fast,hold,allow_hold,b2b,combo,serial=key; start=time.perf_counter()
        # Fully verified canonical targets are selected INSIDE the first
        # Fast result, eliminating the Fast->Deep adoption race. The GUI and
        # input thread do not run a pattern search.
        flags,observed,placed,eligible=(opener_snapshot if opener_snapshot is not None
                                       else ((False,False,False),(),20,False))
        if (eligible and mode=='attack' and not soft_disabled and incoming==0
                and (flags[0] or flags[1])):
            from opener_commit import fast_opening_plan
            q=tuple(observed) if observed and observed[0]==nxt else ((nxt,) if nxt else ())
            proof=fast_opening_plan(board,piece,q,hold=hold,allow_hold=allow_hold,
                                    b2b=b2b,combo=combo,incoming=incoming,placed=placed,
                                    session_ok=eligible,canonical=flags[0],pco=flags[1],
                                    ruleset=ruleset,combo_table=combo_table,rounding=rounding)
            if proof is not None:return proof,(time.perf_counter()-start)*1000
        # Only emergency snapshots invoke a heavier rescue score. Normal
        # gameplay remains byte-for-byte on the old fast planner call.
        if mode=='attack' and soft_disabled:
            from engine import plan_nonsonic
            plan=plan_nonsonic(board,piece,nxt,hold,allow_hold,b2b,combo)
        elif mode=='attack' and incoming>0 and combat_profile(board,incoming)[0]=='EMERGENCY':
            plan=emergency_fast_plan(board,piece,nxt,hold,allow_hold,b2b,combo,incoming,ruleset,combo_table,rounding)
        else:
            plan=plan_move(board,piece,nxt,hold,allow_hold,mode,fast,b2b,combo,
                           incoming=incoming,single_guard=single_guard,
                           tempo_bias=(mode=='attack' and incoming==0 and fast),
                           attack_urgency=(attack_urgency if mode=='attack' and fast
                                           and incoming==0 else 0.0))
        if plan is not None and mode in ('attack','balanced','combo'):
            from engine import guard_fast_stack
            plan=guard_fast_stack(plan,board,piece,nxt,hold,allow_hold,
                                  b2b,combo,incoming,ruleset,combo_table,rounding)
        if (eligible and mode=='attack' and not soft_disabled and incoming==0
                and flags[2]):
            from opener_commit import fast_stacker63_plan
            plan=fast_stacker63_plan(plan,board,piece,hold=hold,allow_hold=allow_hold,
                                     b2b=b2b,combo=combo,incoming=incoming,
                                     placed=placed,session_ok=eligible)
        return plan,(time.perf_counter()-start)*1000
    def closeEvent(self,e):
        if getattr(self,'_calibration_hidden',False):
            self._calibration_hidden=False
            self.color_reader.cancel_recalibration()
        self.stop_auto('종료'); self.timer.stop(); self.overlay.close(); self.cap.close(); self.pool.shutdown(wait=False,cancel_futures=True)
        if hasattr(self,'roi_pool'):self.roi_pool.shutdown(wait=False,cancel_futures=True)
        if hasattr(self,'deep_pool'):self.deep_pool.shutdown(wait=False,cancel_futures=True)
        try: self.save()
        except OSError: pass
        super().closeEvent(e)

if __name__=='__main__':
    mp.freeze_support()
    # Windows PyInstaller startup probe: catches missing imports *after* EXE
    # creation, not just in source. --windowed has no console, so emit a marker
    # that build-exe.cmd can check before claiming success.
    smoke=next((arg.split('=',1)[1] for arg in sys.argv[1:]
                if arg.startswith('--self-test=')),None)
    if smoke is not None:
        # Random per-build token prevents stale/fabricated previous success
        # files from being accepted as proof of the *new* frozen EXE startup.
        import re, tempfile
        if not re.fullmatch(r'[0-9a-f]{32}',smoke):sys.exit(2)
        for module in ('attack_rate','telemetry','combo_rules','combat_rules','opening','legacy_quarantine','pressure_refresh',
                       'garbage_tracking','garbage_transition','tspin_factory','b2b_search','competitive_policy','config_security','input_rhythm','engine','motion',
                       'vision','color_vision','piece_vision','vision_failure','spawn_vision','legacy_integrity','windows_input','autoplay','opener_commit','canonical_openers','perfect_clear_planner'):
            __import__(module)
        marker=Path(tempfile.gettempdir())/('TetrioAssist_smoke_'+smoke+'.txt')
        marker.write_text(smoke+'\n',encoding='ascii')
        sys.exit(0)
    app=QApplication(sys.argv); window=App(); window.show(); sys.exit(app.exec())
