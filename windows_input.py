"""Small Win32 SendInput adapter. Only a verified foreground TETR.IO EXE.
No memory access, process injection, elevation, focus stealing or key hooks.
"""
import ctypes as C
import ntpath
import sys
import time
import threading
from collections import deque
from window_roi import region_issues

U32=C.c_uint32; U16=C.c_uint16; I32=C.c_int32; PTR=C.c_size_t
class KEYBDINPUT(C.Structure):
    _fields_=[('wVk',U16),('wScan',U16),('dwFlags',U32),('time',U32),('dwExtraInfo',PTR)]
class MOUSEINPUT(C.Structure):
    _fields_=[('dx',I32),('dy',I32),('mouseData',U32),('dwFlags',U32),('time',U32),('dwExtraInfo',PTR)]
class HARDWAREINPUT(C.Structure):
    _fields_=[('uMsg',U32),('wParamL',U16),('wParamH',U16)]
class PAYLOAD(C.Union):
    _fields_=[('ki',KEYBDINPUT),('mi',MOUSEINPUT),('hi',HARDWAREINPUT)]
class INPUT(C.Structure):
    _anonymous_=('data',)
    _fields_=[('type',U32),('data',PAYLOAD)]
class RECT(C.Structure):
    _fields_=[('left',I32),('top',I32),('right',I32),('bottom',I32)]

KEYS={'LEFT':0x25,'RIGHT':0x27,'UP':0x26,'DOWN':0x28,'SPACE':0x20,'SHIFT':0x10,'CTRL':0x11}
KEYS.update({chr(i):i for i in range(65,91)})
DEFAULT={'left':'LEFT','right':'RIGHT','cw':'X','ccw':'Z','rot180':'A','soft':'DOWN','drop':'SPACE','hold':'C'}

def valid_binding(value):return value.upper().strip() in KEYS

class WindowsInput:
    def __init__(self):
        self.available=sys.platform=='win32';self.target=None;self.target_pid=None;self.down=None;self.region=None
        self.capture_regions=()
        self.capture_problems=[]
        self.worker=None;self.cancel=threading.Event();self.error=None;self.pulse_ms=8;self.gap_ms=4
        self.abort_count=0;self.last_aborted_keys=()
        # Per-burst receipts distinguish a keydown sent to Windows from a
        # cancelled/unstarted queued command. NEVER equate SendInput=1 with
        # a gameplay lock; the actual lock still requires live pixels.
        self._receipt_lock=threading.Lock()
        self._receipt_seq=0
        self._receipts=deque(maxlen=256)
        # Input worker and emergency key-release share one key-state lock.
        # Cancellation is per burst: a new burst cannot revive an old worker.
        self._key_lock=threading.RLock()
        self.high_resolution=False
        if not self.available:return
        self.u=C.WinDLL('user32',use_last_error=True);self.k=C.WinDLL('kernel32',use_last_error=True)
        self.u.GetForegroundWindow.restype=C.c_void_p
        self.u.GetWindowThreadProcessId.argtypes=[C.c_void_p,C.POINTER(U32)]
        self.u.GetWindowRect.argtypes=[C.c_void_p,C.POINTER(RECT)]
        self.u.GetWindowTextW.argtypes=[C.c_void_p,C.c_wchar_p,C.c_int]
        self.k.OpenProcess.argtypes=[U32,C.c_int,U32];self.k.OpenProcess.restype=C.c_void_p
        self.k.QueryFullProcessImageNameW.argtypes=[C.c_void_p,U32,C.c_wchar_p,C.POINTER(U32)]
        self.k.CloseHandle.argtypes=[C.c_void_p]
        self.u.SendInput.argtypes=[U32,C.POINTER(INPUT),C.c_int];self.u.SendInput.restype=U32
        self.u.MapVirtualKeyW.argtypes=[U32,U32];self.u.MapVirtualKeyW.restype=U32
    def arm(self,region):
        if not self.available:return False,'Windows에서만 자동 입력 가능'
        hwnd=self.u.GetForegroundWindow()
        pid=U32();self.u.GetWindowThreadProcessId(hwnd,C.byref(pid))
        handle=self.k.OpenProcess(0x1000,0,pid.value)
        if not handle:return False,'TETR.IO 프로세스를 확인할 수 없음'
        try:
            buf=C.create_unicode_buffer(32768);size=U32(len(buf))
            if not self.k.QueryFullProcessImageNameW(handle,0,buf,C.byref(size)):return False,'프로세스 경로 확인 실패'
            name=ntpath.basename(buf.value).lower()
            if name not in ('tetr.io.exe','tetrio.exe'):return False,'TETR.IO 게임 창을 클릭한 뒤 F9를 누르세요'
        finally:self.k.CloseHandle(handle)
        self.target=hwnd;self.target_pid=pid.value;self.region=region;self.capture_regions=();self.capture_problems=[]
        if not self.focused():self.target=None;self.target_pid=None;return False,'지정한 보드가 TETR.IO 창 내부에 있지 않음'
        if not self.high_resolution:
            self.high_resolution=C.WinDLL('winmm').timeBeginPeriod(1)==0
        return True,''
    @staticmethod
    def _inside_rect(rect,region):
        if not isinstance(region,dict):return False
        try:
            left,top,w,h=(region[k] for k in ('left','top','width','height'))
            if any(type(v) is not int for v in (left,top,w,h)) or w<=0 or h<=0:return False
            return (rect.left<=left and rect.top<=top and
                    rect.right>=left+w and rect.bottom>=top+h)
        except (KeyError,TypeError,ValueError):
            return False

    def set_capture_regions(self,regions):
        """Validate every ROI, including the sixth SPAWN/queue/meter region.

        Store named failures so F9 reports the *specific* bad selection.
        Never capture or authorize pixels outside the game window.
        """
        self.capture_problems=[]
        if not self.available or not self.target or not self.focused():
            self.capture_problems=[('GAME','게임 창 포커스/대상 확인 실패')]
            return False
        r=RECT()
        if not self.u.GetWindowRect(self.target,C.byref(r)):
            self.capture_problems=[('GAME','게임 창 좌표 확인 실패')]
            return False
        rect={'left':r.left,'top':r.top,'right':r.right,'bottom':r.bottom}
        self.capture_problems=region_issues(rect,regions)
        if self.capture_problems:
            return False
        self.capture_regions=tuple(regions.values()) if isinstance(regions,dict) else tuple(regions)
        return True

    @property
    def busy(self):return self.worker is not None and self.worker.is_alive()
    def focused(self):
        if not self.available or not self.target or self.u.GetForegroundWindow()!=self.target:return False
        # A Win32 HWND can be recycled after the game exits. Validate that
        # the focused HWND still belongs to the PID explicitly armed by F9.
        pid=U32()
        if not self.u.GetWindowThreadProcessId(self.target,C.byref(pid)):
            return False
        if pid.value != self.target_pid:
            return False
        r=RECT()
        if not self.u.GetWindowRect(self.target,C.byref(r)):return False
        return (self._inside_rect(r,self.region) and
                all(self._inside_rect(r,extra) for extra in self.capture_regions))
    def _send(self,vk,up):
        scan=self.u.MapVirtualKeyW(vk,0)
        flags=0x0008|(0x0002 if up else 0)|(0x0001 if vk in (0x25,0x26,0x27,0x28) else 0)
        event=INPUT(type=1,ki=KEYBDINPUT(0,scan,flags,0,0))
        if self.u.SendInput(1,C.byref(event),C.sizeof(INPUT))!=1:raise OSError('SendInput 실패: 게임과 프로그램의 실행 권한/키 설정 확인')
    def drain_receipts(self):
        # Consumed by the GUI thread. Do not call Qt or capture inside worker.
        with self._receipt_lock:
            events=tuple(self._receipts)
            self._receipts.clear()
        return events

    def _receipt(self,token,stage,sent,total,when):
        with self._receipt_lock:
            self._receipts.append((token,stage,sent,total,when))

    def press(self,name,valid=None):
        return self.press_sequence((name,),valid=valid)
    def press_sequence(self,names,valid=None):
        keys=tuple(KEYS[name.upper().strip()] for name in names)
        if not keys:return None
        if self.busy or self.down is not None:raise RuntimeError('이전 입력이 아직 진행 중')
        if not self.focused():raise RuntimeError('TETR.IO 포커스 해제')
        with self._receipt_lock:
            self._receipt_seq+=1
            token=self._receipt_seq
        # Snapshot pulse/gap so GUI changes cannot mutate a running burst.
        pulse_ms=max(1,int(self.pulse_ms))
        gap_ms=max(0,int(self.gap_ms))
        cancel=threading.Event()
        self.cancel=cancel
        def run():
            sent=0
            try:
                for vk in keys:
                    if cancel.is_set():break
                    # Verify focus immediately before each *down*, under the
                    # same lock used by ESC/disarm. Never inject a new down
                    # after emergency release has acquired this lock.
                    with self._key_lock:
                        if cancel.is_set() or not self.focused():break
                        # A stale board/turn must never start a queued key.
                        # The predicate is a cheap immutable ticket check;
                        # do not capture pixels or enter Qt from this thread.
                        if valid is not None and not valid():break
                        self._send(vk,False)
                        sent+=1
                        self.down=(vk,float('inf'))
                        self._receipt(token,'key_down',sent,len(keys),time.perf_counter())
                    try:
                        cancel.wait(pulse_ms/1000)
                    finally:
                        with self._key_lock:
                            if self.down is not None and self.down[0]==vk:
                                self._send(vk,True)
                                self.down=None  # only after acknowledged SendInput
                    if cancel.wait(gap_ms/1000):break
            except Exception as e:
                self.error=e
            finally:
                # A queued-but-unsent hard drop is NOT a real drop. Tell the
                # controller it must replan instead of waiting forever for a
                # lock acknowledgement that cannot arrive. A key-down that
                # was already sent remains acknowledged by live observation.
                if sent<len(keys):
                    self.last_aborted_keys=keys
                    self.abort_count+=1
                self._receipt(token,'finished',sent,len(keys),time.perf_counter())
        self.worker=threading.Thread(target=run,daemon=True,name='TetrioInput')
        self.worker.start()
        return token
    def cancel_pending(self):
        """Cancel all remaining keys on a newly observed board/vision loss."""
        self.release()

    def poll(self):
        if self.busy and not self.focused():self.release()
        if self.error:
            error=self.error;self.error=None;raise error
    def release(self):
        # Ensure a cancelled burst can never resume when the next one starts.
        self.cancel.set()
        worker=self.worker
        if worker and worker is not threading.current_thread():
            worker.join(timeout=.025)
        # Emergency release even if the worker is inside a slow Windows API.
        # A later worker finally-block sees down=None and will NOT send a
        # second key-up to an unrelated foreground application.
        with self._key_lock:
            if self.down:
                vk,_=self.down
                self._send(vk,True)
                self.down=None  # retain key state if SendInput failed
    def disarm(self):
        try:self.release()
        finally:
            self.target=None
            self.target_pid=None
            self.capture_regions=()
            self.capture_problems=[]
            if self.high_resolution:
                C.WinDLL('winmm').timeEndPeriod(1);self.high_resolution=False
