"""Opt-in, local TBP (Tetris Bot Protocol) adapter for Cold Clear 2.

Security boundary: a *separately installed, explicitly opted-in* local binary,
spawned without a shell; bounded JSON framing / wall clock; no network download,
no elevated permissions, no direct input authority. Every move is revalidated
against the observed board and the existing motion route checker.

This implements the TBP MVP 'info -> rules -> ready -> start -> suggest' cycle;
a fresh start is sent for each independently observed board, rather than
trusting a speculative play/new_piece sequence after visual desynchronization.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from engine import BASE, Move, PvpPlan, board_metrics, combat_profile, observed_queue_prefix
from motion import SHAPES, can_route, cells_at

ORIENTATIONS = ('north','east','south','west')
# x,y of TBP piece's true SRS rotation center relative to the corresponding
# motion.SHAPES internal origin (top-left, screen Y positive down).
ANCHORS = {
    'I': ((1,1),(2,1),(2,2),(1,2)),
    'O': ((0,1),(0,0),(1,0),(1,1)),
}
for _p in 'TJLSZ':
    ANCHORS[_p] = ((1,1),)*4
MAX_LINE = 65536
MAX_SUGGESTIONS = 64


def bundled_command():
    """Run the packaged subprocess with the already-installed Python/runtime.

    In frozen builds, a second bundled console EXE handles stdio safely.  The
    result is a true local TBP child; it is NOT Cold Clear 2 or an external
    open-source algorithm and must not be labelled as such.
    """
    if getattr(sys, 'frozen', False):
        return [str(Path(sys.executable).resolve().with_name('TetrioBundledBot.exe'))]
    return [str(Path(sys.executable).resolve()), '-u', str(Path(__file__).resolve().with_name('bundled_tbp_bot.py'))]


def binary_location():
    """The binary must be in a fixed app-local folder, never settings.json."""
    root = (Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False)
            else Path(__file__).resolve().parent)
    name = 'cold-clear-2.exe' if os.name == 'nt' else 'cold-clear-2'
    return root / 'external_solver' / name


def executable_is_ready(path=None):
    path = Path(path or binary_location())
    return path.is_file() and (os.name == 'nt' or os.access(path, os.X_OK))


def validate_snapshot(board, piece, queue_pieces, hold, allow_hold):
    if piece not in BASE or not isinstance(board, (tuple,list)) or len(board)!=20:
        raise ValueError('Invalid observed board/piece')
    if any(type(row) is not int or not (0 <= row <= 1023) for row in board):
        raise ValueError('Invalid board bitmask')
    confirmed = observed_queue_prefix(queue_pieces, limit=8)
    values=tuple(queue_pieces)
    if len(values)>8 or (values[:len(confirmed)]!=confirmed) or any(v in BASE for v in values[len(confirmed):]):
        raise ValueError('Unconfirmed queue')
    # TBP MVP cannot express an unknown or disabled HOLD. Never silently
    # substitute empty HOLD for '?' or allow a bot to choose a disabled swap.
    if hold == '?' or (hold not in BASE and hold is not None):
        raise ValueError('TBP requires a confirmed HOLD state')
    # A hold-disabled turn may still use a proposal for the CURRENT piece.
    # parse_tbp_move() will independently reject every HOLD swap when
    # allow_hold=False; do not reject the whole snapshot here.
    return confirmed


def snapshot_to_tbp(board, piece, queue_pieces, hold, b2b, combo):
    """Convert 20-row top-first masks to 40-row *bottom-first* TBP board."""
    confirmed = validate_snapshot(board, piece, queue_pieces, hold, True)
    filled = [[('G' if (board[y] >> x) & 1 else None) for x in range(10)]
              for y in range(19,-1,-1)]
    filled.extend([[None]*10 for _ in range(20)])
    return {'type':'start', 'hold':hold, 'queue':[piece,*confirmed],
            'combo':max(0,int(combo)+1), 'back_to_back':bool(b2b), 'board':filled}


def parse_tbp_move(board, current, confirmed, hold, allow_hold, raw):
    """Reject impossible, stale, malformed or unrouteable external suggestions."""
    if not isinstance(raw,dict):return None
    location=raw.get('location')
    if not isinstance(location,dict):return None
    p=location.get('type'); orientation=location.get('orientation')
    px=location.get('x'); py=location.get('y')
    spin=raw.get('spin','none')
    if p not in BASE or orientation not in ORIENTATIONS or spin not in ('none','mini','full'):
        return None
    if type(px) is not int or type(py) is not int or not (-5<=px<=15 and -5<=py<=44):
        return None
    use_hold = p != current
    if use_hold:
        expected = confirmed[0] if hold is None and confirmed else hold
        if not allow_hold or hold == '?' or p != expected:return None
    r=ORIENTATIONS.index(orientation)
    ax,ay=ANCHORS[p][r]
    cells=cells_at(p,(px-ax,19-py-ay,r))
    if len(cells)!=4 or len(set(cells))!=4:return None
    if any(not (0<=x<10 and 0<=y<20) or (board[y] & (1<<x)) for x,y in cells):return None
    # Drop target must be grounded, not a pose halfway down.
    if all(y+1<20 and not (board[y+1] & (1<<x)) for x,y in cells):return None
    requested_spin='' if spin=='none' else spin
    if requested_spin and p!='T':return None
    # The external AI never sends keys; actual SRS path feasibility is mandatory.
    if not can_route(board,p,cells,requested_spin):return None
    rows=list(board)
    for x,y in cells:rows[y]|=(1<<x)
    retained=[row for row in rows if row!=1023]
    lines=20-len(retained)
    after=tuple([0]*lines+retained)
    # A claimed spin cannot be rewarded without an actual cleared row.
    if requested_spin and not lines:return None
    return Move(p,r,tuple(sorted(cells)),after,lines,spin=requested_spin,tactical=bool(requested_spin)),use_hold


def as_pvp_plan(board,current,queue_pieces,hold,allow_hold,b2b,combo,incoming,raw_moves,
                revision=0,queue_revision=0,pieces_placed=0,sent_upper_bound=0,
                rejection_counts=None):
    confirmed=validate_snapshot(board,current,queue_pieces,hold,allow_hold)
    if not isinstance(raw_moves,list) or len(raw_moves)>MAX_SUGGESTIONS:return None
    if rejection_counts is not None:
        rejection_counts['received']=len(raw_moves)
    def rejected(reason):
        if rejection_counts is not None:
            rejection_counts[reason]=rejection_counts.get(reason,0)+1
    combat=combat_profile(board,incoming)[0]
    _,height,holes,*_=board_metrics(board)
    for candidate in raw_moves:
        decoded=parse_tbp_move(board,current,confirmed,hold,allow_hold,candidate)
        if decoded is None:
            rejected('invalid_pose_hold_or_srs')
            continue
        move,swapped=decoded
        _,height_after,holes_after,*_=board_metrics(move.board)
        if height_after>13 or height_after>height+4:
            rejected('unsafe_height')
            continue
        if holes_after>holes+1:
            rejected('unsafe_holes')
            continue
        # MVP TBP does not include pending garbage or TETR.IO-specific Surge.
        # Restrict to objectively low pressure, otherwise legacy defense wins.
        if incoming>0:
            if incoming>3 or height_after+incoming>13:
                rejected('unsafe_incoming')
                continue
            from engine import attack_lines
            if attack_lines(move,b2b,combo)<min(2,incoming):
                rejected('insufficient_counter')
                continue
        plan=PvpPlan(move,tuple(board),current,swapped,False,
                     confirmed[0] if confirmed else None,hold,b2b,combo,
                     revision,queue_revision,int(incoming),combat,
                     pieces_placed,sent_upper_bound)
        plan.continuation_source='external_tbp'
        plan.prepared_safe=False  # no NEXT-route proof from TBP
        return plan
    return None


class TBPError(Exception):
    pass


class TBPClient:
    """One Deep-worker TBP child with per-snapshot process isolation."""
    def __init__(self,command,handshake_timeout=.7,require_identity=True,expected_name='Cold Clear 2'):
        if not command or any(not isinstance(a,str) or not a for a in command):
            raise ValueError('Executable argv required')
        # Never invoke a shell or rely on PATH resolution for the executable.
        exe=Path(command[0])
        if not exe.is_absolute() or not exe.is_file():
            raise TBPError('Executable missing or not an absolute local path')
        self.command=tuple(command)
        self.handshake_timeout=handshake_timeout
        self.require_identity=require_identity
        self.expected_name=expected_name
        self.proc=None;self.messages=queue.Queue(maxsize=32)
        self.errors=queue.Queue(maxsize=2)
        self.active=False;self.snapshot_key=None

    def close(self):
        proc=self.proc
        self.proc=None;self.active=False;self.snapshot_key=None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=.5)
            except (OSError,subprocess.TimeoutExpired):
                pass
            for stream in (proc.stdin,proc.stdout):
                try:stream.close()
                except (OSError,AttributeError):pass

    def _reader(self,proc,messages,errors):
        try:
            while True:
                line=proc.stdout.readline(MAX_LINE+1)
                if not line:break
                if len(line)>MAX_LINE or not line.endswith(b'\n'):
                    raise TBPError('Oversized / unterminated TBP response')
                data=json.loads(line)
                if not isinstance(data,dict):raise TBPError('Invalid TBP JSON object')
                try:messages.put_nowait(data)
                except queue.Full:raise TBPError('TBP output flooded')
        except (ValueError,OSError,TBPError) as exc:
            try:errors.put_nowait(str(exc)[:120])
            except queue.Full:pass
        finally:
            try:errors.put_nowait('TBP process closed stdout')
            except queue.Full:pass

    def _send(self,message):
        if self.proc is None or self.proc.poll() is not None:raise TBPError('TBP process exited')
        data=json.dumps(message,separators=(',',':'),ensure_ascii=True).encode('ascii')+b'\n'
        if len(data)>MAX_LINE:raise TBPError('TBP request oversized')
        # Piped stdin may block indefinitely if an untrusted/buggy child stops
        # reading. Bound the write itself, not only the response wait. A timed-
        # out child is killed by the caller's fail-closed exception handler.
        finished=threading.Event()
        write_error=[]
        proc=self.proc
        def writer():
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except (OSError,BrokenPipeError,ValueError) as exc:
                write_error.append(exc)
            finally:
                finished.set()
        threading.Thread(target=writer,daemon=True,name='tbp-write').start()
        if not finished.wait(.06):raise TBPError('TBP stdin write deadline exceeded')
        if write_error:raise TBPError('TBP input closed') from write_error[0]

    def _wait(self,expected,deadline):
        while time.monotonic()<deadline:
            try:msg=self.messages.get(timeout=min(.01,max(.0005,deadline-time.monotonic())))
            except queue.Empty:
                if not self.errors.empty():raise TBPError(self.errors.get_nowait())
                continue
            if msg.get('type')=='error':raise TBPError('TBP bot refused the request')
            if msg.get('type')==expected:return msg
        raise TBPError('TBP '+expected+' deadline exceeded')

    def launch(self):
        if self.proc is not None and self.proc.poll() is None:return
        self.close()
        self.messages=queue.Queue(maxsize=32);self.errors=queue.Queue(maxsize=2)
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        try:
            self.proc=subprocess.Popen(self.command,stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                cwd=str(Path(self.command[0]).parent),
                shell=False,bufsize=0,creationflags=flags)
        except (OSError,ValueError) as exc:raise TBPError('TBP binary failed to start') from exc
        threading.Thread(target=self._reader,args=(self.proc,self.messages,self.errors),daemon=True,name='tbp-read').start()
        deadline=time.monotonic()+self.handshake_timeout
        try:
            info=self._wait('info',deadline)
            if self.require_identity and info.get('name')!=self.expected_name:
                raise TBPError('Unexpected TBP solver identity')
            self._send({'type':'rules'})
            self._wait('ready',deadline)
        except TBPError:
            self.close();raise

    def suggest(self,snapshot,timeout=.08,preserve_on_timeout=False):
        """Poll until a non-empty TBP suggestion or a *hard* deadline.

        Cold Clear 2 may reply with moves=[] while its asynchronous search is
        starting. An empty reply is NOT a definitive illegal position. Never
        expose a speculative / stale result: changed snapshots always use a
        fresh child. Repeated *identical* snapshots may opt to keep searching
        on timeout. The caller still enforces each bounded polling deadline.
        """
        timeout=max(.005,min(float(timeout),5.0))
        self.last_empty_suggestions=0
        self.last_suggest_status='pending'
        try:
            self.launch()
            snapshot_key=json.dumps(snapshot,sort_keys=True,separators=(',',':'))
            same_snapshot=self.active and self.snapshot_key==snapshot_key
            if self.active and not same_snapshot:
                # TBP has no request identifier; a second different board
                # MUST be isolated from delayed suggestions for the first.
                self.close()
                self.launch()
            if not same_snapshot:
                self._send(snapshot)
                self.active=True
                self.snapshot_key=snapshot_key
            # For one exact snapshot, opt-in CC2-only retry may continue the
            # SAME search tree. A timed-out search is never allowed to migrate
            # to another board. This fixes resets after every empty response.
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                self._send({'type':'suggest'})
                try:
                    response=self._wait('suggestion',min(deadline,time.monotonic()+.025))
                except TBPError as exc:
                    if 'deadline exceeded' not in str(exc):raise
                    continue
                moves=response.get('moves')
                if not isinstance(moves,list) or len(moves)>MAX_SUGGESTIONS:
                    raise TBPError('Malformed or oversized TBP suggestions')
                if moves:
                    self.last_suggest_status='received'
                    return moves
                # Cold Clear 2 can return an empty list before it has
                # searched enough; don't falsely declare verification failure.
                self.last_empty_suggestions+=1
                time.sleep(min(.018,max(0,deadline-time.monotonic())))
            self.last_suggest_status=('empty_until_deadline' if
                                      self.last_empty_suggestions else 'no_response')
            if not preserve_on_timeout:self.close()
            return [] if self.last_empty_suggestions else None
        except (TBPError,OSError,ValueError) as exc:
            self.last_suggest_status=('protocol_error:'+str(exc)[:80])
            self.close()
            return None


# The Deep pool has one process: keep the subprocess warm between observed turns.
_WORKER_CLIENT=None
_WORKER_EXECUTABLE=None


def close_external_worker():
    """Bounded explicit release when the user stops an opted-in session."""
    global _WORKER_CLIENT, _WORKER_EXECUTABLE
    if _WORKER_CLIENT is not None:
        _WORKER_CLIENT.close()
    _WORKER_CLIENT=None
    _WORKER_EXECUTABLE=None
    return 'CLOSED'


def prewarm_external_worker():
    """Initialize the opt-in local solver in the Deep process, without input.

    READY proves only a TBP handshake. It does NOT imply a playable move.
    A failed handshake must never be mistaken for CC2 being ready.
    """
    global _WORKER_CLIENT, _WORKER_EXECUTABLE
    executable=binary_location()
    if not executable_is_ready(executable):
        return 'EXECUTABLE_MISSING'
    try:
        if _WORKER_CLIENT is None or _WORKER_EXECUTABLE!=str(executable):
            if _WORKER_CLIENT is not None:
                _WORKER_CLIENT.close()
            _WORKER_CLIENT=TBPClient([str(executable)])
            _WORKER_EXECUTABLE=str(executable)
        _WORKER_CLIENT.launch()
        return 'READY'
    except (TBPError,OSError,ValueError,TypeError):
        if _WORKER_CLIENT is not None:
            _WORKER_CLIENT.close()
        _WORKER_CLIENT=None
        _WORKER_EXECUTABLE=None
        return 'HANDSHAKE_ERROR'


def external_plan_diagnostic(args, timeout=.30, rejection_counts=None):
    """Return a validated CC2 plan plus a bounded, actionable status code.

    This is only a local diagnosis: an installed EXE, protocol output and a
    legally executable placement are different things. No direct key input.
    No automatic fallback to internal strategy is possible here.
    """
    global _WORKER_CLIENT,_WORKER_EXECUTABLE
    (board,piece,queue_pieces,hold,allow_hold,b2b,combo,incoming,budget_ms,
     revision,queue_revision,pieces_placed,sent_upper_bound,*rest)=args
    if incoming>3:return None,'INCOMING_TOO_HIGH'
    if hold=='?':return None,'HOLD_UNKNOWN'
    executable=binary_location()
    if not executable_is_ready(executable):return None,'EXECUTABLE_MISSING'
    try:
        snapshot=snapshot_to_tbp(board,piece,queue_pieces,hold,b2b,combo)
    except (ValueError,TypeError):
        return None,'INVALID_OBSERVATION'
    try:
        if _WORKER_CLIENT is None or _WORKER_EXECUTABLE!=str(executable):
            if _WORKER_CLIENT is not None:_WORKER_CLIENT.close()
            _WORKER_CLIENT=TBPClient([str(executable)])
            _WORKER_EXECUTABLE=str(executable)
        budget=max(.02,min(.5,float(timeout)))
        moves=_WORKER_CLIENT.suggest(snapshot,timeout=budget,
                                     preserve_on_timeout=budget>=.30)
        if moves is None:
            status=str(getattr(_WORKER_CLIENT,'last_suggest_status','no_response'))
            if status.startswith('protocol_error:'):
                return None,'PROTOCOL_ERROR:'+status[15:90]
            return None,'NO_REPLY'
        if not moves:return None,'NO_SUGGESTION'
        plan=as_pvp_plan(board,piece,queue_pieces,hold,allow_hold,b2b,combo,
                         incoming,moves,revision,queue_revision,pieces_placed,sent_upper_bound,
                         rejection_counts=rejection_counts)
        if plan is None:return None,'NO_VALID_ROUTE'
        return plan,'PASS'
    except (ValueError,TypeError,TBPError,OSError) as exc:
        if _WORKER_CLIENT is not None:_WORKER_CLIENT.close()
        return None,'BRIDGE_ERROR:'+type(exc).__name__


def maybe_external_plan(args,enabled=False):
    """Validated opt-in CC2 move. Hybrid retains its original fast deadline."""
    if not enabled:return None
    plan,_=external_plan_diagnostic(args,timeout=.075)
    return plan


# One bundled child per Deep worker, reused for successive independently seen
# boards. The command is a fixed application-local script or the frozen app,
# never user input, an arbitrary PATH search, a shell or a network URL.
_BUNDLED_CLIENT=None


def maybe_bundled_plan(args,enabled=False):
    """Usable opt-in independent process, even if Cold Clear 2 is absent.

    The bot uses our bundled Python attack policy, NOT CC2. UNKNOWN HOLD is
    represented to that bot as unavailable and any swap is rejected here.
    All proposals must pass geometry, route and currently observed revision.
    """
    global _BUNDLED_CLIENT
    if not enabled:return None
    (board,piece,queue_pieces,hold,allow_hold,b2b,combo,incoming,budget_ms,
     revision,queue_revision,pieces_placed,sent_upper_bound,*rest)=args
    if incoming>8:return None
    if budget_ms<6:return None
    confirmed_hold=hold in BASE or hold is None
    allow_swap=bool(allow_hold and confirmed_hold)
    seen_hold=hold if confirmed_hold else None
    try:
        snapshot=snapshot_to_tbp(board,piece,queue_pieces,seen_hold,b2b,combo)
    except (ValueError,TypeError):return None
    snapshot['assist_hold_allowed']=allow_swap
    snapshot['assist_b2b_count']=max(0,min(1000,int(b2b)))
    snapshot['assist_combo']=max(-1,min(1000,int(combo)))
    snapshot['assist_incoming']=max(0,min(20,int(incoming)))
    try:
        if _BUNDLED_CLIENT is None:
            _BUNDLED_CLIENT=TBPClient(bundled_command(),handshake_timeout=1.8,
                                      expected_name='TETRIO Bundled Strategy')
        moves=_BUNDLED_CLIENT.suggest(snapshot,timeout=.15)
        if not isinstance(moves,list) or len(moves)>MAX_SUGGESTIONS:return None
        from engine import attack_lines
        _,old_height,old_holes,*_=board_metrics(board)
        for raw in moves:
            decoded=parse_tbp_move(board,piece,observed_queue_prefix(queue_pieces,8),seen_hold,allow_swap,raw)
            if decoded is None:continue
            move,swap=decoded
            if swap and not allow_swap:continue
            _,height,holes,*_=board_metrics(move.board)
            if height>16 or height>old_height+4 or holes>old_holes+1:continue
            atk=attack_lines(move,b2b,combo)
            if incoming and height+max(0,incoming-atk)>18:continue
            combat=combat_profile(board,incoming)[0]
            plan=PvpPlan(move,tuple(board),piece,swap,False,
                         queue_pieces[0] if queue_pieces else None,hold,b2b,combo,
                         revision,queue_revision,int(incoming),combat,
                         pieces_placed,sent_upper_bound)
            plan.continuation_source='bundled_tbp'
            plan.prepared_safe=False
            return plan
        return None
    except (TBPError,ValueError,TypeError,OSError,OverflowError):
        if _BUNDLED_CLIENT is not None:_BUNDLED_CLIENT.close()
        _BUNDLED_CLIENT=None
        return None
