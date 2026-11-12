"""Armed auto play: fresh observation -> bounded input -> acknowledgement.
Transient failures pause inputs and preserve the user's F9 switch.
"""
from engine import norm, projected_cells, Plan
from opening import opening_decision, OPENING_TARGET, OPENING_PIECES, OPENING_SECONDS
from garbage_tracking import EMPTY_MASK, advance_mask
from garbage_transition import verified_rise, mask_after_rise
from transition_guard import classify_post_drop
from attack_rate import AttackRateWindow


def garbage_rise_rows(expected, observed):
    """Compatibility wrapper: exact visible 1-hole garbage translation only."""
    return verified_rise(expected, observed)


class AutoPlayer:
    def __init__(self, turbo=False):
        self.turbo = turbo
        self.route = None
        self.route_pending = None
        # A 180 with an SRS+ displacement must be confirmed at its actual
        # expected pose. Visual shape + column alone cannot validate a kick.
        self.route_pending_180 = False
        self.route_prev_cells = None
        self.route_ack_reject_reported = False
        self.route_soft_active = False
        self.route_soft_started = None
        self.route_soft_pulses = 0
        self.prepared = None
        self.expected_prepared = None
        self.plan_is_prepared = False
        self.prepared_submitted = 0
        self.prepared_started = 0
        self.prepared_locked = 0
        self.deep_prepared_locked = 0  # legacy exact-"deep" compatibility
        self.prepared_locks_by_source={}
        self.replans_by_reason={}
        self.prepared_failed = 0
        self.enabled = False
        self.serial = 0
        self.held = '?'
        self.phase = 'idle'
        self.plan = None
        self.hold_used = False
        self.pending = None
        self.stable = None
        self.stable_count = 0
        self.paused = False
        self.pause_since = None
        self.rejected_plan = None
        self.message = '자동 OFF'
        self.not_before = 0
        self.failed_signature = None
        self.failed_count = 0
        self.recovery_retry_signature=None
        self.recovery_retry_count=0
        self.recovery_retry_total=0
        self.recovery_blocked_target=None
        self.recovery_blocked_since=None
        self.hold_uncertain = False
        self.verified_holds = 0
        self.first_seen = None
        self.first_lock = None
        self.confirmed_locks = 0
        self.extra_pulse_ms = 0
        self.input_failures = 0
        self.soft_failures = 0
        self.soft_disabled = False  # stop repeatedly commanding a non-sonic SDF setup
        self.input_success_streak = 0
        self.verified_input_streak = 0
        self.input_recoveries = 0
        self.b2b_count = 0
        self.b2b_local_floor = 0  # lower bound under locally observed locks only
        self.b2b_unverified_spins = 0  # server never independently adjudicated
        self.rotation_ack_rejects = 0
        self.rotation_180_suppressed = 0
        self.b2b_max_observed = 0
        self.b2b_breaks_observed = 0
        self.b2b_difficult_observed = 0
        self.b2b_guard_verified_locks = 0
        self.all_mini_planned_locks = 0  # confirmed placement, NOT server spin adjudication
        self.combo = -1
        self.sent_attack_upper_bound = 0
        self.attack_window = AttackRateWindow()
        self.last_attack_sample_at = None
        self.ruleset = 'standard'
        self.combo_table = 'multiplier'
        self.rounding = 'auto'
        self.garbage_rise_events = 0
        self.garbage_fast_replans = 0
        self.unverified_board_resyncs = 0
        self.post_drop_rise_resyncs = 0
        self.post_drop_fresh_resyncs = 0
        self.garbage_drop_ack_ms = 0.0
        self.garbage_rise_lines = 0
        self.garbage_mask = EMPTY_MASK
        self.last_garbage_rise_at = None
        self.drop_confirmation = None
        self.drop_guard_checks = 0
        self.drop_guard_rejected = 0
        self.drop_unacknowledged_replans = 0
        self.drop_ambiguous_watchdog_stops = 0
        self.drop_delivery='none'  # legacy/queued/dispatched/none
        self.drop_token=None
        self.drop_dispatched_count=0
        self.drop_unsent_resyncs=0
        self.last_os_to_lock_ms=None  # real SendInput key-down -> observed lock; else UNKNOWN
        self.input_partial_resyncs=0
        self.last_lock_source = 'fast'
        self.last_lock_prefetch_source = None
        # One-shot opening campaign: automatically ends and never reactivates
        # during this F9 session. Estimate gross generation, not enemy delivery.
        self.opening_enabled = True
        self.opening_target = OPENING_TARGET
        self.opening_eligible = None
        self.opening_finished = False
        self.opening_reason = 'waiting'
        self.opening_generated = 0
        self.opening_adaptive_enabled = True
        self.opener_confirmed = 0
        self.opener_last = ''

    def start(self, held='?'):
        serial = self.serial + 1
        self.__init__(self.turbo)
        self.serial = serial
        self.enabled = True
        self.held = held
        self.message = '자동 ON · 인식 대기'

    def stop(self, reason='사용자 중지'):
        self.prepared = None
        self.enabled = False
        self.phase = 'idle'
        self.plan = None
        self.pending = None
        self.route = None
        self.route_pending = None
        # A 180 with an SRS+ displacement must be confirmed at its actual
        # expected pose. Visual shape + column alone cannot validate a kick.
        self.route_pending_180 = False
        self.route_prev_cells = None
        self.route_ack_reject_reported = False
        self.route_soft_active = False
        self.drop_confirmation = None
        self.drop_delivery='none'
        self.drop_token=None
        self.paused = False
        self.message = '자동 OFF · ' + reason

    def pause(self, reason, now=None):
        if not self.enabled:
            return
        if not self.paused:
            self.pause_since = now
            self.stable = None
            self.stable_count = 0
        self.paused = True
        self.message = '자동 ON · 대기: ' + reason

    def snapshot(self):
        return dict(enabled=self.enabled, phase=self.phase, paused=self.paused,
                    message=self.message, hold=self.held, hold_used=self.hold_used,
                    verified_holds=self.verified_holds,
                    stable_frames=self.stable_count, revision=self.serial,
                    confirmed_locks=self.confirmed_locks, first_seen=self.first_seen,
                    first_lock=self.first_lock, input_failures=self.input_failures,
                    extra_pulse_ms=self.extra_pulse_ms, input_success_streak=self.input_success_streak,
                    input_recoveries=self.input_recoveries,
                    verified_input_streak=self.verified_input_streak,
                    b2b_count=self.b2b_count,
                    b2b_local_floor=self.b2b_local_floor,
                    b2b_unverified_spins=self.b2b_unverified_spins,
                    rotation_ack_rejects=self.rotation_ack_rejects,
                    rotation_180_suppressed=self.rotation_180_suppressed,
                    b2b_max_observed=self.b2b_max_observed,
                    b2b_breaks_observed=self.b2b_breaks_observed,
                    b2b_difficult_observed=self.b2b_difficult_observed,
                    b2b_guard_verified_locks=self.b2b_guard_verified_locks,
                    all_mini_planned_locks=self.all_mini_planned_locks,
                    combo=self.combo, gross_attack_est=self.sent_attack_upper_bound,
                    garbage_rise_events=self.garbage_rise_events, garbage_rise_lines=self.garbage_rise_lines,
                    garbage_fast_replans=self.garbage_fast_replans,
                    unverified_board_resyncs=self.unverified_board_resyncs,
                    post_drop_rise_resyncs=self.post_drop_rise_resyncs,
                    post_drop_fresh_resyncs=self.post_drop_fresh_resyncs,
                    garbage_drop_ack_ms=round(self.garbage_drop_ack_ms, 2),
                    tracked_garbage_minos=sum(r.bit_count() for r in self.garbage_mask),
                    opening_generated=self.opening_generated, opening_target=self.opening_target,
                    opening_finished=self.opening_finished, opening_reason=self.opening_reason,
                    opening_adaptive_enabled=self.opening_adaptive_enabled,
                    opener_confirmed=self.opener_confirmed, opener_last=self.opener_last,
                    prepared_submitted=self.prepared_submitted,prepared_started=self.prepared_started,
                    prepared_locked=self.prepared_locked,deep_prepared_locked=self.deep_prepared_locked,
                    prepared_locks_by_source=dict(self.prepared_locks_by_source),
                    replans_by_reason=dict(self.replans_by_reason),
                    prepared_failed=self.prepared_failed,
                    recovery_retry_total=self.recovery_retry_total,
                    soft_failures=self.soft_failures,soft_disabled=self.soft_disabled,
                    drop_guard_checks=self.drop_guard_checks,
                    drop_guard_rejected=self.drop_guard_rejected,
                    drop_unacknowledged_replans=self.drop_unacknowledged_replans,
                    drop_ambiguous_watchdog_stops=self.drop_ambiguous_watchdog_stops,
                    drop_delivery=self.drop_delivery,
                    drop_dispatched_count=self.drop_dispatched_count,
                    last_os_to_lock_ms=self.last_os_to_lock_ms,
                    drop_unsent_resyncs=self.drop_unsent_resyncs,
                    input_partial_resyncs=self.input_partial_resyncs,
                    modeled_attack=self.attack_window.snapshot(self.last_attack_sample_at).modeled_gross
                        if getattr(self,'last_attack_sample_at',None) is not None else 0.0)

    def attack_pace(self, now, incoming=0, safe=True):
        return self.attack_window.snapshot(now,incoming=incoming,safe=safe)

    def pace(self,now):
        elapsed=0 if self.first_seen is None else max(0,now-self.first_seen)
        return dict(target_pps=1.20, confirmed_locks=self.confirmed_locks,
                    elapsed=elapsed, pps=self.confirmed_locks/elapsed if elapsed>=1 else None,
                    first_lock_ms=None if self.first_lock is None else (self.first_lock-self.first_seen)*1000)


    def prime_opening(self, board, now):
        """Prime first-board evidence before HOLD and Fast planning.

        A zero settled board is only a heuristic, never a verified round-start
        signal. This does not re-arm once the first state was observed.
        """
        if not self.enabled or self.first_seen is not None:
            return False
        self.first_seen=now
        self.attack_window.start(now)
        self.opening_eligible=not any(board)
        if not self.opening_eligible:self.opening_reason='midgame'
        return self.opening_eligible

    def opening_status(self, now, incoming=0, board=None):
        if not self.enabled or not self.opening_enabled or self.opening_finished:
            return False
        # A physically verified garbage rise or an unexplained board resync
        # invalidates the opening premise for the rest of this F9 session.
        if self.garbage_rise_events or self.unverified_board_resyncs:
            self.opening_finished=True
            self.opening_reason='garbage' if self.garbage_rise_events else 'resync'
            return False
        # Keep headroom: an opener cannot outrank emergency survival.
        if board is not None and any(int(row) for row in board[:9]):
            self.opening_finished=True;self.opening_reason='danger'
            return False
        elapsed=(now-self.first_seen) if self.first_seen is not None else 0.0
        decision=opening_decision(self.opening_eligible is True,self.opening_finished,
                                  self.confirmed_locks,elapsed,self.opening_generated,
                                  incoming=incoming,target=self.opening_target)
        if decision.reason not in ('burst','off') and self.opening_eligible is True:
            self.opening_finished=True
        self.opening_reason=decision.reason
        return decision.active

    def _record_lock(self, move, now=None):
        """Advance the locally observable competitive chain state."""
        from engine import attack_lines
        # Upper bound: the true server-sent amount is no larger than gross
        # generated attack if pending garbage has consumed some of it.
        from combo_rules import attack_estimate
        # Include the ceiling under RNG rounding: this must remain an *upper*
        # bound for conservative early-cancellation eligibility.
        estimate = attack_estimate(move,self.b2b_count,self.combo,self.ruleset,
                                   self.combo_table,self.rounding)
        previous_b2b=self.b2b_count
        if move.lines:
            if estimate.next_b2b>previous_b2b:
                self.b2b_difficult_observed+=1
            elif previous_b2b>0 and estimate.next_b2b==0:
                self.b2b_breaks_observed+=1
        self.b2b_max_observed=max(self.b2b_max_observed,estimate.next_b2b)
        self.sent_attack_upper_bound += estimate.ceiling
        # Only acknowledge telemetry after the board-confirmed lock. A locally
        # labelled spin is a hypothesis, not a server-confirmed spin attack.
        if now is not None:
            self.last_attack_sample_at=now
            self.attack_window.record(now, estimate,
                rule_certain=(not bool(move.spin)))
        if self.opening_eligible and not self.opening_finished:
            self.opening_generated += estimate.guaranteed
            if self.opening_generated >= self.opening_target:
                self.opening_finished=True
                self.opening_reason='target'
        if move.lines:
            self.combo += 1
            difficult = move.lines == 4 or bool(move.spin) or not any(move.board)
            increment = 2 if self.ruleset=='quick_play' and not any(move.board) else 1
            if difficult:
                self.b2b_count += increment
                # Quad/all clear is a difficult clear regardless of whether
                # a predicted spin label is accepted by the game. Ordinary
                # spin-only difficulty is a model hypothesis, NOT a verified
                # game-server result. Never invent a verified B2B from it.
                if move.lines==4 or not any(move.board):
                    self.b2b_local_floor += increment
                else:
                    self.b2b_local_floor = 0
                    self.b2b_unverified_spins += 1
            else:
                self.b2b_count = 0
                self.b2b_local_floor = 0
        else:
            self.combo = -1

    def _replan(self, external_plan, reason, now, hold_used=None, verified_transition=False):
        # Distinguish observed transitions from uncertain geometry/input
        # without logging screenshots, text entry, or arbitrary user data.
        classification=('garbage' if 'Garbage' in reason or 'garbage' in reason else
                        'confirmed_lock' if '착지 확인' in reason else
                        'input' if '입력' in reason or '경로' in reason or '드롭' in reason else
                        'vision_or_board')
        self.replans_by_reason[classification]=self.replans_by_reason.get(classification,0)+1
        if self.plan_is_prepared:
            self.prepared_failed+=1
        self.plan_is_prepared=False
        self.expected_prepared=None
        self.prepared = None
        self.rejected_plan = external_plan
        self.plan = None
        self.phase = 'idle'
        self.pending = None
        self.route = None
        self.route_pending = None
        # A 180 with an SRS+ displacement must be confirmed at its actual
        # expected pose. Visual shape + column alone cannot validate a kick.
        self.route_pending_180 = False
        self.route_prev_cells = None
        self.route_ack_reject_reported = False
        self.route_soft_active = False
        self.drop_confirmation = None
        self.drop_delivery='none'
        self.drop_token=None
        if hold_used is not None:
            self.hold_used = hold_used
        self.serial += 1
        self.not_before = now if verified_transition else now + (0 if self.turbo else .016)
        self.message = '자동 ON · 재판단: ' + reason

    def _authorize_drop(self, state, now, pressure=False):
        """Double-observe the target under incoming/recent verified garbage.

        The ordinary low-pressure path retains its original timing. Pressure
        requires a second newly captured independent state at the same pose;
        no speculative hard-drop command is queued before that evidence.
        """
        board,piece,cells=state
        target=tuple(sorted(self.plan.move.cells))
        observed=tuple(sorted(cells))
        # On a second guarded frame, NEVER rely solely on unchanged piece
        # silhouette/x. A physical garbage shift or an invalidated target
        # can change the projected landing without changing the silhouette.
        # Recheck against the freshly observed fixed stack every time.
        if (projected_cells(board,cells)!=target or
                (self.plan.move.spin and observed!=target)):
            self.drop_confirmation=None
            self.drop_guard_rejected+=1
            self.message='자동 ON · 착지 예상 변경 · 하드드롭 보류'
            return None
        recent=(self.last_garbage_rise_at is not None and
                now-self.last_garbage_rise_at<1.0)
        # Once a two-observation confirmation has started, a transient meter
        # confidence change cannot silently downgrade the pending hard drop
        # into an unguarded single-frame authorization.
        need_guard=bool(pressure or recent or self.drop_confirmation is not None)
        if need_guard:
            ticket=(self.serial,board,piece,norm(cells),
                    min(x for x,y in cells),
                    observed if self.plan.move.spin else target)
            if self.drop_confirmation is None or self.drop_confirmation[0]!=ticket:
                if self.drop_confirmation is not None:
                    self.drop_guard_rejected+=1
                self.drop_confirmation=(ticket,now)
                self.drop_guard_checks+=1
                self.message='자동 ON · 압박 중 하드드롭 재확인'
                return None
            if now<=self.drop_confirmation[1]:
                return None
        self.drop_confirmation=None
        self.phase='drop_wait'
        self.before_drop=(board,piece,observed)
        # The controller has only REQUESTED a drop at this point.  The GUI
        # records the real OS key-down receipt separately on Windows.
        self.sent_at=now
        self.drop_delivery='legacy'
        self.drop_token=None
        self.saw_gap=False
        self.message='자동 ON · 하드드롭 전송 준비'
        return 'drop'

    def note_drop_queued(self, token, now):
        if self.phase!='drop_wait' or token is None:return False
        self.drop_token=token
        self.drop_delivery='queued'
        self.sent_at=now
        self.message='자동 ON · 하드드롭 전송 대기'
        return True

    def note_drop_dispatched(self, token, at):
        if (self.phase!='drop_wait' or self.drop_token!=token
                or self.drop_delivery!='queued'):
            return False
        # SendInput success is evidence of OS key-down only, NEVER of a lock.
        self.drop_delivery='dispatched'
        self.sent_at=at
        self.drop_dispatched_count+=1
        self.message='자동 ON · 하드드롭 착수 확인 대기'
        return True

    def note_drop_unsent(self, token, now, external_plan=None):
        if (self.phase!='drop_wait' or self.drop_token!=token
                or self.drop_delivery!='queued'):
            return False
        self.drop_unsent_resyncs+=1
        self._replan(external_plan,'DROP 키다운 전 취소 · 현재 화면 재판단',now,
                     self.hold_used,verified_transition=False)
        return True

    def _acknowledged_input(self):
        """Back off a temporary pulse penalty after repeated *observed* success.

        A successful calculation or command submission does not count as an
        input acknowledgement. Only a changed board/piece/pose does.
        """
        self.verified_input_streak=min(255,self.verified_input_streak+1)
        if not self.turbo or not self.extra_pulse_ms:
            return
        self.input_success_streak += 1
        if self.input_success_streak >= 6:
            self.extra_pulse_ms = max(0, self.extra_pulse_ms - 2)
            self.input_recoveries += 1
            self.input_success_streak = 0

    def _failed_input(self, state, external_plan, now, reason):
        self.input_failures += 1
        self.input_success_streak = 0
        self.verified_input_streak = 0
        # Geometric/routing failures are not evidence that a key pulse was lost.
        # Do not permanently slow the input driver because a route was blocked.
        missed_input = reason not in (
            '도달 가능한 경로 재계산', '착지 경로 재확인', '최종 착지 재확인',
            '배치 지연·실제 위치로 재계산')
        if self.turbo and missed_input:
            self.extra_pulse_ms=min(12,self.extra_pulse_ms+4)
        board, piece, cells = state
        signature = (board, piece, norm(cells), min(x for x, y in cells))
        if signature == self.failed_signature:
            self.failed_count += 1
        else:
            self.failed_signature = signature
            self.failed_count = 1
        if reason=='소닉드롭 경로 미완료':
            self.soft_failures += 1
            # Two independently observed failed descents are stronger evidence
            # than blindly retrying DOWN indefinitely. This is session-local.
            if self.soft_failures >= 2:
                self.soft_disabled=True
        self._replan(external_plan, reason, now)
        # A separately verified geometric/path failure is not a lost key
        # pulse. Long backoff here used to amplify garbage replan stalls.
        recent_garbage=(self.last_garbage_rise_at is not None and
                        now-self.last_garbage_rise_at<1.0)
        if not missed_input:
            self.not_before=now+.016
        elif recent_garbage and self.failed_count<3:
            # Fast retry is allowed only after a verified physical rise and
            # new plan/pose acknowledgement; bounded failed-route protection
            # still applies. Never blindly repeat the old hard drop.
            self.not_before=now+.10
        else:
            self.not_before=now + (.75 if self.failed_count>=3 else .03) if self.turbo else now + .25
        if self.soft_disabled:
            self.message='자동 ON · 소닉드롭 확인 실패 · 안전 일반 배치 전환 · 게임 SDF ∞ 확인'

    def observe(self, state, external_plan, now, foreground=True, pressure=False):
        if not self.enabled:
            return None
        if not foreground:
            self.pause('게임 창으로 돌아오면 재개', now)
            return None
        if state is None:
            if self.drop_confirmation is not None:
                self.drop_guard_rejected+=1
                self.drop_confirmation=None
            if self.phase == 'drop_wait':
                self.saw_gap = True
            self.pause('블록 재인식', now)
            self.stable = None
            self.stable_count = 0
            return None

        board, piece, cells = state
        if self.drop_confirmation is not None:
            ticket=self.drop_confirmation[0]
            pose=(self.serial,board,piece,norm(cells),min(x for x,y in cells),
                  tuple(sorted(cells)) if self.plan is not None and self.plan.move.spin
                  else (tuple(sorted(self.plan.move.cells)) if self.plan is not None else ()))
            if ticket!=pose:
                self.drop_guard_rejected+=1
                self.drop_confirmation=None
        if self.first_seen is None:
            self.prime_opening(board,now)
            # F9 pressed during an existing match must not force an opener.
            # An initially empty settled board is the only local evidence we
            # have of a fresh match; all other starts stay in normal PvP mode.
        signature = (board, piece, norm(cells), min(x for x, y in cells))
        if signature != self.recovery_retry_signature:
            self.recovery_retry_signature=signature
            self.recovery_retry_count=0
            self.recovery_blocked_target=None
            self.recovery_blocked_since=None
        if signature == self.stable:
            self.stable_count += 1
        else:
            self.stable = signature
            self.stable_count = 1

        if self.paused:
            # Only an exact physically verified shifted board permits faster
            # rejoin; all other vision gaps keep the old 3-frame gate.
            rise_ok=(self.plan is not None and
                     verified_rise(self.plan.source_board,board) not in (None,0)
                     and piece==self.plan.move.piece)
            if self.stable_count < (2 if rise_ok else 3):
                return None
            if self.pause_since is not None:
                duration = max(0, now - self.pause_since)
                for name in ('sent_at', 'started_at'):
                    if hasattr(self, name):
                        setattr(self, name, getattr(self, name) + duration)
            self.paused = False
            self.pause_since = None
            self.message = '자동 ON · 재인식 완료'

        if self.phase == 'drop_wait':
            # A queued command is not proof of even one Windows key-down.
            # Avoid scoring an unrelated visual transition as our lock.
            if self.drop_delivery=='queued':
                # If the worker has finished but its receipt was lost, there
                # is no safe basis to treat an observed transition as our
                # lock, or to resend a possibly dispatched SPACE. A bounded
                # explicit stop avoids a permanent silent WAIT. This branch
                # is reached only after the GUI has returned to observe() --
                # the normal worker-busy gate prevents evaluating it mid-send.
                if self.stable_count>=3 and now-self.sent_at>=1.5:
                    self.drop_ambiguous_watchdog_stops+=1
                    self.stop('DROP 전송 결과 1.5초 미확인 · 중복 착수 방지 자동 중지 · 진단 저장')
                else:
                    self.message='자동 ON · DROP 전송 확인 중'
                return None
            completed_source=self.plan.source_board
            transition=classify_post_drop(
                completed_source,self.plan.move.board,
                self.before_drop[1],self.before_drop[2],board,piece,cells,
                stable_frames=self.stable_count,
                elapsed_ms=max(0,(now-self.sent_at)*1000),
                saw_gap=self.saw_gap)
            inserted=transition.inserted
            if transition.kind=='confirmed_lock':
                completed=self.plan
                self.last_lock_source=completed.continuation_source or 'fast'
                self.last_lock_prefetch_source=getattr(completed,'prefetch_source',None)
                if self.last_lock_source in ('b2b_guard','all_mini_b2b_rescue'):
                    self.b2b_guard_verified_locks+=1
                if completed.move.piece!='T' and completed.move.spin=='mini' and completed.move.lines>0:
                    self.all_mini_planned_locks+=1
                following=completed.move.next_move
                opening_profile=getattr(completed,'opening_profile','')
                if opening_profile and self.opening_eligible and self.opening_enabled:
                    self.opener_confirmed+=1
                    self.opener_last=opening_profile
                lock_ack_ms=max(0.0,(now-self.sent_at)*1000.0)
                self.last_os_to_lock_ms=(lock_ack_ms if self.drop_delivery=='dispatched' else None)
                self._record_lock(completed.move,now)
                self.last_locked_piece=completed.move.piece
                if self.plan_is_prepared:
                    self.prepared_locked+=1
                    actual_source=completed.continuation_source or 'fast'
                    self.prepared_locks_by_source[actual_source]=(self.prepared_locks_by_source.get(actual_source,0)+1)
                    if actual_source=='deep':self.deep_prepared_locked+=1
                    self.plan_is_prepared=False
                # Keep a separate mask only for *confirmed* incoming garbage.
                # Normal blocks and garbage without a verified rise remain
                # unknown, rather than corrupting the strategic board state.
                self.garbage_mask=advance_mask(completed.source_board,
                                               self.garbage_mask,completed.move.cells,
                                               inserted,board if inserted else None)
                if inserted:
                    self.garbage_rise_events += 1
                    self.last_garbage_rise_at=now
                    self.garbage_rise_lines += inserted
                self._acknowledged_input()  # verified hard-drop/lock
                self.confirmed_locks+=1
                if self.first_lock is None:self.first_lock=now
                self.failed_count = 0
                self._replan(external_plan, '착지 확인·다음 블록', now, False, verified_transition=True)
                if inserted:
                    self.garbage_fast_replans += 1
                    self.garbage_drop_ack_ms += lock_ack_ms
                if (not inserted and completed.prepared_safe and following is not None
                        and following.piece==piece):
                    self.prepared=Plan(following,board,piece,False,True,
                                       None,self.held,self.b2b_count,self.combo,
                                       continuation_source=(getattr(completed,'prefetch_source',None)
                                                            or completed.continuation_source))
            elif transition.kind=='risen_active':
                # The old active piece is still visible on an exact garbage-
                # raised source stack, even if it fell far below row five.
                # The hard drop was NOT acknowledged, so neither B2B nor the
                # planned clear can be credited. Same-turn HOLD stays used.
                rise=transition.inserted
                try:
                    self.garbage_mask=mask_after_rise(completed_source,self.garbage_mask,board,rise)
                except ValueError:
                    self.garbage_mask=(0,)*(20-rise)+tuple(board[-rise:])
                self.garbage_rise_events+=1
                self.last_garbage_rise_at=now
                self.garbage_rise_lines+=rise
                self.garbage_fast_replans+=1
                self.post_drop_rise_resyncs+=1
                self._replan(external_plan,'DROP 대기 중 Garbage 상승·착지 미확정',now,
                             self.hold_used,verified_transition=True)
            elif transition.kind=='fresh_board':
                # A newly reconstructed *four-cell* active piece + changed
                # settled board is evidence enough to abandon the old plan,
                # even if gravity already moved the new piece below row 5.
                # NEVER attribute the planned attack to this unverified lock.
                same_piece=piece==self.plan.move.piece
                self.unverified_board_resyncs+=1
                self.post_drop_fresh_resyncs+=1
                self.garbage_mask=EMPTY_MASK
                self._replan(external_plan,'보드 전환 재동기화·착지 점수 미확정',now,
                             self.hold_used if same_piece else False,
                             verified_transition=True)
            else:
                # A hard-drop command can be delivered to the OS but never
                # accepted by the game. The old controller waited forever.
                # Recover ONLY when a current, independently observed ACTIVE
                # remains on the *unchanged source board* for multiple frames
                # and there was no visual gap suggesting an unobserved lock.
                # Never infer a lock, credit attack, or resend the old DROP.
                old_piece=self.before_drop[1]
                old_shape=norm(self.before_drop[2])
                old_column=min(x for x,y in self.before_drop[2])
                recoverable=(not self.saw_gap and
                    self.plan.move.board!=completed_source and
                    board==completed_source and piece==old_piece and
                    norm(cells)==old_shape and
                    min(x for x,y in cells)==old_column and
                    self.stable_count>=3 and now-self.sent_at>=.90)
                if recoverable:
                    self.drop_unacknowledged_replans+=1
                    self._failed_input(state,external_plan,now,
                                       '하드드롭 미반영 · 실제 현재 조각으로 재판단')
                elif (self.drop_delivery in ('dispatched','legacy') and
                      self.stable_count>=3 and now-self.sent_at>=2.5):
                    # A vision gap can conceal a Perfect Clear or a new
                    # identical tetromino. With no independently proven lock
                    # or live-old-piece evidence, it is impossible to know
                    # whether retrying SPACE would drop the *next* piece.
                    # An explicit stop is safer than waiting forever or
                    # silently repeating a potentially delivered hard drop.
                    self.drop_ambiguous_watchdog_stops+=1
                    self.stop('하드드롭 확인 불가능 2.5초 · 중복 DROP 방지 자동 중지 · 진단 저장')
                else:
                    self.message = '자동 ON · 대기: 착지/새 블록 확인'
            return None  # No duplicate hard drop while confirmation is pending.

        if self.phase == 'hold_wait':
            if self.stable_count < 2:
                return None
            if board == self.plan.source_board and piece == self.plan.move.piece:
                self.held = self.outgoing
                self.hold_uncertain = False
                self.verified_holds += 1  # observed HOLD swap, not merely planned
                self._acknowledged_input()  # verified hold swap
                self._replan(external_plan, '홀드 완료·현재 위치로 새 계산', now, True)
            elif now - self.sent_at > .7:
                self.held = '?'
                self.hold_uncertain = True
                self._replan(external_plan, '홀드 불확실·보이는 블록 사용', now, True)
            return None

        if self.phase != 'idle' and (board != self.plan.source_board or piece != self.plan.move.piece):
            # An active piece may be on screen when another player's garbage
            # arrives.  Two identical verified frames + exact shifted stack
            # suffice; do not mistake this for a generic 3-frame vision loss.
            rise=(verified_rise(self.plan.source_board, board)
                  if piece==self.plan.move.piece else None)
            if rise is not None and rise>0 and self.stable_count>=2:
                try:
                    self.garbage_mask=mask_after_rise(self.plan.source_board,self.garbage_mask,board,rise)
                except ValueError:
                    # Previous provenance can be stale when vision rejoined
                    # mid-game. Preserve only the *newly observed* garbage.
                    self.garbage_mask=(0,)*(20-rise)+tuple(board[-rise:])
                self.garbage_rise_events+=1
                self.last_garbage_rise_at=now
                self.garbage_rise_lines+=rise
                self.garbage_fast_replans+=1
                self._replan(external_plan,'확인된 Garbage 유입·즉시 재판단',now,
                             self.hold_used,verified_transition=True)
            elif self.stable_count >= 3:
                self._replan(external_plan, '보드/블록 변경', now, True)
            else:
                self.message = '자동 ON · 대기: 변경된 보드 확인'
            return None

        if self.phase == 'idle':
            if self.stable_count < 2 or now < self.not_before:
                return None
            # Formerly: non-turbo entered a permanent idle return after 3
            # missed acknowledgements, while turbo reset the counter and
            # blindly replayed the same stale plan after a 750ms pause.
            # Replan ONLY after 3 identical, freshly observed frames; never
            # repeat a hard drop on a mere timer. Limit same-pose recovery
            # attempts so a broken key binding cannot create an input loop.
            if self.failed_count >= 3 and signature == self.failed_signature:
                if self.stable_count<3 or now<self.not_before:return None
                if self.recovery_retry_count>=2:
                    self.stop('동일 화면에서 입력 3회 연속 실패 · 자동 중지 · 키/SDF 설정 확인')
                    return None
                self.recovery_retry_count+=1
                self.recovery_retry_total+=1
                self.failed_count=0
                self.failed_signature=None
                if external_plan is not None and external_plan.move is not None:
                    self.recovery_blocked_target=(board,piece,
                        tuple(sorted(external_plan.move.cells)),external_plan.use_hold,
                        external_plan.move.spin)
                    self.recovery_blocked_since=now
                # Reject the existing plan and increment serial; the GUI
                # must calculate a fresh one against the observed state.
                self._replan(external_plan,'3회 입력 실패 · 검증된 상태에서 새로운 경로 계산',now,
                             self.hold_used,verified_transition=True)
                return None
            if signature != self.failed_signature:
                self.failed_count = 0
            if external_plan is None or external_plan is self.rejected_plan:
                self.message = '자동 ON · 대기: 새 배치 계산'
                return None
            if external_plan.move is not None and self.recovery_blocked_target is not None:
                candidate=(board,piece,tuple(sorted(external_plan.move.cells)),
                           external_plan.use_hold,external_plan.move.spin)
                if candidate==self.recovery_blocked_target:
                    # NEVER retry the exact failed route after an
                    # observation-only recovery. Wait for a different path,
                    # or fail safe with a visible stop (not a silent hang).
                    if self.recovery_blocked_since is not None and now-self.recovery_blocked_since>=4.0:
                        self.stop('동일 경로 입력 불가 · 자동 중지 · 키 설정/ROI 점검')
                    else:
                        self.message='자동 ON · 반복 실패 경로 차단 · 새 목표 필요'
                    return None
                self.recovery_blocked_target=None
                self.recovery_blocked_since=None
            if (external_plan.source_board, external_plan.source_piece) != (board, piece):
                return None
            if external_plan.use_hold and self.hold_used:
                self._replan(external_plan, '이번 블록 홀드 제외', now, True)
                return None
            self.plan = external_plan
            if external_plan is self.expected_prepared:
                self.plan_is_prepared=True
                self.prepared_started+=1
            else:
                self.expected_prepared=None
            self.phase = 'align'
            self.started_at = now
            if self.plan.use_hold:
                self.phase = 'hold_wait'
                self.outgoing = piece
                self.sent_at = now
                self.message = '자동 ON · HOLD 결과 확인'
                return 'hold'

        if now - self.started_at > 2.5:
            self._failed_input(state, external_plan, now, '배치 지연·실제 위치로 재계산')
            return None

        if self.soft_disabled and self.plan.move.spin:
            self._replan(external_plan,'소닉드롭 비활성 · T-spin 경로 사용 금지',now)
            return None
        self._observed_pressure=bool(pressure)
        if self.turbo or self.plan.move.tactical:
            return self._route_action(state,external_plan,now)

        shape = norm(cells)
        col = min(x for x, y in cells)
        if self.phase == 'rotate_wait':
            if shape == self.pending_rotation:
                self.phase = 'align'
                self._acknowledged_input()
            elif now - self.sent_at > .45:
                self._failed_input(state, external_plan, now, '회전 미확인')
                return None
            else:
                return None
        if self.pending:
            if (shape, col) == self.pending:
                self.pending = None
                self._acknowledged_input()
            elif now - self.sent_at > .45:
                self._failed_input(state, external_plan, now, '이동 미확인')
                return None
            else:
                return None

        target_shape = norm(self.plan.move.cells)
        target_col = min(x for x, y in self.plan.move.cells)
        if shape != target_shape:
            clockwise = norm([(-y, x) for x, y in shape])
            halfturn = norm([(-y, x) for x, y in clockwise])
            # The non-turbo controller uses a single A input for 180s,
            # just as the routed turbo controller does. Do not rotate when
            # the direct turn would collide: the next observation will replan.
            use_180 = (target_shape == halfturn and target_shape != clockwise)
            if use_180:
                from motion import pose_from, step
                pose=pose_from(piece,cells)
                use_180=pose is not None and step(board,piece,pose,'rot180') is not None
            self.pending_rotation = halfturn if use_180 else clockwise
            self.phase = 'rotate_wait'
            self.sent_at = now
            self.message = '자동 ON · 180° 회전' if use_180 else '자동 ON · 회전'
            return 'rot180' if use_180 else 'cw'
        if col != target_col:
            step = 1 if col < target_col else -1
            self.pending = (shape, col + step)
            self.sent_at = now
            self.message = '자동 ON · 이동'
            return 'right' if step == 1 else 'left'
        if projected_cells(board, cells) != tuple(sorted(self.plan.move.cells)):
            self._failed_input(state, external_plan, now, '착지 경로 재확인')
            return None
        return self._authorize_drop(state,now,pressure)

    def _route_action(self,state,external_plan,now):
        from motion import route_to
        board,piece,cells=state
        if self.route_pending is not None:
            expected,grounded=self.route_pending
            matched=(tuple(sorted(cells))==expected if grounded else
                     norm(cells)==norm(expected) and min(x for x,y in cells)==min(x for x,y in expected))
            if matched and self.route_pending_180 and tuple(sorted(cells))!=expected:
                # A geometrically similar state displaced to a different
                # height does not prove the specific SRS+ kick was accepted.
                # Failing closed is safer than issuing a stale next command.
                matched=False
            # A symmetric I/S/Z rotation can return to exactly the previously
            # visible cells without the planned rotation having taken effect.
            # Even if a stale path tries to do this, never accept the unchanged
            # frame as proof of an issued command.
            if matched and self.route_prev_cells is not None and piece in 'ISZ' and \
                    tuple(sorted(cells))==self.route_prev_cells and not self.route_soft_active:
                matched=False
                if not self.route_ack_reject_reported:
                    self.rotation_ack_rejects += 1
                    self.route_ack_reject_reported = True
            if matched:
                self.route_pending=None
                self.route_pending_180=False
                self.route_prev_cells=None
                self.route_ack_reject_reported=False
                self.route_soft_active=False
                self._acknowledged_input()  # observed change/pose after a burst
            elif self.route_soft_active:
                # 'soft' in motion.py is a sonic *target*, while a Windows
                # DOWN pulse may move only part way depending on game SDF.
                # Retry only if the observed piece is on the same column and
                # orientation, strictly ABOVE the verified target. Never spam
                # when vision is uncertain, the pose overshot, or focus changed.
                same_column=(norm(cells)==norm(expected) and
                             min(x for x,y in cells)==min(x for x,y in expected))
                above=min(y for x,y in cells)<min(y for x,y in expected)
                safe=(same_column and above and self.route_soft_pulses<12 and
                      self.route_soft_started is not None and
                      now-self.route_soft_started<.75)
                if safe and now-self.sent_at>=.020:
                    self.route_soft_pulses+=1
                    self.sent_at=now
                    return 'soft'
                if not safe or now-self.route_soft_started>=.75:
                    self._failed_input(state,external_plan,now,'소닉드롭 경로 미완료')
                return None
            elif now-self.sent_at>.35:
                self._failed_input(state,external_plan,now,'묶음 입력 재확인')
                return None
            else:return None
        if self.route is None:
            path=route_to(board,piece,cells,self.plan.move.cells,self.plan.move.spin)
            if self.soft_disabled and path is not None and any(a=='soft' for a,_ in path):
                self._failed_input(state,external_plan,now,'소닉드롭 비활성 · 일반 배치 재계산')
                return None
            if path is None:
                self._failed_input(state,external_plan,now,'도달 가능한 경로 재계산')
                return None
            self.route=list(path)
        if self.route:
            actions=[];expected=None
            # A sonic drop is its own acknowledged segment. Rotations/taps on
            # either side can be batched, never across an unconfirmed descent.
            while self.route:
                action,expected=self.route.pop(0);actions.append(action)
                # Guard against an old/stale plan commanding an unobservable
                # A rotation. A no-kick A is still supported for T/J/L.
                if action=='rot180' and piece not in ('T','J','L'):
                    self.rotation_180_suppressed += 1
                    self._failed_input(state,external_plan,now,'시각 확인 불가능한 180° 경로 차단')
                    return None
                # For symmetric shapes acknowledge each 90° independently;
                # batching two 90s recreates the original visible shape.
                if (piece in 'ISZ' and action in ('cw','ccw')) or \
                        action=='soft' or not self.route or \
                        self.route[0][0]=='soft' or not self.turbo:
                    break
            grounded=('soft' in actions or min(y for x,y in expected)>=min(y for x,y in self.plan.move.cells))
            self.route_prev_cells=tuple(sorted(cells))
            self.route_ack_reject_reported=False
            self.route_pending=(expected,grounded);self.sent_at=now
            self.route_pending_180='rot180' in actions
            self.route_soft_active='soft' in actions
            if self.route_soft_active:
                self.route_soft_started=now
                self.route_soft_pulses=1
            self.message='자동 ON · 고속 경로 '+','.join(actions)
            return tuple(actions)
        target=tuple(sorted(self.plan.move.cells))
        if projected_cells(board,cells)!=target or (self.plan.move.spin and tuple(sorted(cells))!=target):
            self._failed_input(state,external_plan,now,'최종 착지 재확인')
            return None
        return self._authorize_drop(state,now,getattr(self,'_observed_pressure',False))
