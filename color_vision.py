"""Bounded, theme-aware board classifier for 10x20 TETR.IO captures.

This module never sends inputs.  It intentionally keeps the historical
brightness reader available, and fails closed on uncertain color evidence.
Solid-color labels are diagnostic hints, NOT game-state or piece-type proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from vision import read_grid, sample_indices

# GUI-only color hints.  Keep both the default approximations and an
# observed pastel/skin reference (I=teal, S=lime, O=gold, T=magenta,
# J=violet).  Identity for NEXT/HOLD is verified by 4-cell geometry in
# piece_vision.py; these hues NEVER decide board collisions or active piece.
_LETTERS = 'IOTJLSZ'
_RGB_PALETTE = np.array([
    (45,205,220), (225,195,35), (160,70,205),
    (60,85,215), (225,135,35), (80,195,70), (220,65,65),
    (50,179,131), (179,155,51), (167,63,155),
    (79,62,166), (180,99,49), (131,178,55), (180,52,59),
], dtype=np.float32)
_LABEL_INDICES = np.arange(len(_RGB_PALETTE)) % len(_LETTERS)
_PALETTE_UNIT = _RGB_PALETTE / np.maximum(_RGB_PALETTE.max(axis=-1, keepdims=True), 1)


@dataclass(frozen=True)
class VisionReading:
    grid: list[list[bool]] | None
    unknown: list[list[bool]] | None
    colors: list[list[str]] | None
    ghosts: list[list[bool]] | None
    method: str
    confidence: float
    reason: str = ''


@dataclass(frozen=True)
class CalibrationProgress:
    status: str  # pending / success / failed / idle
    accepted: int
    required: int
    reason: str


class ColorBoardVision:
    """Infer an empty-cell color from a mostly visible board and classify.

    Recalibration is explicit: a transient overlay or scene cut must never
    silently replace an established background during live control.  AUTO
    uses legacy only if no trustworthy color calibration has *ever* been
    established; STRICT waits instead.  A broken established reference is
    always treated as UNKNOWN regardless of mode.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.background = None  # BGR float32, from cell interior medians
        self.calibration_support = 0.0
        self._recalibration = None
        self._manual_block = False
        self._initial_samples = []
        self._initial_sample_at = -1e12

    @property
    def recalibrating(self):
        return self._recalibration is not None

    def begin_recalibration(self, now: float):
        """Stage a replacement; never erase a previously confirmed reference.

        Collect four separated game frames, rather than trusting a single
        transient menu/animation frame.  The caller must suspend inputs.
        """
        self._manual_block = True
        self._recalibration = dict(started=float(now), next_at=float(now) + 0.18,
                                   candidates=[], supports=[], required=4)

    def cancel_recalibration(self):
        self._recalibration = None
        # If no prior reference was available, remain fail-closed until a
        # successful explicit calibration or an intentional full reset.

    def collect_recalibration(self, bgr, now: float) -> CalibrationProgress:
        session = self._recalibration
        if session is None:
            return CalibrationProgress('idle', 0, 4, '진행 중인 보정 없음')
        accepted = len(session['candidates'])
        if float(now)-session['started'] > 5.0:
            self.cancel_recalibration()
            return CalibrationProgress('failed', accepted, 4,
                                       '재보정 실패: 유효한 게임 보드 4프레임을 확인하지 못했습니다. 기존 배경은 유지합니다.')
        if float(now) < session['next_at']:
            return CalibrationProgress('pending', accepted, 4,
                                       f'안정적인 게임 화면 확인 중 {accepted}/4')
        session['next_at'] = float(now) + .09
        candidate = self._candidate(bgr)
        if candidate is None:
            session['candidates'].clear(); session['supports'].clear()
            return CalibrationProgress('pending', 0, 4,
                                       '배경 분리 불가: BOARD 내부와 실제 블록이 충분히 보여야 합니다.')
        background, support, foreground = candidate
        # An all-black loading screen or solid overlay may be uniform enough
        # for the ordinary automatic first-frame estimator. Require actual
        # occupied-color evidence during *explicit* recalibration.
        if foreground < 4:
            session['candidates'].clear(); session['supports'].clear()
            return CalibrationProgress('pending', 0, 4,
                                       '단색 화면 또는 빈 보드: 실제 블록 4칸 이상이 보여야 합니다.')
        if session['candidates'] and np.max(np.abs(background-session['candidates'][0])) > 12:
            session['candidates'].clear(); session['supports'].clear()
            # Do not accept a scene transition as one stable calibration.
            return CalibrationProgress('pending', 0, 4,
                                       '화면 색상 변화 감지: 연속 안정 프레임을 다시 확인합니다.')
        session['candidates'].append(background)
        session['supports'].append(support)
        count = len(session['candidates'])
        if count < session['required']:
            return CalibrationProgress('pending', count, 4,
                                       f'배경색 검증 {count}/4 · 입력 중단 상태')
        confirmed = np.median(np.stack(session['candidates']), axis=0).astype(np.float32)
        self.background = confirmed
        self.calibration_support = min(session['supports'])
        self._manual_block = False
        self._initial_samples = []
        self._recalibration = None
        return CalibrationProgress('success', 4, 4,
                                   '컬러 배경 4프레임 검증 완료 · 새 기준 적용')

    @staticmethod
    def _samples(bgr):
        a = np.asarray(bgr)
        if (a.ndim != 3 or a.shape[2] < 3 or a.dtype != np.uint8
                or a.shape[0] < 80 or a.shape[1] < 40
                or a.shape[0] > 8192 or a.shape[1] > 8192
                or a.shape[0]*a.shape[1] > 5_000_000
                or not .42 < a.shape[1]/a.shape[0] < .58):
            raise ValueError('Invalid 10x20 board image: dimensions or pixel format')
        ys, xs = sample_indices(a.shape[0], a.shape[1])
        return a[ys, xs, :3].reshape(20, 10, 9, 3)

    def _candidate(self, bgr):
        """Propose a background without mutating a working reference.

        If a proven reference already exists, use matching *empty* cells from
        anywhere on the board. The top rows can be completely occupied in a
        normal mid-game board. Never infer a replacement skin from saturated
        stack colors; a genuinely different theme still requires a clear view.
        """
        samples = self._samples(bgr)
        cells = np.median(samples, axis=2).astype(np.float32)
        # Existing reference: only actual reference-colored cells may nominate
        # a background, regardless of row. Do not derive the background from
        # the most frequent *piece* color on an almost full board.
        if self.background is not None:
            known = np.max(np.abs(cells-self.background), axis=-1) <= 24
            if int(np.count_nonzero(known)) >= 16:
                trusted = np.median(cells[known], axis=0).astype(np.float32)
                foreground = int(np.count_nonzero(
                    np.max(np.abs(cells-trusted), axis=-1) >= 43))
                return trusted, float(np.mean(known)), foreground
        candidate = np.median(cells[:8].reshape(-1, 3), axis=0)
        diffs = np.max(np.abs(cells-candidate), axis=-1)
        upper = float(np.mean(diffs[:8] <= 24))
        overall = float(np.mean(diffs <= 24))
        if upper < .78 or overall < .54:
            return None
        accepted = cells[diffs <= 24]
        background = np.median(accepted, axis=0).astype(np.float32)
        diffs = np.max(np.abs(cells-background), axis=-1)
        if float(np.mean(diffs[:8] <= 24)) < .78:
            return None
        foreground = int(np.count_nonzero(diffs >= 43))
        return background, overall, foreground

    def calibrate(self, bgr) -> bool:
        """Historical one-frame initial calibration (no active replacement).

        Manual recalibration uses the staged multi-frame collector instead.
        """
        candidate = self._candidate(bgr)
        if candidate is None:
            return False
        background, support, _foreground = candidate
        self.background = background
        self.calibration_support = support
        self._manual_block = False
        return True

    @staticmethod
    def _color_hints(samples, occupied):
        # Approximate hue matching for GUI preview ONLY.  Multiple reference
        # skins help distinguish lime S from gold O, but ambiguous hues are
        # rendered neutrally instead of mislabelling a locked block.
        bgr = np.median(samples, axis=2).astype(np.float32)
        rgb = bgr[..., ::-1]
        normalized = rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1)
        distances = np.linalg.norm(normalized[..., None, :] - _PALETTE_UNIT[None, None, :, :], axis=-1)
        # Find the closest reference for each PIECE, then compare against the
        # runner-up piece. Never use distance to a duplicated same-piece ref
        # as evidence of uncertainty.
        piece_dist = np.stack([np.min(distances[..., _LABEL_INDICES == i], axis=-1)
                               for i in range(7)], axis=-1)
        rank = np.argsort(piece_dist, axis=-1)
        first = np.take_along_axis(piece_dist, rank[..., :1], axis=-1)[..., 0]
        second = np.take_along_axis(piece_dist, rank[..., 1:2], axis=-1)[..., 0]
        chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
        reliable = (first < .25) & ((second - first) >= .075) & (chroma >= 35)
        hints = np.full((20,10), '■', dtype='<U1')
        for i, letter in enumerate(_LETTERS):
            hints[occupied & reliable & (rank[..., 0] == i)] = letter
        hints[~occupied] = ''
        return hints.tolist()

    def _ghost_hints(self, bgr, occupied):
        """Visual-only candidate overlay. Never modifies the collision board."""
        a = np.asarray(bgr)
        offsets = np.array([.10,.50,.90])
        ys = np.minimum(((np.arange(20)[:,None]+offsets)*a.shape[0]/20).astype(int),a.shape[0]-1)
        xs = np.minimum(((np.arange(10)[:,None]+offsets)*a.shape[1]/10).astype(int),a.shape[1]-1)
        pixels = a[ys[:,None,:,None],xs[None,:,None,:],:3].reshape(20,10,9,3)
        intensity = np.max(np.abs(pixels.astype(np.int16)-self.background),axis=-1)
        color = pixels.max(axis=-1).astype(np.int16)-pixels.min(axis=-1).astype(np.int16)
        outer = ((intensity >= 35) & (color >= 35))
        outer[:,:,4] = False
        # Avoid labelling arbitrary grid lines as ghosts: require saturation.
        return ((np.count_nonzero(outer,axis=-1)>=3) & ~occupied).tolist()

    @staticmethod
    def _check_cell_dense(bgr, x, y, background):
        """Recheck ONLY an ambiguous cell with 25 center samples.

        Return True/False only if new evidence is unambiguous. Ghosts,
        animation, clipped pieces and skin effects stay UNKNOWN. There is no
        use of previous/stale board occupancy as a substitute for evidence.
        """
        height, width = bgr.shape[:2]
        offsets = np.array((.30, .40, .50, .60, .70))
        ys = np.minimum(((y+offsets)*height/20).astype(int), height-1)
        xs = np.minimum(((x+offsets)*width/10).astype(int), width-1)
        pixels = bgr[ys[:, None], xs[None, :], :3].astype(np.int16)
        contrast = np.max(np.abs(pixels-background), axis=-1)
        strong = int(np.count_nonzero(contrast >= 43))
        moderate = int(np.count_nonzero(contrast >= 22))
        if strong >= 22 and moderate >= 22:
            return True
        if strong == 0 and moderate <= 1:
            return False
        return None

    @staticmethod
    def _check_cell_surface(bgr, x, y, background):
        """Independent 7x7 interior support for textured/offset real blocks.

        This is a *last* bounded check after 9+25 samples disagree.  A true
        block must cover at least 38/49 interior points, including the center
        and each spatial quadrant. An outline Ghost or partial line-clear
        animation remains UNKNOWN. No temporal or predicted board is reused.
        """
        h, w = bgr.shape[:2]
        # At tiny ROI resolutions 49 sample indices can collapse onto the
        # same pixels, falsely amplifying evidence. Stay UNKNOWN instead.
        if h < 240 or w < 120:
            return None
        offsets = np.array((.22, .31, .40, .50, .60, .69, .78))
        ys = np.minimum(((y+offsets)*h/20).astype(int),h-1)
        xs = np.minimum(((x+offsets)*w/10).astype(int),w-1)
        patch = bgr[ys[:,None],xs[None,:],:3].astype(np.int16)
        diff = np.max(np.abs(patch-background),axis=-1)
        solid = diff >= 43
        if (int(np.count_nonzero(solid))>=38 and solid[3,3]
                and all(np.count_nonzero(solid[y0:y0+3,x0:x0+3])>=6
                        for y0,x0 in ((0,0),(0,4),(4,0),(4,4)))):
            return True
        # An original sparse hit followed by a mostly empty surface could be
        # a Ghost, UI effect or clipped tetromino. Never override the earlier
        # UNKNOWN with EMPTY using this last-resort *positive-only* test.
        return None

    def read(self, bgr, mode='auto', threshold=105, color_hints=False,
             allow_legacy_fallback=True, initial_confirmations=1,
             now=None, spawn_evidence=False, reference_floor=12) -> VisionReading:
        if type(reference_floor) is not int or not 4 <= reference_floor <= 12:
            raise ValueError('reference_floor must be an integer from 4 through 12')
        if mode not in ('auto', 'strict', 'legacy'):
            raise ValueError('Unsupported vision mode')
        if not isinstance(initial_confirmations, int) or not 1 <= initial_confirmations <= 4:
            raise ValueError('Initial confirmations must be 1..4')
        if self._recalibration is not None:
            return VisionReading(None, None, None, None, 'CALIBRATING', 0.0,
                                 '안전한 배경 재보정 진행 중 · 입력 보류')
        if mode == 'legacy':
            return VisionReading(read_grid(bgr, threshold), None, None, None,
                                 'LEGACY', 0.0, '기존 밝기 기준')
        samples = self._samples(bgr)
        if self.background is None:
            if self._manual_block:
                return VisionReading(None, None, None, None, 'UNKNOWN', 0.0,
                                     '재보정 실패 · 컬러 배경을 재확인해야 합니다.')
            if initial_confirmations > 1:
                # GUI/auto path: refuse a one-frame loading screen or an
                # animation masquerading as a new background. Actual
                # foreground OR independently shape-validated SPAWN evidence
                # must accompany several consistent background samples.
                candidate=self._candidate(bgr)
                t=time.perf_counter() if now is None else float(now)
                if candidate is None or (candidate[2] < 4 and not spawn_evidence):
                    self._initial_samples.clear()
                    return VisionReading(None, None, None, None, 'UNKNOWN', 0.0,
                                         '초기 컬러 보정 대기 · 게임 블록 확인 필요')
                proposed=candidate[0]
                if (self._initial_samples and
                        np.max(np.abs(proposed-self._initial_samples[0])) > 12):
                    self._initial_samples.clear()
                if t-self._initial_sample_at >= .035:
                    self._initial_samples.append(proposed)
                    self._initial_sample_at=t
                if len(self._initial_samples) < initial_confirmations:
                    return VisionReading(None, None, None, None, 'UNKNOWN', 0.0,
                                         f'초기 컬러 검증 {len(self._initial_samples)}/{initial_confirmations} · 입력 보류')
                self.background=np.median(np.stack(self._initial_samples),axis=0).astype(np.float32)
                self.calibration_support=candidate[1]
                self._initial_samples.clear()
            elif not self.calibrate(bgr):
                if mode == 'auto' and allow_legacy_fallback:
                    return VisionReading(read_grid(bgr, threshold), None, None, None,
                                         'LEGACY', 0.0, '컬러 배경 보정 불가 · 기존 방식 사용')
                return VisionReading(None, None, None, None, 'UNKNOWN', 0.0,
                                     '배경 보정 불가 · 배경이 충분히 보이는 화면에서 재보정')
        background = self.background
        # Validate intact background-coloured interiors across ALL 200 cells,
        # not the top eight rows (which may be full of real blocks).
        # Reuse the same 9-point contrast tensor for occupancy classification:
        # avoiding a second median+ptp pass preserves the 8-ms capture budget.
        contrast = np.max(np.abs(samples.astype(np.int16)-background), axis=-1)
        reference_cells = np.count_nonzero(contrast <= 24, axis=2) >= 8
        reference_count = int(np.count_nonzero(reference_cells))
        if reference_count < reference_floor:
            return VisionReading(None, None, None, None, 'UNKNOWN', 0.0,
                                 f'배경 검증 부족 ({reference_count}/{reference_floor}칸) · 테마 변경·높은 스택 확인')
        # A ghost or transient overlay can partly affect 9-point interiors.
        # Only strongly supported occupancy is actionable; uncertain cells
        # receive a targeted, bounded dense recheck below.
        strong = np.count_nonzero(contrast >= 43, axis=2)
        moderate = np.count_nonzero(contrast >= 22, axis=2)
        occupied = strong >= 5
        # A partly covered cell could be an animated piece or effect.  It is
        # unsafe to silently turn it into either a filled or an empty square.
        uncertain = ((strong >= 2) & (strong <= 4)) | ((strong == 1) & (moderate >= 5))
        if np.any(uncertain):
            # Full-frame expensive re-scans are unnecessary. Try only a few
            # ambiguous squares and accept a result only with strong evidence.
            coords = np.argwhere(uncertain)
            if len(coords) <= 6:
                for y, x in coords:
                    resolved = self._check_cell_dense(np.asarray(bgr), int(x), int(y), background)
                    if resolved is None:
                        resolved = self._check_cell_surface(np.asarray(bgr), int(x), int(y), background)
                    if resolved is not None:
                        occupied[y, x] = resolved
                        uncertain[y, x] = False
            if np.any(uncertain):
                count = int(np.count_nonzero(uncertain))
                return VisionReading(None, uncertain.tolist(), None, None, 'UNKNOWN',
                                     max(0., 1-count/200),
                                     f'불확실한 칸 {count}개 · 자동입력 보류 후 새 프레임 확인')
        labels = self._color_hints(samples, occupied) if color_hints else None
        ghosts = self._ghost_hints(bgr, occupied) if color_hints else None
        return VisionReading(occupied.tolist(), uncertain.tolist(), labels, ghosts,
                             'COLOR', min(0.99, max(0.60, 0.7+self.calibration_support*.25)),
                             f'컬러 점유 판정 · 배경 검증 {reference_count}칸 · 색상 이름은 참고용')


def suppress_confirmed_white_i_projection(bgr, background, uncertain, previous):
    """Mask a SPECIFIC white placement-guide artifact, never generic ghost pixels.

    Returns an adjusted *copy* or None. Only used for a 4-row, featureless,
    pure-white vertical guide in a black-background scene while the CURRENT
    I piece is independently visible in its cyan horizontal spawn pose.
    Prior LOCKED cells must remain untouched, and the caller must validate
    the resulting full board/active identity against prior before using it.
    This is intentionally narrow to prevent white custom skins becoming empty.
    """
    if background is None or uncertain is None or previous is None:
        return None
    a=np.asarray(bgr)
    if a.ndim!=3 or a.shape[2]<3 or a.dtype!=np.uint8:
        return None
    h,w=a.shape[:2]
    if h<400 or w<200 or not .42<w/h<.58:
        return None
    bg=np.asarray(background,dtype=np.float32)
    if bg.shape!=(3,) or not np.all(np.isfinite(bg)) or np.max(bg)>20:
        return None
    if not isinstance(previous, tuple) or len(previous)!=2 or previous[1]!='I':
        return None
    prior=previous[0]
    if len(prior)!=20:
        return None
    coords=np.argwhere(np.asarray(uncertain,dtype=bool))
    if len(coords)!=1:
        return None
    y,x=map(int,coords[0])
    if y<3 or y>15 or x not in range(10):
        return None
    if any(prior[yy] & (1<<x) for yy in range(y,y+4)):
        return None

    # Require strong, normal-colored CURRENT I at the top of the *original*
    # frame, not as an inference from the white guide or a prior assumption.
    # Never suppress a white custom piece skin: it fails this check.
    for ix in (3,4,5,6):
        y0,y1=int(.27*h/20),int(.74*h/20)
        x0,x1=int((ix+.27)*w/10),int((ix+.74)*w/10)
        med=np.median(a[y0:y1,x0:x1,:3].reshape(-1,3),axis=0)
        b,g,r=med
        if not (g>=145 and b>=75 and g-r>=55 and b-r>=35):
            return None
    # Grid-aligned, textureless white column spanning 3 complete rows and
    # entering the row above partially. Validate each independent cell core.
    def white_fraction(row,col):
        y0,y1=int((row+.22)*h/20),int((row+.78)*h/20)
        x0,x1=int((col+.22)*w/10),int((col+.78)*w/10)
        patch=a[y0:y1,x0:x1,:3]
        if patch.size==0:return 0.
        maxi=patch.max(axis=-1).astype(np.int16)
        mini=patch.min(axis=-1).astype(np.int16)
        return float(np.mean((mini>=249)&(maxi-mini<=5)))
    top=white_fraction(y,x)
    if not .20<=top<=.78:
        return None
    if any(white_fraction(yy,x)<.88 for yy in range(y+1,y+4)):
        return None
    if white_fraction(y-1,x)>.03:
        return None
    for side in (x-1,x+1):
        if 0<=side<10 and any(white_fraction(yy,side)>.06 for yy in range(y,y+4)):
            return None
    # Restrict masking to the affected grid column; never alter known-colored
    # pixels, including green/yellow locked blocks beneath the white stripe.
    out=a.copy()
    y0,y1=int(y*h/20),min(h,int((y+5)*h/20))
    x0,x1=int(x*w/10),min(w,int((x+1)*w/10))
    patch=out[y0:y1,x0:x1,:3]
    maxi=patch.max(axis=-1).astype(np.int16)
    mini=patch.min(axis=-1).astype(np.int16)
    pure_white=(mini>=249)&(maxi-mini<=5)
    patch[pure_white]=np.clip(bg,0,255).astype(np.uint8)
    return out
