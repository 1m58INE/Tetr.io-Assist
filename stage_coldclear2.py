"""Stage a *verified* Cold Clear 2 binary next to PyInstaller's TetrioAssist.exe.

The installer writes to SOURCE_ROOT/external_solver, while the frozen GUI
intentionally loads from EXE_DIR/external_solver.  These are separate folders.
No downloads; source candidate binaries are never promoted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stage(root: Path, validate, optional: bool = False) -> tuple[bool, str]:
    """Copy *only after* the caller's real protocol/route verifier passes.

    A compiled app is not considered installed until dist/TetrioAssist.exe
    exists.  Never silently pick the unverified .candidate.exe.  Never modify
    a source Cold Clear 2 binary.
    """
    root = Path(root).resolve()
    source = root / 'external_solver' / 'cold-clear-2.exe'
    if not source.is_file() or source.is_symlink():
        return False, ('SKIP: verified external_solver/cold-clear-2.exe is absent. '
                       'A compiled .candidate.exe is NOT a verified install.')
    exe = root / 'dist' / 'TetrioAssist.exe'
    if not exe.is_file() or exe.is_symlink():
        return False, 'SKIP: dist/TetrioAssist.exe is absent; run build-exe.cmd first.'
    target_folder = root / 'dist' / 'external_solver'
    if target_folder.is_symlink():
        return False, 'ERROR: refusing symlinked dist/external_solver directory.'
    if not validate(source):
        return False, 'FAIL: Cold Clear 2 did not PASS the real TBP route check; not staged.'
    target_folder.mkdir(parents=True, exist_ok=True)
    target = target_folder / 'cold-clear-2.exe'
    if target.is_symlink():
        return False, 'ERROR: refusing symlinked destination binary.'
    with tempfile.NamedTemporaryFile(prefix='.cc2-stage-',suffix='.tmp',dir=target_folder,delete=False) as f:
        tmp = Path(f.name)
    try:
        shutil.copyfile(source, tmp)
        if sha256(tmp) != sha256(source):
            return False, 'ERROR: staged binary SHA256 mismatch.'
        os.replace(tmp, target)
        for license_name in ('LICENSE-MIT', 'LICENSE-APACHE'):
            lic_source = source.parent / license_name
            if lic_source.is_file() and not lic_source.is_symlink():
                shutil.copyfile(lic_source, target_folder / license_name)
        return True, ('PASS: compiled EXE Cold Clear 2 staged to ' + str(target)
                      + '\nSHA256: ' + sha256(target))
    finally:
        if tmp.exists():
            tmp.unlink()


def _check_actual_protocol(root: Path, exe: Path):
    check = root / 'coldclear_check.py'
    if not check.is_file():
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(check), '--exe', str(exe), '--timeout', '5.0'],
            cwd=str(root), check=False, capture_output=True, text=True,
            timeout=13, shell=False,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        data = json.loads(lines[-1]) if lines else {}
        print('CC2 protocol check:', json.dumps(data, ensure_ascii=False), flush=True)
        return result.returncode == 0 and data.get('status') == 'PASS' and data.get('source') == 'external_tbp'
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as exc:
        print('CC2 protocol check failed:', type(exc).__name__, flush=True)
        return False


def main() -> int:
    p = argparse.ArgumentParser(description='Stage verified Cold Clear 2 for the compiled GUI; no downloads.')
    p.add_argument('--optional', action='store_true', help='Do not fail a normal EXE build without optional CC2.')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    ok, message = stage(root, lambda exe: _check_actual_protocol(root,exe), optional=args.optional)
    print(message, flush=True)
    return 0 if ok or args.optional else 1


if __name__ == '__main__':
    sys.exit(main())
