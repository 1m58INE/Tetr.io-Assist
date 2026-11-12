"""Validate v1.1.10.17 runtime sources and import graph before PyInstaller runs.

Uses the standard library only, so it can run before pip installs dependencies.
"""
from __future__ import annotations

import ast
import io
import tokenize
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME_MODULES = (
    'assist', 'autoplay', 'engine', 'motion', 'vision', 'windows_input', 'legacy_integrity', 'survival_policy1195',
    'attack_rate', 'attack_structure11113', 'observed_attack_sequence', 'telemetry', 'performance_budget', 'legacy_quarantine', 'pressure_refresh', 'combat_rules', 'combo_rules', 'opening', 'adaptive_opener', 'canonical_openers', 'opener_commit', 'perfect_clear_planner', 'garbage_tracking',
    'config_security', 'input_rhythm', 'color_vision', 'piece_vision', 'vision_failure', 'spawn_vision', 'grid_calibration', 'window_roi',
    'garbage_transition', 'engine_policy', 'b2b_search', 'tspin_factory', 'competitive_policy', 'offensive_beam', 'vision_recovery', 'offense_runtime', 'chain_director', 'stall_monitor', 'tspin_initiator', 'transition_guard', 'tbp_bridge', 'bundled_tbp_bot',
)
OTHER_FILES = ('requirements.txt', 'start.cmd', 'build-exe.cmd', 'bundled_check.py', 'coldclear_check.py', 'stage_coldclear2.py')
EXTERNAL_IMPORTS = {'numpy', 'mss', 'PySide6'}


def legacy_fstring_quote_conflicts(source: str) -> list[int]:
    """Flag Python 3.12+ quote reuse that Python 3.11 cannot parse.

    A 3.13 build can accept an f-string whose expression reuses its outer
    quote; Python 3.11 cannot. Detect this before packaging on newer hosts.
    """
    fstart = getattr(tokenize, 'FSTRING_START', None)
    fend = getattr(tokenize, 'FSTRING_END', None)
    if fstart is None:
        return []  # Python 3.11's ast.parse already rejects this syntax.
    stack = []
    conflicts = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == fstart:
                quote = tok.string[-1]
                delimiter = None if tok.string.endswith(quote * 3) else quote
                if stack and delimiter is not None and stack[-1] == delimiter:
                    conflicts.append(tok.start[0])
                stack.append(delimiter)
            elif tok.type == fend:
                if stack:
                    stack.pop()
            elif tok.type == tokenize.STRING and stack and stack[-1]:
                literal = tok.string.lstrip('rRbBuUfF')
                if literal and literal[0] == stack[-1] and not literal.startswith(literal[0] * 3):
                    conflicts.append(tok.start[0])
    except tokenize.TokenError:
        pass  # ast.parse checks malformed syntax.
    return conflicts


def main() -> int:
    errors = []
    for name in (*[m + '.py' for m in RUNTIME_MODULES], *OTHER_FILES):
        # This public package retains only core sources and essential validation scripts.
        # Runtime modules and startup/build scripts must still be
        # present at their original import- and fixture-relative paths.
        candidates = [ROOT / name]
        if '/' not in name and Path(name).suffix.lower() in ('.md', '.json', '.txt'):
            candidates.extend((ROOT / 'docs' / 'release' / name,
                               ROOT / 'docs' / 'history' / name,
                               ROOT / 'docs' / 'history' / 'validation' / name,
                               ROOT / 'docs' / 'history' / 'checksums' / name,
                               ROOT / 'benchmarks' / 'results' / name))
        if not any(path.is_file() for path in candidates):
            errors.append('Missing required file: ' + name)

    for module in RUNTIME_MODULES:
        path = ROOT / (module + '.py')
        if not path.exists():
            continue
        try:
            source = path.read_text('utf-8')
            parsed = ast.parse(source, filename=str(path))
            for line in legacy_fstring_quote_conflicts(source):
                errors.append(f'{path.name}:{line}: Python 3.11-incompatible f-string quote reuse')
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'Cannot parse {path.name}: {exc}')
            continue
        for node in ast.walk(parsed):
            # Do not require third-party dependencies before pip install, but
            # reject any unknown local import omitted from an archive. This is
            # stronger than a hand-maintained list of runtime modules alone.
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root_name=node.module.split('.')[0]
                if (root_name not in sys.stdlib_module_names and
                    root_name not in EXTERNAL_IMPORTS and
                    not (ROOT / (root_name + '.py')).is_file()):
                    errors.append(f'{module}.py imports unbundled {root_name}.py')
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root_name=alias.name.split('.')[0]
                    if (root_name not in sys.stdlib_module_names and
                        root_name not in EXTERNAL_IMPORTS and
                        not (ROOT / (root_name + '.py')).is_file()):
                        errors.append(f'{module}.py imports unbundled {root_name}.py')
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module in RUNTIME_MODULES and not (ROOT / (node.module + '.py')).is_file():
                    errors.append(f'{module}.py imports missing local module {node.module}.py')
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    dependency = alias.name.split('.')[0]
                    if dependency in RUNTIME_MODULES and not (ROOT / (dependency + '.py')).is_file():
                        errors.append(f'{module}.py imports missing local module {dependency}.py')

    if errors:
        print('ERROR: Package check failed:')
        for error in sorted(set(errors)):
            print('  - ' + error)
        return 1

    print(f'PASS: {len(RUNTIME_MODULES)} runtime modules, startup/build files, and local import graph verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
