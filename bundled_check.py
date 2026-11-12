"""Run the REAL included Python or frozen TBP child, not a fake responder.

No game I/O and no internet access; exit 0 only after real move validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from tbp_bridge import (TBPClient,bundled_command,parse_tbp_move,
                        snapshot_to_tbp)


def check(command=None):
    cmd = command or bundled_command()
    client=TBPClient(cmd,handshake_timeout=2.0,
                     expected_name='TETRIO Bundled Strategy')
    try:
        b=(0,)*20
        snap=snapshot_to_tbp(b,'T',('I','O','S'),None,0,-1)
        snap.update(assist_hold_allowed=False, assist_b2b_count=0,
                    assist_combo=-1,assist_incoming=0)
        start=time.perf_counter()
        moves=client.suggest(snap,timeout=.15)
        duration=(time.perf_counter()-start)*1000
        if not moves or parse_tbp_move(b,'T',('I','O','S'),None,False,moves[0]) is None:
            print('FAIL: Local bot did not produce a reachable T placement.')
            return 1
        # Second TURN, not just a one-shot fabricated handshake.
        again=dict(snap,queue=['O','J','L','I'])
        second=client.suggest(again,timeout=.15)
        if not second or parse_tbp_move(b,'O',('J','L','I'),None,False,second[0]) is None:
            print('FAIL: Local bot did not survive the second turn.')
            return 1
        print(f'PASS: REAL local TBP process, two independently validated turns; cold start {duration:.1f}ms.')
        print('NOTE: This is the bundled native Python policy, NOT Cold Clear 2. No live Windows control verified.')
        return 0
    finally:
        client.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,default=None,
                        help='Test compiled Windows console TetrioBundledBot.exe')
    args=parser.parse_args()
    return check([str(args.exe.resolve())] if args.exe is not None else None)

if __name__=='__main__':raise SystemExit(main())
