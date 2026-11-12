"""Read-only local Cold Clear 2 protocol / route smoke check. Never sends keys.

One-turn legacy check is the default. --turns 2 also verifies that a second
independent snapshot can be processed without mixing old TBP suggestions.
"""
import json
import sys
from pathlib import Path
import argparse
import time
from tbp_bridge import (TBPClient, binary_location, executable_is_ready,
                        snapshot_to_tbp, as_pvp_plan)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,default=None)
    parser.add_argument('--timeout',type=float,default=3.0,
                        help='Offline CC2 search deadline per turn, 0.2..5.0 sec')
    parser.add_argument('--turns',type=int,choices=(1,2),default=1,
                        help='1=legacy single-turn check; 2=independent second-turn handshake check')
    opt=parser.parse_args()
    if not (.2<=opt.timeout<=5.0):
        parser.error('--timeout must be in [0.2, 5.0]')
    path=opt.exe.resolve() if opt.exe is not None else binary_location()
    if not executable_is_ready(path):
        print(json.dumps({'status':'NOT_INSTALLED','expected_path':str(path)},ensure_ascii=False))
        return 2
    bot=TBPClient([str(path)],handshake_timeout=2.0)
    board=(0,)*20
    # Independently valid start positions (not speculative successor states).
    snapshots=((board,'T',('I','O','S','Z','J','L')),
               (board,'O',('I','T','S','Z','J','L')))
    checks=[]
    try:
        for index in range(opt.turns):
            board,piece,queue=snapshots[index]
            start=time.monotonic()
            moves=bot.suggest(snapshot_to_tbp(board,piece,queue,None,0,-1),timeout=opt.timeout)
            rejected={}
            plan=as_pvp_plan(board,piece,queue,None,True,0,-1,0,moves,
                             rejection_counts=rejected)
            status=('PASS' if plan is not None else
                    'NO_VALID_ROUTE' if isinstance(moves,list) and moves else
                    'NO_SUGGESTION' if isinstance(moves,list) else 'NO_REPLY')
            checks.append({'turn':index+1,'status':status,
                           'elapsed_ms':round((time.monotonic()-start)*1000,2),
                           'suggestions':len(moves) if isinstance(moves,list) else None,
                           'empty_search_replies':bot.last_empty_suggestions,
                           'rejection_counts':rejected,
                           'reason':bot.last_suggest_status,
                           'source':plan.continuation_source if plan else None,
                           'piece':plan.move.piece if plan else None})
            if status!='PASS':break
        result=dict(checks[-1],turns_requested=opt.turns,
                    turns_checked=len(checks),checks=checks)
        print(json.dumps(result,ensure_ascii=False))
        return 0 if len(checks)==opt.turns and all(c['status']=='PASS' for c in checks) else 1
    finally:bot.close()

if __name__=='__main__':sys.exit(main())
