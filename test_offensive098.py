"""v0.9.8 regressions: committed-next planning, safe use, and opener limits.

These are synthetic tests, not evidence of PvP performance.
"""
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock
from test_app_flow import load_app
from autoplay import AutoPlayer
from engine import placements, Plan, plan_move_pvp, deep_adoption_allowed
from opening import opening_decision, OPENING_PIECES, quad_well_potential
from combo_rules import attack_estimate

EMPTY=(0,)*20

class DummyPool:
    def __init__(self):self.calls=[]
    def submit(self,func,args):
        self.calls.append((func,args))
        fut=Future()
        fut.set_result(func(args))
        return fut


class Offensive098Tests(unittest.TestCase):
    def app(self):
        app=object.__new__(load_app().App)
        app.running=True;app.generation=7;app.deep_revision=12;app.queue_revision=9
        app.continuation_future=None;app.continuation_token=None
        app.continuation_requested=0;app.continuation_applied=0
        app.continuation_ready=MagicMock();app.deep_pool=DummyPool()
        app.deep_stats=MagicMock();app.deep_applied=0;app.deep_received=0;app.deep_stale=0
        app.auto=AutoPlayer(turbo=True);app.auto.start();app.auto.phase='align'
        move=next(m for m in placements(EMPTY,'O') if min(x for x,y in m.cells)==3)
        app.auto.plan=Plan(move,EMPTY,'O',False,False,'T',None)
        app.key=(EMPTY,'O','T','attack',True,None,True,0,-1,app.auto.serial)
        return app

    def test_commit_next_plan_is_precomputed_without_waiting_for_fast(self):
        app=self.app();current=app.auto.plan
        app._schedule_committed_continuation(('T','I'),12,9)
        self.assertEqual(app.continuation_requested,1)
        self.assertEqual(len(app.deep_pool.calls),1)
        context=app.continuation_ready.emit.call_args.args[1]
        proposal=app.continuation_ready.emit.call_args.args[2]
        self.assertEqual((proposal.source_board,proposal.source_piece),(current.move.board,'T'))
        app.accept_continuation_result(app.generation,context,proposal,None)
        self.assertEqual(app.continuation_applied,1)
        self.assertIsNotNone(current.move.next_move)
        self.assertEqual(current.move.next_move.piece,'T')
        self.assertTrue(current.prepared_safe)
        self.assertEqual(current.observed_next,'T')

    def test_stale_or_swapped_current_rejected(self):
        app=self.app();app._schedule_committed_continuation(('T',),12,9)
        c=app.continuation_ready.emit.call_args.args
        app.deep_revision+=1
        app.accept_continuation_result(c[0],c[1],c[2],None)
        self.assertEqual(app.continuation_applied,0)
        self.assertIsNone(app.auto.plan.move.next_move)
        app=self.app();app._schedule_committed_continuation(('T',),12,9)
        c=app.continuation_ready.emit.call_args.args
        app.auto.plan=Plan(app.auto.plan.move,EMPTY,'O')
        app.accept_continuation_result(c[0],c[1],c[2],None)
        self.assertEqual(app.continuation_applied,0)

    def test_no_next_or_consumed_next_never_scheduled(self):
        app=self.app();app._schedule_committed_continuation((),12,9)
        self.assertEqual(app.continuation_requested,0)
        app.auto.plan.use_hold=True;app.auto.plan.observed_hold=None
        app._schedule_committed_continuation(('T',),12,9)
        self.assertEqual(app.continuation_requested,0)

    def test_repeated_same_revision_does_not_double_schedule(self):
        app=self.app();app._schedule_committed_continuation(('T',),12,9)
        c=app.continuation_ready.emit.call_args.args
        app.accept_continuation_result(c[0],c[1],c[2],None)
        app._schedule_committed_continuation(('T',),12,9)
        self.assertEqual(app.continuation_requested,1)

    def test_opener_up_to_24_pieces_but_not_early_rule(self):
        self.assertEqual(OPENING_PIECES,24)
        self.assertTrue(opening_decision(True,False,15,6.,0).active)
        self.assertFalse(opening_decision(True,False,24,6.,0).active)
        from combat_rules import exchange
        self.assertTrue(exchange(3,7,13,0,True).early_double)
        self.assertFalse(exchange(3,7,14,0,True).early_double)

    def test_quad_setup_requires_observed_i(self):
        ready=(0,)*16+(1022,)*4
        self.assertEqual(quad_well_potential(ready,('T',)),0)
        self.assertGreater(quad_well_potential(ready,('I',)),20)
        blocked=(0,)*16+(1023,)*4
        self.assertEqual(quad_well_potential(blocked,('I',)),0)

    def test_committed_continuation_finds_stronger_tspin_followup(self):
        # Real T-double tactical fixture, not a fabricated attack score.
        # Legacy Fast follow-up chooses a weaker T-spin single in this branch.
        from engine import plan_move
        board=(0,)*17+(975,967,1007)
        m=next(m for m in placements(board,'O') if min(x for x,y in m.cells)==0)
        legacy=plan_move(m.board,'T','I','?',False,'attack')
        refined=plan_move_pvp(m.board,'T',('I','S'),'?',False,0,-1,0,15.0,pieces_placed=16)
        self.assertIsNotNone(legacy)
        self.assertIsNotNone(refined)
        self.assertEqual((legacy.move.spin,legacy.move.lines),('full',1))
        self.assertEqual((refined.move.spin,refined.move.lines),('full',2))
        self.assertGreater(attack_estimate(refined.move).guaranteed,
                           attack_estimate(legacy.move).guaranteed)

    def test_combo_mode_submits_same_deep_engine(self):
        source=__import__('pathlib').Path(__file__).with_name('assist.py').read_text(encoding='utf8')
        self.assertIn("key[3] not in ('attack','combo')",source)

if __name__=='__main__':unittest.main()
