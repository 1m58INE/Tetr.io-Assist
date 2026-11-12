"""v1.1.9 fail-closed solver isolation; no Windows input or real CC2 binary."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from engine import Plan, placements, plan_move_pvp_worker
from engine_policy import (configured_mode, normalized_worker_mode,
                           exclusive_cc2_safe, hybrid_wait_for_cc2)
from config_security import sanitize_settings, save_settings, load_settings
from tbp_bridge import as_pvp_plan
from test_tbp_bridge1170 import tbp_coordinates

EMPTY = (0,)*20
ARGS=(EMPTY,'T',('I','O'),None,True,0,-1,0,9,0,0,20,0,False,
      'standard','multiplier','auto',False,0,None)


def external_plan():
    m=next(m for m in placements(EMPTY,'T') if m.rotation==0)
    p=as_pvp_plan(EMPTY,'T',('I','O'),None,True,0,-1,0,
                  [tbp_coordinates('T',m.rotation,m.cells)],1,1)
    assert p is not None
    return p


class EnginePolicyTests(unittest.TestCase):
    def test_four_modes_and_legacy_migration(self):
        self.assertEqual(configured_mode({}), 'internal')
        self.assertEqual(configured_mode({'external_tbp':True}), 'hybrid')
        self.assertEqual(configured_mode({'bundled_tbp':True}), 'bundled')
        self.assertEqual(configured_mode({'external_tbp':True,'engine_mode':'internal'}),'internal')
        self.assertEqual(normalized_worker_mode(True),'hybrid')
        self.assertEqual(normalized_worker_mode(False),'internal')
        self.assertEqual(normalized_worker_mode('__arbitrary__'),'internal')
        self.assertNotIn('engine_mode', sanitize_settings({'engine_mode':'__arbitrary__'}))
        for mode in ('internal','cc2','hybrid','bundled'):
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'config.json'
                save_settings(path,{'engine_mode':mode})
                config,err=load_settings(path)
                self.assertIsNone(err)
                self.assertEqual(config['engine_mode'],mode)

    def test_worker_cc2_only_never_falls_back_to_native(self):
        with patch('tbp_bridge.maybe_external_plan',return_value=None) as external,\
             patch('engine.plan_move_pvp',return_value='NATIVE') as native:
            self.assertIsNone(plan_move_pvp_worker(ARGS,'cc2'))
            external.assert_called_once()
            native.assert_not_called()

    def test_worker_cc2_only_exception_never_falls_back(self):
        with patch('tbp_bridge.maybe_external_plan',side_effect=RuntimeError('timeout')),\
             patch('engine.plan_move_pvp',return_value='NATIVE') as native:
            self.assertIsNone(plan_move_pvp_worker(ARGS,'cc2'))
            native.assert_not_called()

    def test_worker_hybrid_fallback_and_internal_isolation(self):
        with patch('tbp_bridge.maybe_external_plan',return_value=None) as ext,\
             patch('engine.plan_move_pvp',return_value='NATIVE') as native:
            self.assertEqual(plan_move_pvp_worker(ARGS,'hybrid'),'NATIVE')
            ext.assert_called_once();native.assert_called_once()
            ext.reset_mock();native.reset_mock()
            self.assertEqual(plan_move_pvp_worker(ARGS,'internal'),'NATIVE')
            ext.assert_not_called();native.assert_called_once()

    def test_worker_cc2_only_returns_actual_provenance(self):
        plan=external_plan()
        with patch('tbp_bridge.maybe_external_plan',return_value=plan),\
             patch('engine.plan_move_pvp',return_value='NATIVE') as native:
            self.assertIs(plan_move_pvp_worker(ARGS,'cc2'),plan)
            native.assert_not_called()

    def test_cc2_direct_accepts_safe_same_target_without_fast_relative_scoring(self):
        p=external_plan()
        fast=Plan(p.move,p.source_board,p.source_piece,observed_hold=None)
        self.assertTrue(exclusive_cc2_safe(fast,p,0))
        self.assertTrue(exclusive_cc2_safe(None,p,0))
        self.assertFalse(exclusive_cc2_safe(fast,p,4))
        self.assertFalse(exclusive_cc2_safe(fast,Plan(p.move,EMPTY,'T'),0))

    def test_hybrid_no_wait_under_pressure_or_unknown_hold(self):
        kw=dict(pending=True,phase='idle',focused=True,vision_ok=True,
                incoming=0,incoming_known=True,hold='I',allow_hold=True)
        self.assertTrue(hybrid_wait_for_cc2(1.09,1.00,**kw))
        self.assertFalse(hybrid_wait_for_cc2(1.2,1.00,**kw))
        for change in (dict(incoming=1),dict(hold='?'),dict(pending=False),
                       dict(focused=False),dict(vision_ok=False),
                       dict(phase='align'),dict(incoming_known=False)):
            args={**kw,**change}
            self.assertFalse(hybrid_wait_for_cc2(1.09,1.00,**args))

    def test_cc2_only_never_runs_fast_controller_plan(self):
        from test_app_flow import load_app
        from autoplay import AutoPlayer
        mod=load_app()
        app=object.__new__(mod.App)
        app.solver_buttons={'cc2':SimpleNamespace(isChecked=lambda:True)}
        app.auto=AutoPlayer();app.auto.start(None)
        app.inputs=MagicMock();app.inputs.down=None;app.inputs.busy=False
        app.key=(EMPTY,'T','I','attack',True,None,True,0,-1,0)
        app.current_plan=Plan(external_plan().move,EMPTY,'T')  # Fast provenance
        app.detected_at=10.0;app.events=[];app.session_at=10.0
        app.stall_monitor=None;app.cc2_strict_waits=0;app.running=True
        app.auto_status=MagicMock();app._run_state='ON';app.run_chip=MagicMock()
        app.pump_auto((EMPTY,'T',()),10.01)
        self.assertEqual(app.auto.phase,'idle')
        self.assertEqual(app.cc2_strict_waits,1)
        self.assertEqual(app.auto.confirmed_locks,0)
        app.pump_auto((EMPTY,'T',()),12.1)
        self.assertFalse(app.auto.enabled,'CC2-only must stop rather than use Fast')
        self.assertIn('CC2 단독',app.auto.message)

    def test_cc2_only_adopts_exact_fast_target_as_cc2_source(self):
        from test_app_flow import load_app
        from autoplay import AutoPlayer
        mod=load_app()
        app=object.__new__(mod.App)
        app.solver_buttons={'cc2':SimpleNamespace(isChecked=lambda:True)}
        app.auto=AutoPlayer();app.auto.start(None)
        app.cfg={};app.running=True;app.generation=0
        app.deep_revision=1;app.queue_revision=1;app.deep_future=None
        app.deep_pending=None;app.deep_failures=0;app.deep_received=0
        app.deep_stale=0;app.deep_applied=0;app.external_received=0
        app.external_selected=0;app.external_unadopted=0;app.cc2_no_plan=0
        app.bundled_received=0;app.bundled_unadopted=0
        app.continuation_future=None;app.key=(EMPTY,'T','I','attack',True,None,True,0,-1,0)
        app.inputs=MagicMock();app.inputs.down=None;app.inputs.busy=False
        p=external_plan();p.deep_revision=1;p.queue_revision=1
        app.current_plan=Plan(p.move,EMPTY,'T')
        app.preview=MagicMock();app.overlay=MagicMock();app.pvp_status=MagicMock()
        app.latency_status=MagicMock();app.deep_stats=MagicMock()
        app._update_deep_stats=MagicMock();app._update_latency_text=MagicMock()
        app._set_combat_status=MagicMock();app.events=[];app.session_at=0
        app.garbage_filter=MagicMock();app.incoming_garbage=0
        app.accept_deep_result(0,(app.key,1,1,('I',),0),p,40.0,None)
        self.assertIs(app.current_plan,p)
        self.assertEqual(app.external_selected,1)
        self.assertEqual(app.deep_applied,1)
        self.assertEqual(app.external_unadopted,0)

    def test_cc2_only_failure_stops_without_native_source(self):
        from test_app_flow import load_app
        from autoplay import AutoPlayer
        mod=load_app()
        app=object.__new__(mod.App)
        app.solver_buttons={'cc2':SimpleNamespace(isChecked=lambda:True)}
        app.auto=AutoPlayer();app.auto.start(None)
        app.cfg={};app.running=True;app.generation=0
        app.deep_revision=1;app.queue_revision=1;app.deep_future=None
        app.deep_pending=None;app.deep_failures=0;app.deep_received=0
        app.deep_stale=0;app.deep_applied=0;app.external_received=0
        app.external_selected=0;app.external_unadopted=0;app.cc2_no_plan=0
        app.bundled_received=0;app.bundled_unadopted=0
        app.continuation_future=None;app.key=(EMPTY,'T','I','attack',True,None,True,0,-1,0)
        app.current_plan=Plan(external_plan().move,EMPTY,'T')
        app.inputs=MagicMock();app.inputs.down=None;app.inputs.busy=False
        app.auto_status=MagicMock();app.deep_stats=MagicMock()
        app._update_deep_stats=MagicMock();app.events=[];app.session_at=0
        app.accept_deep_result(0,(app.key,1,1,('I',),0),None,100.0,None)
        self.assertEqual(app.cc2_no_plan,1)
        self.assertFalse(app.auto.enabled)
        self.assertIn('내부 엔진 미사용',app.auto.message)


if __name__ == '__main__':
    unittest.main()
