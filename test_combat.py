"""Synthetic rules/controller/input tests; not live TETR.IO validation."""
import time
import unittest
from engine import BASE, Plan, placements, plan_move, projected_cells
from motion import cells_at, pose_from, step, route_to, fits, reachable
from autoplay import AutoPlayer
from windows_input import WindowsInput

EMPTY=(0,)*20
T_DOUBLE=(0,)*17+(1023^48,1023^56,1023^16)

class CombatTests(unittest.TestCase):
    def test_tspin_double_has_executable_descent_then_rotation(self):
        p=plan_move(T_DOUBLE,'T','I',mode='attack')
        self.assertEqual((p.move.spin,p.move.lines),('full',2))
        cells=cells_at('T',(3,0,0))
        route=route_to(T_DOUBLE,'T',cells,p.move.cells,p.move.spin)
        actions=[a for a,c in route]
        self.assertIn('soft',actions)
        self.assertIn(actions[-1],('cw','ccw'))
        pose=pose_from('T',cells)
        for action,expected in route:
            pose,k=step(T_DOUBLE,'T',pose,action)
            self.assertEqual(cells_at('T',pose),expected)
            self.assertTrue(fits(T_DOUBLE,'T',pose))
        self.assertEqual(cells_at('T',pose),p.move.cells)

    def test_all_empty_board_placements_reachable_by_bursts(self):
        for piece in BASE:
            for move in placements(EMPTY,piece):
                c=AutoPlayer(turbo=True);c.start();p=Plan(move,EMPTY,piece)
                pose=(3,0,0);dropped=False
                for frame in range(12):
                    cells=cells_at(piece,pose)
                    action=c.observe((EMPTY,piece,cells),p,frame*.02)
                    if action=='drop':
                        self.assertEqual(projected_cells(EMPTY,cells),tuple(sorted(move.cells)))
                        dropped=True;break
                    if action:
                        for key in action:
                            pose,_=step(EMPTY,piece,pose,key)
                self.assertTrue(dropped,(piece,move.rotation,move.cells,c.message))

    def test_tspin_controller_verifies_each_segment_and_drops_once(self):
        p=plan_move(T_DOUBLE,'T',mode='attack');c=AutoPlayer(turbo=True);c.start()
        pose=(3,0,0);actions=[]
        for frame in range(20):
            state=(T_DOUBLE,'T',cells_at('T',pose))
            action=c.observe(state,p,frame*.02)
            if action=='drop':actions.append(action);break
            if action:
                actions.extend(action)
                for key in action:pose,_=step(T_DOUBLE,'T',pose,key)
        self.assertEqual(actions,['cw','soft','cw','drop'])
        self.assertIsNone(c.observe(state,p,1))

    def test_ignored_burst_never_hard_drops_wrong_column(self):
        move=next(m for m in placements(EMPTY,'O') if min(x for x,y in m.cells)==0)
        p=Plan(move,EMPTY,'O');c=AutoPlayer(turbo=True);c.start()
        s=(EMPTY,'O',cells_at('O',(3,0,0)))
        c.observe(s,p,0)
        self.assertEqual(c.observe(s,p,.01),('left','left','left'))
        self.assertIsNone(c.observe(s,p,.05))
        self.assertIsNone(c.observe(s,p,.4))
        self.assertTrue(c.enabled);self.assertEqual(c.phase,'idle')

    def test_srs_wall_kick_and_blocked_rotation(self):
        pose=(-1,4,1) # T facing right flush against left wall
        self.assertTrue(fits(EMPTY,'T',pose))
        rotated,k=step(EMPTY,'T',pose,'ccw')
        self.assertEqual(k,1);self.assertTrue(fits(EMPTY,'T',rotated))
        self.assertIsNone(step((1023,)*20,'T',(3,1,0),'cw'))

    def test_s_tuck_requires_descent_then_rotation(self):
        # Build a known S cavity from reachable ground states; verify a real
        # overhang fixture with a bottom gap that straight drops cannot fill.
        board=(0,)*17+(564,188,639)
        straight={tuple(sorted(m.cells)) for m in placements(board,'S')}
        tucked=[(p,path) for p,path,spin in reachable(board,'S')
                if cells_at('S',p) not in straight and any(a=='soft' for a,c in path)]
        self.assertTrue(tucked)
        target=((7,19),(8,18),(8,19),(9,18))
        route=route_to(board,'S',cells_at('S',(3,0,0)),target)
        self.assertIn('soft',[a for a,c in route])
        self.assertEqual(route[-1][0],'cw')

    def test_app_sends_batch_with_bindings(self):
        from test_app_flow import load_app
        from unittest.mock import MagicMock
        app=object.__new__(load_app().App)
        move=next(m for m in placements(EMPTY,'O') if min(x for x,y in m.cells)==0)
        app.current_plan=Plan(move,EMPTY,'O');app.auto=AutoPlayer(turbo=True);app.auto.start()
        app.inputs=MagicMock();app.inputs.down=None;app.inputs.busy=False
        app.inputs.focused.return_value=True;app.auto_status=MagicMock()
        app.bindings={'left':'LEFT'}
        state=(EMPTY,'O',cells_at('O',(3,0,0)))
        app.pump_auto(state,0);app.pump_auto(state,.01)
        app.inputs.press_sequence.assert_called_once_with(('LEFT','LEFT','LEFT'))

class InputTests(unittest.TestCase):
    def driver(self):
        d=WindowsInput();d.focused=lambda:True;d.pulse_ms=1;d.gap_ms=1
        self.events=[];d._send=lambda vk,up:self.events.append((vk,up))
        return d
    def test_sequence_keyup_and_order(self):
        d=self.driver();d.press_sequence(('LEFT','LEFT','X'));d.worker.join(1)
        self.assertFalse(d.busy);self.assertIsNone(d.down)
        self.assertEqual(self.events,[(37,False),(37,True),(37,False),(37,True),(88,False),(88,True)])
    def test_cancel_releases_and_prevents_remaining_keys(self):
        d=self.driver();d.pulse_ms=100;d.press_sequence(('LEFT','X','SPACE'))
        deadline=time.perf_counter()+1
        while d.down is None and time.perf_counter()<deadline:time.sleep(.001)
        d.release()
        self.assertFalse(d.busy);self.assertIsNone(d.down)
        self.assertEqual(self.events,[(37,False),(37,True)])
    def test_focus_lost_between_pulses_cancels_tail(self):
        d=self.driver();d.focused=lambda:len(self.events)<2
        d.press_sequence(('LEFT','X','SPACE'));d.worker.join(1)
        self.assertEqual(self.events,[(37,False),(37,True)])

if __name__=='__main__':unittest.main()
