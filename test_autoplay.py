import ctypes, unittest
from autoplay import AutoPlayer
from engine import BASE, Move, Plan, norm, placements, plan_move
from windows_input import INPUT, DEFAULT, valid_binding

EMPTY=(0,)*20

def spawn(piece):return tuple((x+3,y+1) for x,y in BASE[piece])
def plan_for(piece,col=3,rotation=0,board=EMPTY):
    m=next(m for m in placements(board,piece) if m.rotation==rotation and min(x for x,y in m.cells)==col)
    return Plan(m,board,piece)

class HoldTests(unittest.TestCase):
    def test_filled_hold_prefers_four_lines(self):
        board=(0,)*16+(1023^(1<<4),)*4
        plan=plan_move(board,'O',None,'I')
        self.assertTrue(plan.use_hold);self.assertEqual(plan.move.lines,4)
        self.assertEqual(plan.source_piece,'O');self.assertEqual(plan.move.piece,'I')
    def test_empty_hold_consumes_next(self):
        board=(0,)*16+(1023^(1<<4),)*4
        p=plan_move(board,'O','I',None)
        self.assertTrue(p.use_hold);self.assertIsNone(p.move.next_move)
        self.assertEqual(p.move.lines,4)
    def test_unknown_and_unavailable(self):
        for hold,allowed in [('?',True),('I',False),(None,True)]:
            p=plan_move(EMPTY,'T',None,hold,allowed)
            self.assertFalse(p.use_hold)
    def test_same_piece_does_not_hold(self):
        self.assertFalse(plan_move(EMPTY,'T','I','T').use_hold)

class AutoTests(unittest.TestCase):
    def controller(self):
        c=AutoPlayer();c.start();return c
    def test_focus_loss_and_missing_never_send(self):
        c=self.controller();p=plan_for('O')
        self.assertIsNone(c.observe((EMPTY,'O',spawn('O')),p,0,False));self.assertTrue(c.enabled);self.assertTrue(c.paused)
        c.start();self.assertIsNone(c.observe(None,None,0));self.assertIsNone(c.observe(None,None,5));self.assertTrue(c.enabled)
    def test_single_drop_and_acknowledged_new_spawn(self):
        c=self.controller();p=plan_for('O');state=(EMPTY,'O',spawn('O'))
        self.assertIsNone(c.observe(state,p,0))
        self.assertEqual(c.observe(state,p,.02),'drop')
        for t in (.04,.08,.12):self.assertIsNone(c.observe(state,p,t))
        self.assertEqual(c.phase,'drop_wait')
        new=(p.move.board,'T',spawn('T'))
        c.observe(new,None,.2);c.observe(new,None,.22);self.assertEqual(c.phase,'idle');self.assertTrue(c.enabled)
    def test_drop_mismatch_bounded_replan_without_duplicate(self):
        # v1.1.10.4: an unchanged CURRENT after a long, gap-free pending drop
        # now requests a fresh plan instead of waiting forever. No second DROP
        # is emitted from the old plan and no lock is falsely credited.
        c=self.controller();p=plan_for('O');state=(EMPTY,'O',spawn('O'))
        c.observe(state,p,0);self.assertEqual(c.observe(state,p,.02),'drop')
        self.assertIsNone(c.observe(state,p,1.3))
        self.assertTrue(c.enabled);self.assertEqual(c.phase,'idle')
        self.assertEqual(c.confirmed_locks,0)
        self.assertEqual(c.drop_unacknowledged_replans,1)
        self.assertIsNone(c.observe(state,p,5))
    def test_movement_needs_ack(self):
        c=self.controller();p=plan_for('O',5);state=(EMPTY,'O',spawn('O'))
        c.observe(state,p,0);self.assertEqual(c.observe(state,p,.02),'right')
        self.assertIsNone(c.observe(state,p,.1))
        self.assertIsNone(c.observe(state,p,.5));self.assertTrue(c.enabled);self.assertEqual(c.phase,'idle')
    def test_hold_one_time_then_drop(self):
        c=self.controller();m=plan_for('I',3).move;p=Plan(m,EMPTY,'O',True)
        c.start('I');state=(EMPTY,'O',spawn('O'))
        c.observe(state,p,0);self.assertEqual(c.observe(state,p,.02),'hold')
        self.assertIsNone(c.observe(state,p,.05))
        new=(EMPTY,'I',spawn('I'));c.observe(new,None,.1)
        c.observe(new,None,.12)
        self.assertTrue(c.hold_used);self.assertEqual(c.held,'O')
        self.assertIsNone(c.observe(new,p,.14))
        self.assertEqual(c.observe(new,plan_for('I',3),.16),'drop')
        c.observe((m.board,'T',spawn('T')),None,.2)
        c.observe((m.board,'T',spawn('T')),None,.22)
        self.assertFalse(c.hold_used)
    def test_hold_timeout(self):
        c=self.controller();p=Plan(plan_for('I').move,EMPTY,'O',True);state=(EMPTY,'O',spawn('O'))
        c.observe(state,p,0);c.observe(state,p,.02);c.observe(state,p,.8)
        self.assertTrue(c.enabled);self.assertTrue(c.hold_used)
        self.assertEqual(c.observe(state,plan_for('O'),.84),'drop')
    def test_rotation_and_translation_to_target(self):
        c=self.controller();p=plan_for('T',6,1);cells=spawn('T')
        actions=[]
        for frame in range(100):
            action=c.observe((EMPTY,'T',cells),p,frame*.02)
            if not action:continue
            actions.append(action)
            if action=='cw':
                col=min(x for x,y in cells);row=min(y for x,y in cells)
                shape=norm([(-y,x) for x,y in norm(cells)])
                cells=tuple((x+col,y+row) for x,y in shape)
            if action in ('left','right'):
                dx=1 if action=='right' else -1;cells=tuple((x+dx,y) for x,y in cells)
            if action=='drop':break
        self.assertEqual(actions,['cw','right','right','right','drop'])
        self.assertTrue(c.enabled)
    def test_changed_board_requests_fresh_plan(self):
        c=self.controller();p=plan_for('O',5);state=(EMPTY,'O',spawn('O'))
        c.observe(state,p,0);c.observe(state,p,.02)
        c.observe(((0,)*19+(1,),'O',spawn('O')),p,.06)
        new=((0,)*19+(1,),'O',spawn('O'))
        c.observe(new,p,.08);c.observe(new,p,.1)
        self.assertTrue(c.enabled);self.assertEqual(c.phase,'idle')
        self.assertIsNone(c.observe(new,p,.14))
        self.assertEqual(c.observe(new,plan_for('O',3,board=new[0]),.16),'drop')
    def test_invalid_landing_never_drops(self):
        c=self.controller();p=plan_for('O')
        p.move.cells=tuple((x,y-1) for x,y in p.move.cells)
        s=(EMPTY,'O',spawn('O'));c.observe(s,p,0)
        self.assertIsNone(c.observe(s,p,.02));self.assertTrue(c.enabled)
    def test_input_struct_and_bindings(self):
        self.assertEqual(ctypes.sizeof(INPUT),40 if ctypes.sizeof(ctypes.c_void_p)==8 else 28)
        self.assertTrue(all(valid_binding(v) for v in DEFAULT.values()))
        self.assertFalse(valid_binding('F9'))

if __name__=='__main__':unittest.main()
