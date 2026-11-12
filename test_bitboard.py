import random
import unittest
from engine import BASE, SHAPES, placements, value, Plan, plan_move
from autoplay import AutoPlayer
from motion import cells_at

def reference_placements(board,piece):
    out=[]
    for r,shape in SHAPES[piece]:
        for col in range(10-max(x for x,y in shape)):
            def fits(row):return all(y+row<20 and not board[y+row]&(1<<(x+col)) for x,y in shape)
            if not fits(0):continue
            row=0
            while fits(row+1):row+=1
            cells=tuple((x+col,y+row) for x,y in shape);rows=list(board)
            for x,y in cells:rows[y]|=1<<x
            left=[r for r in rows if r!=1023];lines=20-len(left)
            out.append((r,cells,tuple([0]*lines+left),lines))
    return out

class BitboardTests(unittest.TestCase):
    def test_exact_match_on_random_boards_including_overhangs(self):
        rng=random.Random(1707)
        boards=[(0,)*20,(1023,)*20]
        for _ in range(100):
            n=rng.randrange(1,21)
            boards.append((0,)*(20-n)+tuple(rng.randrange(1024) for _ in range(n)))
        for board in boards:
            for p in BASE:
                actual=[(m.rotation,m.cells,m.board,m.lines) for m in placements(board,p)]
                self.assertEqual(actual,reference_placements(board,p),(board,p))
            heights=[];holes=0
            for x in range(10):
                top=next((y for y in range(20) if board[y]&(1<<x)),20)
                heights.append(20-top)
                holes+=sum(not board[y]&(1<<x) for y in range(top,20))
            for mode in ('balanced','survival','attack'):
                expected=-sum(heights)*.48-holes*(10 if mode=='survival' else 7)-sum(abs(a-b) for a,b in zip(heights,heights[1:]))*.4-max(heights)*(1.4 if mode=='survival' else .55)
                if mode=='attack' and max(heights)<10 and holes<=2:expected-=heights[9]*1.8
                self.assertAlmostEqual(value(board,mode),expected)

    def test_cached_moves_do_not_share_mutable_scores(self):
        a=list(placements((0,)*20,'T'));a[0].score=999;a[0].next_move=a[-1]
        b=list(placements((0,)*20,'T'))
        self.assertEqual(b[0].score,0);self.assertIsNone(b[0].next_move)

    def confirm(self,wrong_piece=False,wrong_board=False):
        empty=(0,)*20;c=AutoPlayer(turbo=True);c.start()
        m=next(m for m in placements(empty,'O') if min(x for x,y in m.cells)==3)
        m.next_move=next(placements(m.board,'T'));p=Plan(m,empty,'O')
        s=(empty,'O',cells_at('O',(3,0,0)))
        c.observe(s,p,0);self.assertEqual(c.observe(s,p,.01),'drop')
        board=(0,)*19+(1,) if wrong_board else m.board
        piece='J' if wrong_piece else 'T'
        s=(board,piece,cells_at(piece,(3,0,0)))
        for t in (.4,.41,.42):c.observe(s,p,t)
        return c

    def test_prepared_requires_confirmed_predicted_board_and_piece(self):
        good=self.confirm();self.assertIsNotNone(good.prepared)
        self.assertEqual(good.prepared.source_piece,'T')
        self.assertIsNone(self.confirm(wrong_piece=True).prepared)
        self.assertIsNone(self.confirm(wrong_board=True).prepared)
        good.stop();self.assertIsNone(good.prepared)

if __name__=='__main__':unittest.main()
