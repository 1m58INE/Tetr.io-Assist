import unittest, random
from engine import *

class EngineTests(unittest.TestCase):
    def test_rotations(self):
        self.assertEqual({p:len(v) for p,v in SHAPES.items()},dict(I=2,O=1,T=4,J=4,L=4,S=2,Z=2))
        for p,rs in SHAPES.items():
            for _,s in rs: self.assertEqual(identify(s),p)
    def test_tetris(self):
        board=(0,)*16+(1023^(1<<4),)*4
        move=recommend(board,'I')
        self.assertEqual(move.lines,4); self.assertEqual(move.board,(0,)*20)
        self.assertEqual(set(move.cells),{(4,y) for y in range(16,20)})
    def test_no_overlap_and_conservation(self):
        rng=random.Random(17)
        for _ in range(30):
            b=tuple([0]*12+[rng.randrange(1023) for y in range(8)])
            for p in BASE:
                for m in placements(b,p):
                    self.assertTrue(all(0<=x<10 and 0<=y<20 and not b[y]&(1<<x) for x,y in m.cells))
                    self.assertEqual(sum(r.bit_count() for r in b)+4,sum(r.bit_count() for r in m.board)+m.lines*10)
    def test_preview(self):
        m=recommend((0,)*20,'T','I','combo')
        self.assertIsNotNone(m.next_move)
        self.assertTrue(all(not m.board[y]&(1<<x) for x,y in m.next_move.cells))
    def test_separation_and_ambiguity(self):
        g=[[0]*10 for _ in range(20)]
        for x,y in BASE['T']: g[y+1][x+3]=1
        g[19]=[1]*8+[0,0]
        b,p,c=separate(g); self.assertEqual(p,'T'); self.assertEqual(b[-1],255); self.assertEqual(sum(r.bit_count() for r in b),8)
        for x,y in BASE['O']: g[y+1][x]=1
        self.assertIsNone(separate(g))
    def test_topout(self):
        self.assertIsNone(recommend((1023,)*20,'T'))

if __name__=='__main__': unittest.main()
