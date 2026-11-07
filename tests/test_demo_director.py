"""Actual director timing without audio devices, GPU or fabricated assistant replies."""
import ast
from pathlib import Path
import unittest
import numpy as np


class DirectorPacing(unittest.TestCase):
    def simulate(self, intervals):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'demo_call.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Director')
        rate=48000;block=960
        ns={'Path':Path,'np':np,'SAMPLE_RATE':rate,'REPLY_TIMEOUT_S':5,
            '_load_wav':lambda path:np.zeros(int((2 if path.name=='first.wav' else .4)*rate))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'demo_call.py','exec'),ns)
        events=[]
        d=ns['Director']({'lead_in_s':0,'tail_s':.2,'reply_timeout_s':5,'lines':[
            {'wav':'first.wav','gap_s':0},{'wav':'next.wav','wait_reply':True,'gap_s':.2}]},
            Path('.'),lambda kind,**data:events.append((kind,data)))
        while not d.finished and d.t<8:
            now=d.t;d.next_block(block);d.agent_block(any(start<=now<end for start,end in intervals))
        return next(data['t'] for kind,data in events if kind=='line_start' and data['line']==1)

    def test_early_acknowledgment_does_not_release_followup(self):
        self.assertGreaterEqual(self.simulate([(.4,.8),(2.5,3.2)]),3.39)

    def test_reply_crossing_human_line_end_still_counts(self):
        started=self.simulate([(1.7,2.4)])
        self.assertGreaterEqual(started,2.59)
        self.assertLess(started,3)
