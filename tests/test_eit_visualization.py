import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_eit_visualization import prepare
from export_multimodal_video import draw_eit


class EitVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.origin=1700000000000
        (self.root/'timestamp.json').write_text(json.dumps(dict(startedAtUnixMs=self.origin,stoppedAtUnixMs=self.origin+1000)))
        self.records=[]
        for i,t in enumerate([0,.01,.02,.03,1.]):
            real=[3.]*256;imag=[4.]*256
            if i==1:real[2]=float('nan')
            self.records.append(dict(t_unix=self.origin/1000+t,t_perf=t,frame_idx=i,
                shape=[16,1,16],freqs=[30000],re=real,im=imag,n_missing=int(i==1),
                inj_sequence=[[j+1,(j+1)%16+1] for j in range(16)],mea_mode='diff-skip0'))
        self.write()

    def tearDown(self):self.temp.cleanup()

    def write(self):
        (self.root/'eit_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))

    def test_preserves_missing_cells_and_real_time_gap(self):
        m=prepare(self.root);a=np.fromfile(self.root/'visualization/eit/magnitude.f32',dtype='<f4').reshape(-1,16,16)
        self.assertEqual(a.shape,(5,16,16));self.assertEqual(a[0,0,0],5)
        self.assertTrue(np.isnan(a[1,0,2]));self.assertEqual(m['invalidCells'],1)
        self.assertEqual(m['clockStatus'],'unverified');self.assertEqual(len(m['gaps']),1)
        self.assertAlmostEqual(m['gaps'][0]['durationSeconds'],.97,places=4)
        times=np.array(m['rawRelativeSeconds'])
        self.assertTrue(draw_eit(m,a,times,.01)[1])
        self.assertFalse(draw_eit(m,a,times,.5)[1])
        self.assertFalse(draw_eit(m,a,times,-.2)[1])

    def test_offset_is_explicit_and_raw_times_preserved(self):
        m=prepare(self.root,offset_ms=100,clock_status='user_offset')
        self.assertEqual(m['rawRelativeSeconds'][0],0)
        self.assertAlmostEqual(m['rawFrameCoverageSeconds'][0],.1)
        self.assertEqual(m['offsetMs'],100);self.assertEqual(m['clockStatus'],'user_offset')

    def test_rejects_nonmonotonic_timestamps(self):
        self.records[2]['t_unix']=self.records[1]['t_unix'];self.write()
        with self.assertRaisesRegex(ValueError,'increase'):prepare(self.root)

    def test_rejects_changing_frame_schema(self):
        self.records[2]['shape']=[16,2,8];self.write()
        with self.assertRaisesRegex(ValueError,'change'):prepare(self.root)


if __name__=='__main__':unittest.main()
