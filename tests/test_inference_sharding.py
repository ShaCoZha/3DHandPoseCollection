import unittest
from infer_hand_reliability import assigned_chunks


class ShardingTests(unittest.TestCase):
    def test_all_frames_owned_once_including_partial_last_chunk(self):
        for n in (0, 1, 63, 64, 65, 6394):
            for workers in (1, 2, 4):
                covered=[]
                for index in range(workers):
                    for start in assigned_chunks(n,64,index,workers):
                        covered.extend(range(start,min(start+64,n)))
                self.assertEqual(sorted(covered),list(range(n)))

    def test_invalid_assignments_fail(self):
        for index,count in ((-1,2),(2,2),(0,0)):
            with self.assertRaises(ValueError):assigned_chunks(100,64,index,count)
