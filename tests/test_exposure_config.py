import json
import tempfile
import unittest
from pathlib import Path
from multicamera_capture import CameraConfig


class ExposureConfigTests(unittest.TestCase):
    def test_fixed_exposure_units_and_reject_invalid_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dummy=root/'dependency';dummy.touch()
            config={k:str(dummy) for k in ('python','calibration','wilorPython','aniposePython','detectorScript')}
            config['cameras']=[dict(name=f'cam{i:02d}',serial=str(i)) for i in range(1,5)]
            path=root/'config.json'
            config['exposureTimeUs']=2000.;path.write_text(json.dumps(config))
            self.assertEqual(CameraConfig.load(path).data['exposureTimeUs'],2000.)
            for bad in (0,-1,float('inf'),float('nan'),True,'2000'):
                config['exposureTimeUs']=bad;path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):CameraConfig.load(path)
            del config['exposureTimeUs'];path.write_text(json.dumps(config))
            self.assertNotIn('exposureTimeUs',CameraConfig.load(path).data)


if __name__=='__main__':unittest.main()
