import unittest
import numpy as np
from hand_reliability import project
from fit_weighted_handpose import ReprojectionProblem, SETTINGS, guard_reprojection, fit_windows
from regularize_handpose import EDGES
from test_regularize_handpose import hand
from test_hand_reliability import rig


class WeightedQualityTests(unittest.TestCase):
    def inputs(self,n=4):
        x=np.repeat(hand()[None],n,axis=0);cams=rig()
        xy=np.array([project(x,c).reshape(n,21,2) for c in cams])
        return x,cams,xy,np.ones((4,n,21))

    def test_motion_loss_derivatives_and_bounded_influence(self):
        x,cams,xy,w=self.inputs(3)
        lengths=np.linalg.norm(x[0,EDGES[:,0]]-x[0,EDGES[:,1]],axis=-1)
        p=ReprojectionProblem(x,np.array([0.,.05,.11]),xy,w,cams,lengths,SETTINGS)
        z=np.linspace(.1,100,p.nres+p.obsres);h=1e-4
        rho=p.loss(z);plus=p.loss(z+h);minus=p.loss(z-h)
        np.testing.assert_allclose(rho[1],(plus[0]-minus[0])/(2*h),rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(rho[2],(plus[1]-minus[1])/(2*h),rtol=1e-6,atol=1e-8)
        motion=p.observed.size+len(p.ba)
        self.assertTrue((rho[1,motion:p.nres]<.2).all())
        np.testing.assert_array_equal(rho[1,p.observed.size:motion],1.)

    def test_guard_reverts_whole_frame_and_catches_single_bad_joint(self):
        x,cams,xy,w=self.inputs();x[0,8]=np.nan;xy[:,0,8]=np.nan
        candidate=x.copy();candidate[1,:,0]+=.05;candidate[2,8,0]+=.10
        candidate[0,8]=0 # A missing input point must never be invented.
        out,rollback,checked,r=guard_reprojection(x,candidate,xy,w,cams,SETTINGS)
        np.testing.assert_array_equal(rollback,[False,True,True,False])
        np.testing.assert_allclose(out,x,equal_nan=True)
        self.assertTrue(checked.all());self.assertEqual(r['fallbackFrames'],2)
        self.assertIn('joint_8_regressed',r['details'][1]['reasons'])

    def test_guard_keeps_improvement_and_reports_unchecked_views(self):
        truth,cams,xy,w=self.inputs();initial=truth+.01
        out,rollback,checked,r=guard_reprojection(initial,truth,xy,w,cams,SETTINGS)
        self.assertFalse(rollback.any());np.testing.assert_allclose(out,truth)
        w[:]=0
        _,rollback,checked,r=guard_reprojection(initial,truth,xy,w,cams,SETTINGS)
        self.assertFalse(checked.any());self.assertEqual(r['uncheckedFrames'],4)

    def test_fast_observed_motion_remains_close_to_images(self):
        n=12;x,cams,_,w=self.inputs(n);times=np.arange(n)*.05
        x[:,:,0]+=(.04*np.sin(2*np.pi*times*4))[:,None]
        xy=np.array([project(x,c).reshape(n,21,2) for c in cams])
        lengths=np.linalg.norm(x[0,EDGES[:,0]]-x[0,EDGES[:,1]],axis=-1)
        result,reports=fit_windows(x,times,xy,w,cams,lengths,SETTINGS)
        self.assertTrue(all(r['converged'] for r in reports))
        self.assertLess(np.max(np.linalg.norm(result-x,axis=-1)),.005)


if __name__=='__main__':unittest.main()
