import unittest
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from fit_hand_trajectory import basis,fill_short_gaps,TrajectoryProblem,TRAJECTORY_SETTINGS,evaluate,align_observations,validate_trajectory,prepare
from hand_reliability import project
from regularize_handpose import EDGES
from test_regularize_handpose import hand
from test_hand_reliability import rig

class TrajectoryTests(unittest.TestCase):
    def inputs(self,n=7):
        t=np.arange(n)*.05;c=rig();velocity=np.array([.5,-.15,.1])
        x=hand()[None]+t[:,None,None]*velocity
        vt=t[None]+np.array([0.,-.025,-.008,-.026])[:,None]
        xy=np.array([project(hand()[None]+v[:,None,None]*velocity,cam).reshape(n,21,2) for v,cam in zip(vt,c)])
        lengths=np.linalg.norm(hand()[EDGES[:,0]]-hand()[EDGES[:,1]],axis=-1)
        return t,c,x,vt,xy,np.ones((4,n,21)),lengths

    def test_native_times_remove_known_motion_error(self):
        t,c,x,vt,xy,w,lengths=self.inputs()
        p=TrajectoryProblem(x,t,xy,w,c,lengths,TRAJECTORY_SETTINGS,vt)
        np.testing.assert_allclose(p.residual(x.ravel()),0,atol=1e-8)
        sync=TrajectoryProblem(x,t,xy,w,c,lengths,TRAJECTORY_SETTINGS,np.broadcast_to(t,vt.shape))
        self.assertGreater(np.linalg.norm(sync.residual(x.ravel())),10)
        positions,e=evaluate(x,t,vt,xy,c)
        self.assertLess(np.nanmax(e),1e-8)
        self.assertTrue(np.isnan(positions[1,0]).all()) # No endpoint extrapolation.

    def test_interpolated_projection_jacobian(self):
        t,c,x,vt,xy,w,lengths=self.inputs(4)
        p=TrajectoryProblem(x,t,xy,w,c,lengths,TRAJECTORY_SETTINGS,vt)
        numeric=approx_derivative(p.residual,p.observed.ravel(),method='3-point')
        np.testing.assert_allclose(p.jacobian(p.observed.ravel()).toarray(),numeric,atol=.02,rtol=1e-4)

    def test_bounded_gap_recovery_and_no_extrapolation(self):
        t,c,x,vt,xy,w,lengths=self.inputs(12);seed=x.copy()
        seed[3:5,8]=np.nan;seed[3:8,12]=np.nan;seed[:2,4]=np.nan;seed[-2:,16]=np.nan
        initial,inferred=fill_short_gaps(seed,t,.15)
        np.testing.assert_allclose(initial[3:5,8],x[3:5,8]);self.assertEqual(inferred.sum(),2)
        self.assertTrue(np.isnan(initial[3:8,12]).all());self.assertTrue(np.isnan(initial[:2,4]).all())
        self.assertTrue(np.isnan(initial[-2:,16]).all())
        # Recovery remains possible without observations during the short gap.
        xy[:,3:5,8]=np.nan;w[:,3:5,8]=0
        p=TrajectoryProblem(initial,t,xy,w,c,lengths,TRAJECTORY_SETTINGS,vt)
        f=least_squares(p.residual,p.observed.ravel(),jac=p.jacobian,loss=p.loss,max_nfev=100)
        self.assertTrue(f.success);out=initial.copy();out[p.valid]=f.x.reshape(-1,3)
        np.testing.assert_allclose(out[3:5,8],x[3:5,8],atol=1e-5)

    def test_physical_rejection_does_not_revert_good_frames_or_bridge_bad_knots(self):
        t,c,x,vt,xy,w,lengths=self.inputs()
        initial=x+.005;candidate=x.copy();candidate[3,8,2]=-2
        out,report=validate_trajectory(initial,candidate,t,xy,w,c,lengths,vt)
        self.assertTrue(report['accepted']);self.assertEqual(report['rejectedKnots'],1)
        self.assertTrue(np.isnan(out[3,8]).all());np.testing.assert_allclose(out[2],x[2])
        positions,_=evaluate(out,t,vt,xy,c)
        self.assertTrue(np.isnan(positions[1,3,8]).all())

    def test_native_seed_fallback_does_not_require_interpolated_2d(self):
        t,c,x,vt,xy,w,lengths=self.inputs(1)
        result=prepare(xy,w,c,t,vt)
        initial,seed,inferred=result[:3];fallback=result[-1]
        self.assertTrue(np.isfinite(seed).all())
        self.assertTrue(fallback.all());self.assertFalse(inferred.any())
        np.testing.assert_array_equal(initial,seed)

    def test_initializer_alignment_and_long_gap_policy(self):
        t,c,x,vt,xy,w,lengths=self.inputs()
        aligned,aw=align_observations(xy,w,vt,t)
        for k in range(4):
            truth=project(x,c[k]).reshape(xy[k].shape);good=aw[k]>0
            # Perspective motion is only approximately linear in image space.
            self.assertLess(np.max(np.linalg.norm(aligned[k][good]-truth[good],axis=-1)),.1)
        with self.assertRaises(ValueError):basis(np.array([0.,0.]),np.ones((2,1),bool),[0.],.15)

if __name__=='__main__':unittest.main()
