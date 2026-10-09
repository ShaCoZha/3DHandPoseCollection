import unittest
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from fit_joint_handpose import JointProblem, JOINT_SETTINGS, initialize, validate_solution, objective
from hand_reliability import project
from regularize_handpose import EDGES
from test_regularize_handpose import hand
from test_hand_reliability import rig


class JointReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.cams=rig();self.truth=np.repeat(hand()[None],3,0)
        self.times=np.array([0.,.05,.11]);self.w=np.ones((4,3,21))
        self.xy=np.array([project(self.truth,c).reshape(3,21,2) for c in self.cams])
        self.lengths=np.linalg.norm(hand()[EDGES[:,0]]-hand()[EDGES[:,1]],axis=-1)

    def problem(self,seed,weights=None):
        return JointProblem(seed,self.times,self.xy,self.w if weights is None else weights,
                            self.cams,self.lengths,JOINT_SETTINGS)

    def test_seed_is_not_a_three_dimensional_target(self):
        a=self.problem(self.truth);b=self.problem(self.truth+.04)
        np.testing.assert_allclose(a.residual(self.truth.ravel()),b.residual(self.truth.ravel()))
        self.assertEqual(objective(b,self.truth)['anchor'],0)
        fit=least_squares(b.residual,b.observed.ravel(),jac=b.jacobian,loss=b.loss,max_nfev=120)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.x.reshape(self.truth.shape),self.truth,atol=1e-5)

    def test_initialization_handles_occlusion_outlier_and_degenerate_pair(self):
        xy=self.xy.copy();w=self.w.copy();xy[0,:,:,0]+=100;w[0]=.001
        # Joint 8 has only one observed view: no depth is invented.
        xy[1:,:,8]=np.nan;w[1:,:,8]=0
        seed,_=initialize(xy,w,self.cams)
        self.assertTrue(np.isnan(seed[:,8]).all())
        np.testing.assert_allclose(np.delete(seed,8,axis=1),np.delete(self.truth,8,axis=1),atol=1e-6)
        cams=[self.cams[0]]*4;xy=np.repeat(self.xy[:1],4,axis=0)
        with self.assertRaises(ValueError):initialize(xy,self.w,cams)

    def test_jacobian_loss_derivatives_and_weight_outside_robust_loss(self):
        w=self.w.copy();w[0]=.01;p=self.problem(self.truth,w)
        numeric=approx_derivative(p.residual,p.observed.ravel(),method='3-point')
        np.testing.assert_allclose(p.jacobian(p.observed.ravel()).toarray(),numeric,rtol=1e-4,atol=.02)
        z=np.full(p.nres+p.obsres,100.);h=1e-4
        rho=p.loss(z)
        np.testing.assert_allclose(rho[1],(p.loss(z+h)[0]-p.loss(z-h)[0])/(2*h),rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(rho[2],(p.loss(z+h)[1]-p.loss(z-h)[1])/(2*h),rtol=1e-6,atol=1e-8)
        other=p.nres+3*21*2
        self.assertAlmostEqual(rho[0,p.nres]/rho[0,other],.01)

    def test_guard_uses_joint_objective_and_rejects_invalid_solution(self):
        seed=self.truth+.01
        out,report=validate_solution(seed,self.truth,self.times,self.xy,self.w,self.cams,self.lengths,JOINT_SETTINGS)
        self.assertTrue(report['accepted']);np.testing.assert_equal(out,self.truth)
        for bad in (self.truth+.2, np.full_like(self.truth,np.nan)):
            out,report=validate_solution(seed,bad,self.times,self.xy,self.w,self.cams,self.lengths,JOINT_SETTINGS)
            self.assertFalse(report['accepted']);np.testing.assert_equal(out,seed)


if __name__=='__main__':unittest.main()
