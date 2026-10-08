import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative

from regularize_handpose import DEFAULTS, EDGES, WindowProblem, constrain


def hand():
    x=np.zeros((21,3))
    for k,base in enumerate((1,5,9,13,17)):
        for j in range(4): x[base+j]=[(k-2)*.018,.045+j*.025,.01*k]
    return x


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.h=hand(); self.lengths=np.linalg.norm(self.h[EDGES[:,0]]-self.h[EDGES[:,1]],axis=1)

    def test_jacobian_matches_finite_difference(self):
        rng=np.random.default_rng(4)
        x=np.repeat(self.h[None],4,axis=0)+rng.normal(0,.001,(4,21,3));x[1,8]=np.nan
        p=WindowProblem(x,np.array([0,.05,.11,.16]),np.full((4,21),3),self.lengths,DEFAULTS)
        numerical=approx_derivative(p.residual,p.observed.ravel(),method='3-point')
        np.testing.assert_allclose(p.jacobian(p.observed.ravel()).toarray(),numerical,atol=.02,rtol=1e-4)

    def test_irregular_time_constant_velocity_has_zero_acceleration(self):
        t=np.array([0,.05,.13,.18,.28])
        x=self.h[None]+t[:,None,None]*np.array([.13,-.07,.02])
        p=WindowProblem(x,t,np.full((5,21),3),self.lengths,DEFAULTS)
        residual=p.residual(p.observed.ravel())
        np.testing.assert_allclose(residual,0,atol=1e-10)

    def test_gaps_do_not_create_temporal_constraints(self):
        t=np.array([0,.05,.5,.55]);x=np.repeat(self.h[None],4,axis=0)
        p=WindowProblem(x,t,np.full((4,21),3),self.lengths,DEFAULTS)
        self.assertEqual(len(p.tids),0)

    def test_noise_reduced_without_filling_missing_points(self):
        rng=np.random.default_rng(19);n=24;t=np.arange(n)*.05
        truth=self.h[None]+t[:,None,None]*np.array([.02,.01,0])
        noisy=truth+rng.normal(0,.005,truth.shape);noisy[8:12,8]=np.nan
        result,report=constrain(noisy,t,np.full((n,21),3),self.lengths,DEFAULTS)
        self.assertTrue(np.isnan(result[8:12,8]).all())
        self.assertLess(np.nanmean((result-truth)**2),np.nanmean((noisy-truth)**2))
        before=np.abs(np.linalg.norm(noisy[:,EDGES[:,0]]-noisy[:,EDGES[:,1]],axis=-1)-self.lengths)
        after=np.abs(np.linalg.norm(result[:,EDGES[:,0]]-result[:,EDGES[:,1]],axis=-1)-self.lengths)
        self.assertLess(np.nanmedian(after),np.nanmedian(before))

    def test_short_missing_sample_keeps_observed_points_connected(self):
        t=np.arange(5)*.05;x=np.repeat(self.h[None],5,axis=0);x[1,8]=np.nan
        p=WindowProblem(x,t,np.full((5,21),3),self.lengths,DEFAULTS)
        expected=[p.ids[0,8],p.ids[2,8],p.ids[3,8]]
        self.assertTrue(any(np.array_equal(row,expected) for row in p.tids))
        self.assertEqual(p.ids[1,8],-1)


if __name__=='__main__':unittest.main()
