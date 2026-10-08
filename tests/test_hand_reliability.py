import unittest
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from hand_reliability import project,triangulate_pair,leave_one_out,mesh_surface_visibility,reliability,fast_crop_gaussian
from fit_weighted_handpose import ReprojectionProblem,SETTINGS
from regularize_handpose import EDGES
from test_regularize_handpose import hand


def rig():
    return [dict(R=np.eye(3),t=np.array([x,y,1.]),K=np.array([[800.,0,400],[0,800,300],[0,0,1]]),d=np.array([.03,-.01,.001,0,0]),
                 P=np.column_stack([np.eye(3),[x,y,1.]])) for x,y in [(-.2,0),(.2,0),(0,.2),(0,-.2)]]

class ReliabilityTests(unittest.TestCase):
    def test_fast_antialias_matches_original_pipeline(self):
        from skimage.filters import gaussian
        im=np.random.default_rng(8).integers(0,256,(64,79,3),dtype=np.uint8)
        for sigma in (.05,.4,1.,1.9):
            np.testing.assert_allclose(fast_crop_gaussian(im,sigma),gaussian(im,sigma=sigma,channel_axis=2,preserve_range=True),atol=3e-13,rtol=0)

    def test_projection_and_joint_optimizer_jacobian(self):
        cams=rig();x=np.repeat(hand()[None],3,0);x[1,8]=np.nan
        lengths=np.linalg.norm(hand()[EDGES[:,0]]-hand()[EDGES[:,1]],axis=-1)
        xy=np.array([project(x,c).reshape(3,21,2) for c in cams]);w=np.ones((4,3,21));w[:,1,8]=0
        p=ReprojectionProblem(x,np.array([0,.05,.12]),xy,w,cams,lengths,SETTINGS)
        numerical=approx_derivative(p.residual,p.observed.ravel(),method='3-point')
        np.testing.assert_allclose(p.jacobian(p.observed.ravel()).toarray(),numerical,atol=.02,rtol=1e-4)
        self.assertEqual(p.ids[1,8],-1)

    def test_leave_one_out_does_not_use_held_view(self):
        cams=rig();truth=hand()[None];xy=np.array([project(truth,c).reshape(1,21,2) for c in cams]);corrupt=xy.copy();corrupt[0]+=80
        refs,q=leave_one_out(corrupt,cams)
        np.testing.assert_allclose(refs[0],xy[0],atol=1e-6)
        np.testing.assert_allclose(triangulate_pair(xy[1],xy[2],cams[1],cams[2]),truth,atol=1e-7)

    def test_surface_occlusion_front_plane_hides_back_plane(self):
        front=np.array([[-1.,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
        back=front.copy();back[:,:2]*=.5;back[:,2]=2
        vertices=np.concatenate([front,back]);faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
        v=mesh_surface_visibility(vertices,np.array([vertices[0],vertices[4]]),np.zeros(3),faces,samples=1)
        self.assertGreater(v[0],v[1]);self.assertEqual(v[1],0)

    def test_stability_and_image_border_reduce_weights(self):
        aug=np.full((1,1,4,21,2),50.);aug[0,0,1:,8,0]+=20;aug[0,0,:,9,0]=-10
        weights,deviation,stable=reliability(aug,np.ones((1,1,21))*.5,np.ones((1,1)),(100,100))
        self.assertLess(weights[0,0,8],weights[0,0,7]);self.assertEqual(weights[0,0,9],0)

    def test_low_weight_corrupt_view_influences_fit_less(self):
        cams=rig();truth=hand()[None];initial=truth+.005
        lengths=np.linalg.norm(hand()[EDGES[:,0]]-hand()[EDGES[:,1]],axis=-1)
        xy=np.array([project(truth,c).reshape(1,21,2) for c in cams]);xy[0,:,:,0]+=40
        def fit(weight):
            p=ReprojectionProblem(initial,np.array([0.]),xy,weight,cams,lengths,SETTINGS)
            return least_squares(p.residual,p.observed.ravel(),jac=p.jacobian,max_nfev=100).x.reshape(truth.shape)
        w=np.ones((4,1,21));equal=fit(w);w[0]=.01;weighted=fit(w)
        self.assertLess(np.mean((weighted-truth)**2),np.mean((equal-truth)**2)/4)

    def test_robust_loss_derivatives_and_bone_penalties(self):
        x=hand()[None];cams=rig();xy=np.array([project(x,c).reshape(1,21,2) for c in cams])
        lengths=np.linalg.norm(hand()[EDGES[:,0]]-hand()[EDGES[:,1]],axis=-1)
        p=ReprojectionProblem(x,np.array([0.]),xy,np.ones((4,1,21)),cams,lengths,SETTINGS)
        z=np.linspace(.1,25,p.nres+p.obsres);h=1e-5
        rho=p.loss(z);plus=p.loss(z+h);minus=p.loss(z-h)
        np.testing.assert_allclose(rho[1],(plus[0]-minus[0])/(2*h),rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(rho[2],(plus[1]-minus[1])/(2*h),rtol=1e-7,atol=1e-8)
        np.testing.assert_array_equal(rho[1,x.size:p.nres],1)
        self.assertTrue((rho[1,p.nres:]<.2).all())

if __name__=='__main__':unittest.main()
