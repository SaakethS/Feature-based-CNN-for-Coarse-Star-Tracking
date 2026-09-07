"""Run after make -C c all bridge. No network needed."""
import ctypes as C
import tempfile,unittest,json
from pathlib import Path
import numpy as np
from st_runtime import select,detect,library,pipeline
from st_catalog import Catalog
from st_features import Camera
from st_sim import NoiseModel,render_image
from st_matcher import fit_rotation,attitude_error_deg
from st_experiment import load_model

class RegressionTests(unittest.TestCase):
    def test_selector_caps_and_confidence(self):
        p=np.zeros(529,np.float32);p[:20]=.2
        self.assertEqual(select(p).tolist(),list(range(8)))
        p[12]=.95;self.assertEqual(select(p).tolist(),[12])
        self.assertEqual(len(select(p,confident=1.01)),8)
        self.assertEqual(len(select(np.zeros(529))),0)
    def test_selector_boundary_and_nonfinite(self):
        p=np.zeros(529,np.float32);p[13]=np.float32(.15)
        self.assertEqual(select(p).tolist(),[13])
        p[2]=np.nan
        with self.assertRaises(ValueError):select(p)
    def test_catalog_bad_path(self):
        with self.assertRaises(FileNotFoundError):Catalog.load('/does-not-exist/catalog.csv')
    def test_catalog_bad_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.csv';p.write_text('ra_deg,dec_deg,mag\n0,95,4\n')
            with self.assertRaises(ValueError):Catalog.load(str(p))
    def test_blank_low_count_overflow(self):
        image=np.full((960,1280),20,np.uint8)
        self.assertEqual(detect(image)[0],2)
        for n in range(3):image[10:13,10+10*n:13+10*n]=100
        self.assertEqual(detect(image)[0],2)
        for n in range(520):
            x=10+(n%100)*10;y=30+(n//100)*10;image[y:y+3,x:x+3]=100
        self.assertEqual(detect(image)[0],3)
        status,prob,hist,counts=pipeline(image,Camera(1800,1800,640,480))
        self.assertEqual(status,3)
    def test_detector_threshold_effect(self):
        rng=np.random.default_rng(2);image=np.clip(rng.normal(20,2,(960,1280)),0,255).astype(np.uint8)
        for n in range(10):image[20:23,20+15*n:23+15*n]=30
        self.assertEqual(detect(image,3)[0],0)
        self.assertEqual(detect(image,10)[0],2)
    def test_fractional_psf_centers(self):
        cam=Camera(1800,1800,640,480)
        uv=np.array([[200.2,200.3],[400.7,220.8],[700.3,500.6],[900.8,700.2]])
        vec=cam.pixels_to_vectors(uv).astype(float)
        cat=Catalog(vec,np.full(4,2.))
        noise=NoiseModel();noise.read_noise=(0,0);noise.hot_pixel_rate=(0,0)
        noise.psf_sigma_px=(1.2,1.2);noise.background=(20,20)
        image,_,_,_=render_image(cat,cam,1280,960,np.random.default_rng(1),noise,q=np.array([0,0,0,1]))
        status,got,_,_=detect(image,5)
        self.assertEqual(status,0)
        errors=np.linalg.norm(got[:,None,:]-uv[None,:,:],axis=2).min(1)
        self.assertLess(errors.max(),.15)
    def test_rotation_fit(self):
        a=np.array([[1.,0,0],[0,1,0],[0,0,1],[1,1,1]])
        a/=np.linalg.norm(a,axis=1)[:,None]
        theta=.7;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        self.assertLess(attitude_error_deg(fit_rotation(a,a@R.T),R),1e-5)
    def test_bins_integrity(self):
        import os,shutil
        gen=Path(__file__).resolve().parents[1]/'gen'
        with tempfile.TemporaryDirectory() as d:
            for name in ('model.npz','bins.npy'):shutil.copy2(gen/name,Path(d)/name)
            load_model(d)
            np.save(Path(d)/'bins.npy',np.zeros(26))
            with self.assertRaises(ValueError):load_model(d)
    def test_equal_flux_selection(self):
        # The old C selection sort changed tie order after swapping a brighter star.
        class Star(C.Structure):
            _fields_=[('x',C.c_float),('y',C.c_float),('flux',C.c_float),('n_pixels',C.c_uint16)]
        stars=(Star*4)(Star(0,0,1,3),Star(1,0,1,3),Star(2,0,2,3),Star(3,0,1,3))
        lib=library();lib.st_select_brightest.argtypes=[C.POINTER(Star),C.c_int,C.c_int]
        lib.st_select_brightest(stars,4,3)
        self.assertEqual([stars[i].x for i in range(3)],[2,0,1])

if __name__=='__main__':unittest.main()
