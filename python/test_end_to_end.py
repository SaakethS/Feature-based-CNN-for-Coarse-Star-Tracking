"""Rendered images through C inference vs C detections + Python inference."""
import json
from pathlib import Path
import numpy as np
from st_catalog import Catalog
from st_sim import NoiseModel,render_image
from st_features import Camera,histogram
from st_runtime import detect,pipeline,select
from st_experiment import load_model
from evaluate import forward

def main():
    root=Path(__file__).resolve().parents[1];m,meta=load_model(root/'gen')
    cat=Catalog.load(str(root/'data'/meta['catalog']['name']));cam=Camera(**meta['camera'])
    noise=NoiseModel();rng=np.random.default_rng(772);edges=np.load(root/'gen/bins.npy')
    max_hist=max_prob=0.;usable=0
    for _ in range(30):
        im,_,_,p=render_image(cat,cam,1280,960,rng,noise)
        status,uv,flux,n=detect(im,p['detect_sigma'])
        rc,cp,ch,counts=pipeline(im,cam,p['detect_sigma'])
        assert status==rc,(status,rc)
        if status:continue
        ph=histogram(cam.pixels_to_vectors(uv),edges);pp=forward(m,ph.astype(float))
        max_hist=max(max_hist,float(np.abs(ph-ch).max()))
        max_prob=max(max_prob,float(np.abs(pp-cp).max()))
        assert select(cp).tolist()==select(pp).tolist()
        usable+=1
    assert usable>=20
    assert max_hist<1e-6 and max_prob<2e-5,(max_hist,max_prob)
    result=dict(attempted=30,usable=usable,max_histogram_difference=max_hist,max_probability_difference=max_prob,status='PASS')
    (root/'gen/end_to_end_test.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
