"""Experiment identity and shared frame acquisition."""
import hashlib,json
from pathlib import Path
import numpy as np
from st_sim import simulate_frame,render_image,quat_to_matrix
from st_features import select_brightest, NUM_BINS

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metadata(cam,cat,noise,mode,seed):
    from st_cells import SIN_EDGES,RING_COUNTS,NUM_CELLS
    return dict(schema_version=2,simulator_version=2,mode=mode,seed=seed,
                camera={k:float(getattr(cam,k)) for k in ('fx','fy','cx','cy','k1','k2')},
                width=1280,height=960,max_stars=24,min_stars=4,num_bins=NUM_BINS,
                num_cells=NUM_CELLS,partition=dict(sin_edges=SIN_EDGES.tolist(),counts=RING_COUNTS.tolist()),
                catalog=cat.identity,noise=vars(noise))
def acquire(cat,cam,rng,noise,mode,q=None):
    if mode=='rendered':
        from st_runtime import detect
        image,q,truth,p=render_image(cat,cam,1280,960,rng,noise,q=q)
        status,uv,flux,n=detect(image,p['detect_sigma'])
        return dict(status=status,uv=uv,flux=flux,n_detected=n,q=q,R=quat_to_matrix(q))
    fr=simulate_frame(cat,cam,1280,960,rng,noise,q=q)
    fr['n_detected']=len(fr['uv']);fr['status']=0 if len(fr['uv'])>=4 else 2
    fr['uv'],fr['flux']=select_brightest(fr['uv'],fr['flux'])
    return fr

def load_model(directory):
    directory=Path(directory);m=dict(np.load(directory/'model.npz',allow_pickle=False))
    if 'metadata_json' not in m:
        raise ValueError('Legacy model has no provenance. Retrain before using the new evaluator.')
    meta=json.loads(str(m['metadata_json']))
    if digest(directory/'bins.npy')!=meta['bins_sha256']:
        raise ValueError('Bin hash does not match trained model')
    if m['W1'].shape[1]!=meta['num_bins'] or m['W3'].shape[0]!=meta['num_cells']:
        raise ValueError('Model shapes disagree with metadata')
    return m,meta
