"""make_dataset.py — build the training set.

Produces:
  gen/bins.npy      cosine-space bin edges (descending), float32
  gen/dataset.npz   X (n, NUM_BINS) histograms, Y (n, NUM_CELLS) multi-hot

Run:
  python3 make_dataset.py --n 40000 --seed 1

Two-pass design: the first pass collects raw pairwise angles so the quantile
bin edges can be fitted to the distribution the sensor actually produces; the
second pass builds histograms using those edges. Fitting the bins on the same
data you then histogram is fine here — the bins are a property of the optics
and the sky, not of any individual label.
"""

import argparse
import os
import numpy as np

import st_features as F
from st_cells import vectors_to_cells, NUM_CELLS
from st_catalog import Catalog
from st_sim import NoiseModel, simulate_frame

IMG_W, IMG_H = 1280, 960
GEN = os.environ.get("ST_GEN_DIR", os.path.join(os.path.dirname(__file__), "..", "gen"))


def frame_cells(cam, R_ci, width, height, grid=12):
    """Every cell overlapping the field of view for this attitude.

    Rather than testing all 529 cells for intersection with the frame, sample a
    grid of pixel positions across the image, map each to the sky, and take the
    set of cells hit. With a cell about 9 degrees across and a field of view of
    roughly the same scale, a 12x12 grid cannot skip a cell that covers any
    meaningful part of the frame.
    """
    us = np.linspace(0, width - 1, grid)
    vs = np.linspace(0, height - 1, grid)
    uu, vv = np.meshgrid(us, vs)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    v_cam = cam.pixels_to_vectors(uv).astype(np.float64)
    v_in = v_cam @ R_ci.T                      # camera -> inertial
    return np.unique(vectors_to_cells(v_in))


def main():
    from st_experiment import acquire,metadata,digest
    import json
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=20000)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--catalog',default=None)
    ap.add_argument('--fx',type=float,default=1800.)
    ap.add_argument('--mode',choices=['analytic','rendered'],default='rendered')
    args=ap.parse_args()
    if args.n<20 or args.fx<=0: ap.error('Need n >= 20 and fx > 0')
    os.makedirs(GEN,exist_ok=True)
    cam=F.Camera(args.fx,args.fx,IMG_W/2,IMG_H/2)
    cat=Catalog.load(args.catalog);noise=NoiseModel();rng=np.random.default_rng(args.seed)
    meta=metadata(cam,cat,noise,args.mode,args.seed)
    # Independent fit stream. Validation/test samples never set the bin edges.
    fit_rng=np.random.default_rng(np.random.SeedSequence([args.seed,5821]))
    angles=[]
    for _ in range(min(args.n,1000)):
        fr=acquire(cat,cam,fit_rng,noise,args.mode)
        if fr['status']==0: angles.append(F.pairwise_angles(cam.pixels_to_vectors(fr['uv'])))
    if not angles: raise ValueError('No usable bin-fitting frames')
    edges,_=F.fit_quantile_bins(np.concatenate(angles));np.save(os.path.join(GEN,'bins.npy'),edges)
    meta['bins_sha256']=digest(os.path.join(GEN,'bins.npy'))
    X=[];Y=[];counts=[];used=[];vectors=[];quaternions=[];failures={}
    for i in range(args.n):
        fr=acquire(cat,cam,rng,noise,args.mode)
        if fr['status']:
            key=str(fr['status']);failures[key]=failures.get(key,0)+1;continue
        vec=cam.pixels_to_vectors(fr['uv']);h=F.histogram(vec,edges)
        if h is None: failures['histogram']=failures.get('histogram',0)+1;continue
        y=np.zeros(NUM_CELLS,np.float32);y[frame_cells(cam,fr['R'],IMG_W,IMG_H)]=1
        padded=np.zeros((24,3),np.float32);padded[:len(vec)]=vec
        X.append(h);Y.append(y);counts.append(fr['n_detected']);used.append(len(vec))
        vectors.append(padded);quaternions.append(fr['q'])
        if (i+1)%500==0: print(f'generated {i+1}/{args.n}',flush=True)
    if len(X)<20: raise ValueError('Fewer than 20 usable training frames')
    meta.update(attempted=args.n,kept=len(X),failures=failures)
    np.savez_compressed(os.path.join(GEN,'dataset.npz'),X=np.asarray(X),Y=np.asarray(Y),
        n_detected=np.asarray(counts),n_used=np.asarray(used),vectors=np.asarray(vectors),
        q=np.asarray(quaternions),metadata_json=json.dumps(meta,sort_keys=True))
    print(f'kept {len(X)}/{args.n} frames; failures {failures}')
    print(f'labels per frame: mean {np.asarray(Y).sum(1).mean():.2f}')
    print(f'cells never seen: {(np.asarray(Y).sum(0)==0).sum()}/{NUM_CELLS}')

if __name__=='__main__': main()
