"""Evaluate exact C shortlist policy; failures count in unconditional hit rate."""
import argparse,json,os
from pathlib import Path
import numpy as np
import st_features as F
from st_cells import NUM_CELLS
from st_catalog import Catalog
from st_sim import NoiseModel
from st_experiment import acquire,load_model,metadata
from st_runtime import select
from make_dataset import frame_cells
GEN=os.environ.get('ST_GEN_DIR',str(Path(__file__).resolve().parents[1]/'gen'))

def forward(m,X):
    x=(X-m['in_mean'])/np.where(m['in_scale']==0,1,m['in_scale'])
    a=np.maximum(x@m['W1'].T+m['b1'],0)
    b=np.maximum(a@m['W2'].T+m['b2'],0)
    z=b@m['W3'].T+m['b3'];out=np.empty_like(z);pos=z>=0
    out[pos]=1/(1+np.exp(-z[pos]));e=np.exp(z[~pos]);out[~pos]=e/(1+e)
    return out

def random_baseline_topk(Y,k,rng=None):
    v=Y.sum(1);miss=np.ones(len(Y));N=Y.shape[1]
    for i in range(k):miss*=np.maximum(N-v-i,0)/(N-i)
    return float((1-miss).mean())

def build_set(n,seed,model_meta,catalog=None,mode=None):
    if n<1:raise ValueError('n must be positive')
    cam=F.Camera(**model_meta['camera']);noise=NoiseModel()
    for key,value in model_meta['noise'].items():setattr(noise,key,tuple(value))
    if catalog is None and model_meta['catalog']['kind']=='csv':
        catalog=str(Path(__file__).resolve().parents[1]/'data'/model_meta['catalog']['name'])
    cat=Catalog.load(catalog)
    if cat.identity!=model_meta['catalog']:raise ValueError('Evaluation catalog differs from trained model')
    mode=mode or model_meta['mode'];rng=np.random.default_rng(seed)
    edges=np.load(Path(GEN)/'bins.npy');X=[];Y=[];frames=[];failed=0
    for i in range(n):
        fr=acquire(cat,cam,rng,noise,mode)
        if fr['status']:failed+=1;continue
        vec=cam.pixels_to_vectors(fr['uv']);h=F.histogram(vec,edges)
        if h is None:failed+=1;continue
        y=np.zeros(NUM_CELLS,np.float32);y[frame_cells(cam,fr['R'],1280,960)]=1
        fr['vectors']=vec;X.append(h);Y.append(y);frames.append(fr)
    return np.asarray(X).reshape(-1,F.NUM_BINS),np.asarray(Y).reshape(-1,NUM_CELLS),frames,failed

def metrics(P,Y,frames,attempted,accept=.15,confident=.90):
    selections=[select(p,accept,confident) for p in P]
    hits=np.array([bool(Y[i,ids].any()) for i,ids in enumerate(selections)])
    sizes=np.array([len(ids) for ids in selections]);valid=len(P)
    result=dict(attempted=attempted,usable=valid,failed_before_classifier=attempted-valid,
        operational_hit_all_frames=float(hits.sum()/attempted),
        operational_hit_usable=float(hits.mean()) if valid else None,
        mean_shortlist_usable=float(sizes.mean()) if valid else None,
        empty_shortlists=int((sizes==0).sum()),accept=accept,confident=confident)
    if valid:
        top=np.argsort(-P,axis=1,kind='stable')[:,:8];got=np.take_along_axis(Y,top,axis=1)
        result.update(top8_hit_usable=float(got.any(1).mean()),random_top8=random_baseline_topk(Y,8),
            coverage_at8=float((got.sum(1)/Y.sum(1)).mean()),mean_visible_cells=float(Y.sum(1).mean()))
        counts=np.array([fr['n_detected'] for fr in frames]);cut=np.quantile(counts,.1)
        result['low_star_cutoff']=float(cut);result['low_star_hit_usable']=float(hits[counts<=cut].mean())
        high=P.max(1)>=confident
        result['high_confidence_frames']=int(high.sum())
        result['high_confidence_wrong']=int((high & ~hits).sum())
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=999);ap.add_argument('--catalog')
    ap.add_argument('--mode',choices=['analytic','rendered']);ap.add_argument('--thr',type=float,default=.15)
    ap.add_argument('--confident',type=float,default=.90);ap.add_argument('--out',default=None)
    a=ap.parse_args();m,meta=load_model(GEN)
    if a.seed==meta['seed']:ap.error('Evaluation seed must differ from dataset seed')
    X,Y,frames,failed=build_set(a.n,a.seed,meta,a.catalog,a.mode)
    P=forward(m,X);r=metrics(P,Y,frames,a.n,a.thr,a.confident)
    r.update(seed=a.seed,mode=a.mode or meta['mode'],catalog=meta['catalog'])
    # Show policy tradeoff using the actual C function, never an unlimited threshold list.
    r['policy_sweep']=[metrics(P,Y,frames,a.n,t,c) for t,c in ((.05,.9),(.15,1.01),(.15,.9),(.3,.9))]
    print(json.dumps(r,indent=2))
    if a.out:Path(a.out).write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
