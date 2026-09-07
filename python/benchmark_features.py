"""Controlled research comparison; alternative features do NOT change flight ABI.
Uses one recorded detection dataset and one split for all feature variants.
"""
import argparse,json,os
from pathlib import Path
import numpy as np
from train import MLP
from evaluate import metrics
from st_features import fit_quantile_bins,pairwise_angles
GEN=Path(os.environ.get('ST_GEN_DIR',str(Path(__file__).resolve().parents[1]/'gen')))

def features(vectors,counts,kind,train_end):
    if kind.startswith('hist'):
        bins=int(kind[4:]);angles=[pairwise_angles(v[:n]) for v,n in zip(vectors,counts)]
        _,edges=fit_quantile_bins(np.concatenate(angles[:train_end]),n_bins=bins)
        out=[]
        for a in angles:
            h=np.histogram(a,bins=edges)[0].astype(float);out.append(h/max(h.sum(),1))
        return np.asarray(out,np.float32)
    out=[]
    for v,n in zip(vectors,counts):
        a=np.arccos(np.clip(v[:n]@v[:n].T,-1,1));np.fill_diagonal(a,np.inf)
        # Sorted nearest-neighbor separations, padded; normalized star count retained.
        nearest=np.sort(a.min(1));row=np.zeros(25);row[:n]=nearest;row[-1]=n/24
        out.append(row)
    return np.asarray(out,np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--out',default=str(GEN/'feature_comparison.json'));a=ap.parse_args()
    d=np.load(GEN/'dataset.npz');Y=d['Y'];n=len(Y);end=int(.70*n);val_end=int(.85*n)
    results=[]
    for kind in ('hist25','hist50','hist100','nearest'):
        X=features(d['vectors'],d['n_used'],kind,end);mean=X[:end].mean(0);std=X[:end].std(0);std[std<1e-6]=1
        Z=(X-mean)/std;rng=np.random.default_rng(12);model=MLP(X.shape[1],Y.shape[1],rng)
        best=float('inf');params=None
        for ep in range(a.epochs):
            order=rng.permutation(end)
            for s in range(0,end,256):
                b=order[s:s+256];model.step(Z[b],Y[b],.003)
            z=model.forward(Z[end:val_end])[4];loss=float((np.logaddexp(0,z)-Y[end:val_end]*z).mean())
            if loss<best:best=loss;params={k:getattr(model,k).copy() for k in model.params}
        for k,v in params.items():setattr(model,k,v)
        P=model.predict(Z[val_end:]);frames=[dict(n_detected=int(x)) for x in d['n_detected'][val_end:]]
        r=metrics(P,Y[val_end:],frames,len(P));r['feature']=kind;r['input_dimensions']=X.shape[1];results.append(r)
        print(kind,r['operational_hit_usable'],flush=True)
    Path(a.out).write_text(json.dumps(dict(note='Separate 70/15/15 split; usable dataset frames only, research features.',results=results),indent=2)+'\n')
if __name__=='__main__':main()
