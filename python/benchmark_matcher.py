"""Measure geometric acquisition with/without ML and with full-search fallback.
Uses rendered images -> C detections, and independent test attitudes.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from evaluate import GEN,build_set,forward
from st_experiment import load_model
from st_catalog import Catalog
from st_features import Camera
from st_runtime import select
from st_matcher import Matcher,attitude_error_deg

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=20)
    ap.add_argument('--seed',type=int,default=2087);ap.add_argument('--max-catalog',type=int,default=2000)
    ap.add_argument('--out',default=str(Path(GEN)/'matcher_benchmark.json'));a=ap.parse_args()
    m,meta=load_model(GEN)
    if a.seed==meta['seed']:ap.error('Test seed must differ from training')
    catalog=Path(__file__).resolve().parents[1]/'data'/meta['catalog']['name']
    cat=Catalog.load(str(catalog));cam=Camera(**meta['camera'])
    started=time.perf_counter();matcher=Matcher(cat,a.max_catalog);index_seconds=time.perf_counter()-started
    X,Y,frames,failed=build_set(a.n,a.seed,meta,str(catalog),'rendered');rows=[]
    for x,fr in zip(X,frames):
        t=time.perf_counter();prob=forward(m,x);cells=select(prob);overhead=time.perf_counter()-t
        full=matcher.solve(fr['vectors'],diagonal_deg=cam.fov_deg(1280,960))
        short=matcher.solve(fr['vectors'],cells,cam.fov_deg(1280,960))
        for r in (full,short):
            r['error_deg']=attitude_error_deg(r.pop('rotation'),fr['R']) if r['solved'] else None
            r['correct']=bool(r['solved'] and r['error_deg']<.1)
        fallback=None
        if not short['solved']:fallback=full # identical deterministic full-search invocation
        rows.append(dict(full=full,shortlist=short,cells=cells.tolist(),classifier_seconds=overhead,
            assisted_seconds=overhead+short['seconds']+(full['seconds'] if fallback else 0),
            assisted_correct=short['correct'] if fallback is None else full['correct']))
        print('frame',len(rows),'full',full['correct'],'short',short['correct'],flush=True)
    result=dict(attempted=a.n,usable=len(rows),failed_before_matching=failed,seed=a.seed,
        catalog_limit=a.max_catalog,index_build_seconds=index_seconds,
        note='Ground reference; online times exclude shared detection and offline index construction. '
             'Fallback cost reuses measured deterministic full solve. No flight-runtime claim.',
        full_success_all=sum(r['full']['correct'] for r in rows)/a.n,
        assisted_success_all=sum(r['assisted_correct'] for r in rows)/a.n,
        mean_full_seconds=float(np.mean([r['full']['seconds'] for r in rows])) if rows else None,
        mean_assisted_seconds=float(np.mean([r['assisted_seconds'] for r in rows])) if rows else None,rows=rows)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
