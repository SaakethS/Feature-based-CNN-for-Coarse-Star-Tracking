"""Ground-only geometric acquisition reference, not flight-qualified software.
Pair-angle index -> two-vector rotation hypotheses -> unique star matches -> SVD.
The catalog subset and hypothesis budget are explicit benchmark limitations.
"""
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from st_cells import cell_centers,SIN_EDGES,RING_COUNTS,RING_START

def fit_rotation(camera,inertial):
    u,_,vt=np.linalg.svd(camera.T@inertial)
    d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
    return (u@d@vt).T  # camera-column -> inertial-column

def attitude_error_deg(estimate,truth):
    return float(np.degrees(np.arccos(np.clip((np.trace(estimate@truth.T)-1)/2,-1,1))))

def basis(a,b):
    second=b-np.dot(a,b)*a;norm=np.linalg.norm(second)
    if norm<1e-7:return None
    second/=norm
    return np.stack([a,second,np.cross(a,second)],axis=1)

def max_cell_radius():
    # Conservative bound using meridional + azimuthal arc lengths.
    # It intentionally over-expands polar cells rather than excluding true stars.
    edges=np.arcsin(SIN_EDGES)
    radii=[]
    for i,count in enumerate(RING_COUNTS):
        lat=np.arcsin((SIN_EDGES[i]+SIN_EDGES[i+1])/2)
        dlat=max(lat-edges[i],edges[i+1]-lat)
        radii.extend([min(np.pi,dlat+np.pi/count)]*int(count))
    return np.asarray(radii)

class Matcher:
    def __init__(self,catalog,max_catalog=2000):
        self.original_ids=np.argsort(catalog.mag,kind='stable')[:max_catalog]
        self.vec=catalog.vec[self.original_ids]
        self.tree=cKDTree(self.vec);self.centers=cell_centers();self.radii=max_cell_radius()
        i,j=np.triu_indices(len(self.vec),1)
        angles=np.arccos(np.clip(np.einsum('ij,ij->i',self.vec[i],self.vec[j]),-1,1))
        order=np.argsort(angles);self.angles=angles[order]
        self.i=i[order].astype(np.int32);self.j=j[order].astype(np.int32)

    def candidates(self,cells=None,diagonal_deg=48.):
        if cells is None:return np.ones(len(self.vec),bool)
        mask=np.zeros(len(self.vec),bool)
        for cell in cells:
            radius=min(np.pi,np.radians(diagonal_deg)+self.radii[cell])
            mask|=self.vec@self.centers[cell]>=np.cos(radius)
        return mask

    def solve(self,observed,cells=None,diagonal_deg=48.,tolerance_deg=.05,max_hypotheses=5000):
        start=perf_counter();observed=np.asarray(observed,dtype=float)
        mask=self.candidates(cells,diagonal_deg);hypotheses=0;pair_candidates=0
        result=dict(solved=False,rotation=None,matches=0,catalog_stars=int(mask.sum()),
                    hypotheses=0,pair_candidates=0,seconds=0.)
        if len(observed)<4 or mask.sum()<4:
            result['seconds']=perf_counter()-start;return result
        tol=np.radians(tolerance_deg);chord=2*np.sin(tol/2)
        for a,b in combinations(range(min(6,len(observed))),2):
            base=basis(observed[a],observed[b])
            if base is None:continue
            angle=np.arccos(np.clip(observed[a]@observed[b],-1,1))
            lo,hi=np.searchsorted(self.angles,[angle-2*tol,angle+2*tol])
            indices=np.arange(lo,hi);indices=indices[mask[self.i[indices]]&mask[self.j[indices]]]
            # Closer pair angles first; identical order for shortlisted/full search.
            indices=indices[np.argsort(np.abs(self.angles[indices]-angle),kind='stable')]
            pair_candidates+=len(indices)
            for k in indices:
                for i,j in ((self.i[k],self.j[k]),(self.j[k],self.i[k])):
                    hypotheses+=1
                    if hypotheses>max_hypotheses:break
                    other=basis(self.vec[i],self.vec[j]);R=other@base.T
                    predicted=observed@R.T
                    distances,ids=self.tree.query(predicted,distance_upper_bound=chord)
                    good=np.flatnonzero(np.isfinite(distances))
                    # One observation per catalog star; no duplicate match inflation.
                    unique=[];taken=set()
                    for g in good[np.argsort(distances[good])]:
                        if int(ids[g]) not in taken:unique.append(g);taken.add(int(ids[g]))
                    if len(unique)<6:continue
                    unique=np.asarray(unique);refined=fit_rotation(observed[unique],self.vec[ids[unique]])
                    residual=np.arccos(np.clip(np.einsum('ij,ij->i',observed[unique]@refined.T,self.vec[ids[unique]]),-1,1))
                    if np.max(residual)>tol:continue
                    result.update(solved=True,rotation=refined,matches=len(unique),
                                  residual_deg=float(np.degrees(np.sqrt(np.mean(residual**2)))))
                    break
                if result['solved'] or hypotheses>=max_hypotheses:break
            if result['solved'] or hypotheses>=max_hypotheses:break
        result.update(hypotheses=min(hypotheses,max_hypotheses),pair_candidates=pair_candidates,
                      seconds=perf_counter()-start)
        return result
