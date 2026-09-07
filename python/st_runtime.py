"""Call the actual C detector/selector on a Linux/WSL host. Calls are serialized."""
import ctypes as C
from pathlib import Path
import numpy as np
from st_cells import NUM_CELLS
from st_features import NUM_BINS
_LIB = None

def library():
    global _LIB
    if _LIB is None:
        path = Path(__file__).resolve().parents[1] / 'c/libstartracker.so'
        if not path.exists():
            raise RuntimeError('Build the host bridge first: make -C c bridge (Linux/WSL)')
        lib=C.CDLL(str(path)); fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
        up=np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS')
        lib.st_bridge_detect.argtypes=[up,C.c_float,fp,ip]
        lib.st_bridge_select.argtypes=[fp,C.c_float,C.c_float,ip]
        lib.st_bridge_pipeline.argtypes=[up,fp,C.c_float,fp,fp,ip]
        if lib.st_bridge_cells()!=NUM_CELLS or lib.st_bridge_bins()!=NUM_BINS:
            raise ValueError('C cell count differs from Python; rebuild matching artifacts')
        _LIB=lib
    return _LIB

def image_array(image):
    a=np.ascontiguousarray(image,dtype=np.uint8)
    if a.shape!=(960,1280): raise ValueError('Expected image shape (960,1280)')
    return a

def detect(image,threshold=5.0):
    out=np.zeros((24,3),np.float32);counts=np.zeros(2,np.int32)
    status=library().st_bridge_detect(image_array(image),threshold,out,counts)
    return status,out[:counts[1],:2].copy(),out[:counts[1],2].copy(),int(counts[0])

def select(prob,accept=.15,confident=.90):
    p=np.ascontiguousarray(prob,dtype=np.float32)
    if p.shape!=(NUM_CELLS,): raise ValueError('Wrong probability shape')
    ids=np.zeros(8,np.int32)
    n=library().st_bridge_select(p,accept,confident,ids)
    if n<0: raise ValueError('C rejected invalid probabilities or thresholds')
    return ids[:n].copy()

def pipeline(image,cam,threshold=5.0):
    camera=np.array([cam.fx,cam.fy,cam.cx,cam.cy,cam.k1,cam.k2],np.float32)
    prob=np.zeros(NUM_CELLS,np.float32);hist=np.zeros(NUM_BINS,np.float32);counts=np.zeros(2,np.int32)
    status=library().st_bridge_pipeline(image_array(image),camera,threshold,prob,hist,counts)
    return status,prob,hist,counts
