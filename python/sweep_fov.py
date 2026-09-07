"""Isolated FOV experiments. Cell count stays at the compiled 529-cell ABI.
Never overwrite the active gen/model.npz or gen/bins.npy.
"""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=6000)
    ap.add_argument('--epochs',type=int,default=120);ap.add_argument('--eval-n',type=int,default=500)
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--catalog',default=str(root/'data/hipparcos_mag65.csv'))
    a=ap.parse_args()
    for fx in (1800,4000,6000):
        out=root/'experiments'/f'fx_{fx}';out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,ST_GEN_DIR=str(out))
        commands=[['make_dataset.py','--n',str(a.n),'--fx',str(fx),'--catalog',a.catalog],
                  ['train.py','--epochs',str(a.epochs)],
                  ['evaluate.py','--n',str(a.eval_n),'--out',str(out/'evaluation.json')]]
        for cmd in commands:
            subprocess.run([sys.executable,str(root/'python'/cmd[0]),*cmd[1:]],env=env,check=True)
if __name__=='__main__':main()
