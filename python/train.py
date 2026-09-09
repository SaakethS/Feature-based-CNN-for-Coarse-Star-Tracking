"""train.py — train the multi-label cell classifier.

Deliberately written in plain NumPy. The network is 100 -> 96 -> 96 -> 529, which
is small enough that a hand-written Adam loop trains it in seconds on a laptop,
and writing it out has two real advantages for this project:

  1. No framework dependency, so the exported weights cannot be reshaped or
     reordered by a version change between now and flight.
  2. Every operation here appears one-for-one in c/src/st_mlp.c, so the
     cross-check compares two things you can read side by side.

If you later want a bigger model, port this to PyTorch — but keep the export
format and the cross-check.

Loss: binary cross-entropy summed over the 529 outputs, not softmax
cross-entropy. Several cells are genuinely in the field of view at once, so
the outputs are independent yes/no questions, not a single choice.

Also writes gen/training_history.json: the per-epoch training and validation
loss, so the learning curve can be plotted (python/plot_loss.py) and two runs on
different dataset sizes can be compared without retraining either of them.

Best-validation-epoch weights are restored at the end regardless of --epochs, and
the RNG stream is seeded, so a short run that reaches the minimum produces the
same weights as a long one that passes it. Survey the curve with a long run once,
then set --epochs to the epoch it identified so the recorded provenance matches.

Run:
  python3 train.py --epochs 21
"""

import argparse
import json
from st_experiment import digest
import os
import numpy as np

GEN = os.environ.get("ST_GEN_DIR", os.path.join(os.path.dirname(__file__), "..", "gen"))
H1, H2 = 96, 96


def he_init(rng, n_out, n_in):
    """He initialization: variance 2/n_in. Scaled for ReLU, which zeroes half
    the activations and so would otherwise halve the signal variance at every
    layer."""
    return rng.normal(0.0, np.sqrt(2.0 / n_in), (n_out, n_in)).astype(np.float32)


class MLP:
    def __init__(self, n_in, n_out, rng):
        self.W1, self.b1 = he_init(rng, H1, n_in), np.zeros(H1, np.float32)
        self.W2, self.b2 = he_init(rng, H2, H1), np.zeros(H2, np.float32)
        self.W3 = he_init(rng, n_out, H2)
        # Bias the output layer negative so training starts by predicting
        # "this cell is not in view", which is true for about 95% of cells.
        # Without this the first few epochs are spent unlearning a uniform 0.5.
        self.b3 = np.full(n_out, -3.0, np.float32)
        self.params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0

    def forward(self, X):
        z1 = X @ self.W1.T + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2.T + self.b2
        a2 = np.maximum(z2, 0.0)
        z3 = a2 @ self.W3.T + self.b3
        return z1, a1, z2, a2, z3

    def predict(self, X):
        z3 = self.forward(X)[4]
        # Stable sigmoid, matching st_sigmoid() in st_mlp.c.
        out = np.empty_like(z3)
        pos = z3 >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z3[pos]))
        e = np.exp(z3[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def step(self, X, Y, lr, wd=1e-5, b1=0.9, b2=0.999, eps=1e-8):
        n = len(X)
        z1, a1, z2, a2, z3 = self.forward(X)
        p = self.predict(X)

        # d(BCE)/d(logit) for a sigmoid output is simply (p - y).
        dz3 = (p - Y) / n
        gW3, gb3 = dz3.T @ a2, dz3.sum(0)
        da2 = dz3 @ self.W3
        dz2 = da2 * (z2 > 0)
        gW2, gb2 = dz2.T @ a1, dz2.sum(0)
        da1 = dz2 @ self.W2
        dz1 = da1 * (z1 > 0)
        gW1, gb1 = dz1.T @ X, dz1.sum(0)

        grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2, W3=gW3, b3=gb3)
        self.t += 1
        for k in self.params:
            g = grads[k]
            if k.startswith("W"):
                g = g + wd * getattr(self, k)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            setattr(self, k, (getattr(self, k) -
                              lr * mh / (np.sqrt(vh) + eps)).astype(np.float32))

        # Mean binary cross-entropy, for reporting.
        eps_l = 1e-7
        return float(-(Y * np.log(p + eps_l) +
                       (1 - Y) * np.log(1 - p + eps_l)).mean())


def evaluate(model, X, Y, k=8, thr=0.15):
    """Training diagnostics: top-k hit and exact default C-policy shortlist size.
    Python implementation here avoids loading a stale C library while retraining.
    Final evaluation invokes C directly.
    """
    p=model.predict(X);topk=np.argsort(-p,axis=1,kind='stable')[:,:k]
    hit=np.take_along_axis(Y,topk,axis=1).max(axis=1)
    sizes=[]
    for row,indices in zip(p,topk):
        chosen=indices[row[indices]>=thr]
        sizes.append(1 if len(chosen) and row[chosen[0]]>=.9 else len(chosen))
    return float(hit.mean()),float(np.mean(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history", default=None,
                    help="where to write the per-epoch loss curve "
                         "(default: gen/training_history.json)")
    ap.add_argument("--tag", default=None,
                    help="label for this run inside the history file")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.lr <= 0:
        ap.error("epochs, batch and lr must be positive")

    d = np.load(os.path.join(GEN, "dataset.npz"))
    metadata = json.loads(str(d["metadata_json"]))
    if digest(os.path.join(GEN, "bins.npy")) != metadata["bins_sha256"]:
        raise ValueError("Dataset/bin mismatch")
    X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)

    # Hold out the last 15% for validation. Frames are independent random
    # attitudes, so a plain tail split is a valid holdout here.
    n_val = max(1, int(0.15 * len(X)))
    Xtr, Ytr, Xva, Yva = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]

    # Input standardization, computed on the training split only and exported
    # with the model so C applies exactly the same transform.
    mean = Xtr.mean(0).astype(np.float32)
    scale = Xtr.std(0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    Ztr, Zva = (Xtr - mean) / scale, (Xva - mean) / scale

    rng = np.random.default_rng(args.seed)
    model = MLP(X.shape[1], Y.shape[1], rng)

    best_loss = float("inf")
    best_params = None
    best_epoch = 0
    # Per-epoch record. Training loss is the mean over the epoch's minibatches,
    # so it lags the validation loss by roughly half an epoch of progress; that
    # is the usual reason an early train curve sits above the validation curve.
    history = []
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(Ztr))
        loss = 0.0
        for s in range(0, len(idx), args.batch):
            b = idx[s:s + args.batch]
            loss += model.step(Ztr[b], Ytr[b], args.lr)
        loss /= max(1, (len(idx) + args.batch - 1) // args.batch)
        logits = model.forward(Zva)[4]
        val_loss = float((np.logaddexp(0, logits) - Yva * logits).mean())
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, ep
            best_params = {k: getattr(model, k).copy() for k in model.params}
        record = dict(epoch=ep, train_loss=loss, val_loss=val_loss)
        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            rec, sl = evaluate(model, Zva, Yva)
            record.update(val_top8_recall=rec, val_mean_shortlist=sl)
            print(f"epoch {ep:3d}  loss {loss:.4f}  val loss {val_loss:.4f}  "
                  f"top-8 recall {rec:.3f}  mean shortlist {sl:.1f}")
        history.append(record)

    for k, value in best_params.items(): setattr(model, k, value)
    metadata["training"] = dict(seed=args.seed, epochs=args.epochs, best_epoch=best_epoch,
                                batch=args.batch, lr=args.lr, validation_fraction=0.15)
    metadata["dataset_sha256"] = digest(os.path.join(GEN, "dataset.npz"))
    rec, sl = evaluate(model, Zva, Yva)
    print(f"\nfinal: top-8 recall {rec:.3f}, mean shortlist {sl:.1f} of 529")

    np.savez(os.path.join(GEN, "model.npz"),
             metadata_json=json.dumps(metadata,sort_keys=True),
             W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
             W3=model.W3, b3=model.b3, in_mean=mean, in_scale=scale)
    print("wrote gen/model.npz")

    # The learning curve is a separate artifact, not part of model.npz: it is
    # diagnostic output, and nothing in the C export path may depend on it.
    history_path = args.history or os.path.join(GEN, "training_history.json")
    curve = dict(tag=args.tag or f"{len(X)} frames",
                 train_frames=int(len(Xtr)), val_frames=int(len(Xva)),
                 attempted=metadata.get("attempted"), kept=metadata.get("kept"),
                 epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed,
                 best_epoch=best_epoch, best_val_loss=best_loss,
                 final_top8_recall=rec, final_mean_shortlist=sl,
                 dataset_sha256=metadata["dataset_sha256"],
                 bins_sha256=metadata["bins_sha256"], history=history)
    with open(history_path, "w") as fh:
        json.dump(curve, fh, indent=2)
        fh.write("\n")
    print(f"wrote {history_path}")


if __name__ == "__main__":
    main()
