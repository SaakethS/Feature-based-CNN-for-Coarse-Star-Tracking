"""plot_loss.py — draw the learning curves recorded by train.py.

Writes a standalone SVG. Deliberately no matplotlib: requirements.txt is NumPy
and SciPy only, and a loss plot is not a good reason to add a plotting stack to
a project whose point is that the training path has no framework dependency.
SVG opens in any browser and in VS Code, and diffs as text.

Each run file is gen/training_history.json as written by train.py. Passing more
than one lets two dataset sizes be compared on the same axes.

Run:
  python3 plot_loss.py --runs gen/training_history.json --out gen/loss_curve.svg
  python3 plot_loss.py --out gen/loss_curve.svg \
      --runs gen/training_history_5819.json gen/training_history_11643.json \
             gen/training_history_23319_ep60.json
"""

import argparse
import json
import os
from pathlib import Path

GEN = os.environ.get("ST_GEN_DIR", str(Path(__file__).resolve().parents[1] / "gen"))

# Chosen to stay distinguishable in both light and dark viewers and when
# printed in greyscale, which the dashed validation stroke also helps with.
COLORS = ["#2f6fbf", "#c2571a", "#3f8f5b", "#8a4fb5"]
W, H = 980, 560
L, R, T, B = 76, 320, 56, 64          # margins; R leaves room for the legend


def escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def nice_ticks(lo, hi, count=6):
    """Round tick positions covering [lo, hi]. A plain linspace would put ticks
    at values like 0.03871, which makes two runs harder to read against each
    other than ticks on a 1/2/5 decade grid."""
    if hi <= lo:
        hi = lo + 1e-9
    raw = (hi - lo) / max(1, count)
    mag = 10.0 ** int(__import__("math").floor(__import__("math").log10(raw)))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    start = step * int(lo / step) - (step if lo < 0 else 0)
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 12))
        v += step
    return ticks, step


def load(paths):
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if not data.get("history"):
            raise ValueError(f"{path} has no recorded epochs")
        runs.append(data)
    return runs


def render(runs, log_scale=True):
    epochs = [e["epoch"] for r in runs for e in r["history"]]
    losses = [v for r in runs for e in r["history"]
              for v in (e["train_loss"], e["val_loss"]) if v > 0]
    x_lo, x_hi = min(epochs), max(epochs)
    y_lo, y_hi = min(losses), max(losses)

    import math
    if log_scale:
        # Binary cross-entropy on 529 mostly-zero outputs falls by more than an
        # order of magnitude in the first few epochs. On a linear axis that
        # first drop flattens everything after it into an unreadable line.
        ty_lo, ty_hi = math.log10(y_lo), math.log10(y_hi)
        pad = (ty_hi - ty_lo) * 0.06 or 0.1
        ty_lo, ty_hi = ty_lo - pad, ty_hi + pad

        def ty(v):
            return math.log10(max(v, 1e-12))
    else:
        pad = (y_hi - y_lo) * 0.06 or 0.1
        ty_lo, ty_hi = y_lo - pad, y_hi + pad

        def ty(v):
            return v

    def px(e):
        return L + (e - x_lo) / max(1e-9, x_hi - x_lo) * (W - L - R)

    def py(v):
        return H - B - (ty(v) - ty_lo) / max(1e-9, ty_hi - ty_lo) * (H - T - B)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{L}" y="30" font-size="17" font-weight="600" fill="#1b1b1b">'
           'Cell-classifier training loss per epoch</text>',
           f'<text x="{L}" y="47" font-size="12" fill="#5a5a5a">'
           'Binary cross-entropy over 529 outputs'
           f'{" — log scale" if log_scale else ""}. '
           'Solid: training. Dashed: held-out validation.</text>']

    # Y grid.
    if log_scale:
        decade = math.floor(ty_lo)
        yticks = []
        while decade <= ty_hi:
            for m in (1, 2, 5):
                v = m * 10.0 ** decade
                if ty_lo <= math.log10(v) <= ty_hi:
                    yticks.append(v)
            decade += 1
    else:
        yticks, _ = nice_ticks(ty_lo, ty_hi)
    for v in yticks:
        y = py(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                   'stroke="#e6e6e6" stroke-width="1"/>')
        label = f"{v:g}" if v >= 0.01 else f"{v:.0e}"
        out.append(f'<text x="{L - 10}" y="{y + 4:.1f}" font-size="11" '
                   f'fill="#5a5a5a" text-anchor="end">{label}</text>')

    # X grid.
    xticks, _ = nice_ticks(x_lo, x_hi, 8)
    for v in xticks:
        if v < x_lo or v > x_hi:
            continue
        x = px(v)
        out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" '
                   'stroke="#f0f0f0" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{H - B + 20}" font-size="11" '
                   f'fill="#5a5a5a" text-anchor="middle">{v:g}</text>')

    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#9a9a9a"/>')
    out.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#9a9a9a"/>')
    out.append(f'<text x="{W - R - (W - L - R) / 2:.0f}" y="{H - 18}" font-size="12" '
               'fill="#3a3a3a" text-anchor="middle">epoch</text>')
    out.append(f'<text transform="translate(22,{(T + H - B) / 2:.0f}) rotate(-90)" '
               'font-size="12" fill="#3a3a3a" text-anchor="middle">'
               'binary cross-entropy</text>')

    legend_y = T + 6
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        hist = run["history"]
        for key, dash in (("train_loss", ""), ("val_loss", ' stroke-dasharray="5 4"')):
            pts = " ".join(f"{px(e['epoch']):.1f},{py(e[key]):.1f}" for e in hist)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                       f'stroke-width="1.9" stroke-linejoin="round"{dash}/>')
        # Mark the epoch whose weights were actually exported.
        best = run.get("best_epoch")
        if best:
            b = next(e for e in hist if e["epoch"] == best)
            bx, by = px(best), py(b["val_loss"])
            out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" fill="#ffffff" '
                       f'stroke="{color}" stroke-width="2"/>')

        out.append(f'<rect x="{W - R + 8}" y="{legend_y}" width="14" height="3" '
                   f'fill="{color}"/>')
        out.append(f'<text x="{W - R + 28}" y="{legend_y + 5}" font-size="12.5" '
                   f'font-weight="600" fill="#1b1b1b">{escape(run.get("tag", "run"))}</text>')
        legend_y += 19
        # Whether the run actually turned the corner matters more than the
        # minimum itself. A best epoch sitting on the last epoch means this run
        # did not bracket the minimum: the curve was still falling when the
        # budget ended, so the minimum has to come from somewhere else (a longer
        # survey run) rather than from this curve.
        last = hist[-1]["epoch"]
        if best and best >= last:
            verdict = f"best epoch {best} = last: minimum not bracketed here"
        elif best:
            verdict = (f"best epoch {best}, then {last - best} epochs of "
                       f"rising val loss")
        else:
            verdict = "no best epoch recorded"

        for line in (f"{run.get('train_frames', '?')} train / "
                     f"{run.get('val_frames', '?')} val frames",
                     f"best val BCE {run.get('best_val_loss', float('nan')):.5f}",
                     verdict,
                     f"top-8 recall {run.get('final_top8_recall', float('nan')):.4f}, "
                     f"shortlist {run.get('final_mean_shortlist', float('nan')):.2f}"):
            out.append(f'<text x="{W - R + 28}" y="{legend_y + 5}" font-size="11.5" '
                       f'fill="#5a5a5a">{escape(line)}</text>')
            legend_y += 16
        legend_y += 12

    out.append(f'<text x="{W - R + 8}" y="{H - B + 4}" font-size="10.5" fill="#8a8a8a">'
               'circle = exported epoch (lowest validation loss)</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[os.path.join(GEN, "training_history.json")])
    ap.add_argument("--out", default=os.path.join(GEN, "loss_curve.svg"))
    ap.add_argument("--linear", action="store_true",
                    help="linear loss axis instead of the default log axis")
    args = ap.parse_args()

    runs = load(args.runs)
    Path(args.out).write_text(render(runs, log_scale=not args.linear), encoding="utf-8")
    print(f"wrote {args.out}")
    for run in runs:
        first, last = run["history"][0], run["history"][-1]
        print(f"  {run.get('tag')}: epoch 1 val {first['val_loss']:.5f} -> "
              f"epoch {last['epoch']} val {last['val_loss']:.5f}, "
              f"best epoch {run.get('best_epoch')} at {run.get('best_val_loss'):.5f}")


if __name__ == "__main__":
    main()
