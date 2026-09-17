import numpy as np, pandas as pd, os
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
o = dict(np.load(os.path.join(D, "cache", "oof_aux2.npz")))
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values for c in COLS)
static = vcl == 0; mover = ~static
lin1 = (lin == 1) & mover; turn0 = (turn == 0) & mover
N = len(tt)
ps, pl1, pt0, linc = o["ps"], o["pl1"], o["pt0"], o["linc"]


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def sweep(fn, grid):
    b = None
    for g in grid:
        s = kt(fn(g), lin)
        if b is None or s > b[0]: b = (round(s, 4), g)
    return b


print("=== LIN: isolate which stage breaks ===")
print("A oracle static + oracle lin1     ",
      round(kt(np.where(static, -1e9, np.where(lin1, 1e9, linc)), lin), 4))
print("B oracle static + real pl1        ",
      sweep(lambda t: np.where(static, -1e9, np.where(pl1 > t, 1e9, linc)), np.arange(.2, .95, .05)))
print("C real static + oracle lin1       ",
      sweep(lambda t: np.where(ps > t, -1e9, np.where(lin1, 1e9, linc)), np.arange(.2, .95, .05)))
print("D real static + real pl1 (best)   ",
      round(max(kt(np.where(ps > a, -1e9, np.where(pl1 > b, 1e9, linc)), lin)
                for a in np.arange(.2, .95, .05) for b in np.arange(.2, .95, .05)), 4))
print()
print("pl1 distribution on STATIC cells: p50 %.3f p90 %.3f  frac>0.5 %.3f" %
      (np.median(pl1[static]), np.percentile(pl1[static], 90), (pl1[static] > 0.5).mean()))
print("pl1 on movers lin1=1: p50 %.3f | movers lin<1: p50 %.3f" %
      (np.median(pl1[lin1]), np.median(pl1[mover & ~lin1])))
print("ps on static: p50 %.3f | on movers: p50 %.3f" % (np.median(ps[static]), np.median(ps[mover])))
print()
print("=== how costly is each error type on lin? ===")
v = np.where(static, -1e9, np.where(lin1, 1e9, linc))
print("  perfect groups tau", round(kt(v, lin), 4))
rng = np.random.default_rng(0)
for frac in [0.02, 0.05, 0.1, 0.2]:
    w = v.copy()
    idx = rng.choice(np.where(static)[0], int(frac * static.sum()), replace=False)
    w[idx] = 1e9
    print(f"  {frac:.0%} of STATIC wrongly put in top group -> {kt(w, lin):.4f}")
for frac in [0.02, 0.05, 0.1, 0.2]:
    w = v.copy()
    idx = rng.choice(np.where(mover & ~lin1)[0], int(frac * (mover & ~lin1).sum()), replace=False)
    w[idx] = 1e9
    print(f"  {frac:.0%} of mover-lin<1 wrongly in top group -> {kt(w, lin):.4f}")
for frac in [0.05, 0.1, 0.2, 0.3]:
    w = v.copy()
    idx = rng.choice(np.where(lin1)[0], int(frac * lin1.sum()), replace=False)
    w[idx] = linc[idx]
    print(f"  {frac:.0%} of true-lin1 missed (put in middle) -> {kt(w, lin):.4f}")

print("\n=== does a 3-way joint model fix it? P(bottom)/P(mid)/P(top) ===")
pbot = ps
ptop = (1 - ps) * pl1
pmid = (1 - ps) * (1 - pl1)
argm = np.argmax(np.stack([pbot, pmid, ptop], 1), 1)
print("  joint argmax 3-group          ",
      round(kt(np.where(argm == 0, -1e9, np.where(argm == 2, 1e9, linc)), lin), 4))
print("  confusion vs truth group:")
tg = np.where(static, 0, np.where(lin1, 2, 1))
print(pd.crosstab(pd.Series(tg, name="true"), pd.Series(argm, name="pred")))

print("\n=== SANITY: is linc itself sane on the middle group? ===")
m = mover & ~lin1
print("  tau(linc, lin) on middle group only:", round(kt(linc[m], lin[m]), 4))
print("  AUC static", round(roc_auc_score(static, ps), 4),
      " AUC lin1|mover", round(roc_auc_score(lin1[mover], pl1[mover]), 4),
      " AUC lin1|all", round(roc_auc_score(lin1, pl1), 4))
