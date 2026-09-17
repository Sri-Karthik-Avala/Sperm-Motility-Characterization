import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from core2 import build_all, zero_feats, peak_path

tt = pd.read_csv("train_tracks.csv").reset_index(drop=True)
C = np.load("cache/train_cost.npy")
static = (tt.vcl.values == 0)
lin1 = (tt.lin.values == 1) & ~static
c = C[:, 0].astype(np.float32)
pk, c0 = peak_path(c)
zf = zero_feats(c, c0)
names = ["zmean", "zmin", "zstd", "z0", "zlast", "zdelta", "bz", "bz_n", "tstat",
         "zhalf", "gapmean", "gapmax", "gapdelta", "r3mean", "r3min", "atanh"]
print("single-feature AUC(static vs all):")
for i, n in enumerate(names):
    a = roc_auc_score(static, zf[:, i])
    print(f"  {n:9s} {max(a, 1 - a):.4f}")
print("\nAUC(static vs lin1 movers only) - the confusable pair:")
m = static | lin1
for i, n in enumerate(names):
    a = roc_auc_score(static[m], zf[m, i])
    print(f"  {n:9s} {max(a, 1 - a):.4f}")
A, SY, SX = build_all(C)
print("\ntotal aux2 dims", A.shape, "sy nonzero", int(SY.sum()), "sx nonzero", int(SX.sum()))
