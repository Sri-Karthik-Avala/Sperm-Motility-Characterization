import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

tt = pd.read_csv("train_tracks.csv").reset_index(drop=True)
fine = np.load("cache/train_fine.npy")
vcl = tt.vcl.values
static = vcl == 0
F = fine.astype(np.int16)
c = 24
for R, nm in ((3, "7x7"), (6, "13x13")):
    p = F[:, :, c - R:c + R + 1, c - R:c + R + 1]
    dmax = np.abs(p - p[:, :1]).max(axis=(1, 2, 3))
    dmean = np.abs(p - p[:, :1]).mean(axis=(1, 2, 3))
    print(f"--- central {nm} ---")
    print("  STATIC   : max|f_t-f_0| p50 %5.1f p90 %5.1f | frac EXACTLY identical %.4f" %
          (np.median(dmax[static]), np.percentile(dmax[static], 90), (dmax[static] == 0).mean()))
    for lo, hi in [(0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 99)]:
        m = (~static) & (vcl >= lo) & (vcl < hi)
        print(f"  vcl[{lo},{hi}) n={m.sum():5d}: p50 {np.median(dmax[m]):5.1f} "
              f"p10 {np.percentile(dmax[m], 10):5.1f} | frac identical {(dmax[m]==0).mean():.4f}")
    print("  AUC(static) from -dmax:", round(roc_auc_score(static, -dmax), 4),
          " from -dmean:", round(roc_auc_score(static, -dmean), 4))

print("\n--- is the WHOLE 48x48 patch identical for static cells? ---")
dall = np.abs(F - F[:, :1]).max(axis=(1, 2, 3))
print("  static frac identical:", (dall[static] == 0).mean().round(4),
      " mover frac identical:", (dall[~static] == 0).mean().round(4))

print("\n--- per-frame noise level: std of a flat background region ---")
bg = F[:, :, 0:8, 0:8].astype(np.float32)
print("  temporal std of a corner 8x8 patch, median over cells:",
      round(float(np.median(bg.std(axis=1).mean(axis=(1, 2)))), 4))

print("\n--- do OTHER cells move through the patch? count of changed pixels ---")
ch = (np.abs(F - F[:, :1]).max(axis=1) > 3).sum(axis=(1, 2))
print("  static: p50 %d changed px | movers: p50 %d" %
      (np.median(ch[static]), np.median(ch[~static])))
