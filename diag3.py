import numpy as np, pandas as pd, os, time
from scipy.stats import kendalltau

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
C = np.load(os.path.join(D, "cache", "train_cost.npy"))
print("cost", C.shape)
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
static = tt.vcl.values == 0
mover = ~static


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else t


def peak_path(c):
    N, T, S, _ = c.shape
    f = c.reshape(N, T, -1)
    k = f.argmax(2)
    py, px = k // S, k % S
    ar = np.arange(N)[:, None]
    tr = np.arange(T)[None, :]
    def val(dy, dx):
        return c[ar, tr, np.clip(py + dy, 0, S - 1), np.clip(px + dx, 0, S - 1)]
    c0 = val(0, 0)
    dyv = (val(-1, 0) - val(1, 0)) / (2 * (val(-1, 0) - 2 * c0 + val(1, 0)) + 1e-9)
    dxv = (val(0, -1) - val(0, 1)) / (2 * (val(0, -1) - 2 * c0 + val(0, 1)) + 1e-9)
    dyv = np.clip(dyv, -1, 1); dxv = np.clip(dxv, -1, 1)
    off = S // 2
    return np.stack([py + dyv - off, px + dxv - off], -1).astype(np.float32), c0


def descr(p, split=10):
    step = np.linalg.norm(np.diff(p, axis=1), axis=2)
    early = step[:, :split - 1].mean(1)
    late = step[:, split:].mean(1)
    net_l = p[:, -1] - p[:, split]
    path_l = step[:, split:].sum(1)
    lin = np.linalg.norm(net_l, axis=1) / (path_l + 1e-9)
    de = p[:, split] - p[:, 0]
    ne = np.linalg.norm(de, axis=1) + 1e-9
    nl = np.linalg.norm(net_l, axis=1) + 1e-9
    turn = np.arccos(np.clip((de * net_l).sum(1) / (ne * nl), -1, 1))
    net = p[:, -1] - p[:, 0]
    nn = np.linalg.norm(net, axis=1) + 1e-9
    return dict(vcl=late, lin=lin, accel=late / (early + 1e-6), turn=turn,
                vbias=net[:, 0] / nn, netmag=nn, path=step.sum(1), early=early)


t0 = time.time()
pf, cf = peak_path(C[:, 0].astype(np.float32))
pc, cc = peak_path(C[:, 1].astype(np.float32))
print("paths in %.1fs" % (time.time() - t0))
far = (np.abs(pf).max(axis=(1, 2)) > 14)[:, None, None]
pm = np.where(far, pc * 4.0, pf)
print("frac using coarse:", far.mean().round(4))

for nm, p in (("fine", pf), ("coarse*4", pc * 4), ("merged", pm)):
    d = descr(p)
    print(f"\n  {nm}: " + "  ".join(f"{c} {kt(d[c], tt[c].values):+.4f}" for c in COLS))
    print(f"    movers only: " + "  ".join(
        f"{c} {kt(d[c][mover], tt[c].values[mover]):+.4f}" for c in COLS))

d = descr(pm)
print("\n=== vs the old greedy tracker (vcl 0.439 / vbias 0.257 / lin -0.075 / turn -0.093) ===")
print("   cost-volume tracker beats it on:",
      [c for c in COLS if abs(kt(d[c], tt[c].values)) > 0.30])

print("\n=== static discrimination from cost-volume path ===")
mag = np.linalg.norm(pm, axis=2).max(1)
tot = np.linalg.norm(np.diff(pm, axis=1), axis=2).sum(1)
from sklearn.metrics import roc_auc_score
print("  AUC(static) from max|disp|:", round(roc_auc_score(static, -mag), 4))
print("  AUC(static) from path len :", round(roc_auc_score(static, -tot), 4))
print("  AUC(static) from peakconf var:", round(roc_auc_score(static, -cf.std(1)), 4))
print("  old aux best AUC was 0.7773")

print("\n=== vbias by speed stratum (cost-volume) ===")
for lo, hi in [(0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 99)]:
    m = mover & (tt.vcl.values >= lo) & (tt.vcl.values < hi)
    if m.sum() > 30:
        print(f"  vcl [{lo},{hi}) n={m.sum():5d} vbias tau {kt(d['vbias'][m], tt.vbias.values[m]):+.4f}"
              f"  vcl tau {kt(d['vcl'][m], tt.vcl.values[m]):+.4f}")

np.save(os.path.join(D, "cache", "train_cvpath.npy"), pm)
np.save(os.path.join(D, "cache", "train_cvconf.npy"), np.stack([cf, cc], 1))
