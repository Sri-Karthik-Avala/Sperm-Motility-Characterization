import numpy as np, pandas as pd, os
from scipy.stats import kendalltau, rankdata
from core import build_aux
from core2 import build_all

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values.astype(float) for c in COLS)
static = vcl == 0
N = len(tt)


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def cdfv(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


print("=== INFORMATION BOUND: what can PERFECT speed knowledge alone buy? ===")
order = np.argsort(vcl)
nb = 60
bins = np.array_split(order, nb)
for c, y in (("lin", lin), ("accel", accel), ("turn", turn), ("vbias", vbias)):
    f = cdfv(y)
    pred = np.zeros(N)
    for b in bins:
        pred[b] = f[b].mean()
    print(f"  E[F({c}) | TRUE vcl]  tau = {kt(pred, y):.4f}")
print("  (this is the ceiling for any model that only recovers speed)")

print()
print("=== bound given TRUE vcl AND TRUE static flag, with within-bin tie structure ===")
for c, y in (("lin", lin), ("accel", accel), ("turn", turn)):
    f = cdfv(y)
    pred = np.zeros(N)
    for b in bins:
        pred[b] = f[b].mean()
    pred = np.where(static, -1.0, pred)
    print(f"  {c}: {kt(pred, y):.4f}")

print()
print("=== how much EXTRA is in trajectory shape beyond speed? ===")
resid_lin = cdfv(lin) - np.array([cdfv(lin)[b].mean() for b in bins for _ in b])[np.argsort(order)]
print("  lin oracle 0.944 vs speed-only bound above -> the gap is pure shape information")

print()
print("=== TEST vs TRAIN distribution check (is the query population the same?) ===")
tf = np.load(os.path.join(D, "cache", "test_fine.npy"))
tc = np.load(os.path.join(D, "cache", "test_coarse.npy"))
tz = np.load(os.path.join(D, "cache", "test_cost.npy"))
rf = np.load(os.path.join(D, "cache", "train_fine.npy"))
rc = np.load(os.path.join(D, "cache", "train_coarse.npy"))
rz = np.load(os.path.join(D, "cache", "train_cost.npy"))
A1t, _, _ = build_aux(tf, tc); A2t, _, _ = build_all(tz)
At = np.concatenate([A1t, A2t], 1)
A1r, _, _ = build_aux(rf, rc); A2r, _, _ = build_all(rz)
Ar = np.concatenate([A1r, A2r], 1)
print("  train feats", Ar.shape, "test feats", At.shape)
mu, sd = Ar.mean(0), Ar.std(0) + 1e-9
z = (At.mean(0) - mu) / (sd / np.sqrt(len(At)))
z = np.nan_to_num(z)
print("  |z| of test-mean vs train-mean: median %.2f  p90 %.2f  max %.2f" %
      (np.median(np.abs(z)), np.percentile(np.abs(z), 90), np.abs(z).max()))
print("  worst 5 feature indices:", np.argsort(-np.abs(z))[:5], np.round(np.sort(-np.abs(z))[:5] * -1, 1))

zm = np.load(os.path.join(D, "cache", "train_cost.npy"))[:, 0, :, 18, 18].astype(np.float32)
zt = tz[:, 0, :, 18, 18].astype(np.float32)
print()
print("  zero-disp ZNCC mean:  train p25 %.4f p50 %.4f p75 %.4f" %
      tuple(np.percentile(zm.mean(1), [25, 50, 75])))
print("                        test  p25 %.4f p50 %.4f p75 %.4f" %
      tuple(np.percentile(zt.mean(1), [25, 50, 75])))
thr = np.percentile(zm.mean(1)[static], 50)
print("  frac above the train-static median (proxy static rate): train %.3f  test %.3f" %
      ((zm.mean(1) > thr).mean(), (zt.mean(1) > thr).mean()))
print("  actual train static rate: %.3f" % static.mean())
