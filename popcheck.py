import numpy as np, pandas as pd, os
from core2 import peak_path

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
rz = np.load(os.path.join(D, "cache", "train_cost.npy"))
tz = np.load(os.path.join(D, "cache", "test_cost.npy"))
vcl = tt.vcl.values
static = vcl == 0


def stats(cost, nm):
    c = cost[:, 0].astype(np.float32)
    pk, c0 = peak_path(c)
    disp = np.linalg.norm(pk, axis=2)
    mx = disp.max(1)
    z = c[:, :, 18, 18]
    cc = cost[:, 1].astype(np.float32)
    pk2, _ = peak_path(cc)
    mx2 = np.linalg.norm(pk2, axis=2).max(1) * 4
    print(f"{nm}: n={len(c)}")
    print("   max|disp| fine  p10 %6.2f p50 %6.2f p90 %6.2f p99 %6.2f" %
          tuple(np.percentile(mx, [10, 50, 90, 99])))
    print("   max|disp| coarse p10 %6.2f p50 %6.2f p90 %6.2f p99 %6.2f" %
          tuple(np.percentile(mx2, [10, 50, 90, 99])))
    print("   zero-disp ZNCC mean over t: p10 %.3f p50 %.3f p90 %.3f" %
          tuple(np.percentile(z.mean(1), [10, 50, 90])))
    print("   peak ZNCC mean over t:      p10 %.3f p50 %.3f p90 %.3f" %
          tuple(np.percentile(c0.mean(1), [10, 50, 90])))
    print("   frac max|disp| < 1px: %.3f   < 3px: %.3f   > 10px: %.3f" %
          ((mx < 1).mean(), (mx < 3).mean(), (mx2 > 10).mean()))
    return mx, mx2, z.mean(1), c0.mean(1)


rm, rm2, rz0, rc0 = stats(rz, "TRAIN (all cells)")
print()
tm, tm2, tz0, tc0 = stats(tz, "TEST (queries)")

print()
print("=== which TRAIN subpopulation matches TEST? ===")
for lo in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
    m = vcl >= lo
    print(f"  train vcl>={lo:<5}: n={m.sum():5d} static_frac={0.0 if lo>0 else static.mean():.3f} "
          f"zerodisp p50 {np.median(rz0[m]):.3f}  max|disp| p50 {np.median(rm[m]):6.2f}")
print(f"  TEST                : n={len(tz0):5d} {'':21s}zerodisp p50 {np.median(tz0):.3f}  "
      f"max|disp| p50 {np.median(tm):6.2f}")

print()
print("=== match by quantile: find train vcl cutoff whose zero-disp median matches test ===")
target = np.median(tz0)
best = None
for q in np.arange(0.0, 0.95, 0.01):
    thr = np.quantile(vcl, q)
    m = vcl >= thr
    if m.sum() < 200: break
    d = abs(np.median(rz0[m]) - target)
    if best is None or d < best[0]:
        best = (d, q, thr, m.sum(), np.median(rz0[m]))
print(f"  best match: drop bottom {best[1]:.0%} of train by vcl (vcl>={best[2]:.4f}), "
      f"n={best[3]}, zerodisp p50 {best[4]:.3f} vs test {target:.3f}")

print()
print("=== implied test composition ===")
q = best[1]
sub = vcl >= best[2]
print("  if test ~ train[vcl>=%.4f]: static frac %.3f, lin==1 frac %.3f, turn==0 frac %.3f" %
      (best[2], (vcl[sub] == 0).mean(), (tt.lin.values[sub] == 1).mean(),
       (tt.turn.values[sub] == 0).mean()))
print("  train overall            : static frac %.3f, lin==1 frac %.3f, turn==0 frac %.3f" %
      (static.mean(), (tt.lin.values == 1).mean(), (tt.turn.values == 0).mean()))
np.save(os.path.join(D, "cache", "train_maxdisp.npy"), rm)
np.save(os.path.join(D, "cache", "test_maxdisp.npy"), tm)
np.save(os.path.join(D, "cache", "train_z0.npy"), rz0)
np.save(os.path.join(D, "cache", "test_z0.npy"), tz0)
