import numpy as np, pandas as pd, os
from scipy.stats import kendalltau, rankdata

D = r"C:\Users\srika\Downloads\eris_sperm"
W = os.path.join(D, "working")
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
F = np.load(os.path.join(W, "dump_ftr.npy"))
names = open(os.path.join(W, "dump_fnames.txt")).read().split("\n")
O = dict(np.load(os.path.join(W, "dump_oof.npz")))
O.pop("fold")
w = np.load(os.path.join(W, "iw.npy"))
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values.astype(float) for c in COLS)
TRUTH = dict(zip(COLS, [vcl, lin, accel, turn, vbias]))
static = vcl == 0
mv = ~static
N = len(tt)
rng = np.random.default_rng(1)
IW = [rng.choice(np.arange(N), size=808, replace=True, p=w) for _ in range(12)]
print("features", F.shape, "names", len(names))


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else t


def iws(v, c):
    return float(np.mean([max(0.0, kt(v[s], TRUTH[c][s])) for s in IW]))


def iws_signed(v, c):
    return float(np.mean([kt(v[s], TRUTH[c][s]) for s in IW]))


print()
print("=== top raw FEATURES per axis under the importance-weighted (test-like) estimate ===")
for c in COLS:
    sc = []
    for j, nm in enumerate(names):
        s = iws_signed(F[:, j], c)
        sc.append((abs(s), s, nm))
    sc.sort(reverse=True)
    head = O["d_" + c]
    print(f"\n  --- {c} --- model head IW={iws(head,c):.4f} (signed {iws_signed(head,c):+.4f})")
    for a, s, nm in sc[:6]:
        print(f"     {nm:22s} IW tau {s:+.4f}")

print()
print("=== ACCEL deep dive ===")
print("  masked head accc IW:", round(iws(O["accc"], "accel"), 4),
      "signed", round(iws_signed(O["accc"], "accel"), 4))
print("  -accc               :", round(iws(-O["accc"], "accel"), 4))
for nm in ["s1p_spratio", "s2p_spratio", "s5p_spratio", "s1p_steprat", "s2p_steprat",
           "s5p_steprat", "s1q_spratio", "s2q_spratio", "s5q_spratio"]:
    if nm in names:
        j = names.index(nm)
        print(f"  {nm:14s} IW {iws_signed(F[:, j], 'accel'):+.4f}   movers "
              f"{kt(F[mv, j], accel[mv]):+.4f}   all {kt(F[:, j], accel):+.4f}")

print()
print("  accel truth on the IW population:")
s = IW[0]
print("    frac ==1:", round(float((np.abs(accel[s] - 1) < 1e-3).mean()), 4),
      " <1:", round(float((accel[s] < 1 - 1e-3).mean()), 4),
      " >1:", round(float((accel[s] > 1 + 1e-3).mean()), 4),
      " ==0:", round(float((accel[s] == 0).mean()), 4))
print("    accel p10/50/90:", np.percentile(accel[s], [10, 50, 90]).round(4))

print()
print("=== TURN deep dive ===")
for nm in ["s1p_turn", "s2p_turn", "s5p_turn", "s1p_cos", "s2p_cos", "s5p_cos",
           "s1q_turn", "s2q_turn", "s5q_turn"]:
    if nm in names:
        j = names.index(nm)
        print(f"  {nm:12s} IW {iws_signed(F[:, j], 'turn'):+.4f}  movers {kt(F[mv, j], turn[mv]):+.4f}")

print()
print("=== LIN deep dive ===")
for nm in ["s1p_linraw", "s2p_linraw", "s5p_linraw", "s1p_linfit", "s2p_linfit", "s5p_linfit",
           "s1p_q_l", "s2p_q_l", "s5p_q_l", "s1p_r_l", "s2p_r_l", "s5p_r_l"]:
    if nm in names:
        j = names.index(nm)
        print(f"  {nm:12s} IW {iws_signed(F[:, j], 'lin'):+.4f}  movers {kt(F[mv, j], lin[mv]):+.4f}")

print()
print("=== best single feature vs model head, summary ===")
tot_f, tot_m = 0, 0
for c in COLS:
    bf = max(((abs(iws_signed(F[:, j], c)), iws_signed(F[:, j], c), names[j])
              for j in range(F.shape[1])))
    hm = iws(O["d_" + c], c)
    tot_f += max(0.0, bf[0])
    tot_m += hm
    print(f"  {c:6s} best-feature |IW| {bf[0]:.4f} ({bf[2]})   model head {hm:.4f}")
print(f"  sum/5: best-features {tot_f/5:.4f}   model heads {tot_m/5:.4f}")
