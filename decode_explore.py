import numpy as np, pandas as pd, os
from scipy.stats import kendalltau

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
o = dict(np.load(os.path.join(D, "cache", "oof_aux2.npz")))
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values for c in COLS)
static = vcl == 0
mover = ~static
lin1 = (lin == 1) & mover
turn0 = (turn == 0) & mover
N = len(tt)


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def F(v, x):
    return (np.sum(v < x) + 0.5 * np.sum(v == x)) / len(v)


print("marginal CDF anchors: lin(0)=%.4f lin(1)=%.4f turn(0)=%.4f vcl(0)=%.4f" %
      (F(lin, 0), F(lin, 1), F(turn, 0), F(vcl, 0)))

ps, pl1, pt0 = o["ps"], o["pl1"], o["pt0"]
pmid_l = (1 - ps) * (1 - pl1)
plin1 = (1 - ps) * pl1

print("\n=== LIN axis: decode variants ===")
print("  oracle group decode            ",
      round(kt(np.where(static, -1e9, np.where(lin1, 1e9, lin)), lin), 4))
print("  rank by p_lin1 alone           ", round(kt(pl1 * (1 - ps), lin), 4))
print("  rank by expected-quantile      ",
      round(kt(ps * F(lin, 0) + pmid_l * 0.30 + plin1 * F(lin, 1), lin), 4))
eq = ps * F(lin, 0) + pmid_l * (0.10 + 0.30 * (1 / (1 + np.exp(-o["linc"])))) + plin1 * F(lin, 1)
print("  expected-quantile + linc       ", round(kt(eq, lin), 4))
best = None
for ts in np.arange(0.2, 0.85, 0.05):
    for tl in np.arange(0.3, 0.95, 0.05):
        v = np.where(ps > ts, -1e9, np.where(pl1 > tl, 1e9, o["linc"]))
        s = kt(v, lin)
        if best is None or s > best[0]:
            best = (round(s, 4), round(ts, 2), round(tl, 2))
print("  best hard-threshold decode     ", best)
best2 = None
for tl in np.arange(0.3, 0.98, 0.04):
    v = eq.copy()
    v[pl1 > tl] = 10.0
    v[ps > 0.5] = -10.0
    s = kt(v, lin)
    if best2 is None or s > best2[0]:
        best2 = (round(s, 4), round(tl, 2))
print("  expected-quantile + snap top   ", best2)

print("\n=== is the ceiling the classifier? sweep synthetic AUC ===")
from sklearn.metrics import roc_auc_score
print("  actual AUC(lin1|mover) =", round(roc_auc_score(lin1[mover], pl1[mover]), 4))
rng = np.random.default_rng(0)
for noise in [0.0, 0.3, 0.6, 1.0, 1.5, 2.5]:
    sc = lin1.astype(float) + rng.normal(0, noise, N)
    p = 1 / (1 + np.exp(-(sc - 0.5) * 4))
    auc = roc_auc_score(lin1[mover], p[mover])
    bb = None
    for tl in np.arange(0.2, 0.95, 0.05):
        v = np.where(static, -1e9, np.where(p > tl, 1e9, o["linc"]))
        s = kt(v, lin)
        if bb is None or s > bb: bb = s
    print(f"    synthetic AUC {auc:.3f} -> lin tau {bb:.4f}  (with ORACLE static)")

print("\n=== TURN axis ===")
print("  oracle group decode            ",
      round(kt(np.where(static | turn0, -1e9, turn), turn), 4))
bb = None
for ts in np.arange(0.2, 0.85, 0.05):
    for t2 in np.arange(0.2, 0.95, 0.05):
        v = np.where((ps > ts) | (pt0 > t2), -1e9, o["turnc"])
        s = kt(v, turn)
        if bb is None or s > bb[0]: bb = (round(s, 4), round(ts, 2), round(t2, 2))
print("  best hard-threshold            ", bb)
eqt = ps * F(turn, 0) + (1 - ps) * (pt0 * F(turn, 0) + (1 - pt0) * (0.65 + 0.3 * (1 / (1 + np.exp(-o["turnc"])))))
print("  expected-quantile              ", round(kt(eqt, turn), 4))
bb = None
for t2 in np.arange(0.2, 0.98, 0.04):
    v = eqt.copy(); v[(pt0 > t2) | (ps > 0.5)] = -10.0
    s = kt(v, turn)
    if bb is None or s > bb[0]: bb = (round(s, 4), round(t2, 2))
print("  expected-quantile + snap       ", bb)

print("\n=== ACCEL axis ===")
a0, a1, a2 = o["a0"], o["a1"], o["a2"]
acls = np.full(N, 1); acls[mover & (accel < 1 - 1e-3)] = 0; acls[mover & (accel > 1 + 1e-3)] = 2
print("  oracle 3-group                 ",
      round(kt(np.where(static, -1e9, np.where(acls == 0, -1e6, np.where(acls == 1, 0, 1e6))), accel), 4))
print("  argmax 3-group                 ",
      round(kt(np.where(ps > .5, -1e9, np.where(np.argmax(np.stack([a0, a1, a2], 1), 1) == 0, -1e6,
                np.where(np.argmax(np.stack([a0, a1, a2], 1), 1) == 1, 0, 1e6))), accel), 4))
Fa = {k: F(accel, v) for k, v in [("lo", 0.5), ("one", 1.0), ("hi", 2.0)]}
eqa = ps * F(accel, 0) + (1 - ps) * (a0 * 0.30 + a1 * F(accel, 1.0) + a2 * 0.90)
print("  expected-quantile              ", round(kt(eqa, accel), 4))
bb = None
for t1 in np.arange(0.2, 0.95, 0.05):
    v = eqa.copy(); v[(a1 > t1) & (ps <= 0.5)] = F(accel, 1.0); v[ps > 0.5] = -10
    s = kt(v, accel)
    if bb is None or s > bb[0]: bb = (round(s, 4), round(t1, 2))
print("  expected-quantile + snap ==1   ", bb)
print("  AUC(a1 | mover) =", round(roc_auc_score((acls == 1)[mover], a1[mover]), 4))

print("\n=== VCL / VBIAS ===")
print("  vcl continuous          ", round(kt(o["vcl"], vcl), 4))
print("  vcl + static tie        ", round(kt(np.where(ps > 0.5, -1e9, o["vcl"]), vcl), 4))
eqv = ps * F(vcl, 0) + (1 - ps) * (0.15 + 0.85 / (1 + np.exp(-o["vcl"])))
print("  vcl expected-quantile   ", round(kt(eqv, vcl), 4))
print("  vbias continuous        ", round(kt(o["vbias"], vbias), 4))
print("  vbias * (1-ps)          ", round(kt(o["vbias"] * (1 - ps), vbias), 4))
print("  vbias, static->0        ", round(kt(np.where(ps > 0.5, 0.0, o["vbias"]), vbias), 4))
