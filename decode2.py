import numpy as np, pandas as pd, os
from scipy.stats import kendalltau, rankdata

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
o = dict(np.load(os.path.join(D, "cache", "oof3.npz")))
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values for c in COLS)
static = vcl == 0
N = len(tt)
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where(turn == 0, 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def cdf(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


def qbar(y, g, k):
    f = cdf(y)
    return np.array([f[g == i].mean() for i in range(k)])


print("class-mean quantiles (the Bayes-optimal score under class-only info):")
qL, qT, qA = qbar(lin, g_lin, 3), qbar(turn, g_turn, 2), qbar(accel, g_acc, 4)
print("  lin ", np.round(qL, 4), "\n  turn", np.round(qT, 4), "\n  acc ", np.round(qA, 4))

PL = o["lin3"]; PA = o["acc4"]; pt = o["turn2"]
PT = np.stack([1 - pt, pt], 1)

print("\n=== LIN decode comparison ===")
print("  argmax -> extremes      ",
      round(kt(np.where(np.argmax(PL, 1) == 0, -1e9,
                        np.where(np.argmax(PL, 1) == 2, 1e9, o["linc"])), lin), 4))
eq = PL @ qL
print("  posterior-mean quantile ", round(kt(eq, lin), 4))
eqr = PL @ qL + PL[:, 1] * 0.05 * np.tanh(o["linc"])
print("  + within-class refine   ", round(kt(eqr, lin), 4))
best = None
for t0 in np.arange(0.5, 1.0, 0.05):
    for t2 in np.arange(0.5, 1.0, 0.05):
        v = eqr.copy()
        v[PL[:, 0] > t0] = -1.0
        v[PL[:, 2] > t2] = 2.0
        s = kt(v, lin)
        if best is None or s > best[0]: best = (round(s, 4), round(t0, 2), round(t2, 2))
print("  + confident-snap ties   ", best)

print("\n=== TURN decode comparison ===")
print("  hard threshold best     ",
      round(max(kt(np.where(pt < t, -1e9, o["turnc"]), turn) for t in np.arange(.1, .95, .05)), 4))
eqt = PT @ qT + PT[:, 1] * 0.05 * np.tanh(o["turnc"])
print("  posterior-mean quantile ", round(kt(eqt, turn), 4))
bb = None
for t0 in np.arange(0.3, 0.98, 0.04):
    v = eqt.copy(); v[PT[:, 0] > t0] = -1.0
    s = kt(v, turn)
    if bb is None or s > bb[0]: bb = (round(s, 4), round(t0, 2))
print("  + confident-snap tie    ", bb)

print("\n=== ACCEL decode comparison ===")
print("  argmax                  ",
      round(kt(np.where(np.argmax(PA, 1) == 0, -1e9,
               np.where(np.argmax(PA, 1) == 1, -1e6, np.where(np.argmax(PA, 1) == 2, 0, 1e6))), accel), 4))
eqa = PA @ qA + (PA[:, 1] + PA[:, 3]) * 0.05 * np.tanh(o["accc"])
print("  posterior-mean quantile ", round(kt(eqa, accel), 4))
bb = None
for t2 in np.arange(0.3, 0.98, 0.04):
    for t0 in np.arange(0.4, 0.98, 0.06):
        v = eqa.copy(); v[PA[:, 2] > t2] = qA[2]; v[PA[:, 0] > t0] = -1.0
        s = kt(v, accel)
        if bb is None or s > bb[0]: bb = (round(s, 4), round(t2, 2), round(t0, 2))
print("  + confident-snap ties   ", bb)

print("\n=== VCL / VBIAS ===")
pv = o["vcl0"]
qV = qbar(vcl, static.astype(int), 2)
print("  vcl: continuous only    ", round(kt(o["vcl"], vcl), 4))
eqv = (1 - pv) * qV[0] + pv * qV[1] + (1 - pv) * 0.30 * np.tanh(o["vcl"])
print("  vcl: post-mean + refine ", round(kt(eqv, vcl), 4))
bb = None
for t0 in np.arange(0.3, 0.98, 0.04):
    v = eqv.copy(); v[pv > t0] = -1.0
    s = kt(v, vcl)
    if bb is None or s > bb[0]: bb = (round(s, 4), round(t0, 2))
print("  vcl: + snap static tie  ", bb)
pb = o["vb0"]
print("  vbias continuous        ", round(kt(o["vbias"], vbias), 4))
bb = None
for t0 in np.arange(0.3, 0.98, 0.04):
    v = np.where(pb > t0, 0.0, o["vbias"])
    s = kt(v, vbias)
    if bb is None or s > bb[0]: bb = (round(s, 4), round(t0, 2))
print("  vbias + zero-tie snap   ", bb)
print("  vbias * (1-pb)          ", round(kt(o["vbias"] * (1 - pb), vbias), 4))
