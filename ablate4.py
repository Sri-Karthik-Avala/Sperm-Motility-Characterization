import numpy as np, pandas as pd, os, time
from scipy.stats import kendalltau, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from core import build_aux, rank_gauss
from core2 import build_all

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt["video"] = tt["clip"].map(ti.drop_duplicates("clip").set_index("clip").video)
A1, _, _ = build_aux(np.load(os.path.join(D, "cache", "train_fine.npy")),
                     np.load(os.path.join(D, "cache", "train_coarse.npy")))
A2, SY, SX = build_all(np.load(os.path.join(D, "cache", "train_cost.npy")))
AUX = np.concatenate([A1, A2], 1)
print("features", AUX.shape)
N = len(tt)
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values for c in COLS)
static = vcl == 0
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where(turn == 0, 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))
g_vb = (vbias == 0).astype(int)

sizes = pd.Series(tt.video.values).value_counts()
load = np.zeros(4); asg = {}
for v, c in sizes.items():
    j = int(np.argmin(load)); asg[v] = j; load[j] += c
fold = np.array([asg[v] for v in tt.video.values])

out = {}
for k, d in [("lin3", 3), ("acc4", 4), ("turn2", 2), ("vcl0", 2), ("vb0", 2)]:
    out[k] = np.zeros((N, d))
for k in ["d_vcl", "d_lin", "d_accel", "d_turn", "d_vbias", "linc", "turnc", "accc", "vbias_m"]:
    out[k] = np.zeros(N)
t0 = time.time()
for f in range(4):
    tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
    def C(y, m=None):
        i = tr if m is None else tr[m[tr]]
        return HGBC(max_iter=140, learning_rate=0.08, max_leaf_nodes=15,
                    l2_regularization=1.0, random_state=0).fit(AUX[i], y[i]).predict_proba(AUX[va])
    def R(y, m=None):
        i = tr if m is None else tr[m[tr]]
        return HGBR(max_iter=140, learning_rate=0.08, max_leaf_nodes=15,
                    l2_regularization=1.0, random_state=0).fit(AUX[i], y[i]).predict(AUX[va])
    out["lin3"][va] = C(g_lin); out["acc4"][va] = C(g_acc)
    out["turn2"][va] = C(g_turn); out["vcl0"][va] = C(static.astype(int))
    out["vb0"][va] = C(g_vb)
    for c, arr in (("vcl", vcl), ("lin", lin), ("turn", turn), ("vbias", vbias)):
        out["d_" + c][va] = R(rank_gauss(arr))
    out["d_accel"][va] = R(rank_gauss(np.log(np.clip(accel, 1e-6, None))))
    out["linc"][va] = R(rank_gauss(lin, g_lin == 1), g_lin == 1)
    out["turnc"][va] = R(rank_gauss(turn, g_turn == 1), g_turn == 1)
    ma = (g_acc == 1) | (g_acc == 3)
    out["accc"][va] = R(rank_gauss(np.log(np.clip(accel, 1e-6, None)), ma), ma)
    out["vbias_m"][va] = R(vbias, g_vb == 0)
    print(f"  fold{f} {time.time()-t0:.0f}s", flush=True)
np.savez(os.path.join(D, "cache", "oof4.npz"), fold=fold, **out)


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def cdfv(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


def qbar(y, g, k):
    f = cdfv(y)
    return np.array([f[g == i].mean() for i in range(k)])


qL, qT, qA = qbar(lin, g_lin, 3), qbar(turn, g_turn, 2), qbar(accel, g_acc, 4)
qV = qbar(vcl, static.astype(int), 2)
res = {}
print("\n=== per-axis: DIRECT regression vs GROUP decode vs BLEND ===")
for c, truth in (("vcl", vcl), ("lin", lin), ("accel", accel), ("turn", turn), ("vbias", vbias)):
    direct = kt(out["d_" + c], truth)
    res[c] = ("direct", direct, out["d_" + c])
    print(f"  {c:6s} direct {direct:.4f}", end="")
    if c == "lin":
        g = out["lin3"] @ qL + out["lin3"][:, 1] * 0.05 * np.tanh(out["linc"])
    elif c == "turn":
        g = out["turn2"] @ qT + out["turn2"][:, 1] * 0.05 * np.tanh(out["turnc"])
    elif c == "accel":
        g = out["acc4"] @ qA + (out["acc4"][:, 1] + out["acc4"][:, 3]) * 0.05 * np.tanh(out["accc"])
    elif c == "vcl":
        g = out["vcl0"] @ qV[::-1] if False else (out["vcl0"][:, 0] * qV[0] + out["vcl0"][:, 1] * qV[1])
        g = g + out["vcl0"][:, 0] * 0.30 * np.tanh(out["d_vcl"])
    else:
        g = np.where(out["vb0"][:, 1] > 0.5, 0.0, out["vbias_m"])
    gs = kt(g, truth)
    print(f" | group {gs:.4f}", end="")
    bb = (max(direct, gs), "direct" if direct > gs else "group",
          out["d_" + c] if direct > gs else g)
    for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
        bl = w * rankdata(out["d_" + c]) / N + (1 - w) * rankdata(g) / N
        s = kt(bl, truth)
        if s > bb[0]: bb = (s, f"blend{w}", bl)
    print(f" | best {bb[0]:.4f} ({bb[1]})")
    res[c] = bb
print("\nMotilityScore (best per axis, no tie-snapping):",
      round(np.mean([res[c][0] for c in COLS]), 4))

print("\n=== add confident tie-snapping on top of the best per-axis score ===")
final = {}
tot = 0
for c, truth in (("vcl", vcl), ("lin", lin), ("accel", accel), ("turn", turn), ("vbias", vbias)):
    base = res[c][2].astype(float).copy()
    b0 = kt(base, truth)
    best = (b0, None)
    if c in ("vcl", "lin", "accel"):
        pl = out["vcl0"][:, 1]
        for t in np.arange(0.3, 0.95, 0.05):
            v = base.copy(); v[pl > t] = base.min() - 1.0
            s = kt(v, truth)
            if s > best[0]: best = (s, ("static", round(t, 2)))
    if c == "lin":
        for t in np.arange(0.4, 0.95, 0.05):
            v = best[1] and base.copy() or base.copy()
            v[out["vcl0"][:, 1] > 0.5] = base.min() - 1
            v[out["lin3"][:, 2] > t] = base.max() + 1
            s = kt(v, truth)
            if s > best[0]: best = (s, ("lin1", round(t, 2)))
    if c == "turn":
        for t in np.arange(0.3, 0.95, 0.05):
            v = base.copy(); v[out["turn2"][:, 0] > t] = base.min() - 1
            s = kt(v, truth)
            if s > best[0]: best = (s, ("turn0", round(t, 2)))
    if c == "accel":
        for t in np.arange(0.4, 0.95, 0.05):
            v = base.copy(); v[out["acc4"][:, 2] > t] = np.median(base)
            s = kt(v, truth)
            if s > best[0]: best = (s, ("acc1", round(t, 2)))
    if c == "vbias":
        for t in np.arange(0.3, 0.95, 0.05):
            v = base.copy(); v[out["vb0"][:, 1] > t] = 0.0
            s = kt(v, truth)
            if s > best[0]: best = (s, ("vb0", round(t, 2)))
    print(f"  {c:6s} {b0:.4f} -> {best[0]:.4f}  via {best[1]}")
    tot += best[0]
print("\nFINAL MotilityScore:", round(tot / 5, 4))
