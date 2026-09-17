import numpy as np, pandas as pd, os, time
from scipy.stats import kendalltau
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from core import build_aux, rank_gauss
from core2 import build_all

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt["video"] = tt["clip"].map(ti.drop_duplicates("clip").set_index("clip").video)
fine = np.load(os.path.join(D, "cache", "train_fine.npy"))
coarse = np.load(os.path.join(D, "cache", "train_coarse.npy"))
A1, S1, X1 = build_aux(fine, coarse)
A2, S2, X2 = build_all(np.load(os.path.join(D, "cache", "train_cost.npy")))
AUX = np.concatenate([A1, A2], 1)
N = len(tt)
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values for c in COLS)
static = vcl == 0
mover = ~static

g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where((turn == 0), 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))
g_vcl = static.astype(int)
g_vb = (vbias == 0).astype(int)
print("group sizes lin", np.bincount(g_lin), "turn", np.bincount(g_turn),
      "acc", np.bincount(g_acc), "vcl0", np.bincount(g_vcl), "vb0", np.bincount(g_vb))

m_linc = g_lin == 1
m_turnc = g_turn == 1
m_accc = (g_acc == 1) | (g_acc == 3)
m_vb = g_vb == 0

sizes = pd.Series(tt.video.values).value_counts()
load = np.zeros(4); asg = {}
for v, c in sizes.items():
    j = int(np.argmin(load)); asg[v] = j; load[j] += c
fold = np.array([asg[v] for v in tt.video.values])

P = {k: None for k in ["lin3", "turn2", "acc4", "vcl0", "vb0"]}
P["lin3"] = np.zeros((N, 3)); P["acc4"] = np.zeros((N, 4))
P["turn2"] = np.zeros(N); P["vcl0"] = np.zeros(N); P["vb0"] = np.zeros(N)
R = {k: np.zeros(N) for k in ["vcl", "vbias", "linc", "turnc", "accc"]}
t0 = time.time()
for f in range(4):
    tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
    def C(y, m=None):
        i = tr if m is None else tr[m[tr]]
        return HGBC(max_iter=160, learning_rate=0.07, max_leaf_nodes=15,
                    l2_regularization=1.0, random_state=0).fit(AUX[i], y[i]).predict_proba(AUX[va])
    def R_(y, m=None):
        i = tr if m is None else tr[m[tr]]
        return HGBR(max_iter=160, learning_rate=0.07, max_leaf_nodes=15,
                    l2_regularization=1.0, random_state=0).fit(AUX[i], y[i]).predict(AUX[va])
    P["lin3"][va] = C(g_lin)
    P["acc4"][va] = C(g_acc)
    P["turn2"][va] = C(g_turn)[:, 1]
    P["vcl0"][va] = C(g_vcl)[:, 1]
    P["vb0"][va] = C(g_vb)[:, 1]
    R["vcl"][va] = R_(rank_gauss(vcl))
    R["vbias"][va] = R_(vbias, m_vb)
    R["linc"][va] = R_(rank_gauss(lin, m_linc), m_linc)
    R["turnc"][va] = R_(rank_gauss(turn, m_turnc), m_turnc)
    R["accc"][va] = R_(rank_gauss(np.log(np.clip(accel, 1e-6, None)), m_accc), m_accc)
    print(f"  fold{f} {time.time()-t0:.0f}s", flush=True)

np.savez(os.path.join(D, "cache", "oof3.npz"), lin3=P["lin3"], acc4=P["acc4"],
         turn2=P["turn2"], vcl0=P["vcl0"], vb0=P["vb0"], fold=fold, **R)


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def decode(P, R, t_vcl0, t_lin, t_turn, t_acc, t_vb):
    cl = P["lin3"]; ca = P["acc4"]
    lin_cls = np.where(cl[:, 0] > t_lin, 0, np.where(cl[:, 2] > cl[:, 1], 2, 1))
    acc_cls = np.where(ca[:, 0] > t_acc, 0, np.argmax(ca[:, 1:], 1) + 1)
    return {
        "vcl": np.where(P["vcl0"] > t_vcl0, -1e9, R["vcl"]),
        "lin": np.where(lin_cls == 0, -1e9, np.where(lin_cls == 2, 1e9, R["linc"])),
        "turn": np.where(P["turn2"] < t_turn, -1e9, R["turnc"]),
        "accel": np.where(acc_cls == 0, -1e9,
                          np.where(acc_cls == 1, -1e6 + R["accc"],
                                   np.where(acc_cls == 2, 0.0, 1e6 + R["accc"]))),
        "vbias": np.where(P["vb0"] > t_vb, 0.0, R["vbias"]),
    }


truth = {c: tt[c].values for c in COLS}
gr = np.arange(0.2, 0.9, 0.1)
best = None
for a in gr:
    for b in gr:
        for c in gr:
            for d in gr:
                for e in [0.4, 0.6, 0.8, 1.1]:
                    p = decode(P, R, a, b, c, d, e)
                    s = np.mean([kt(p[x], truth[x]) for x in COLS])
                    if best is None or s > best[0]:
                        best = (s, a, b, c, d, e)
print("BEST", round(best[0], 4), "thr", [round(x, 2) for x in best[1:]])
p = decode(P, R, *best[1:])
print("per-axis:", {c: round(kt(p[c], truth[c]), 4) for c in COLS})
from sklearn.metrics import roc_auc_score
print("AUC vcl0", round(roc_auc_score(g_vcl, P["vcl0"]), 4),
      "turn2", round(roc_auc_score(g_turn, P["turn2"]), 4),
      "vb0", round(roc_auc_score(g_vb, P["vb0"]), 4))
print("lin3 confusion:")
print(pd.crosstab(pd.Series(g_lin, name="true"), pd.Series(np.argmax(P["lin3"], 1), name="pred")))
