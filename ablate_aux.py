import numpy as np, pandas as pd, os, time
from scipy.stats import kendalltau
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from core import build_aux, rank_gauss

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt["video"] = tt["clip"].map(ti.drop_duplicates("clip").set_index("clip").video)
fine = np.load(os.path.join(D, "cache", "train_fine.npy"))
coarse = np.load(os.path.join(D, "cache", "train_coarse.npy"))
AUX, SY, SX = build_aux(fine, coarse)
N = len(tt)
COLS = ["vcl", "lin", "accel", "turn", "vbias"]

vcl = tt.vcl.values; lin = tt.lin.values; accel = tt.accel.values
turn = tt.turn.values; vbias = tt.vbias.values
static = vcl == 0; mover = ~static
lin1 = (lin == 1) & mover; turn0 = (turn == 0) & mover
acls = np.full(N, 1, dtype=int)
acls[mover & (accel < 1 - 1e-3)] = 0
acls[mover & (accel > 1 + 1e-3)] = 2
acls[static] = 1

sizes = pd.Series(tt.video.values).value_counts()
load = np.zeros(4); asg = {}
for v, c in sizes.items():
    j = int(np.argmin(load)); asg[v] = j; load[j] += c
fold = np.array([asg[v] for v in tt.video.values])

oof = {k: np.zeros(N) for k in ["ps", "pl1", "pt0", "a0", "a1", "a2", "vcl", "vbias", "linc", "turnc", "accc"]}
t0 = time.time()
for f in range(4):
    tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
    def clf(y, m=None, multi=False):
        idx = tr if m is None else tr[m[tr]]
        c = HGBC(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                 random_state=0).fit(AUX[idx], y[idx])
        return c.predict_proba(AUX[va])
    def reg(y, m=None):
        idx = tr if m is None else tr[m[tr]]
        r = HGBR(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                 random_state=0).fit(AUX[idx], y[idx])
        return r.predict(AUX[va])
    oof["ps"][va] = clf(static.astype(int))[:, 1]
    oof["pl1"][va] = clf(lin1.astype(int), mover)[:, 1]
    oof["pt0"][va] = clf(turn0.astype(int), mover)[:, 1]
    pa = clf(acls, mover, True)
    oof["a0"][va], oof["a1"][va], oof["a2"][va] = pa[:, 0], pa[:, 1], pa[:, 2]
    oof["vcl"][va] = reg(rank_gauss(vcl))
    oof["vbias"][va] = reg(vbias, mover)
    oof["linc"][va] = reg(rank_gauss(lin, mover & (lin < 1)), mover & (lin < 1))
    oof["turnc"][va] = reg(rank_gauss(turn, mover & (turn > 0)), mover & (turn > 0))
    oof["accc"][va] = reg(rank_gauss(np.log(np.clip(accel, 1e-6, None)), mover & (acls != 1)), mover & (acls != 1))
    print(f"fold{f} {time.time()-t0:.0f}s", flush=True)

np.savez(os.path.join(D, "cache", "oof_aux.npz"), **oof, fold=fold)


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def decode(o, ts, tl, tt_, ta):
    s = o["ps"] > ts
    cls = np.argmax(np.stack([o["a0"], o["a1"], o["a2"]], 1), 1)
    conf = np.max(np.stack([o["a0"], o["a1"], o["a2"]], 1), 1)
    val = np.where(cls == 1, 0.0, np.where(cls == 0, -1e6, 1e6) + o["accc"])
    val = np.where((cls == 1) & (conf > ta), 0.0, np.where(cls == 1, o["accc"] * 1e-3, val))
    return {"vcl": np.where(s, -1e9, o["vcl"]),
            "lin": np.where(s, -1e9, np.where(o["pl1"] > tl, 1e9, o["linc"])),
            "turn": np.where(s | (o["pt0"] > tt_), -1e9, o["turnc"]),
            "accel": np.where(s, -1e9, val),
            "vbias": np.where(s, 0.0, o["vbias"])}


truth = {c: tt[c].values for c in COLS}
best = None
for ts in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for tl in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for tt_ in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for ta in [0.0, 0.4, 0.5, 0.6, 0.7]:
                p = decode(oof, ts, tl, tt_, ta)
                sc = np.mean([kt(p[c], truth[c]) for c in COLS])
                if best is None or sc > best[0]:
                    best = (sc, ts, tl, tt_, ta)
print("AUX-ONLY best decode", best)
p = decode(oof, *best[1:])
print("per-axis:", {c: round(kt(p[c], truth[c]), 4) for c in COLS})
print("continuous only vcl", round(kt(oof["vcl"], truth["vcl"]), 4),
      "vbias", round(kt(oof["vbias"], truth["vbias"]), 4))
from sklearn.metrics import roc_auc_score
print("AUC static", round(roc_auc_score(static, oof["ps"]), 4),
      "lin1|mover", round(roc_auc_score(lin1[mover], oof["pl1"][mover]), 4),
      "turn0|mover", round(roc_auc_score(turn0[mover], oof["pt0"][mover]), 4))
