import numpy as np, pandas as pd, os
from scipy.stats import kendalltau, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.metrics import roc_auc_score

D = r"C:\Users\srika\Downloads\eris_sperm"
W = os.path.join(D, "working")
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt["video"] = tt["clip"].map(ti.drop_duplicates("clip").set_index("clip").video)
OOF = dict(np.load(os.path.join(W, "dump_oof.npz")))
TP = dict(np.load(os.path.join(W, "dump_test.npz")))
FTR = np.load(os.path.join(W, "dump_ftr.npy"))
FTE = np.load(os.path.join(W, "dump_fte.npy"))
fold = OOF.pop("fold")
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values.astype(float) for c in COLS)
static = vcl == 0
NTR, NTE = len(FTR), len(FTE)
TRUTH = dict(zip(COLS, [vcl, lin, accel, turn, vbias]))
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where(turn == 0, 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def cdfv(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


print("=== 1. DOMAIN CLASSIFIER: how different is test from train? ===")
X = np.vstack([FTR, FTE])
y = np.r_[np.zeros(NTR), np.ones(NTE)]
p = np.zeros(len(y))
vids = tt.video.values
uv = np.unique(vids)
rng = np.random.default_rng(0)
for k in range(4):
    hv = uv[k::4]
    trm = np.r_[~np.isin(vids, hv), np.ones(NTE, bool)]
    vam = ~trm
    trm[NTR:] = rng.random(NTE) < 0.75
    vam = ~trm
    if vam.sum() < 10:
        continue
    clf = HGBC(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=0).fit(X[trm], y[trm])
    p[vam] = clf.predict_proba(X[vam])[:, 1]
msk = p > 0
print("  domain AUC:", round(roc_auc_score(y[msk], p[msk]), 4), " (1.0 = fully separable)")
ptr = np.clip(p[:NTR], 1e-4, 1 - 1e-4)
w = ptr / (1 - ptr)
w = np.where(p[:NTR] > 0, w, 0.0)
w = w / (w.sum() + 1e-12)
ess = 1.0 / np.sum(w ** 2)
print(f"  importance-weight effective sample size: {ess:.0f} of {NTR}")
print("  top-weight train cells: median vcl %.3f  vs all-train median vcl %.4f" %
      (np.median(vcl[np.argsort(-w)[:800]]), np.median(vcl)))
print("  static frac among top-800 weighted:", round(float(static[np.argsort(-w)[:800]].mean()), 4))

qb = {}
for k, g, nd in (("lin", g_lin, 3), ("turn", g_turn, 2), ("accel", g_acc, 4),
                 ("vcl", static.astype(int), 2)):
    f = cdfv(TRUTH[k])
    qb[k] = np.array([f[g == i].mean() for i in range(nd)])


def group_score(st):
    return {"lin": st["lin3"] @ qb["lin"], "turn": st["turn2"] @ qb["turn"],
            "accel": st["acc4"] @ qb["accel"], "vcl": st["vcl0"] @ qb["vcl"],
            "vbias": np.where(st["vb0"][:, 1] > 0.5, 0.0, np.tanh(st["vbm"]))}


GO = group_score(OOF)


def shipped_decode(st, GS):
    return {"vcl": st["d_vcl"], "lin": st["d_lin"], "accel": st["d_accel"],
            "turn": 0.75 * cdfv(st["d_turn"]) + 0.25 * cdfv(GS["turn"]),
            "vbias": st["d_vbias"]}


P = shipped_decode(OOF, GO)

print()
print("=== 2. LB ESTIMATE by importance resampling (target: LB=0.3957) ===")
idx = np.arange(NTR)
ests = []
for r in range(40):
    s = rng.choice(idx, size=808, replace=True, p=w)
    ests.append(np.mean([kt(P[c][s], TRUTH[c][s]) for c in COLS]))
print(f"  resampled estimate: {np.mean(ests):.4f} +/- {np.std(ests):.4f}")
print(f"  plain all-train OOF: {np.mean([kt(P[c], TRUTH[c]) for c in COLS]):.4f}")

print()
print("=== 3. which TRAIN-LABEL subpopulation reproduces that estimate? ===")
target = float(np.mean(ests))
best = None
for q in np.arange(0.0, 0.92, 0.04):
    thr = np.quantile(vcl, q)
    m = vcl >= thr
    if m.sum() < 300:
        break
    sc = np.mean([kt(P[c][m], TRUTH[c][m]) for c in COLS])
    d = abs(sc - target)
    print(f"  vcl>=q{q:.2f} (n={m.sum():5d}) score {sc:.4f}")
    if best is None or d < best[0]:
        best = (d, q, thr, m.sum(), sc)
print(f"  -> closest train-label population: vcl>=q{best[1]:.2f} "
      f"(thr {best[2]:.4f}, n={best[3]}) score {best[4]:.4f} vs target {target:.4f}")

print()
print("=== 4. per-axis on that population (where is the score actually lost?) ===")
m = vcl >= best[2]
for c in COLS:
    print(f"  {c:6s} all {kt(P[c], TRUTH[c]):.4f}   test-like {kt(P[c][m], TRUTH[c][m]):.4f}")

print()
print("=== 5. per-axis ORACLE on that population (headroom) ===")
for c in COLS:
    print(f"  {c:6s} oracle 1.0 | group-only "
          f"{kt({'vcl': static.astype(float)*-1, 'lin': g_lin, 'accel': g_acc, 'turn': g_turn, 'vbias': np.zeros(NTR)}[c][m], TRUTH[c][m]):.4f}")

print()
print("=== 6. TARGET JOINT STRUCTURE on movers: are there discrete cell types? ===")
mv = ~static
key = (np.round(lin, 6) == 1).astype(int) * 4 + (np.round(turn, 6) == 0).astype(int) * 2 + \
      (np.abs(accel - 1) < 1e-3).astype(int)
vc = pd.Series(key[mv]).value_counts()
print("  (lin==1, turn==0, accel==1) signature counts on movers:")
for k, n in vc.items():
    print(f"    lin1={bool(k & 4)} turn0={bool(k & 2)} acc1={bool(k & 1)}  n={n}")

print()
print("=== 7. vbias detail: is the ceiling measurement or definition? ===")
for lo, hi in [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 99)]:
    mm = (vcl >= lo) & (vcl < hi)
    if mm.sum() > 50:
        print(f"  vcl[{lo},{hi}) n={mm.sum():5d}  vbias tau {kt(P['vbias'][mm], vbias[mm]):.4f}"
              f"  vcl tau {kt(P['vcl'][mm], vcl[mm]):.4f}"
              f"  turn tau {kt(P['turn'][mm], turn[mm]):.4f}"
              f"  lin tau {kt(P['lin'][mm], lin[mm]):.4f}")

print()
print("=== 8. is the test prediction distribution sane vs OOF? ===")
PT = shipped_decode(TP, group_score(TP))
for c in COLS:
    print(f"  {c:6s} oof p10/50/90 {np.percentile(P[c],[10,50,90]).round(3)}   "
          f"test {np.percentile(PT[c],[10,50,90]).round(3)}")
print("  predicted static prob: oof p50 %.3f  test p50 %.3f" %
      (np.median(OOF["vcl0"][:, 1]), np.median(TP["vcl0"][:, 1])))
print("  predicted lin1 prob:   oof p50 %.3f  test p50 %.3f" %
      (np.median(OOF["lin3"][:, 2]), np.median(TP["lin3"][:, 2])))
print("  predicted turn0 prob:  oof p50 %.3f  test p50 %.3f" %
      (np.median(OOF["turn2"][:, 0]), np.median(TP["turn2"][:, 0])))
np.save(os.path.join(W, "iw.npy"), w)
