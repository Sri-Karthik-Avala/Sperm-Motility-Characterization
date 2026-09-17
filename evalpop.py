import numpy as np, pandas as pd, os, glob
from scipy.stats import kendalltau, rankdata

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values.astype(float) for c in COLS)
static = vcl == 0
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where(turn == 0, 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))
truth = dict(zip(COLS, [vcl, lin, accel, turn, vbias]))


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def cdfv(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


POPS = {
    "ALL (train mix)": np.ones(len(tt), bool),
    "movers": ~static,
    "vcl>=0.05": vcl >= 0.05,
    "vcl>=0.33 (test-like)": vcl >= 0.3292,
}
print("population sizes:", {k: int(v.sum()) for k, v in POPS.items()})
print("oracle group-only score per population:")
for nm, m in POPS.items():
    o = {
        "vcl": np.where(static, 0, 1)[m], "lin": g_lin[m],
        "accel": np.where(g_acc == 0, 0, np.where(g_acc == 1, 1, np.where(g_acc == 2, 2, 3)))[m],
        "turn": g_turn[m], "vbias": np.zeros(m.sum()),
    }
    s = np.mean([kt(o[c], truth[c][m]) for c in COLS])
    print(f"  {nm:22s} {s:.4f}")

print()
print("oracle FULL score is 1.0 by construction; here is what our snapshots give:")


def score(O, m, tune_mask=None):
    if tune_mask is None: tune_mask = m
    qL = np.array([cdfv(lin)[g_lin == i].mean() for i in range(3)])
    qT = np.array([cdfv(turn)[g_turn == i].mean() for i in range(2)])
    qA = np.array([cdfv(accel)[g_acc == i].mean() for i in range(4)])
    qV = np.array([cdfv(vcl)[static.astype(int) == i].mean() for i in range(2)])
    grp = {"lin": O["lin3"] @ qL, "turn": O["turn2"] @ qT, "accel": O["acc4"] @ qA,
           "vcl": O["vcl0"] @ qV, "vbias": np.where(O["vb0"][:, 1] > 0.5, 0.0, np.tanh(O["vbm"]))}
    out = {}
    for c in COLS:
        cands = [("direct", O["d_" + c]), ("group", grp[c])]
        for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
            cands.append((f"bl{w}", w * cdfv(O["d_" + c]) + (1 - w) * cdfv(grp[c])))
        best = max(cands, key=lambda kv: kt(kv[1][tune_mask], truth[c][tune_mask]))
        base = best[1].astype(float)
        cur = kt(base[m], truth[c][m])
        pool = {"vcl": ("vcl0", 1), "lin": ("vcl0", 1), "accel": ("vcl0", 1),
                "turn": ("turn2", 0), "vbias": ("vb0", 1)}[c]
        bt, bv = None, cur
        for t in np.arange(0.3, 0.95, 0.05):
            v = base.copy(); sel = O[pool[0]][:, pool[1]] > t
            v[sel] = 0.0 if c == "vbias" else base.min() - 1.0
            s = kt(v[tune_mask], truth[c][tune_mask])
            if bt is None or s > bt: bt, bv2 = s, v
        v = base.copy(); sel = O[pool[0]][:, pool[1]] > 0.5
        v[sel] = 0.0 if c == "vbias" else base.min() - 1.0
        out[c] = max(cur, kt(v[m], truth[c][m]))
    return out, float(np.mean(list(out.values())))


files = sorted(glob.glob(os.path.join(D, "cache", "oof_snap_*.npz")),
               key=lambda p: int(p.split("_")[-1].split(".")[0]))
for p in files:
    e = int(p.split("_")[-1].split(".")[0])
    O = dict(np.load(p))
    row = []
    for nm, m in POPS.items():
        per, s = score(O, m)
        row.append(f"{nm.split()[0]}={s:.4f}")
    print(f"  ep{e:3d}  " + "  ".join(row), flush=True)

print()
print("=== best snapshot, per-axis on the test-like population ===")
O = dict(np.load(os.path.join(D, "cache", "oof_snap_3.npz")))
m = POPS["vcl>=0.33 (test-like)"]
per, s = score(O, m)
print("  ep3 :", {c: round(per[c], 4) for c in COLS}, "mean", round(s, 4))
O = dict(np.load(files[-1]))
per, s = score(O, m)
print("  last:", {c: round(per[c], 4) for c in COLS}, "mean", round(s, 4))

print()
print("=== SWA average of all snapshots, per population ===")
AV = None
for p in files:
    Ox = dict(np.load(p))
    AV = {k: Ox[k].astype(float) for k in Ox} if AV is None else {k: AV[k] + Ox[k] for k in AV}
AV = {k: v / len(files) for k, v in AV.items()}
for nm, mm in POPS.items():
    per, s = score(AV, mm)
    print(f"  {nm:22s} {s:.4f}   " + "  ".join(f"{c} {per[c]:.3f}" for c in COLS))
