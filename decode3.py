import numpy as np, pandas as pd, os
from scipy.stats import kendalltau, rankdata

D = r"C:\Users\srika\Downloads\eris_sperm"
W = os.path.join(D, "working")
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
O = dict(np.load(os.path.join(W, "dump_oof.npz")))
O.pop("fold")
w = np.load(os.path.join(W, "iw.npy"))
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values.astype(float) for c in COLS)
static = vcl == 0
mv = ~static
TRUTH = dict(zip(COLS, [vcl, lin, accel, turn, vbias]))
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where(turn == 0, 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))
N = len(tt)
rng = np.random.default_rng(1)
IW = [rng.choice(np.arange(N), size=808, replace=True, p=w) for _ in range(40)]


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def cdfv(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


def iw_score(v, c):
    return float(np.mean([kt(v[s], TRUTH[c][s]) for s in IW]))


def qbar(y, g, k):
    f = cdfv(y)
    return np.array([f[g == i].mean() for i in range(k)])


qL = qbar(lin, g_lin, 3)
qT = qbar(turn, g_turn, 2)
qA = qbar(accel, g_acc, 4)
qV = qbar(vcl, static.astype(int), 2)

PL, PA, PT2, PV, PB = O["lin3"], O["acc4"], O["turn2"], O["vcl0"], O["vb0"]
ns = 1.0 - PV[:, 1] + 1e-9

print("=== LIN candidates (all / movers / importance-weighted) ===")
cands = {
    "direct": O["d_lin"],
    "group(3cls posterior)": PL @ qL,
    "P(lin1) raw": PL[:, 2],
    "P(lin1|not static) renorm": PL[:, 2] / (PL[:, 1] + PL[:, 2] + 1e-9),
    "renorm + linc refine": PL[:, 2] / (PL[:, 1] + PL[:, 2] + 1e-9) + 0.02 * np.tanh(O["linc"]),
    "bl(direct,renorm)": 0.5 * cdfv(O["d_lin"]) + 0.5 * cdfv(PL[:, 2] / (PL[:, 1] + PL[:, 2] + 1e-9)),
}
for k, v in cands.items():
    print(f"  {k:28s} all {kt(v, lin):.4f}  movers {kt(v[mv], lin[mv]):.4f}  IW {iw_score(v, 'lin'):.4f}")

print()
print("=== TURN candidates ===")
cands = {
    "direct": O["d_turn"],
    "group posterior": PT2 @ qT,
    "P(turn>0) raw": PT2[:, 1],
    "P(turn>0|not static)": PT2[:, 1] / np.clip(ns, 1e-3, None),
    "P(turn>0)+turnc": PT2[:, 1] + 0.02 * np.tanh(O["turnc"]),
    "bl(direct,Pturn)": 0.5 * cdfv(O["d_turn"]) + 0.5 * cdfv(PT2[:, 1]),
}
for k, v in cands.items():
    print(f"  {k:28s} all {kt(v, turn):.4f}  movers {kt(v[mv], turn[mv]):.4f}  IW {iw_score(v, 'turn'):.4f}")

print()
print("=== ACCEL candidates ===")
rn = PA[:, 1:] / (PA[:, 1:].sum(1, keepdims=True) + 1e-9)
cands = {
    "direct": O["d_accel"],
    "group posterior(4)": PA @ qA,
    "renorm 3cls posterior": rn @ qA[1:],
    "renorm + accc": rn @ qA[1:] + 0.02 * np.tanh(O["accc"]),
    "bl(direct,renorm)": 0.5 * cdfv(O["d_accel"]) + 0.5 * cdfv(rn @ qA[1:]),
}
for k, v in cands.items():
    print(f"  {k:28s} all {kt(v, accel):.4f}  movers {kt(v[mv], accel[mv]):.4f}  IW {iw_score(v, 'accel'):.4f}")

print()
print("=== VCL candidates ===")
cands = {
    "direct": O["d_vcl"],
    "group posterior": PV @ qV,
    "bl": 0.5 * cdfv(O["d_vcl"]) + 0.5 * cdfv(PV @ qV),
    "direct+static snap0.7": np.where(PV[:, 1] > 0.7, O["d_vcl"].min() - 1, O["d_vcl"]),
}
for k, v in cands.items():
    print(f"  {k:28s} all {kt(v, vcl):.4f}  movers {kt(v[mv], vcl[mv]):.4f}  IW {iw_score(v, 'vcl'):.4f}")

print()
print("=== VBIAS candidates ===")
cands = {
    "direct": O["d_vbias"],
    "vbm tanh": np.tanh(O["vbm"]),
    "mix": 0.5 * cdfv(O["d_vbias"]) + 0.5 * cdfv(np.tanh(O["vbm"])),
    "zero-snap0.7": np.where(PB[:, 1] > 0.7, 0.0, O["d_vbias"]),
}
for k, v in cands.items():
    print(f"  {k:28s} all {kt(v, vbias):.4f}  movers {kt(v[mv], vbias[mv]):.4f}  IW {iw_score(v, 'vbias'):.4f}")

print()
print("=== BEST-BY-IW combination ===")
best = {}
allc = {
    "lin": {"direct": O["d_lin"], "renorm": PL[:, 2] / (PL[:, 1] + PL[:, 2] + 1e-9),
            "group": PL @ qL,
            "bl": 0.5 * cdfv(O["d_lin"]) + 0.5 * cdfv(PL[:, 2] / (PL[:, 1] + PL[:, 2] + 1e-9))},
    "turn": {"direct": O["d_turn"], "P": PT2[:, 1], "group": PT2 @ qT,
             "bl": 0.5 * cdfv(O["d_turn"]) + 0.5 * cdfv(PT2[:, 1])},
    "accel": {"direct": O["d_accel"], "renorm": rn @ qA[1:], "group": PA @ qA,
              "bl": 0.5 * cdfv(O["d_accel"]) + 0.5 * cdfv(rn @ qA[1:])},
    "vcl": {"direct": O["d_vcl"], "group": PV @ qV,
            "bl": 0.5 * cdfv(O["d_vcl"]) + 0.5 * cdfv(PV @ qV)},
    "vbias": {"direct": O["d_vbias"], "vbm": np.tanh(O["vbm"]),
              "bl": 0.5 * cdfv(O["d_vbias"]) + 0.5 * cdfv(np.tanh(O["vbm"]))},
}
tot_iw, tot_all = 0, 0
for c in COLS:
    k, v = max(allc[c].items(), key=lambda z: iw_score(z[1], c))
    best[c] = k
    tot_iw += iw_score(v, c)
    tot_all += kt(v, TRUTH[c])
    print(f"  {c:6s} -> {k:8s} IW {iw_score(v,c):.4f}  all {kt(v, TRUTH[c]):.4f}")
print(f"  TOTAL  IW {tot_iw/5:.4f}   all-train {tot_all/5:.4f}")
print("  (shipped v2 was IW ~0.385 / all 0.4066, LB 0.3957)")
