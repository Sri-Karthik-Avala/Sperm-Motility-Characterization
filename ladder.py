import numpy as np, pandas as pd, os
from scipy.stats import kendalltau, rankdata

D = r"C:\Users\srika\Downloads\eris_sperm"
W = os.path.join(D, "working")
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
O = dict(np.load(os.path.join(W, "dump_oof.npz")))
O.pop("fold")
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
vcl, lin, accel, turn, vbias = (tt[c].values.astype(float) for c in COLS)
TRUTH = dict(zip(COLS, [vcl, lin, accel, turn, vbias]))
static = vcl == 0
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where(turn == 0, 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))
N = len(tt)


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
PL, PA, PT2, PV, PB = O["lin3"], O["acc4"], O["turn2"], O["vcl0"], O["vb0"]
rnL = PL[:, 2] / (PL[:, 1] + PL[:, 2] + 1e-9)
rnA = PA[:, 1:] / (PA[:, 1:].sum(1, keepdims=True) + 1e-9)
rnT = PT2[:, 1] / np.clip(1 - PV[:, 1], 1e-3, None)

CAND = {
    "vcl": {"direct": O["d_vcl"], "grp": PV @ qV,
            "bl50": 0.5 * cdfv(O["d_vcl"]) + 0.5 * cdfv(PV @ qV)},
    "lin": {"direct": O["d_lin"], "grp": PL @ qL, "renorm": rnL,
            "renorm+ref": rnL + 0.02 * np.tanh(O["linc"]),
            "bl50": 0.5 * cdfv(O["d_lin"]) + 0.5 * cdfv(rnL),
            "bl25": 0.25 * cdfv(O["d_lin"]) + 0.75 * cdfv(rnL)},
    "accel": {"direct": O["d_accel"], "grp": PA @ qA, "renorm": rnA @ qA[1:],
              "renorm+ref": rnA @ qA[1:] + 0.02 * np.tanh(O["accc"]),
              "bl50": 0.5 * cdfv(O["d_accel"]) + 0.5 * cdfv(rnA @ qA[1:]),
              "accc": O["accc"]},
    "turn": {"direct": O["d_turn"], "grp": PT2 @ qT, "renorm": rnT,
             "P+ref": PT2[:, 1] + 0.02 * np.tanh(O["turnc"]),
             "bl50": 0.5 * cdfv(O["d_turn"]) + 0.5 * cdfv(PT2[:, 1])},
    "vbias": {"direct": O["d_vbias"], "vbm": np.tanh(O["vbm"]),
              "bl50": 0.5 * cdfv(O["d_vbias"]) + 0.5 * cdfv(np.tanh(O["vbm"]))},
}

qs = [0.0, 0.20, 0.40, 0.55, 0.65, 0.75, 0.85]
pops = []
for q in qs:
    thr = np.quantile(vcl, q)
    m = vcl >= thr
    if m.sum() > 250:
        pops.append((f"q{q:.2f}(n={m.sum()})", m))

for c in COLS:
    print(f"\n=== {c} ===")
    hdr = "  " + f"{'candidate':14s}" + "".join(f"{p[0]:>16s}" for p in pops)
    print(hdr)
    for k, v in CAND[c].items():
        row = "  " + f"{k:14s}"
        for _, m in pops:
            row += f"{kt(v[m], TRUTH[c][m]):16.4f}"
        print(row)

print("\n=== best-per-axis totals, chosen ON EACH population ===")
for nm, m in pops:
    tot = 0
    pick = []
    for c in COLS:
        k, v = max(CAND[c].items(), key=lambda z: kt(z[1][m], TRUTH[c][m]))
        tot += kt(v[m], TRUTH[c][m])
        pick.append(f"{c}:{k}")
    print(f"  {nm:16s} {tot/5:.4f}   " + " ".join(pick))

print("\n=== if we FIX the choice using the movers population, how does it score elsewhere? ===")
mm = vcl >= np.quantile(vcl, 0.20)
fixed = {}
for c in COLS:
    k, v = max(CAND[c].items(), key=lambda z: kt(z[1][mm], TRUTH[c][mm]))
    fixed[c] = (k, v)
print("  choice:", {c: fixed[c][0] for c in COLS})
for nm, m in pops:
    tot = np.mean([kt(fixed[c][1][m], TRUTH[c][m]) for c in COLS])
    print(f"  {nm:16s} {tot:.4f}")
print("  shipped v2 choice was all-direct (+turn blend). For reference:")
shipped = {"vcl": O["d_vcl"], "lin": O["d_lin"], "accel": O["d_accel"],
           "turn": 0.75 * cdfv(O["d_turn"]) + 0.25 * cdfv(PT2 @ qT), "vbias": O["d_vbias"]}
for nm, m in pops:
    tot = np.mean([kt(shipped[c][m], TRUTH[c][m]) for c in COLS])
    print(f"  {nm:16s} {tot:.4f}")
