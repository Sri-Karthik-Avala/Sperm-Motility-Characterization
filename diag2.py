import numpy as np, pandas as pd, os
from scipy.stats import kendalltau

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
pos = np.load(os.path.join(D, "cache", "train_track_fine.npy"))
COLS = ["vcl", "lin", "accel", "turn", "vbias"]


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else t


print("=== hardware ===")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print("torch missing", e)
print("cpu count", os.cpu_count())

print("\n=== exact tie masses in the truth ===")
static = (tt.vcl == 0).values
for c in COLS:
    v = tt[c].values
    vc = pd.Series(v).value_counts()
    top = vc.head(4)
    n0 = len(v) * (len(v) - 1) / 2
    n_ties = sum(k * (k - 1) / 2 for k in vc.values)
    print(f"  {c:6s} tie-pair fraction {n_ties/n0:.4f}  top: "
          + ", ".join(f"{k:.6g}x{n}" for k, n in top.items()))

acc1 = (np.abs(tt.accel.values - 1) < 1e-3) & ~static
lin1 = (tt.lin.values == 1) & ~static
turn0 = (tt.turn.values == 0) & ~static
print(f"\n  group sizes (movers): acc~1 {acc1.sum()} lin==1 {lin1.sum()} turn==0 {turn0.sum()}")
print("  overlap acc1&lin1", (acc1 & lin1).sum(), " acc1&turn0", (acc1 & turn0).sum(),
      " lin1&turn0", (lin1 & turn0).sum())

print("\n=== REFINED oracle ladder with all tie groups ===")


def sc(pred):
    d = {c: max(0.0, kt(pred[c], tt[c].values)) for c in COLS}
    return round(float(np.mean(list(d.values()))), 4), {k: round(v, 4) for k, v in d.items()}


z = np.zeros(len(tt))
base = {
    "vcl": np.where(static, 0, 1),
    "lin": np.where(static, 0, np.where(lin1, 2, 1)),
    "accel": np.where(static, 0, 1),
    "turn": np.where(static, 0, np.where(turn0, 0, 1)),
    "vbias": z,
}
print("  groups only (static/lin1/turn0), vbias=0        ", sc(base))
b2 = dict(base)
b2["accel"] = np.where(static, 0, np.where(acc1, 1, np.where(tt.accel.values > 1, 2, 0.5)))
print("  + accel 3-group (0 / <1 / ==1 / >1)             ", sc(b2))
b3 = dict(b2)
b3["vcl"] = np.where(static, 0, tt.vcl.values)
print("  + oracle vcl continuous                         ", sc(b3))
b4 = dict(b3)
b4["vbias"] = np.where(static, 0, tt.vbias.values)
print("  + oracle vbias                                  ", sc(b4))

print("\n=== accel sub-structure among movers ===")
a = tt.accel.values[~static]
print("  frac ==1 (1e-3):", (np.abs(a - 1) < 1e-3).mean().round(4),
      " frac <1:", (a < 1 - 1e-3).mean().round(4), " frac >1:", (a > 1 + 1e-3).mean().round(4))

print("\n=== phase-split search using the classical track ===")
step = np.linalg.norm(np.diff(pos, axis=1), axis=2)
best = None
for split in range(4, 17):
    early = step[:, :split].mean(axis=1)
    late = step[:, split:].mean(axis=1)
    t_vcl = kt(late, tt.vcl.values)
    t_acc = kt(late / (early + 1e-6), tt.accel.values)
    net_l = pos[:, 19] - pos[:, split]
    path_l = step[:, split:].sum(axis=1)
    lin = np.linalg.norm(net_l, axis=1) / (path_l + 1e-9)
    t_lin = kt(lin, tt.lin.values)
    de = pos[:, split] - pos[:, 0]
    ne = np.linalg.norm(de, axis=1) + 1e-9
    nl = np.linalg.norm(net_l, axis=1) + 1e-9
    turn = np.arccos(np.clip((de * net_l).sum(1) / (ne * nl), -1, 1))
    t_turn = kt(turn, tt.turn.values)
    print(f"  split@{split:2d}  vcl {t_vcl:+.4f}  accel {t_acc:+.4f}  lin {t_lin:+.4f}  turn {t_turn:+.4f}")

print("\n=== vbias sign convention ===")
net = pos[:, 19] - pos[:, 0]
nn = np.linalg.norm(net, axis=1) + 1e-9
print("  tau(+dy/|net|, vbias) =", round(kt(net[:, 0] / nn, tt.vbias.values), 4))
print("  tau(-dy/|net|, vbias) =", round(kt(-net[:, 0] / nn, tt.vbias.values), 4))
print("  tau(+dx/|net|, vbias) =", round(kt(net[:, 1] / nn, tt.vbias.values), 4))
print("  tau(raw dy, vbias)    =", round(kt(net[:, 0], tt.vbias.values), 4))
mv = ~static
print("  movers only, +dy/|net|:", round(kt((net[:, 0] / nn)[mv], tt.vbias.values[mv]), 4))
print("  movers with vcl>0.5, +dy/|net|:",
      round(kt((net[:, 0] / nn)[tt.vcl.values > 0.5], tt.vbias.values[tt.vcl.values > 0.5]), 4))

print("\n=== late-phase net direction instead of whole-clip ===")
for split in [10, 12]:
    nl_ = pos[:, 19] - pos[:, split]
    d = np.linalg.norm(nl_, axis=1) + 1e-9
    print(f"  split{split} tau(+dy_late/|.|, vbias) =", round(kt(nl_[:, 0] / d, tt.vbias.values), 4),
          " movers:", round(kt((nl_[:, 0] / d)[mv], tt.vbias.values[mv]), 4))
