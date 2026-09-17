import numpy as np, pandas as pd, os, time, sys, math
import torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.stats import kendalltau, rankdata
from core import build_aux, rank_gauss
from core2 import build_all

D = r"C:\Users\srika\Downloads\eris_sperm"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 22
NFOLD = 4
BS = 96
LR = 2.0e-3
WD = 3e-4
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
torch.manual_seed(7); np.random.seed(7)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(10, os.cpu_count()))
print("device", dev)

tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt["video"] = tt["clip"].map(ti.drop_duplicates("clip").set_index("clip").video)
fine = np.load(os.path.join(D, "cache", "train_fine.npy"))
coarse = np.load(os.path.join(D, "cache", "train_coarse.npy"))
COST = np.load(os.path.join(D, "cache", "train_cost.npy"))
A1, Y1, X1 = build_aux(fine, coarse)
A2, Y2, X2 = build_all(COST)
AUX = np.concatenate([A1, A2], 1).astype(np.float32)
SY = np.concatenate([Y1, Y2]); SX = np.concatenate([X1, X2])
am = np.where((SY > 0) | (SX > 0), 0.0, AUX.mean(0))
AUX = np.nan_to_num((AUX - am) / (AUX.std(0) + 1e-6), nan=0.0, posinf=0.0, neginf=0.0)
AUX = np.clip(AUX, -8, 8).astype(np.float32)
N = len(tt)
print("aux", AUX.shape)

vcl, lin, accel, turn, vbias = (tt[c].values.astype(np.float64) for c in COLS)
static = vcl == 0
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1))
g_turn = np.where(turn == 0, 0, 1)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1, np.where(accel <= 1 + 1e-3, 2, 3)))
g_vb = (vbias == 0).astype(np.int64)
g_st = static.astype(np.int64)

d_t = {c: torch.from_numpy(rank_gauss(v).astype(np.float32))
       for c, v in (("vcl", vcl), ("lin", lin), ("turn", turn), ("vbias", vbias),
                    ("accel", np.log(np.clip(accel, 1e-6, None))))}
m_lin = g_lin == 1; m_turn = g_turn == 1; m_acc = (g_acc == 1) | (g_acc == 3); m_vb = g_vb == 0
c_t = {
    "linc": torch.from_numpy(rank_gauss(lin, m_lin).astype(np.float32)),
    "turnc": torch.from_numpy(rank_gauss(turn, m_turn).astype(np.float32)),
    "accc": torch.from_numpy(rank_gauss(np.log(np.clip(accel, 1e-6, None)), m_acc).astype(np.float32)),
    "vbm": torch.from_numpy(vbias.astype(np.float32)),
}
msk = {k: torch.from_numpy(v.astype(np.float32)) for k, v in
       (("linc", m_lin), ("turnc", m_turn), ("accc", m_acc), ("vbm", m_vb))}
gt = {k: torch.from_numpy(np.asarray(v, dtype=np.int64)) for k, v in
      (("lin3", g_lin), ("acc4", g_acc), ("turn2", g_turn), ("vcl0", g_st), ("vb0", g_vb))}

FT = torch.from_numpy(fine); ZT = torch.from_numpy(COST); AT = torch.from_numpy(AUX)
SYt = torch.from_numpy(SY); SXt = torch.from_numpy(SX)


def prep_crop(u8, vf, hf):
    x = u8.to(dev).float().div_(255.0)
    if vf.any(): x[vf] = torch.flip(x[vf], dims=[2])
    if hf.any(): x[hf] = torch.flip(x[hf], dims=[3])
    m = x.mean(dim=(1, 2, 3), keepdim=True); s = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    x = (x - m) / s
    med = x.median(dim=1, keepdim=True).values
    return torch.cat([(x - med) * 2.0, (x[:, 1:] - x[:, :-1]) * 4.0, x[:, :1]], 1)


def prep_cost(z, vf, hf, j):
    x = z[:, j].to(dev).float()
    if vf.any(): x[vf] = torch.flip(x[vf], dims=[2])
    if hf.any(): x[hf] = torch.flip(x[hf], dims=[3])
    return x * 2.0


def blk(a, b, st):
    return nn.Sequential(nn.Conv2d(a, b, 3, st, 1, bias=False), nn.GroupNorm(8, b), nn.SiLU())


class Enc(nn.Module):
    def __init__(self, cin, w):
        super().__init__()
        self.f = nn.Sequential(blk(cin, w, 2), blk(w, w, 1), blk(w, 2 * w, 2), blk(2 * w, 2 * w, 1),
                               blk(2 * w, 4 * w, 2), blk(4 * w, 4 * w, 1), blk(4 * w, 8 * w, 2))
        self.out = 16 * w

    def forward(self, x):
        h = self.f(x)
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)


class Net(nn.Module):
    def __init__(self, naux):
        super().__init__()
        self.ez1 = Enc(20, 32); self.ez2 = Enc(20, 32); self.ec = Enc(40, 24)
        self.ax = nn.Sequential(nn.Linear(naux, 192), nn.SiLU(), nn.Dropout(0.15),
                                nn.Linear(192, 128), nn.SiLU())
        d = self.ez1.out * 2 + self.ec.out + 128
        self.trunk = nn.Sequential(nn.Linear(d, 384), nn.SiLU(), nn.Dropout(0.25),
                                   nn.Linear(384, 256), nn.SiLU())
        self.g = nn.ModuleDict({"lin3": nn.Linear(256, 3), "acc4": nn.Linear(256, 4),
                                "turn2": nn.Linear(256, 2), "vcl0": nn.Linear(256, 2),
                                "vb0": nn.Linear(256, 2)})
        self.d = nn.ModuleDict({k: nn.Linear(256, 1) for k in COLS})
        self.c = nn.ModuleDict({k: nn.Linear(256, 1) for k in ["linc", "turnc", "accc", "vbm"]})

    def forward(self, zf, zc, xc, a):
        h = torch.cat([self.ez1(zf), self.ez2(zc), self.ec(xc), self.ax(a)], 1)
        h = self.trunk(h)
        return ({k: m(h) for k, m in self.g.items()},
                {k: m(h)[:, 0] for k, m in self.d.items()},
                {k: m(h)[:, 0] for k, m in self.c.items()})


sizes = pd.Series(tt.video.values).value_counts()
load = np.zeros(NFOLD); asg = {}
for v, cnt in sizes.items():
    j = int(np.argmin(load)); asg[v] = j; load[j] += cnt
fold = np.array([asg[v] for v in tt.video.values])
print("fold loads", load)


def new_store():
    s = {"lin3": np.zeros((N, 3)), "acc4": np.zeros((N, 4)), "turn2": np.zeros((N, 2)),
         "vcl0": np.zeros((N, 2)), "vb0": np.zeros((N, 2))}
    for k in COLS: s["d_" + k] = np.zeros(N)
    for k in ["linc", "turnc", "accc", "vbm"]: s[k] = np.zeros(N)
    return s


EVAL_AT = sorted(set([e for e in [1, 2, 3, 4, 6, 8, 11, 15, 21, 29] if e < EPOCHS] + [EPOCHS - 1]))
SNAP = {e: new_store() for e in EVAL_AT}


def run_eval(model, va, store):
    model.eval()
    with torch.no_grad():
        for i in range(0, len(va), 256):
            b = va[i:i + 256]; bt = torch.from_numpy(b)
            acc = None
            for vfb, hfb in ((0, 0), (0, 1), (1, 0), (1, 1)):
                vf = torch.full((len(b),), bool(vfb)); hf = torch.full((len(b),), bool(hfb))
                a = AT[bt].clone()
                if vfb: a *= torch.where(SYt > 0, -1.0, 1.0)
                if hfb: a *= torch.where(SXt > 0, -1.0, 1.0)
                G, Dh, Ch = model(prep_cost(ZT[bt], vf, hf, 0), prep_cost(ZT[bt], vf, hf, 1),
                                  prep_crop(FT[bt], vf, hf), a.to(dev))
                g = {k: torch.softmax(v, 1).cpu().numpy() for k, v in G.items()}
                dh = {k: v.cpu().numpy() for k, v in Dh.items()}
                ch = {k: v.cpu().numpy() for k, v in Ch.items()}
                if vfb:
                    dh["vbias"] = -dh["vbias"]; ch["vbm"] = -ch["vbm"]
                cur = (g, dh, ch)
                if acc is None:
                    acc = cur
                else:
                    for dd, ss in zip(acc, cur):
                        for k in dd: dd[k] = dd[k] + ss[k]
            g, dh, ch = acc
            for k in g: store[k][b] = g[k] / 4
            for k in dh: store["d_" + k][b] = dh[k] / 4
            for k in ch: store[k][b] = ch[k] / 4
    model.train()


GW = {"lin3": 1.4, "acc4": 1.4, "turn2": 1.2, "vcl0": 1.2, "vb0": 1.0}
for f in range(NFOLD):
    tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
    net = Net(AUX.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(tr) / BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.3)
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(tr); tl = 0.0
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]; bt = torch.from_numpy(b)
            vf = torch.rand(len(b)) < 0.5; hf = torch.rand(len(b)) < 0.5
            zf = prep_cost(ZT[bt], vf, hf, 0); zc = prep_cost(ZT[bt], vf, hf, 1)
            xc = prep_crop(FT[bt], vf, hf)
            a = AT[bt].clone()
            a[vf] *= torch.where(SYt > 0, -1.0, 1.0); a[hf] *= torch.where(SXt > 0, -1.0, 1.0)
            sgn = torch.where(vf, -1.0, 1.0).to(dev)
            G, Dh, Ch = net(zf, zc, xc, a.to(dev))
            L = 0.0
            for k, w in GW.items():
                L = L + w * Fn.cross_entropy(G[k], gt[k][bt].to(dev))
            for k in ["vcl", "lin", "accel", "turn"]:
                L = L + 0.8 * Fn.mse_loss(Dh[k], d_t[k][bt].to(dev))
            L = L + 0.8 * Fn.mse_loss(Dh["vbias"], d_t["vbias"][bt].to(dev) * sgn)
            for k in ["linc", "turnc", "accc"]:
                m = msk[k][bt].to(dev)
                L = L + 0.5 * (Fn.mse_loss(Ch[k], c_t[k][bt].to(dev), reduction="none") * m).sum() / (m.sum() + 1)
            m = msk["vbm"][bt].to(dev)
            L = L + 0.8 * (Fn.mse_loss(torch.tanh(Ch["vbm"]), c_t["vbm"][bt].to(dev) * sgn,
                                       reduction="none") * m).sum() / (m.sum() + 1)
            opt.zero_grad(set_to_none=True); L.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); tl += float(L) * len(b)
        if ep in SNAP:
            run_eval(net, va, SNAP[ep])
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  f{f} ep{ep:2d} {tl/len(perm):.4f} {time.time()-t0:.0f}s", flush=True)
    print(f"fold{f} {time.time()-t0:.0f}s", flush=True)

for e in EVAL_AT:
    np.savez(os.path.join(D, "cache", f"oof_snap_{e}.npz"), fold=fold, **SNAP[e])


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def cdfv(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


def qbar(y, g, k):
    fv = cdfv(y)
    return np.array([fv[g == i].mean() for i in range(k)])


qL, qT, qA = qbar(lin, g_lin, 3), qbar(turn, g_turn, 2), qbar(accel, g_acc, 4)
qV = qbar(vcl, g_st, 2)
truth = dict(zip(COLS, [vcl, lin, accel, turn, vbias]))


def score_store(O):
    grp = {"lin": O["lin3"] @ qL, "turn": O["turn2"] @ qT, "accel": O["acc4"] @ qA,
           "vcl": O["vcl0"] @ qV, "vbias": np.where(O["vb0"][:, 1] > 0.5, 0.0, np.tanh(O["vbm"]))}
    out = {}
    for c in COLS:
        d = kt(O["d_" + c], truth[c]); g = kt(grp[c], truth[c])
        bb = (max(d, g), O["d_" + c] if d >= g else grp[c])
        for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
            bl = w * cdfv(O["d_" + c]) + (1 - w) * cdfv(grp[c])
            sv = kt(bl, truth[c])
            if sv > bb[0]: bb = (sv, bl)
        base = bb[1].astype(float); snap = bb[0]
        pool = {"vcl": ("vcl0", 1), "lin": ("vcl0", 1), "accel": ("vcl0", 1),
                "turn": ("turn2", 0), "vbias": ("vb0", 1)}[c]
        for t in np.arange(0.3, 0.95, 0.05):
            v = base.copy(); sel = O[pool[0]][:, pool[1]] > t
            v[sel] = 0.0 if c == "vbias" else base.min() - 1.0
            snap = max(snap, kt(v, truth[c]))
        if c == "lin":
            for t in np.arange(0.4, 0.95, 0.05):
                v = base.copy(); v[O["vcl0"][:, 1] > 0.5] = base.min() - 1
                v[O["lin3"][:, 2] > t] = base.max() + 1
                snap = max(snap, kt(v, truth[c]))
        out[c] = snap
    return out, float(np.mean(list(out.values())))


print("=== score by epoch snapshot ===")
for e in EVAL_AT:
    per, m = score_store(SNAP[e])
    print(f"  ep{e:3d}  {m:.4f}   " + "  ".join(f"{c} {per[c]:.3f}" for c in COLS), flush=True)

print("=== trailing-window snapshot average (SWA-style) ===")
for j in range(1, len(EVAL_AT)):
    es = EVAL_AT[max(0, j - 3):j + 1]
    AV = {k: np.mean([SNAP[e][k] for e in es], axis=0) for k in SNAP[es[0]]}
    per, m = score_store(AV)
    print(f"  avg{es}  {m:.4f}   " + "  ".join(f"{c} {per[c]:.3f}" for c in COLS), flush=True)
