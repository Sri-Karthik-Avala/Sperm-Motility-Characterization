# made by - Karthik
import sys
import os
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.stats import rankdata, norm, kendalltau

T0 = time.time()
BUDGET = 4500.0
TRAIN_END = 3750.0
SEED = 7
S = 48
TS = 13
TLEN = 20
SCALES = [1, 2, 5]
OS_ = S - TS + 1
NFOLD = 4
MAX_EPOCHS = 28
MIN_EPOCHS = 5
BS = 96
LR = 2.0e-3
WD = 3e-4
COLS = ["vcl", "lin", "accel", "turn", "vbias"]

pub = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/public")
sub_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("working/submission.csv")
sub_out.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
ncpu = os.cpu_count() or 4
torch.set_num_threads(max(1, min(10, ncpu)))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, "threads", torch.get_num_threads(), flush=True)

tracks = pd.read_csv(pub / "train_tracks.csv").reset_index(drop=True)
tidx = pd.read_csv(pub / "train_index.csv")
xidx = pd.read_csv(pub / "test_index.csv")
queries = pd.read_csv(pub / "mot_queries.csv").reset_index(drop=True)
samp = pd.read_csv(pub / "sample_submission.csv")
NTR = len(tracks)
NTE = len(queries)
print("train cells", NTR, "queries", NTE, flush=True)

samp[["id"]].assign(**{c: 0.0 for c in COLS}).to_csv(sub_out, index=False)

tracks["video"] = tracks["clip"].map(tidx.drop_duplicates("clip").set_index("clip").video)


def zncc(frames):
    B, TT, SS, _ = frames.shape
    c = SS // 2
    h = TS // 2
    tpl = frames[:, 0, c - h:c + h + 1, c - h:c + h + 1].reshape(B, 1, TS * TS)
    tpl = tpl - tpl.mean(2, keepdim=True)
    tpl = tpl / (tpl.norm(dim=2, keepdim=True) + 1e-6)
    u = Fn.unfold(frames.reshape(B * TT, 1, SS, SS), TS)
    u = u - u.mean(1, keepdim=True)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-6)
    r = torch.bmm(tpl.repeat_interleave(TT, dim=0), u)
    return r.reshape(B, TT, SS - TS + 1, SS - TS + 1)


def build_inputs(index_df, cells, img_path, tag):
    im = np.load(img_path, mmap_mode="r")
    H, W = im.shape[1], im.shape[2]
    order = {int(c): g.row.values.astype(np.int64)
             for c, g in index_df.sort_values("t").groupby("clip")}
    n = len(cells)
    crops = {s: np.zeros((n, TLEN, S, S), dtype=np.uint8) for s in SCALES}
    rmax = (S // 2) * max(SCALES)
    t0 = time.time()
    for ci, (clip, grp) in enumerate(cells.groupby("clip")):
        rows = order[int(clip)]
        f = np.asarray(im[rows]).astype(np.uint8)
        if f.shape[0] < TLEN:
            f = np.concatenate([f, np.repeat(f[-1:], TLEN - f.shape[0], 0)], 0)
        f = f[:TLEN]
        fp = np.pad(f, ((0, 0), (rmax, rmax), (rmax, rmax)), mode="reflect")
        for idx, r in zip(grp.index.values, grp.itertuples()):
            cx = int(round(r.ref_cx * W)) + rmax
            cy = int(round(r.ref_cy * H)) + rmax
            for s in SCALES:
                R = (S // 2) * s
                blk = fp[:, cy - R:cy + R, cx - R:cx + R]
                if s == 1:
                    crops[s][idx] = blk
                else:
                    crops[s][idx] = blk.reshape(TLEN, S, s, S, s).mean(axis=(2, 4)).astype(np.uint8)
    cost = np.zeros((n, len(SCALES), TLEN, OS_, OS_), dtype=np.float16)
    for j, s in enumerate(SCALES):
        for i in range(0, n, 16):
            x = torch.from_numpy(crops[s][i:i + 16].astype(np.float32)).to(dev)
            cost[i:i + 16, j] = zncc(x).cpu().numpy().astype(np.float16)
    print(f"{tag} inputs built {time.time()-t0:.0f}s", flush=True)
    return crops[1], cost


C1_TR, COST_TR = build_inputs(tidx, tracks, pub / "train_images.npy", "train")
C1_TE, COST_TE = build_inputs(xidx, queries, pub / "test_images.npy", "test")


def peak_path(c):
    N, TT, SS, _ = c.shape
    k = c.reshape(N, TT, -1).argmax(2)
    py, px = k // SS, k % SS
    ar = np.arange(N)[:, None]
    tr = np.arange(TT)[None, :]

    def val(dy, dx):
        return c[ar, tr, np.clip(py + dy, 0, SS - 1), np.clip(px + dx, 0, SS - 1)]

    c0 = val(0, 0)
    dy = (val(-1, 0) - val(1, 0)) / (2 * (val(-1, 0) - 2 * c0 + val(1, 0)) - 1e-9)
    dx = (val(0, -1) - val(0, 1)) / (2 * (val(0, -1) - 2 * c0 + val(0, 1)) - 1e-9)
    dy = np.clip(np.nan_to_num(dy), -1, 1)
    dx = np.clip(np.nan_to_num(dx), -1, 1)
    return np.stack([py + dy - SS // 2, px + dx - SS // 2], -1).astype(np.float32), c0


def soft_path(c, temp=0.05):
    N, TT, SS, _ = c.shape
    m = c.reshape(N, TT, -1)
    w = np.exp((m - m.max(2, keepdims=True)) / temp)
    w /= w.sum(2, keepdims=True)
    g = np.arange(SS, dtype=np.float32) - SS // 2
    wr = w.reshape(N, TT, SS, SS)
    return np.stack([(wr.sum(3) * g).sum(2), (wr.sum(2) * g).sum(2)], -1).astype(np.float32)


def linfit(p, t):
    tc = t - t.mean()
    den = (tc ** 2).sum() + 1e-12
    b = np.einsum("k,nkd->nd", tc, p) / den
    pred = p.mean(1, keepdims=True) + b[:, None, :] * tc[None, :, None]
    res = np.sqrt((((p - pred) ** 2).sum(2)).mean(1))
    var = np.sqrt((((p - p.mean(1, keepdims=True)) ** 2).sum(2)).mean(1)) + 1e-9
    return b, res, res / var, den


def path_feats(p, pre, split=10):
    TT = p.shape[1]
    t = np.arange(TT, dtype=np.float32)
    b_a, r_a, q_a, sxx = linfit(p, t)
    b_e, r_e, q_e, _ = linfit(p[:, :split], t[:split])
    b_l, r_l, q_l, _ = linfit(p[:, split:], t[split:])
    sp_a = np.linalg.norm(b_a, 1e-30 + 0 * b_a[:, :1].squeeze(-1) * 0 + 1, axis=1) if False else np.linalg.norm(b_a, axis=1)
    sp_e = np.linalg.norm(b_e, axis=1)
    sp_l = np.linalg.norm(b_l, axis=1)
    cosang = (b_e * b_l).sum(1) / (sp_e * sp_l + 1e-9)
    step = np.linalg.norm(np.diff(p, axis=1), axis=2)
    late_net = np.linalg.norm(p[:, -1] - p[:, split], axis=1)
    late_path = step[:, split:].sum(1)
    net = p[:, -1] - p[:, 0]
    nn_ = np.linalg.norm(net, axis=1) + 1e-9
    out = [
        (pre + "sp_a", np.log1p(sp_a)), (pre + "sp_e", np.log1p(sp_e)),
        (pre + "sp_l", np.log1p(sp_l)),
        (pre + "spratio", np.log((sp_l + 1e-4) / (sp_e + 1e-4))),
        (pre + "r_a", np.log1p(r_a)), (pre + "r_e", np.log1p(r_e)), (pre + "r_l", np.log1p(r_l)),
        (pre + "q_a", q_a), (pre + "q_e", q_e), (pre + "q_l", q_l),
        (pre + "turn", np.arccos(np.clip(cosang, -1, 1))), (pre + "cos", cosang),
        (pre + "linraw", late_net / (late_path + 1e-9)),
        (pre + "linfit", np.clip(sp_l * (TT - split - 1) / (late_path + 1e-9), 0, 3)),
        (pre + "latenet", np.log1p(late_net)), (pre + "latepath", np.log1p(late_path)),
        (pre + "stepm", np.log1p(step.mean(1))),
        (pre + "stepe", np.log1p(step[:, :split - 1].mean(1))),
        (pre + "stepl", np.log1p(step[:, split:].mean(1))),
        (pre + "steprat", np.log((step[:, split:].mean(1) + 1e-4) / (step[:, :split - 1].mean(1) + 1e-4))),
        (pre + "net_dy", net[:, 0] / nn_), (pre + "net_dx", net[:, 1] / nn_),
        (pre + "bl_dy", b_l[:, 0] / (sp_l + 1e-9)), (pre + "bl_dx", b_l[:, 1] / (sp_l + 1e-9)),
        (pre + "ba_dy", b_a[:, 0] / (sp_a + 1e-9)), (pre + "ba_dx", b_a[:, 1] / (sp_a + 1e-9)),
        (pre + "netmag", np.log1p(nn_)), (pre + "maxd", np.log1p(np.abs(p).max(1).max(1))),
        (pre + "stepcv", step.std(1) / (step.mean(1) + 1e-6)),
        (pre + "tstat", sp_a * np.sqrt(sxx) / (r_a + 1e-6)),
        (pre + "tstat_l", sp_l * np.sqrt(45.0) / (r_l + 1e-6)),
    ]
    return out


def zero_feats(c, c0, pre):
    N, TT = c.shape[0], c.shape[1]
    o = c.shape[2] // 2
    z = c[:, :, o, o]
    t = np.arange(TT, dtype=np.float32)
    tc = t - t.mean()
    sxx = (tc ** 2).sum()
    bz = (z * tc).sum(1) / sxx
    rz = z - (z.mean(1, keepdims=True) + bz[:, None] * tc[None, :])
    sz = np.sqrt((rz ** 2).mean(1)) + 1e-6
    gap = c0 - z
    r3 = c[:, :, o - 1:o + 2, o - 1:o + 2].reshape(N, TT, 9).mean(2)
    return [
        (pre + "zmean", z.mean(1)), (pre + "zmin", z.min(1)), (pre + "zstd", z.std(1)),
        (pre + "zlast", z[:, -1]), (pre + "zdelta", z[:, -1] - z[:, 0]),
        (pre + "bz", bz), (pre + "bzn", np.abs(bz) * np.sqrt(sxx) / sz),
        (pre + "zhalf", z[:, :10].mean(1) - z[:, 10:].mean(1)),
        (pre + "gapm", gap.mean(1)), (pre + "gapx", gap.max(1)),
        (pre + "gaphalf", gap[:, 10:].mean(1) - gap[:, :10].mean(1)),
        (pre + "r3m", r3.mean(1)), (pre + "r3min", r3.min(1)),
        (pre + "c0m", c0.mean(1)), (pre + "c0min", c0.min(1)), (pre + "c0std", c0.std(1)),
    ]


def energy_feats(crop, pre):
    v = crop.astype(np.float32)
    med = np.median(v, axis=1, keepdims=True)
    r = np.abs(v - med)
    d = np.abs(np.diff(v, axis=1))
    ctr = r[:, :, 20:28, 20:28]
    return [
        (pre + "rm", np.log1p(r.mean((1, 2, 3)))), (pre + "rx", np.log1p(r.max((1, 2, 3)))),
        (pre + "dm", np.log1p(d.mean((1, 2, 3)))), (pre + "cm", np.log1p(ctr.mean((1, 2, 3)))),
        (pre + "cx", np.log1p(ctr.max((1, 2, 3)))), (pre + "sd", np.log1p(v.std((1, 2, 3)))),
    ]


def build_feats(crop1, cost):
    items = []
    for j, s in enumerate(SCALES):
        c = cost[:, j].astype(np.float32)
        pk, c0 = peak_path(c)
        items += path_feats(pk * s, f"s{s}p_")
        items += path_feats(soft_path(c) * s, f"s{s}q_")
        items += zero_feats(c, c0, f"s{s}z_")
    items += energy_feats(crop1, "e_")
    names = [k for k, _ in items]
    arr = np.stack([np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    for _, v in items], 1)
    return arr, names


t0 = time.time()
FTR, FNAMES = build_feats(C1_TR, COST_TR)
FTE, _ = build_feats(C1_TE, COST_TE)
print("features", FTR.shape, f"{time.time()-t0:.0f}s", flush=True)
SY = np.array([1.0 if n.endswith("_dy") else 0.0 for n in FNAMES], dtype=np.float32)
SX = np.array([1.0 if n.endswith("_dx") else 0.0 for n in FNAMES], dtype=np.float32)
fmu = np.where((SY > 0) | (SX > 0), 0.0, FTR.mean(0))
fsd = FTR.std(0) + 1e-6
FTR = np.clip((FTR - fmu) / fsd, -8, 8).astype(np.float32)
FTE = np.clip((FTE - fmu) / fsd, -8, 8).astype(np.float32)

vcl = tracks.vcl.values.astype(np.float64)
lin = tracks.lin.values.astype(np.float64)
accel = tracks.accel.values.astype(np.float64)
turn = tracks.turn.values.astype(np.float64)
vbias = tracks.vbias.values.astype(np.float64)
static = vcl == 0
g_lin = np.where(static, 0, np.where(lin == 1, 2, 1)).astype(np.int64)
g_turn = np.where(turn == 0, 0, 1).astype(np.int64)
g_acc = np.where(static, 0, np.where(accel < 1 - 1e-3, 1,
                                     np.where(accel <= 1 + 1e-3, 2, 3))).astype(np.int64)
g_vb = (vbias == 0).astype(np.int64)
g_st = static.astype(np.int64)
GT = {"lin3": g_lin, "acc4": g_acc, "turn2": g_turn, "vcl0": g_st, "vb0": g_vb}
GDIM = {"lin3": 3, "acc4": 4, "turn2": 2, "vcl0": 2, "vb0": 2}


def rank_gauss(v, mask=None):
    out = np.zeros(len(v))
    idx = np.arange(len(v)) if mask is None else np.where(mask)[0]
    if len(idx) < 2:
        return out
    r = rankdata(v[idx], method="average")
    out[idx] = norm.ppf((r - 0.5) / len(idx))
    return out


logacc = np.log(np.clip(accel, 1e-6, None))
DTGT = {c: rank_gauss(v) for c, v in
        (("vcl", vcl), ("lin", lin), ("turn", turn), ("vbias", vbias), ("accel", logacc))}
m_lin = g_lin == 1
m_turn = g_turn == 1
m_acc = (g_acc == 1) | (g_acc == 3)
m_vb = g_vb == 0
CTGT = {"linc": rank_gauss(lin, m_lin), "turnc": rank_gauss(turn, m_turn),
        "accc": rank_gauss(logacc, m_acc), "vbm": vbias}
CMSK = {"linc": m_lin, "turnc": m_turn, "accc": m_acc, "vbm": m_vb}

CTR = torch.from_numpy(C1_TR)
CTE = torch.from_numpy(C1_TE)
ZTR = torch.from_numpy(COST_TR)
ZTE = torch.from_numpy(COST_TE)
ATR = torch.from_numpy(FTR)
ATE = torch.from_numpy(FTE)
SYt = torch.from_numpy(SY)
SXt = torch.from_numpy(SX)
tGT = {k: torch.from_numpy(v) for k, v in GT.items()}
tD = {k: torch.from_numpy(v.astype(np.float32)) for k, v in DTGT.items()}
tC = {k: torch.from_numpy(v.astype(np.float32)) for k, v in CTGT.items()}
tM = {k: torch.from_numpy(v.astype(np.float32)) for k, v in CMSK.items()}


def prep_crop(u8, vf, hf):
    x = u8.to(dev).float().div_(255.0)
    if vf.any():
        x[vf] = torch.flip(x[vf], dims=[2])
    if hf.any():
        x[hf] = torch.flip(x[hf], dims=[3])
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    x = (x - m) / s
    med = x.median(dim=1, keepdim=True).values
    return torch.cat([(x - med) * 2.0, (x[:, 1:] - x[:, :-1]) * 4.0, x[:, :1]], 1)


def prep_cost(z, vf, hf):
    x = z.to(dev).float()
    x = x.reshape(x.shape[0], -1, OS_, OS_)
    if vf.any():
        x[vf] = torch.flip(x[vf], dims=[2])
    if hf.any():
        x[hf] = torch.flip(x[hf], dims=[3])
    return x * 2.0


def blk(a, b, st):
    return nn.Sequential(nn.Conv2d(a, b, 3, st, 1, bias=False), nn.GroupNorm(8, b), nn.SiLU())


class Enc(nn.Module):
    def __init__(self, cin, w):
        super().__init__()
        self.f = nn.Sequential(blk(cin, w, 2), blk(w, w, 1), blk(w, 2 * w, 2),
                               blk(2 * w, 2 * w, 1), blk(2 * w, 4 * w, 2),
                               blk(4 * w, 4 * w, 1), blk(4 * w, 8 * w, 2))
        self.out = 16 * w

    def forward(self, x):
        h = self.f(x)
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)


class Net(nn.Module):
    def __init__(self, naux):
        super().__init__()
        self.ez = Enc(len(SCALES) * TLEN, 40)
        self.ec = Enc(2 * TLEN, 24)
        self.ax = nn.Sequential(nn.Linear(naux, 192), nn.SiLU(), nn.Dropout(0.15),
                                nn.Linear(192, 128), nn.SiLU())
        d = self.ez.out + self.ec.out + 128
        self.trunk = nn.Sequential(nn.Linear(d, 384), nn.SiLU(), nn.Dropout(0.25),
                                   nn.Linear(384, 256), nn.SiLU())
        self.g = nn.ModuleDict({k: nn.Linear(256, v) for k, v in GDIM.items()})
        self.d = nn.ModuleDict({k: nn.Linear(256, 1) for k in COLS})
        self.c = nn.ModuleDict({k: nn.Linear(256, 1) for k in CTGT})

    def forward(self, z, xc, a):
        h = torch.cat([self.ez(z), self.ec(xc), self.ax(a)], 1)
        h = self.trunk(h)
        return ({k: m(h) for k, m in self.g.items()},
                {k: m(h)[:, 0] for k, m in self.d.items()},
                {k: m(h)[:, 0] for k, m in self.c.items()})


def new_store(n):
    s = {k: np.zeros((n, v)) for k, v in GDIM.items()}
    for k in COLS:
        s["d_" + k] = np.zeros(n)
    for k in CTGT:
        s[k] = np.zeros(n)
    return s


def predict(model, Zs, Cs, As, idx, store, wsum):
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            bt = torch.from_numpy(b)
            acc = None
            for vfb, hfb in ((0, 0), (0, 1), (1, 0), (1, 1)):
                vf = torch.full((len(b),), bool(vfb))
                hf = torch.full((len(b),), bool(hfb))
                a = As[bt].clone()
                if vfb:
                    a = a * torch.where(SYt > 0, -1.0, 1.0)
                if hfb:
                    a = a * torch.where(SXt > 0, -1.0, 1.0)
                G, Dh, Ch = model(prep_cost(Zs[bt], vf, hf), prep_crop(Cs[bt], vf, hf), a.to(dev))
                g = {k: torch.softmax(v, 1).float().cpu().numpy() for k, v in G.items()}
                dh = {k: v.float().cpu().numpy() for k, v in Dh.items()}
                ch = {k: v.float().cpu().numpy() for k, v in Ch.items()}
                if vfb:
                    dh["vbias"] = -dh["vbias"]
                    ch["vbm"] = -ch["vbm"]
                cur = (g, dh, ch)
                if acc is None:
                    acc = cur
                else:
                    for dd, ss in zip(acc, cur):
                        for k in dd:
                            dd[k] = dd[k] + ss[k]
            g, dh, ch = acc
            for k in g:
                store[k][b] += g[k] / 4 * wsum
            for k in dh:
                store["d_" + k][b] += dh[k] / 4 * wsum
            for k in ch:
                store[k][b] += ch[k] / 4 * wsum
    model.train()


sizes = pd.Series(tracks.video.values).value_counts()
loadv = np.zeros(NFOLD)
asg = {}
for v, cnt in sizes.items():
    j = int(np.argmin(loadv))
    asg[v] = j
    loadv[j] += cnt
fold = np.array([asg[v] for v in tracks.video.values])
print("fold loads", loadv, flush=True)

GW = {"lin3": 1.4, "acc4": 1.4, "turn2": 1.2, "vcl0": 1.2, "vb0": 1.0}
OOF = new_store(NTR)
TESTP = new_store(NTE)
nmodels = 0


def train_steps(model, opt, sch, order, nstep):
    tot = 0.0
    cnt = 0
    for i in range(0, len(order), BS):
        b = order[i:i + BS]
        bt = torch.from_numpy(b)
        vf = torch.rand(len(b)) < 0.5
        hf = torch.rand(len(b)) < 0.5
        a = ATR[bt].clone()
        a[vf] *= torch.where(SYt > 0, -1.0, 1.0)
        a[hf] *= torch.where(SXt > 0, -1.0, 1.0)
        sgn = torch.where(vf, -1.0, 1.0).to(dev)
        G, Dh, Ch = model(prep_cost(ZTR[bt], vf, hf), prep_crop(CTR[bt], vf, hf), a.to(dev))
        L = 0.0
        for k, w in GW.items():
            L = L + w * Fn.cross_entropy(G[k], tGT[k][bt].to(dev))
        for k in ["vcl", "lin", "accel", "turn"]:
            L = L + 0.8 * Fn.mse_loss(Dh[k], tD[k][bt].to(dev))
        L = L + 0.8 * Fn.mse_loss(Dh["vbias"], tD["vbias"][bt].to(dev) * sgn)
        for k in ["linc", "turnc", "accc"]:
            m = tM[k][bt].to(dev)
            L = L + 0.5 * (Fn.mse_loss(Ch[k], tC[k][bt].to(dev), reduction="none") * m).sum() / (m.sum() + 1)
        m = tM["vbm"][bt].to(dev)
        L = L + 0.8 * (Fn.mse_loss(torch.tanh(Ch["vbm"]), tC["vbm"][bt].to(dev) * sgn,
                                   reduction="none") * m).sum() / (m.sum() + 1)
        opt.zero_grad(set_to_none=True)
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sch.step()
        tot += float(L) * len(b)
        cnt += len(b)
    return tot / max(cnt, 1)


probe = Net(FTR.shape[1]).to(dev)
po = torch.optim.AdamW(probe.parameters(), lr=1e-4)
ps = torch.optim.lr_scheduler.OneCycleLR(po, 1e-4, total_steps=6)
tb = time.time()
train_steps(probe, po, ps, np.random.permutation(NTR)[:BS * 5], 5)
per_sample = (time.time() - tb) / (BS * 5)
del probe, po, ps
epoch_cost = per_sample * NTR * (NFOLD - 1) / NFOLD
avail = TRAIN_END - (time.time() - T0)
budget_epochs = int(max(MIN_EPOCHS, min(MAX_EPOCHS, avail / max(epoch_cost, 1e-6) / NFOLD)))
print(f"per-sample {per_sample*1000:.1f}ms  epoch~{epoch_cost:.0f}s  epochs/fold={budget_epochs}",
      flush=True)

nsnap_total = 0
for f in range(NFOLD):
    if time.time() - T0 > TRAIN_END:
        print("deadline: skipping remaining folds", flush=True)
        break
    tr = np.where(fold != f)[0]
    va = np.where(fold == f)[0]
    left = TRAIN_END - (time.time() - T0)
    infer_cost = per_sample / 3.0 * (len(va) + NTE) * 4
    nsnap = int(max(1, min(6, (left * 0.30) / max(infer_cost, 1e-6))))
    left = left - nsnap * infer_cost
    ep_here = int(max(MIN_EPOCHS, min(budget_epochs, left / max(epoch_cost, 1e-6))))
    snap_at = sorted(set(np.linspace(max(1, ep_here // 3), ep_here - 1, nsnap).astype(int).tolist()))
    print(f"fold{f} epochs {ep_here} snapshots at {snap_at}", flush=True)
    net = Net(FTR.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = ep_here * math.ceil(len(tr) / BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.3)
    t0 = time.time()
    nsf = 0
    for ep in range(ep_here):
        loss = train_steps(net, opt, sch, np.random.permutation(tr), ep)
        if ep % 5 == 0 or ep == ep_here - 1:
            print(f"  fold{f} ep{ep}/{ep_here} loss {loss:.4f} {time.time()-t0:.0f}s "
                  f"total {time.time()-T0:.0f}s", flush=True)
        if ep in snap_at:
            predict(net, ZTR, CTR, ATR, va, OOF, 1.0)
            predict(net, ZTE, CTE, ATE, np.arange(NTE), TESTP, 1.0)
            nsf += 1
            nsnap_total += 1
        if time.time() - T0 > BUDGET - 420:
            break
    if nsf == 0:
        predict(net, ZTR, CTR, ATR, va, OOF, 1.0)
        predict(net, ZTE, CTE, ATE, np.arange(NTE), TESTP, 1.0)
        nsf = 1
        nsnap_total += 1
    for k in OOF:
        OOF[k][va] /= nsf
    nmodels += 1
    del net, opt, sch
    print(f"fold{f} done {time.time()-t0:.0f}s snaps {nsf}", flush=True)

if nmodels == 0:
    print("no model trained; keeping fallback", flush=True)
    sys.exit(0)
for k in TESTP:
    TESTP[k] /= max(nsnap_total, 1)
done = fold < nmodels
print("models", nmodels, "oof rows", int(done.sum()), flush=True)


def cdfv(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


TRUTH = {"vcl": vcl, "lin": lin, "accel": accel, "turn": turn, "vbias": vbias}
mv = vcl > 0
med_mv = np.median(vcl[mv]) if mv.any() else 0.0
q75 = np.quantile(vcl[mv], 0.75) if mv.any() else 0.0
POPS = [mv, vcl >= med_mv, vcl >= np.quantile(vcl, 0.55)]
POPS = [p & done for p in POPS]
POPS = [p for p in POPS if p.sum() > 50]
if not POPS:
    POPS = [done]
ALLP = done
qb = {}
for k, g in (("lin", g_lin), ("turn", g_turn), ("accel", g_acc), ("vcl", g_st)):
    f = cdfv(TRUTH[k])
    qb[k] = np.array([f[g == i].mean() for i in range(GDIM[{"lin": "lin3", "turn": "turn2",
                                                            "accel": "acc4", "vcl": "vcl0"}[k]])])


def group_score(store):
    return {
        "lin": store["lin3"] @ qb["lin"], "turn": store["turn2"] @ qb["turn"],
        "accel": store["acc4"] @ qb["accel"], "vcl": store["vcl0"] @ qb["vcl"],
        "vbias": np.where(store["vb0"][:, 1] > 0.5, 0.0, np.tanh(store["vbm"])),
    }


def alt_score(store):
    L = store["lin3"]
    A = store["acc4"]
    ra = A[:, 1:] / (A[:, 1:].sum(1, keepdims=True) + 1e-9)
    return {
        "lin": L[:, 2] / (L[:, 1] + L[:, 2] + 1e-9) + 0.02 * np.tanh(store["linc"]),
        "accel": ra @ qb["accel"][1:] + 0.02 * np.tanh(store["accc"]),
        "turn": store["turn2"][:, 1] / np.clip(1.0 - store["vcl0"][:, 1], 1e-3, None),
        "vcl": store["vcl0"][:, 0] + 0.02 * np.tanh(store["d_vcl"]),
        "vbias": np.tanh(store["vbm"]),
    }


def obj(v, c):
    return float(np.mean([kt(v[p], TRUTH[c][p]) for p in POPS]))


def fit_cdf(ref):
    xs = np.sort(np.asarray(ref, dtype=np.float64))
    ys = (np.arange(len(xs)) + 0.5) / len(xs)
    return xs, ys


def apply_cdf(fit, v):
    xs, ys = fit
    return np.interp(np.asarray(v, dtype=np.float64), xs, ys)


GO = group_score(OOF)
GT_ = group_score(TESTP)
AO = alt_score(OOF)
AT_ = alt_score(TESTP)
POOL = {"vcl": ("vcl0", 1), "lin": ("vcl0", 1), "accel": ("vcl0", 1),
        "turn": ("turn2", 0), "vbias": ("vb0", 1)}
final = {}
report = {}
for c in COLS:
    fd = fit_cdf(OOF["d_" + c])
    fg = fit_cdf(GO[c])
    fa = fit_cdf(AO[c])
    cands = [("direct", OOF["d_" + c], TESTP["d_" + c]), ("group", GO[c], GT_[c]),
             ("alt", AO[c], AT_[c])]
    for w in [0.25, 0.5, 0.75]:
        cands.append((f"blendG{w}",
                      w * apply_cdf(fd, OOF["d_" + c]) + (1 - w) * apply_cdf(fg, GO[c]),
                      w * apply_cdf(fd, TESTP["d_" + c]) + (1 - w) * apply_cdf(fg, GT_[c])))
        cands.append((f"blendA{w}",
                      w * apply_cdf(fd, OOF["d_" + c]) + (1 - w) * apply_cdf(fa, AO[c]),
                      w * apply_cdf(fd, TESTP["d_" + c]) + (1 - w) * apply_cdf(fa, AT_[c])))
    nm, ov, tv = max(cands, key=lambda z: obj(z[1], c))
    bs = obj(ov, c)
    ov = np.asarray(ov, dtype=float)
    tv = np.asarray(tv, dtype=float)
    chosen = (nm, None, ov, tv)
    lo_fill = 0.0 if c == "vbias" else float(ov.min()) - 1.0
    hi_fill = float(ov.max()) + 1.0
    pk, pi = POOL[c]
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        o2 = ov.copy()
        t2 = tv.copy()
        o2[OOF[pk][:, pi] > t] = lo_fill
        t2[TESTP[pk][:, pi] > t] = lo_fill
        s = obj(o2, c)
        if s > bs + 5e-3:
            bs = s
            chosen = (nm + f"+snap{t}", t, o2, t2)
    if c == "lin":
        for t in [0.7, 0.8, 0.9]:
            o2 = ov.copy()
            t2 = tv.copy()
            o2[OOF["lin3"][:, 2] > t] = hi_fill
            t2[TESTP["lin3"][:, 2] > t] = hi_fill
            s = obj(o2, c)
            if s > bs + 5e-3:
                bs = s
                chosen = (nm + f"+top{t}", t, o2, t2)
    final[c] = chosen[3]
    report[c] = (chosen[0], round(bs, 4), round(kt(chosen[2][ALLP], TRUTH[c][ALLP]), 4),
                 round(kt(chosen[2][POPS[0]], TRUTH[c][POPS[0]]), 4))

print("=== OOF decode selection ===", flush=True)
for c in COLS:
    r = report[c]
    print(f"  {c:6s} {r[0]:16s} motile-obj {r[1]:.4f}  movers {r[3]:.4f}  all {r[2]:.4f}", flush=True)
print("OOF MotilityScore  movers:", round(float(np.mean([report[c][3] for c in COLS])), 4),
      " all-train:", round(float(np.mean([report[c][2] for c in COLS])), 4), flush=True)

out = pd.DataFrame({"id": queries["query_id"].values})
for c in COLS:
    v = np.asarray(final[c], dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
    out[c] = v
out = samp[["id"]].merge(out, on="id", how="left")
for c in COLS:
    out[c] = out[c].fillna(0.0).astype(np.float64)
assert len(out) == len(samp) and out["id"].is_unique
assert np.isfinite(out[COLS].values).all()
out.to_csv(sub_out, index=False)
print("wrote", sub_out, out.shape, f"total {time.time()-T0:.0f}s", flush=True)
