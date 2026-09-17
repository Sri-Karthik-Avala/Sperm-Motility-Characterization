import numpy as np, pandas as pd, os, time, sys, math
import torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.stats import kendalltau
from core import build_aux, rank_gauss, HALF

D = r"C:\Users\srika\Downloads\eris_sperm"
SEED = 1234
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 26
NFOLD = 4
BS = 96
LR = 2.4e-3
WD = 1e-4
W = 32
COLS = ["vcl", "lin", "accel", "turn", "vbias"]

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(10, os.cpu_count()))
print("device", dev, "threads", torch.get_num_threads())

tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt["video"] = tt["clip"].map(ti.drop_duplicates("clip").set_index("clip").video)
fine = np.load(os.path.join(D, "cache", "train_fine.npy"))
coarse = np.load(os.path.join(D, "cache", "train_coarse.npy"))
N = len(tt)

t0 = time.time()
AUX, SY, SX = build_aux(fine, coarse)
print("aux", AUX.shape, "in %.1fs" % (time.time() - t0))
AUX, SY, SX = AUX, SY, SX
am, asd = AUX.mean(0), AUX.std(0) + 1e-6
am = np.where((SY > 0) | (SX > 0), 0.0, am)
AUXn = (AUX - am) / asd

vcl = tt.vcl.values.astype(np.float32)
lin = tt.lin.values.astype(np.float32)
accel = tt.accel.values.astype(np.float32)
turn = tt.turn.values.astype(np.float32)
vbias = tt.vbias.values.astype(np.float32)
static = vcl == 0
mover = ~static
lin1 = (lin == 1) & mover
turn0 = (turn == 0) & mover
acls = np.full(N, -1, dtype=np.int64)
acls[mover & (accel < 1 - 1e-3)] = 0
acls[mover & (np.abs(accel - 1) <= 1e-3)] = 1
acls[mover & (accel > 1 + 1e-3)] = 2

FT = torch.from_numpy(fine)
CT = torch.from_numpy(coarse)
AT = torch.from_numpy(AUXn)
assert SY.shape[0] == AUX.shape[1], (SY.shape, AUX.shape)
SYt = torch.from_numpy(SY)
SXt = torch.from_numpy(SX)


def chans(u8, vflip, hflip):
    x = u8.to(dev, non_blocking=True).float().div_(255.0)
    if vflip.any():
        x[vflip] = torch.flip(x[vflip], dims=[2])
    if hflip.any():
        x[hflip] = torch.flip(x[hflip], dims=[3])
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    x = (x - m) / s
    med = x.median(dim=1, keepdim=True).values
    r = (x - med) * 2.0
    d = (x[:, 1:] - x[:, :-1]) * 4.0
    return torch.cat([r, d, x[:, :1]], dim=1)


class Enc(nn.Module):
    def __init__(self, cin, w):
        super().__init__()
        def blk(a, b, st):
            return nn.Sequential(nn.Conv2d(a, b, 3, st, 1, bias=False),
                                 nn.GroupNorm(8, b), nn.SiLU())
        self.f = nn.Sequential(
            blk(cin, w, 2), blk(w, w, 1),
            blk(w, 2 * w, 2), blk(2 * w, 2 * w, 1),
            blk(2 * w, 4 * w, 2), blk(4 * w, 4 * w, 1),
            blk(4 * w, 8 * w, 2))
        self.out = 16 * w

    def forward(self, x):
        h = self.f(x)
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)


class Net(nn.Module):
    def __init__(self, cin, naux, w=W):
        super().__init__()
        self.e1 = Enc(cin, w)
        self.e2 = Enc(cin, w)
        self.ax = nn.Sequential(nn.Linear(naux, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        d = self.e1.out * 2 + 96
        self.trunk = nn.Sequential(nn.Linear(d, 320), nn.SiLU(), nn.Dropout(0.1),
                                   nn.Linear(320, 256), nn.SiLU())
        self.h_static = nn.Linear(256, 1)
        self.h_lin1 = nn.Linear(256, 1)
        self.h_turn0 = nn.Linear(256, 1)
        self.h_acls = nn.Linear(256, 3)
        self.h_vcl = nn.Linear(256, 1)
        self.h_vbias = nn.Linear(256, 1)
        self.h_linc = nn.Linear(256, 1)
        self.h_turnc = nn.Linear(256, 1)
        self.h_accc = nn.Linear(256, 1)

    def forward(self, xf, xc, a):
        h = torch.cat([self.e1(xf), self.e2(xc), self.ax(a)], 1)
        h = self.trunk(h)
        return (self.h_static(h)[:, 0], self.h_lin1(h)[:, 0], self.h_turn0(h)[:, 0],
                self.h_acls(h), self.h_vcl(h)[:, 0], torch.tanh(self.h_vbias(h)[:, 0]),
                self.h_linc(h)[:, 0], self.h_turnc(h)[:, 0], self.h_accc(h)[:, 0])


def folds_by_video(vids, k):
    sizes = pd.Series(vids).value_counts()
    load = np.zeros(k); assign = {}
    for v, c in sizes.items():
        j = int(np.argmin(load)); assign[v] = j; load[j] += c
    return np.array([assign[v] for v in vids]), load


fold_id, load = folds_by_video(tt.video.values, NFOLD)
print("fold sizes", load, "videos/fold", pd.Series(fold_id).value_counts().to_dict())

oof = {k: np.zeros(N, dtype=np.float32) for k in
       ["ps", "pl1", "pt0", "a0", "a1", "a2", "vcl", "vbias", "linc", "turnc", "accc"]}
cin = 20 + 19 + 1

for f in range(NFOLD):
    tr = np.where(fold_id != f)[0]
    va = np.where(fold_id == f)[0]
    y_vcl = rank_gauss(vcl)
    y_linc = rank_gauss(lin, mover & (lin < 1))
    y_turnc = rank_gauss(turn, mover & (turn > 0))
    y_accc = rank_gauss(np.log(np.clip(accel, 1e-6, None)), mover & (acls != 1))
    tens = {
        "vcl": torch.from_numpy(y_vcl.astype(np.float32)),
        "vbias": torch.from_numpy(vbias),
        "static": torch.from_numpy(static.astype(np.float32)),
        "lin1": torch.from_numpy(lin1.astype(np.float32)),
        "turn0": torch.from_numpy(turn0.astype(np.float32)),
        "acls": torch.from_numpy(acls),
        "linc": torch.from_numpy(y_linc.astype(np.float32)),
        "turnc": torch.from_numpy(y_turnc.astype(np.float32)),
        "accc": torch.from_numpy(y_accc.astype(np.float32)),
        "m_lin": torch.from_numpy((mover & (lin < 1)).astype(np.float32)),
        "m_turn": torch.from_numpy((mover & (turn > 0)).astype(np.float32)),
        "m_acc": torch.from_numpy((mover & (acls != 1)).astype(np.float32)),
        "m_mv": torch.from_numpy(mover.astype(np.float32)),
    }
    net = Net(cin, AUX.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    nstep = EPOCHS * math.ceil(len(tr) / BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=nstep, pct_start=0.25)
    tf0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            bt = torch.from_numpy(b)
            vf = torch.rand(len(b)) < 0.5
            hf = torch.rand(len(b)) < 0.5
            xf = chans(FT[bt], vf, hf)
            xc = chans(CT[bt], vf, hf)
            a = AT[bt].clone()
            a[vf] = a[vf] * torch.where(SYt > 0, -1.0, 1.0)
            a[hf] = a[hf] * torch.where(SXt > 0, -1.0, 1.0)
            a = a.to(dev)
            sgn = torch.where(vf, -1.0, 1.0).to(dev)
            ps, pl1, pt0, pac, pv, pvb, plc, ptc, pac2 = net(xf, xc, a)
            g = {k: v[bt].to(dev) for k, v in tens.items()}
            L = 0.0
            L = L + 1.5 * Fn.binary_cross_entropy_with_logits(ps, g["static"])
            mm = g["m_mv"]
            L = L + 1.2 * (Fn.binary_cross_entropy_with_logits(pl1, g["lin1"], reduction="none") * mm).sum() / (mm.sum() + 1)
            L = L + 1.2 * (Fn.binary_cross_entropy_with_logits(pt0, g["turn0"], reduction="none") * mm).sum() / (mm.sum() + 1)
            ac = g["acls"].clamp(min=0)
            L = L + 1.2 * (Fn.cross_entropy(pac, ac, reduction="none") * mm).sum() / (mm.sum() + 1)
            L = L + 1.0 * Fn.mse_loss(pv, g["vcl"])
            L = L + 1.5 * (Fn.mse_loss(pvb, g["vbias"] * sgn, reduction="none") * mm).sum() / (mm.sum() + 1)
            for p_, k_, m_ in ((plc, "linc", "m_lin"), (ptc, "turnc", "m_turn"), (pac2, "accc", "m_acc")):
                mk = g[m_]
                L = L + 0.5 * (Fn.mse_loss(p_, g[k_], reduction="none") * mk).sum() / (mk.sum() + 1)
            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()
            tot += float(L) * len(b)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  fold{f} ep{ep:3d} loss {tot/len(perm):.4f} {time.time()-tf0:.0f}s", flush=True)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(va), 256):
            b = va[i:i + 256]
            bt = torch.from_numpy(b)
            zf = torch.zeros(len(b), dtype=torch.bool)
            acc = None
            for vfb, hfb in ((False, False), (False, True), (True, False), (True, True)):
                vf = torch.full((len(b),), vfb); hf = torch.full((len(b),), hfb)
                xf = chans(FT[bt], vf, hf); xc = chans(CT[bt], vf, hf)
                a = AT[bt].clone()
                if vfb: a = a * torch.where(SYt > 0, -1.0, 1.0)
                if hfb: a = a * torch.where(SXt > 0, -1.0, 1.0)
                o = net(xf, xc, a.to(dev))
                o = list(o)
                if vfb: o[5] = -o[5]
                o = [x.float().cpu().numpy() for x in o]
                acc = o if acc is None else [a_ + b_ for a_, b_ in zip(acc, o)]
            o = [x / 4.0 for x in acc]
            oof["ps"][b] = 1 / (1 + np.exp(-o[0]))
            oof["pl1"][b] = 1 / (1 + np.exp(-o[1]))
            oof["pt0"][b] = 1 / (1 + np.exp(-o[2]))
            sm = np.exp(o[3] - o[3].max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
            oof["a0"][b], oof["a1"][b], oof["a2"][b] = sm[:, 0], sm[:, 1], sm[:, 2]
            oof["vcl"][b] = o[4]; oof["vbias"][b] = o[5]
            oof["linc"][b] = o[6]; oof["turnc"][b] = o[7]; oof["accc"][b] = o[8]
    print(f"fold{f} done {time.time()-tf0:.0f}s", flush=True)

np.savez(os.path.join(D, "cache", "oof.npz"), **oof, fold=fold_id)
print("saved oof")


def kt(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def decode(o, ts, tl, tt_, ta):
    s = o["ps"] > ts
    p = {}
    p["vcl"] = np.where(s, -1e9, o["vcl"])
    p["lin"] = np.where(s, -1e9, np.where(o["pl1"] > tl, 1e9, o["linc"]))
    p["turn"] = np.where(s | (o["pt0"] > tt_), -1e9, o["turnc"])
    cls = np.argmax(np.stack([o["a0"], o["a1"], o["a2"]], 1), 1)
    conf = np.max(np.stack([o["a0"], o["a1"], o["a2"]], 1), 1)
    base = np.where(cls == 1, 0.0, np.where(cls == 0, -1e6, 1e6))
    tie1 = (cls == 1) & (conf > ta)
    val = base + np.where(cls == 1, 0.0, o["accc"])
    val = np.where(tie1, 0.0, np.where(cls == 1, o["accc"] * 1e-3, val))
    p["accel"] = np.where(s, -1e9, val)
    p["vbias"] = np.where(s, 0.0, o["vbias"])
    return p


truth = {c: tt[c].values for c in COLS}
best = None
for ts in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for tl in [0.4, 0.5, 0.6, 0.7]:
        for tt_ in [0.4, 0.5, 0.6, 0.7]:
            for ta in [0.0, 0.4, 0.5, 0.6]:
                p = decode(oof, ts, tl, tt_, ta)
                sc = np.mean([kt(p[c], truth[c]) for c in COLS])
                if best is None or sc > best[0]:
                    best = (sc, ts, tl, tt_, ta)
print("best decode", best)
p = decode(oof, *best[1:])
print("per-axis:", {c: round(kt(p[c], truth[c]), 4) for c in COLS})
print("raw continuous only:", {
    "vcl": round(kt(oof["vcl"], truth["vcl"]), 4),
    "vbias": round(kt(oof["vbias"], truth["vbias"]), 4)})
print("group AUC-ish: static acc", ((oof["ps"] > 0.5) == static).mean().round(4),
      " lin1 acc(mover)", ((oof["pl1"][mover] > 0.5) == lin1[mover]).mean().round(4),
      " turn0 acc(mover)", ((oof["pt0"][mover] > 0.5) == turn0[mover]).mean().round(4))
