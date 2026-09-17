import numpy as np, pandas as pd, os, time
from scipy.stats import kendalltau

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt["video"] = tt["clip"].map(ti.drop_duplicates("clip").set_index("clip").video)
fine = np.load(os.path.join(D, "cache", "train_fine.npy"))
coarse = np.load(os.path.join(D, "cache", "train_coarse.npy"))
print("fine", fine.shape, "coarse", coarse.shape)

F = fine.astype(np.float32)
med = np.median(F, axis=1)
resid = np.abs(F - med[:, None]).mean(axis=(1, 2, 3))
d1 = np.abs(np.diff(F, axis=1)).mean(axis=(1, 2, 3))
cen = F[:, :, 20:28, 20:28]
cenres = np.abs(cen - np.median(cen, axis=1)[:, None]).mean(axis=(1, 2, 3))

vcl = tt.vcl.values
static = vcl == 0
print("\n=== static separability from simple energy features ===")
for nm, v in [("patch resid", resid), ("frame-diff", d1), ("centre8 resid", cenres)]:
    a, b = v[static], v[~static]
    t = kendalltau(v, (~static).astype(int)).correlation
    print(f"  {nm:14s} static p50 {np.median(a):7.3f} p95 {np.percentile(a,95):7.3f} | "
          f"mover p05 {np.percentile(b,5):7.3f} p50 {np.median(b):7.3f} | tau_vs_static {t:.3f}")
    thr = np.percentile(a, 95)
    print(f"     at static-p95 threshold: movers above = {(b>thr).mean():.3f} (recall of movers)")

print("\n=== energy vs vcl decile (are slow movers separable from static at all?) ===")
mv = tt[~static]
q = pd.qcut(mv.vcl, 10, labels=False, duplicates="drop")
print(f"  STATIC          n={static.sum():5d}  centre8resid p50 {np.median(cenres[static]):7.3f}")
for k in sorted(np.unique(q)):
    m = mv.index.values[q.values == k]
    print(f"  vcl decile {k}  n={len(m):5d}  vcl p50 {np.median(vcl[m]):8.4f}  "
          f"centre8resid p50 {np.median(cenres[m]):7.3f}  patchresid p50 {np.median(resid[m]):7.3f}")

print("\n=== is lin==1 / turn==0 a function of speed? ===")
mv2 = tt[~static].copy()
mv2["q"] = pd.qcut(mv2.vcl, 10, labels=False, duplicates="drop")
g = mv2.groupby("q").agg(n=("vcl", "size"), vcl=("vcl", "median"),
                         p_lin1=("lin", lambda s: (s == 1).mean()),
                         p_turn0=("turn", lambda s: (s == 0).mean()),
                         p_acc1=("accel", lambda s: (np.abs(s - 1) < 1e-3).mean()))
print(g.round(4))

print("\n=== classical sub-pixel tracker baseline ===")
t0 = time.time()
n, T, S, _ = F.shape
Cn = coarse.astype(np.float32)


def track(vol, half, rad):
    N, T, S, _ = vol.shape
    bgs = np.median(vol, axis=(2, 3), keepdims=True)
    v = vol - bgs
    pos = np.zeros((N, T, 2), dtype=np.float32)
    yy, xx = np.mgrid[-2:3, -2:3].astype(np.float32)
    cur = np.full((N, 2), half, dtype=np.float32)
    for t in range(T):
        ci = np.rint(cur).astype(np.int32)
        for i in range(N):
            y, x = ci[i]
            y0, y1 = max(0, y - rad), min(S, y + rad + 1)
            x0, x1 = max(0, x - rad), min(S, x + rad + 1)
            w = v[i, t, y0:y1, x0:x1]
            k = np.unravel_index(np.argmax(w), w.shape)
            py, px = y0 + k[0], x0 + k[1]
            a0, a1 = max(0, py - 2), min(S, py + 3)
            b0, b1 = max(0, px - 2), min(S, px + 3)
            p = v[i, t, a0:a1, b0:b1]
            p = np.clip(p, 0, None)
            sw = p.sum()
            if sw <= 1e-6:
                pos[i, t] = (py, px)
            else:
                gy, gx = np.mgrid[a0:a1, b0:b1].astype(np.float32)
                pos[i, t] = ((p * gy).sum() / sw, (p * gx).sum() / sw)
            cur[i] = pos[i, t]
    return pos


pos = track(F, 24.0, 5)
print(f"  tracked fine in {time.time()-t0:.1f}s")


def descriptors(pos, scale=1.0):
    p = pos * scale
    step = np.linalg.norm(np.diff(p, axis=1), axis=2)
    early = step[:, :9].mean(axis=1)
    late = step[:, 10:].mean(axis=1)
    net_l = p[:, 19] - p[:, 10]
    path_l = step[:, 10:].sum(axis=1)
    lin = np.linalg.norm(net_l, axis=1) / (path_l + 1e-9)
    de = p[:, 9] - p[:, 0]
    dl = net_l
    ne, nl = np.linalg.norm(de, axis=1) + 1e-9, np.linalg.norm(dl, axis=1) + 1e-9
    cosang = np.clip((de * dl).sum(axis=1) / (ne * nl), -1, 1)
    turn = np.arccos(cosang)
    net = p[:, 19] - p[:, 0]
    vb = -net[:, 0] / (np.linalg.norm(net, axis=1) + 1e-9)
    return dict(vcl=late, lin=lin, accel=late / (early + 1e-6), turn=turn, vbias=vb)


COLS = ["vcl", "lin", "accel", "turn", "vbias"]
pred = descriptors(pos)
print("\n  classical tracker (fine crop) tau per axis:")
tot = 0
for c in COLS:
    t = kendalltau(pred[c], tt[c].values).correlation
    t = 0.0 if not np.isfinite(t) else t
    print(f"    {c:6s} tau {t:+.4f}  (clipped {max(0,t):.4f})")
    tot += max(0, t)
print(f"    MotilityScore = {tot/5:.4f}")

print("\n  same, restricted to movers only (how good is the ranking where it matters):")
m = ~static
for c in COLS:
    t = kendalltau(pred[c][m], tt[c].values[m]).correlation
    print(f"    {c:6s} tau {t:+.4f}")

print("\n  hybrid: classical tracker + ORACLE static mask forced to ties:")
tot = 0
for c in COLS:
    p = pred[c].copy()
    if c != "vbias":
        p = np.where(static, -1e9, p)
    else:
        p = np.where(static, 0.0, p)
    t = kendalltau(p, tt[c].values).correlation
    t = 0.0 if not np.isfinite(t) else t
    print(f"    {c:6s} tau {max(0,t):.4f}")
    tot += max(0, t)
print(f"    MotilityScore = {tot/5:.4f}")

np.save(os.path.join(D, "cache", "train_track_fine.npy"), pos)
