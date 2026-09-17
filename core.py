import numpy as np
from scipy.stats import norm, rankdata

S = 48
HALF = 24
T = 20


def rank_gauss(v, mask=None):
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros_like(v)
    idx = np.arange(len(v)) if mask is None else np.where(mask)[0]
    if len(idx) < 2:
        return out
    r = rankdata(v[idx], method="average")
    out[idx] = norm.ppf((r - 0.5) / len(idx))
    return out


def track_batch(vol, rad=5, cw=2):
    N, TT, SS, _ = vol.shape
    v = vol - np.median(vol, axis=(2, 3), keepdims=True)
    pos = np.zeros((N, TT, 2), dtype=np.float32)
    cy = np.full(N, SS // 2, dtype=np.int64)
    cx = np.full(N, SS // 2, dtype=np.int64)
    ar = np.arange(N)
    off = np.arange(-rad, rad + 1)
    coff = np.arange(-cw, cw + 1)
    for t in range(TT):
        ys = np.clip(cy[:, None] + off[None, :], 0, SS - 1)
        xs = np.clip(cx[:, None] + off[None, :], 0, SS - 1)
        win = v[ar[:, None, None], t, ys[:, :, None], xs[:, None, :]]
        flat = win.reshape(N, -1).argmax(1)
        py = ys[ar, flat // (2 * rad + 1)]
        px = xs[ar, flat % (2 * rad + 1)]
        gy = np.clip(py[:, None] + coff[None, :], 0, SS - 1)
        gx = np.clip(px[:, None] + coff[None, :], 0, SS - 1)
        w2 = v[ar[:, None, None], t, gy[:, :, None], gx[:, None, :]]
        w2 = np.clip(w2, 0, None)
        sw = w2.sum(axis=(1, 2))
        ok = sw > 1e-6
        my = (w2.sum(2) * gy).sum(1) / np.where(ok, sw, 1)
        mx = (w2.sum(1) * gx).sum(1) / np.where(ok, sw, 1)
        pos[:, t, 0] = np.where(ok, my, py)
        pos[:, t, 1] = np.where(ok, mx, px)
        cy = np.rint(pos[:, t, 0]).astype(np.int64)
        cx = np.rint(pos[:, t, 1]).astype(np.int64)
    return pos


def track_feats(vol, scale, split=10):
    p = track_batch(vol) * scale
    step = np.linalg.norm(np.diff(p, axis=1), axis=2)
    early = step[:, :split - 1].mean(1)
    late = step[:, split:].mean(1)
    net_l = p[:, -1] - p[:, split]
    path_l = step[:, split:].sum(1)
    lin = np.linalg.norm(net_l, axis=1) / (path_l + 1e-9)
    de = p[:, split] - p[:, 0]
    ne = np.linalg.norm(de, axis=1) + 1e-9
    nl = np.linalg.norm(net_l, axis=1) + 1e-9
    turn = np.arccos(np.clip((de * net_l).sum(1) / (ne * nl), -1, 1))
    net = p[:, -1] - p[:, 0]
    nn = np.linalg.norm(net, axis=1) + 1e-9
    f = np.stack([
        np.log1p(late), np.log1p(early), np.log((late + 1e-6) / (early + 1e-6)),
        lin, turn, net[:, 0] / nn, net[:, 1] / nn,
        np.log1p(nn), np.log1p(step.sum(1)),
    ], axis=1)
    return f.astype(np.float32)


def energy_feats(vol):
    v = vol.astype(np.float32)
    med = np.median(v, axis=1, keepdims=True)
    r = np.abs(v - med)
    d = np.abs(np.diff(v, axis=1))
    c = r[:, :, HALF - 4:HALF + 4, HALF - 4:HALF + 4]
    f = np.stack([
        np.log1p(r.mean((1, 2, 3))), np.log1p(r.max((1, 2, 3))),
        np.log1p(d.mean((1, 2, 3))), np.log1p(c.mean((1, 2, 3))),
        np.log1p(c.max((1, 2, 3))), np.log1p(v.std((1, 2, 3))),
    ], axis=1)
    return f.astype(np.float32)


def build_aux(fine, coarse):
    parts, sy, sx = [], [], []
    for vol, sc in ((fine, 1.0), (coarse, 4.0)):
        v = vol.astype(np.float32)
        tf = track_feats(v, sc)
        ef = energy_feats(v)
        parts += [tf, ef]
        sy += [0, 0, 0, 0, 0, 1, 0] + [0, 0] + [0] * 6
        sx += [0, 0, 0, 0, 0, 0, 1] + [0, 0] + [0] * 6
    a = np.concatenate(parts, axis=1)
    return a, np.array(sy, dtype=np.float32), np.array(sx, dtype=np.float32)


def make_channels(fine_u8, coarse_u8):
    out = []
    for vol in (fine_u8, coarse_u8):
        x = vol.astype(np.float32) / 255.0
        m = x.mean(axis=(1, 2, 3), keepdims=True)
        s = x.std(axis=(1, 2, 3), keepdims=True) + 1e-6
        x = (x - m) / s
        med = np.median(x, axis=1, keepdims=True)
        r = x - med
        d = np.diff(x, axis=1)
        out.append(np.concatenate([r * 2.0, d * 4.0, x[:, :1]], axis=1))
    return out
