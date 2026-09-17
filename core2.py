import numpy as np

OFF = 18


def peak_path(c):
    N, T, S, _ = c.shape
    f = c.reshape(N, T, -1)
    k = f.argmax(2)
    py, px = k // S, k % S
    ar = np.arange(N)[:, None]
    tr = np.arange(T)[None, :]

    def val(dy, dx):
        return c[ar, tr, np.clip(py + dy, 0, S - 1), np.clip(px + dx, 0, S - 1)]

    c0 = val(0, 0)
    dy = (val(-1, 0) - val(1, 0)) / (2 * (val(-1, 0) - 2 * c0 + val(1, 0)) - 1e-9)
    dx = (val(0, -1) - val(0, 1)) / (2 * (val(0, -1) - 2 * c0 + val(0, 1)) - 1e-9)
    dy = np.clip(np.nan_to_num(dy), -1, 1)
    dx = np.clip(np.nan_to_num(dx), -1, 1)
    return np.stack([py + dy - S // 2, px + dx - S // 2], -1).astype(np.float32), c0


def soft_path(c, temp):
    N, T, S, _ = c.shape
    m = c.reshape(N, T, -1)
    w = np.exp((m - m.max(2, keepdims=True)) / temp)
    w /= w.sum(2, keepdims=True)
    g = np.arange(S, dtype=np.float32) - S // 2
    wy = w.reshape(N, T, S, S).sum(3)
    wx = w.reshape(N, T, S, S).sum(2)
    return np.stack([(wy * g).sum(2), (wx * g).sum(2)], -1).astype(np.float32)


def _fit(p, t):
    n = len(t)
    tm = t.mean()
    tc = t - tm
    den = (tc ** 2).sum() + 1e-12
    b = np.einsum("k,nkd->nd", tc, p) / den
    a = p.mean(1) - b * tm
    pred = a[:, None, :] + b[:, None, :] * t[None, :, None]
    res = np.sqrt((((p - pred) ** 2).sum(2)).mean(1))
    var = np.sqrt((((p - p.mean(1, keepdims=True)) ** 2).sum(2)).mean(1)) + 1e-9
    return b, res, res / var


def path_feats(p, split=10):
    N, T, _ = p.shape
    t = np.arange(T, dtype=np.float32)
    b_all, r_all, q_all = _fit(p, t)
    b_e, r_e, q_e = _fit(p[:, :split], t[:split])
    b_l, r_l, q_l = _fit(p[:, split:], t[split:])
    sp_all = np.linalg.norm(b_all, axis=1)
    sp_e = np.linalg.norm(b_e, axis=1)
    sp_l = np.linalg.norm(b_l, axis=1)
    cosang = (b_e * b_l).sum(1) / (sp_e * sp_l + 1e-9)
    turn = np.arccos(np.clip(cosang, -1, 1))
    step = np.linalg.norm(np.diff(p, axis=1), axis=2)
    late_net = np.linalg.norm(p[:, -1] - p[:, split], axis=1)
    late_path = step[:, split:].sum(1)
    lin_raw = late_net / (late_path + 1e-9)
    lin_fit = (sp_l * (T - split - 1)) / (late_path + 1e-9)
    net = p[:, -1] - p[:, 0]
    nn = np.linalg.norm(net, axis=1) + 1e-9
    tq = np.arange(T, dtype=np.float32)
    tq = (tq - tq.mean()) / tq.std()
    A = np.stack([np.ones(T, np.float32), tq, tq ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, np.ones((T, 1), np.float32), rcond=None)
    pin = np.linalg.pinv(A)
    cq = np.einsum("kt,ntd->nkd", pin, p)
    predq = np.einsum("tk,nkd->ntd", A, cq)
    rq = np.sqrt((((p - predq) ** 2).sum(2)).mean(1))
    curv = np.linalg.norm(cq[:, 2], axis=1)
    f = np.stack([
        np.log1p(sp_all), np.log1p(sp_e), np.log1p(sp_l),
        np.log((sp_l + 1e-4) / (sp_e + 1e-4)),
        np.log1p(r_all), np.log1p(r_e), np.log1p(r_l),
        q_all, q_e, q_l,
        turn, cosang, lin_raw, np.clip(lin_fit, 0, 3),
        np.log1p(late_net), np.log1p(late_path), np.log1p(step.mean(1)),
        np.log1p(step[:, :split - 1].mean(1)), np.log1p(step[:, split:].mean(1)),
        np.log((step[:, split:].mean(1) + 1e-4) / (step[:, :split - 1].mean(1) + 1e-4)),
        net[:, 0] / nn, net[:, 1] / nn,
        b_l[:, 0] / (sp_l + 1e-9), b_l[:, 1] / (sp_l + 1e-9),
        np.log1p(nn), np.log1p(rq), np.log1p(curv),
        np.log1p(r_all) - np.log1p(rq),
        np.log1p(np.abs(p).max(1).max(1)),
        step.std(1) / (step.mean(1) + 1e-6),
    ], axis=1).astype(np.float32)
    sy = np.zeros(f.shape[1], np.float32)
    sx = np.zeros(f.shape[1], np.float32)
    sy[20] = 1; sy[22] = 1
    sx[21] = 1; sx[23] = 1
    return f, sy, sx


def conf_feats(c0):
    return np.stack([
        c0.mean(1), c0.std(1), c0.min(1), c0[:, 0], c0[:, -1],
        c0[:, :10].mean(1) - c0[:, 10:].mean(1),
    ], 1).astype(np.float32)


def zero_feats(c, c0):
    N, T = c.shape[0], c.shape[1]
    o = c.shape[2] // 2
    z = c[:, :, o, o]
    t = np.arange(T, dtype=np.float32)
    tc = t - t.mean()
    sxx = (tc ** 2).sum()
    bz = (z * tc).sum(1) / sxx
    rz = z - (z.mean(1, keepdims=True) + bz[:, None] * tc[None, :])
    sz = np.sqrt((rz ** 2).mean(1)) + 1e-6
    gap = c0 - z
    r3 = c[:, :, o - 1:o + 2, o - 1:o + 2].reshape(N, T, 9).mean(2)
    return np.stack([
        z.mean(1), z.min(1), z.std(1), z[:, 0], z[:, -1], z[:, -1] - z[:, 0],
        bz, bz / sz, np.abs(bz) * np.sqrt(sxx) / sz,
        z[:, :10].mean(1) - z[:, 10:].mean(1),
        gap.mean(1), gap.max(1), gap[:, 10:].mean(1) - gap[:, :10].mean(1),
        r3.mean(1), r3.min(1), np.arctanh(np.clip(z.mean(1), -0.999, 0.999)),
    ], 1).astype(np.float32)


def tstat_feats(p, split=10):
    T = p.shape[1]
    out = []
    for a, b in ((0, T), (0, split), (split, T)):
        q = p[:, a:b]
        t = np.arange(b - a, dtype=np.float32)
        tc = t - t.mean()
        sxx = (tc ** 2).sum() + 1e-9
        bb = np.einsum("k,nkd->nd", tc, q) / sxx
        pred = q.mean(1, keepdims=True) + bb[:, None, :] * tc[None, :, None]
        res = np.sqrt((((q - pred) ** 2).sum(2)).mean(1)) + 1e-6
        sp = np.linalg.norm(bb, axis=1)
        out += [sp * np.sqrt(sxx) / res, np.log1p(res), sp / (res + 1e-6)]
        for d in (0, 1):
            num = (tc[None, :] * (q[:, :, d] - q[:, :, d].mean(1, keepdims=True))).sum(1)
            den = np.sqrt(sxx * ((q[:, :, d] - q[:, :, d].mean(1, keepdims=True)) ** 2).sum(1) + 1e-12)
            out.append(num / den)
    return np.stack(out, 1).astype(np.float32)


def build_all(cost):
    F, SY, SX = [], [], []
    for j, scale in ((0, 1.0), (1, 4.0)):
        c = cost[:, j].astype(np.float32)
        pk, c0 = peak_path(c)
        pk = pk * scale
        f1, sy1, sx1 = path_feats(pk)
        sp = soft_path(c, 0.05) * scale
        f2, sy2, sx2 = path_feats(sp)
        cf = conf_feats(c0)
        zf = zero_feats(c, c0)
        tf = tstat_feats(pk)
        tf2 = tstat_feats(sp)
        ty = np.zeros(tf.shape[1], np.float32); tx = np.zeros(tf.shape[1], np.float32)
        for w in range(3):
            ty[5 * w + 3] = 1; tx[5 * w + 4] = 1
        F += [f1, f2, cf, zf, tf, tf2]
        SY += [sy1, sy2, np.zeros(cf.shape[1], np.float32),
               np.zeros(zf.shape[1], np.float32), ty, ty.copy()]
        SX += [sx1, sx2, np.zeros(cf.shape[1], np.float32),
               np.zeros(zf.shape[1], np.float32), tx, tx.copy()]
    return (np.concatenate(F, 1), np.concatenate(SY), np.concatenate(SX))
