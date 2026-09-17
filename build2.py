import numpy as np, pandas as pd, os, time, torch, torch.nn.functional as Fn

D = r"C:\Users\srika\Downloads\eris_sperm"
OUT = os.path.join(D, "cache")
H, W, T = 480, 640, 20
S = 48
TS = 13
SCALES = [1, 2, 5]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(10, os.cpu_count()))


def zncc(frames):
    B, TT, SS, _ = frames.shape
    c, h = SS // 2, TS // 2
    tpl = frames[:, 0, c - h:c + h + 1, c - h:c + h + 1].reshape(B, 1, TS * TS)
    tpl = tpl - tpl.mean(2, keepdim=True)
    tpl = tpl / (tpl.norm(dim=2, keepdim=True) + 1e-6)
    u = Fn.unfold(frames.reshape(B * TT, 1, SS, SS), TS)
    u = u - u.mean(1, keepdim=True)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-6)
    r = torch.bmm(tpl.repeat_interleave(TT, dim=0), u)
    return r.reshape(B, TT, SS - TS + 1, SS - TS + 1)


def build(index_csv, cells, images, tag):
    ti = pd.read_csv(os.path.join(D, index_csv))
    im = np.load(os.path.join(D, images), mmap_mode="r")
    order = {int(c): g.row.values.astype(np.int64)
             for c, g in ti.sort_values("t").groupby("clip")}
    n = len(cells)
    crops = {s: np.zeros((n, T, S, S), dtype=np.uint8) for s in SCALES}
    RMAX = (S // 2) * max(SCALES)
    t0 = time.time()
    for ci, (clip, grp) in enumerate(cells.groupby("clip")):
        f = np.asarray(im[order[int(clip)]]).astype(np.uint8)
        fp = np.pad(f, ((0, 0), (RMAX, RMAX), (RMAX, RMAX)), mode="reflect")
        for idx, r in zip(grp.index.values, grp.itertuples()):
            cx = int(round(r.ref_cx * W)) + RMAX
            cy = int(round(r.ref_cy * H)) + RMAX
            for s in SCALES:
                R = (S // 2) * s
                blk = fp[:, cy - R:cy + R, cx - R:cx + R]
                if s == 1:
                    crops[s][idx] = blk
                else:
                    crops[s][idx] = blk.reshape(T, S, s, S, s).mean(axis=(2, 4)).astype(np.uint8)
        if ci % 60 == 0:
            print(f"  {tag} clip {ci} {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, f"{tag}_c1.npy"), crops[1])
    cost = np.zeros((n, len(SCALES), T, S - TS + 1, S - TS + 1), dtype=np.float16)
    for j, s in enumerate(SCALES):
        for i in range(0, n, 16):
            x = torch.from_numpy(crops[s][i:i + 16].astype(np.float32)).to(dev)
            cost[i:i + 16, j] = zncc(x).cpu().numpy().astype(np.float16)
    np.save(os.path.join(OUT, f"{tag}_cost3.npy"), cost)
    print(f"{tag}: crops {crops[1].shape} cost {cost.shape} {cost.nbytes/1e6:.0f}MB in {time.time()-t0:.0f}s")


tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
mq = pd.read_csv(os.path.join(D, "mot_queries.csv")).reset_index(drop=True)
build("train_index.csv", tt, "train_images.npy", "train")
build("test_index.csv", mq, "test_images.npy", "test")
