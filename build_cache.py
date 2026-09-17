import numpy as np, pandas as pd, os, time

D = r"C:\Users\srika\Downloads\eris_sperm"
OUT = os.path.join(D, "cache")
os.makedirs(OUT, exist_ok=True)

H, W, T = 480, 640, 20
RF = 24          # fine half-width -> 48x48 full res
RC = 96          # coarse half-width -> 192x192 -> /4 -> 48x48
DS = 4
S = 2 * RF


def build(index_csv, cells_df, images_npy, cx_col, cy_col, tag):
    ti = pd.read_csv(os.path.join(D, index_csv))
    im = np.load(os.path.join(D, images_npy), mmap_mode="r")
    order = {}
    for c, g in ti.sort_values("t").groupby("clip"):
        order[int(c)] = g.row.values.astype(np.int64)
    n = len(cells_df)
    fine = np.zeros((n, T, S, S), dtype=np.uint8)
    coarse = np.zeros((n, T, S, S), dtype=np.uint8)
    t0 = time.time()
    for ci, (clip, grp) in enumerate(cells_df.groupby("clip")):
        rows = order[int(clip)]
        f = np.asarray(im[rows]).astype(np.uint8)
        fp = np.pad(f, ((0, 0), (RC, RC), (RC, RC)), mode="reflect")
        for idx, r in zip(grp.index.values, grp.itertuples()):
            cx = int(round(getattr(r, cx_col) * W))
            cy = int(round(getattr(r, cy_col) * H))
            py, px = cy + RC, cx + RC
            fine[idx] = fp[:, py - RF:py + RF, px - RF:px + RF]
            cc = fp[:, py - RC:py + RC, px - RC:px + RC]
            coarse[idx] = cc.reshape(T, S, DS, S, DS).mean(axis=(2, 4)).astype(np.uint8)
        if ci % 50 == 0:
            print(f"  {tag} clip {ci} elapsed {time.time()-t0:.1f}s", flush=True)
    np.save(os.path.join(OUT, f"{tag}_fine.npy"), fine)
    np.save(os.path.join(OUT, f"{tag}_coarse.npy"), coarse)
    print(f"{tag}: fine {fine.shape} coarse {coarse.shape} in {time.time()-t0:.1f}s")


tt = pd.read_csv(os.path.join(D, "train_tracks.csv")).reset_index(drop=True)
mq = pd.read_csv(os.path.join(D, "mot_queries.csv")).reset_index(drop=True)
build("train_index.csv", tt, "train_images.npy", "ref_cx", "ref_cy", "train")
build("test_index.csv", mq, "test_images.npy", "ref_cx", "ref_cy", "test")
