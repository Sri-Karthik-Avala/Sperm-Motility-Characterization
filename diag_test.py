import numpy as np, pandas as pd, os

D = r"C:\Users\srika\Downloads\eris_sperm"
xi = pd.read_csv(os.path.join(D, "test_index.csv"))
mq = pd.read_csv(os.path.join(D, "mot_queries.csv"))
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
te = np.load(os.path.join(D, "test_images.npy"), mmap_mode="r")
tr = np.load(os.path.join(D, "train_images.npy"), mmap_mode="r")

print("test_index head:\n", xi.head(4))
print("row range", xi.row.min(), xi.row.max(), "n", len(xi), "images", te.shape[0])
print("rows are a permutation of 0..N-1:", sorted(xi.row.values) == list(range(len(xi))))
print("is test_index already sorted by row?", (xi.row.values == np.arange(len(xi))).all())
print()
print("queries clips subset of index clips:", set(mq["clip"]).issubset(set(xi["clip"])))
print("n query clips", mq["clip"].nunique(), "n index clips", xi["clip"].nunique())
print()

c0 = int(mq["clip"].iloc[0])
g = xi[xi["clip"] == c0].sort_values("t")
print(f"test clip {c0}: rows {g.row.values[:6]}... t {g.t.values[:6]}... n={len(g)}")
f = np.asarray(te[g.row.values]).astype(np.float32)
print("consecutive |diff| mean:", np.round(np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))[:6], 3))
print("|f_t - f_0| mean:", np.round(np.abs(f - f[0]).mean(axis=(1, 2))[:6], 3))
print("frame means:", np.round(f.mean(axis=(1, 2))[:6], 2))

gt = ti[ti["clip"] == 0].sort_values("t")
ft = np.asarray(tr[gt.row.values]).astype(np.float32)
print()
print("TRAIN clip 0 for comparison:")
print("consecutive |diff| mean:", np.round(np.abs(np.diff(ft, axis=0)).mean(axis=(1, 2))[:6], 3))
print("|f_t - f_0| mean:", np.round(np.abs(ft - ft[0]).mean(axis=(1, 2))[:6], 3))

print()
print("=== is the test clip contiguous in row order? ===")
for c in list(mq["clip"].unique())[:5]:
    g = xi[xi["clip"] == c].sort_values("t")
    r = g.row.values
    print(f"  clip {c}: rows {r.min()}..{r.max()} contiguous={np.array_equal(r, np.arange(r.min(), r.min()+len(r)))} n={len(r)}")

print()
print("=== bright blob at the query ref position in frame 0? ===")
H, W = 480, 640
for _, q in mq.head(6).iterrows():
    g = xi[xi["clip"] == int(q["clip"])].sort_values("t")
    f0 = np.asarray(te[g.row.values[0]]).astype(np.float32)
    cx, cy = int(round(q.ref_cx * W)), int(round(q.ref_cy * H))
    loc = f0[max(0, cy - 10):cy + 10, max(0, cx - 10):cx + 10]
    print(f"  q {q.query_id} clip{int(q['clip'])} cx{cx} cy{cy} pix {f0[cy,cx]:.0f} "
          f"localmed {np.median(loc):.0f} z {(f0[cy,cx]-np.median(loc))/(loc.std()+1e-6):+.2f}")

print()
print("=== same check on TRAIN cells (should look identical) ===")
tt = pd.read_csv(os.path.join(D, "train_tracks.csv"))
for _, q in tt.head(6).iterrows():
    g = ti[ti["clip"] == int(q["clip"])].sort_values("t")
    f0 = np.asarray(tr[g.row.values[0]]).astype(np.float32)
    cx, cy = int(round(q.ref_cx * W)), int(round(q.ref_cy * H))
    loc = f0[max(0, cy - 10):cy + 10, max(0, cx - 10):cx + 10]
    print(f"  clip{int(q['clip'])} cx{cx} cy{cy} pix {f0[cy,cx]:.0f} "
          f"localmed {np.median(loc):.0f} z {(f0[cy,cx]-np.median(loc))/(loc.std()+1e-6):+.2f}")
