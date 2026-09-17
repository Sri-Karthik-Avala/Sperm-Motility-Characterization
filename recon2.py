import numpy as np, pandas as pd, os

D = r"C:\Users\srika\Downloads\eris_sperm"
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
tt = pd.read_csv(os.path.join(D, "train_tracks.csv"))
tr = np.load(os.path.join(D, "train_images.npy"), mmap_mode="r")

print("=== discrete structure of the 5 targets ===")
z = (tt.vcl == 0)
print("all-zero-vcl rows:", z.sum())
print("  of those: lin==0", (tt.lin[z] == 0).sum(), "accel==0", (tt.accel[z] == 0).sum(),
      "turn==0", (tt.turn[z] == 0).sum(), "vbias==0", (tt.vbias[z] == 0).sum())
nz = ~z
print("nonzero-vcl rows:", nz.sum())
print("  lin==1 exactly:", (tt.lin[nz] == 1).sum(), " turn==0 exactly:", (tt.turn[nz] == 0).sum())
print("  lin==1 AND turn==0:", ((tt.lin == 1) & (tt.turn == 0) & nz).sum())
print("  0<lin<1:", ((tt.lin > 0) & (tt.lin < 1)).sum())
print()
print("joint (lin==1, turn==0) crosstab on nonzero rows:")
print(pd.crosstab(tt.lin[nz] == 1, tt.turn[nz] == 0))
print()
print("=== vcl on log scale, nonzero only ===")
v = tt.vcl[nz].values
print("percentiles:", {p: round(float(np.percentile(v, p)), 5) for p in [1, 5, 10, 25, 50, 75, 90, 99]})
print("frac nonzero vcl < 0.1 px/frame:", (v < 0.1).mean().round(3),
      " < 0.5:", (v < 0.5).mean().round(3), " > 1:", (v > 1).mean().round(3))
print()
print("=== accel tail ===")
a = tt.accel[nz].values
print("accel percentiles:", {p: round(float(np.percentile(a, p)), 4) for p in [1, 25, 50, 75, 99, 100]})
print("frac accel > 10:", (a > 10).mean().round(4), " frac accel==0 among nonzero vcl:", (a == 0).mean().round(4))
print()
print("=== vbias structure ===")
b = tt.vbias.values
print("frac exactly 0:", (b == 0).mean().round(3), " frac |b|==1:", (np.abs(b) == 1).mean().round(3))
print("vbias among nonzero vcl: frac 0:", (tt.vbias[nz] == 0).mean().round(3))
print()

print("=== can we see the cells? patch inspection ===")
ti_s = ti.sort_values(["clip", "t"])
clip_rows = {c: g.row.values for c, g in ti_s.groupby("clip")}
H, W = 480, 640
R = 24

fast = tt[tt.vcl > 2.0].head(4)
slow = tt[tt.vcl == 0].head(4)
for name, sel in [("FAST", fast), ("STATIC", slow)]:
    print(f"--- {name} cells ---")
    for _, r in sel.iterrows():
        rows = clip_rows[int(r["clip"])]
        f = np.asarray(tr[rows]).astype(np.float32)
        cx, cy = int(round(r.ref_cx * W)), int(round(r.ref_cy * H))
        y0, y1 = max(0, cy - R), min(H, cy + R)
        x0, x1 = max(0, cx - R), min(W, cx + R)
        p = f[:, y0:y1, x0:x1]
        bg = np.median(f, axis=0)[y0:y1, x0:x1]
        d = p - bg[None]
        cen = p[:, R - 3:R + 3, R - 3:R + 3] if p.shape[1] >= 2 * R else p
        print(f"  clip{int(r['clip']):3d} cx{cx:4d} cy{cy:4d} vcl {r.vcl:7.4f} lin {r.lin:6.3f} "
              f"turn {r.turn:6.3f} | patch mean {p.mean():6.1f} std {p.std():5.2f} | "
              f"centre3x3 t0 {cen[0].mean():6.1f} tN {cen[-1].mean():6.1f} | "
              f"bgsub |d| mean {np.abs(d).mean():5.2f} max {np.abs(d).max():6.1f}")

print()
print("=== background model: are cells bright or dark vs local background? ===")
rows = clip_rows[0]
f = np.asarray(tr[rows]).astype(np.float32)
bg = np.median(f, axis=0)
sub = tt[tt["clip"] == 0]
print("cells in clip0:", len(sub))
for _, r in sub.head(6).iterrows():
    cx, cy = int(round(r.ref_cx * W)), int(round(r.ref_cy * H))
    val0 = f[0, cy, cx]
    loc = f[0, max(0, cy - 10):cy + 10, max(0, cx - 10):cx + 10]
    print(f"  cx{cx:4d} cy{cy:4d} vcl {r.vcl:7.4f} pix@ref {val0:6.1f} localmed {np.median(loc):6.1f} "
          f"localmax {loc.max():6.1f} localmin {loc.min():6.1f} z {(val0-np.median(loc))/(loc.std()+1e-6):6.2f}")

print()
print("=== per-frame global background drift (does frame median move?) ===")
print("frame means clip0:", np.round(f.mean(axis=(1, 2)), 2))
print("global-median-subtracted residual energy per frame:",
      np.round(np.abs(f - bg[None]).mean(axis=(1, 2)), 3))

print()
print("=== how many blobs? threshold the bg-subtracted frame ===")
d0 = np.abs(f[0] - bg)
for thr in [5, 10, 20, 40]:
    print(f"  |f0-bg| > {thr}: {(d0 > thr).sum()} px")
