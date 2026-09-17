import numpy as np, pandas as pd, os
from scipy.stats import kendalltau, rankdata

D = r"C:\Users\srika\Downloads\eris_sperm"
tt = pd.read_csv(os.path.join(D, "train_tracks.csv"))
ti = pd.read_csv(os.path.join(D, "train_index.csv"))
clip2vid = ti.drop_duplicates("clip").set_index("clip").video.to_dict()
tt["video"] = tt["clip"].map(clip2vid)
COLS = ["vcl", "lin", "accel", "turn", "vbias"]
rng = np.random.default_rng(0)


def tau(a, b):
    t = kendalltau(a, b).correlation
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def score(pred_df, true_df):
    return np.mean([tau(pred_df[c], true_df[c]) for c in COLS]), \
           {c: round(tau(pred_df[c], true_df[c]), 4) for c in COLS}


Y = tt[COLS].copy()
static = (tt.vcl == 0).values
straight = ((tt.lin == 1) & (tt.vcl > 0)).values
noturn = ((tt.turn == 0) & (tt.vcl > 0)).values
print("n =", len(tt), " static", static.mean().round(3),
      " straight(mover)", straight.mean().round(3), " noturn(mover)", noturn.mean().round(3))
print()

print("=== ORACLE STRATEGY LADDER (full train set, tau-b) ===")


def mk(vals):
    return pd.DataFrame({c: vals[i] for i, c in enumerate(COLS)})


n = len(tt)
z = np.zeros(n)
o = np.ones(n)

s1 = mk([np.where(static, 0, 1)] * 5)
print("1. perfect static/mover binary, constant for movers      ", score(s1, Y))

s2 = mk([np.where(static, 0, 1),
         np.where(static, 0, 1),
         np.where(static, 0, 1),
         np.where(static | noturn, 0, 1),
         z])
print("2. + perfect no-turn group on turn axis                  ", score(s2, Y))

s3 = mk([np.where(static, 0, 1),
         np.where(static, 0, np.where(straight, 2, 1)),
         np.where(static, 0, 1),
         np.where(static | noturn, 0, 1),
         z])
print("3. + perfect straight group on lin axis                  ", score(s3, Y))

s4 = mk([np.where(static, 0.0, tt.vcl.values),
         np.where(static, 0, np.where(straight, 2, 1)),
         np.where(static, 0, 1),
         np.where(static | noturn, 0, 1),
         z])
print("4. + ORACLE vcl values                                   ", score(s4, Y))

s5 = mk([tt.vcl.values, tt.lin.values, tt.accel.values, tt.turn.values, z])
print("5. oracle everything except vbias (=0)                   ", score(s5, Y))

s6 = mk([tt.vcl.values, tt.lin.values, tt.accel.values, tt.turn.values, tt.vbias.values])
print("6. full oracle                                           ", score(s6, Y))
print()

print("=== value of each axis: replace one axis with noise, keep rest oracle ===")
for i, c in enumerate(COLS):
    vals = [tt[cc].values.copy() for cc in COLS]
    vals[i] = rng.normal(size=n)
    print(f"  randomise {c:6s} -> {score(mk(vals), Y)[0]:.4f}")
print()

print("=== marginal value of a GOOD (not perfect) static detector ===")
for acc in [1.0, 0.98, 0.95, 0.9, 0.8]:
    p = static.astype(float).copy()
    flip = rng.random(n) > acc
    p[flip] = 1 - p[flip]
    pred = mk([np.where(p > 0.5, 0, 1)] * 4 + [z])
    print(f"  static-detector acc {acc:.2f} -> {score(pred, Y)[0]:.4f}")
print()

print("=== does TYING actually help under tau-b? (vcl axis) ===")
noisy = tt.vcl.values + rng.normal(0, 0.05, n)
print("  raw noisy vcl, no ties          tau", round(tau(noisy, tt.vcl), 4))
tied = noisy.copy()
tied[static] = -1.0
print("  same, static forced to a tie    tau", round(tau(tied, tt.vcl), 4))
tied2 = np.where(static, -1.0, rankdata(noisy))
print("  static tied + ranks elsewhere   tau", round(tau(tied2, tt.vcl), 4))
print()
print("=== same question on lin (52% of truth tied at 1.0) ===")
lin_noisy = tt.lin.values + rng.normal(0, 0.05, n)
print("  noisy lin, no ties              tau", round(tau(lin_noisy, tt.lin), 4))
lt = lin_noisy.copy()
lt[static] = -1
lt[straight] = 2
print("  static & straight groups tied   tau", round(tau(lt, tt.lin), 4))
print()

print("=== TEST-SIZED subsample (808 queries, 7 videos) — variance of the estimate ===")
vids = tt.video.unique()
for trial in range(5):
    vs = rng.choice(vids, 7, replace=False)
    sub = tt[tt.video.isin(vs)]
    if len(sub) > 808:
        sub = sub.sample(808, random_state=trial)
    st = (sub.vcl == 0).values
    nt = ((sub.turn == 0) & (sub.vcl > 0)).values
    sg = ((sub.lin == 1) & (sub.vcl > 0)).values
    pred = pd.DataFrame({
        "vcl": np.where(st, 0, 1), "lin": np.where(st, 0, np.where(sg, 2, 1)),
        "accel": np.where(st, 0, 1), "turn": np.where(st | nt, 0, 1),
        "vbias": np.zeros(len(sub))})
    m, d = score(pred, sub[COLS])
    print(f"  trial{trial} n={len(sub)} static={st.mean():.3f} strategy3 score={m:.4f} {d}")
print()
print("=== per-video static fraction (is the population stable across videos?) ===")
g = tt.groupby("video").agg(n=("vcl", "size"), static=("vcl", lambda s: (s == 0).mean()),
                            straight=("lin", lambda s: (s == 1).mean()),
                            medvcl=("vcl", "median"))
print(g.round(3))
