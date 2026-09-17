import pandas as pd, numpy as np
d = pd.read_csv("working/submission.csv")
s = pd.read_csv("sample_submission.csv")
print("shape", d.shape, "cols", list(d.columns))
print("ids match sample order:", list(d.id) == list(s.id))
print("ids unique:", d.id.is_unique, "n dup", d.id.duplicated().sum())
for c in ["vcl", "lin", "accel", "turn", "vbias"]:
    v = d[c].values
    print(f"{c:6s} nuniq {len(np.unique(v)):5d} min {v.min():12.4f} max {v.max():12.4f} "
          f"finite {bool(np.isfinite(v).all())}")
