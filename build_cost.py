import numpy as np, os, time, torch, torch.nn.functional as Fn

D = r"C:\Users\srika\Downloads\eris_sperm"
TS = 13
OUTS = 36
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(10, os.cpu_count()))


def zncc(frames):
    B, T, S, _ = frames.shape
    c, h = S // 2, TS // 2
    tpl = frames[:, 0, c - h:c + h + 1, c - h:c + h + 1].reshape(B, 1, TS * TS)
    tpl = tpl - tpl.mean(2, keepdim=True)
    tpl = tpl / (tpl.norm(dim=2, keepdim=True) + 1e-6)
    x = frames.reshape(B * T, 1, S, S)
    u = Fn.unfold(x, TS)
    u = u - u.mean(1, keepdim=True)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-6)
    w = tpl.repeat_interleave(T, dim=0)
    r = torch.bmm(w, u).reshape(B, T, S - TS + 1, S - TS + 1)
    return r


def build(tag):
    fine = np.load(os.path.join(D, "cache", f"{tag}_fine.npy"))
    coarse = np.load(os.path.join(D, "cache", f"{tag}_coarse.npy"))
    N = fine.shape[0]
    out = np.zeros((N, 2, 20, OUTS, OUTS), dtype=np.float16)
    t0 = time.time()
    CH = 16
    for i in range(0, N, CH):
        for j, arr in enumerate((fine, coarse)):
            f = torch.from_numpy(arr[i:i + CH].astype(np.float32)).to(dev)
            r = zncc(f)
            out[i:i + CH, j] = r.cpu().numpy().astype(np.float16)
        if i % 1600 == 0:
            print(f"  {tag} {i}/{N} {time.time()-t0:.1f}s", flush=True)
    np.save(os.path.join(D, "cache", f"{tag}_cost.npy"), out)
    print(f"{tag} cost {out.shape} {out.nbytes/1e6:.0f}MB in {time.time()-t0:.1f}s")


build("train")
build("test")
