import sys, numpy as np, hashlib
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32)
d = np.abs(a - b)
print(f"{sys.argv[1]} vs {sys.argv[2]}: n={a.size} md5 {hashlib.md5(a.tobytes()).hexdigest()[:10]} / {hashlib.md5(b.tobytes()).hexdigest()[:10]} max|diff| {d.max():.5f} mean {d.mean():.5f} nan {np.isnan(a).sum()}")
