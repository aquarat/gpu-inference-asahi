import sys, numpy as np, os
ref, out = sys.argv[1], sys.argv[2]
cs = []
for img in ["shapes", "noise", "checker", "t1"]:
    p = f"{out}/{img}.bin"
    if not os.path.exists(p): continue
    a = np.fromfile(p, np.float32); b = np.load(f"{ref}/{img}.ort.npy")
    c = float(a @ b); cs.append(c)
    print(f"  cos {img}: {c:.6f}  maxabs {float(np.abs(a-b).max()):.5f}")
print(f"  cos min {min(cs):.6f} mean {np.mean(cs):.6f}")
