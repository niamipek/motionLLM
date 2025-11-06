import torch

path = "/mnt/d/motionLLM/checkpoints/net_best_fid_trans.pth"
ckpt = torch.load(path, map_location="cpu")

print("Top-level keys:", ckpt.keys())
state = ckpt.get("trans", ckpt)

for k in list(state.keys())[:20]:
    print(k)
