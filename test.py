import torch

path = "/mnt/c/Users/ADMIN/Downloads/net_last.pth"  # hoặc net_last.pth nếu bạn muốn
ckpt = torch.load(path, map_location="cpu")
state_dict = ckpt["net"]

tensor = state_dict["vqvae.quantizer.codebook"]
print("Codebook shape:", tensor.shape)
