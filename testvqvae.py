from options.option import get_args_parser
from models.vqvae import VQVAE_251
import torch

# --- cấu hình GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🟩 Using device:", device)

# --- chuẩn bị args ---
args = get_args_parser()
args.dataname = "humanml3d"
args.quantizer = "ema_reset"

# --- load checkpoint ---
path = "/mnt/c/Users/ADMIN/Downloads/net_best_fid.pth"
vqvae = VQVAE_251(args, nb_code=512)
ckpt = torch.load(path, map_location=device)

vqvae.load_state_dict(ckpt["net"], strict=False)
vqvae = vqvae.to(device)      # 🔹 ép toàn model lên GPU
vqvae.eval()

# --- test encode/decode ---
motion_tensor = torch.randn(1, 196, 263, device=device)   # 🔹 dữ liệu cũng phải nằm trên GPU
motion_token = vqvae.encode(motion_tensor)
motion_recon = vqvae.forward_decoder(motion_token)

print("Token shape:", motion_token.shape)
print("Recon shape:", motion_recon.shape)
