import numpy as np, torch, os
CKPT_DIR = "/data2/user/zyq/checkpoints/PRE/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2"

print("=== loss.dat (time train val_rel) ===")
arr = np.loadtxt(os.path.join(CKPT_DIR, "loss.dat")).reshape(-1, 3)
for i, row in enumerate(arr):
    print(f"ep{i+1}: t={row[0]:.1f}s train={row[1]:.5f} val_rel={row[2]:.5f}")

print("\n=== checkpoint Ep3 ===")
ck = torch.load(os.path.join(CKPT_DIR, "Ep3.pth"), map_location="cpu", weights_only=True)
print("keys:", list(ck.keys()))
print("epoch:", ck.get("epoch"), "best_val:", ck.get("best_val"))
print("config:", ck.get("config"))
sd = ck["model_state_dict"]
print("first keys:", list(sd.keys())[:6])
print("n params tensors:", len(sd))
w = sd["net.patch_embed.proj.weight"]
print("patch_embed.proj.weight:", tuple(w.shape), "mean_abs per input channel:")
ma = w.abs().mean(dim=(2,3,4))  # (embed_dim, in_chans)
for c in range(w.shape[1]):
    tag = "COND" if c < 14 else "TGT"
    print(f"  ch{c:2d} [{tag}]: mean|w|={ma[:,c].mean().item():.5f} std={w[:,c].std().item():.5f}")
print("head.weight:", tuple(sd["net.head.weight"].shape), "std:", sd["net.head.weight"].std().item())
print("pos_embed:", tuple(sd["net.pos_embed"].shape), "std:", sd["net.pos_embed"].std().item())
for k in ("upproj.weight","downproj.weight"):
    print(k, tuple(sd[k].shape), "std:", sd[k].std().item())
b1 = sd["net.blocks.0.filter.w1"]
print("blocks.0.filter.w1:", tuple(b1.shape), "std:", b1.std().item())

print("\n=== eval npz (val h1 ch0 e1 s123 ckptEp3) ===")
z = np.load(os.path.join(CKPT_DIR, "eval_val_h1_ch0_e1_s123_ckptEp3.npz"), allow_pickle=True)
for k in z.files:
    v = z[k]
    if v.size <= 4:
        print(f"{k}: {v}")
print("rmse_model d1:", z["rmse_model"].ravel()[:2], "...")
def pooled(se, n):
    return float(np.sqrt(np.asarray(se).sum() / np.asarray(n).sum()))
print("pooled day1 model :", pooled(z["rmse_model"][0]**2 * z["valid_count"][0], z["valid_count"][0]))
print("n_windows:", z["n_windows"], "stride:", z["stride"], "sigma_data:", z["sigma_data"], "sampling_steps:", z["sampling_steps"])

print("\n=== norm stats ===")
s = np.load("/data2/user/zyq/data_processed/PRE/norm/stats_d29_clipnone.npz")
print("lo:", s["lo"], "hi:", s["hi"], "sigma:", s["sigma"].item())
print("EDM sigma_data (2x):", 2*s["sigma"].item())
print("(lo+hi)/2 physical mean-field value: u:", (s['lo'][0]+s['hi'][0])/2, " v:", (s['lo'][1]+s['hi'][1])/2)
