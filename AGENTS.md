# AGENTS.md

Research code (one CPU smoke test, no lint config, no CI) for DiAFNO: IAFNO (Fourier-neural-operator backbone) + Elucidated diffusion model for autoregressive 3D turbulence prediction. This fork is being adapted for weather/ocean data (branch `adapt-weather-ocean`).

## Environment & run

- The server target is the conda env `diafno` (Python 3.10, torch 2.4.1+cu124, AMP supported) on a machine with 4x RTX 4090. A local environment may be CPU-only; verify with `torch.__version__` and `torch.cuda.is_available()` before interpreting test results.
- Training entrypoint is `trainer.py` (run with `python trainer.py` from repo root — it is a script, not a package).
- Run `python smoke_test.py` for the minimal CPU device/checkpoint check. Full verification still requires a short training (`trainset_num`/`count` scale data size; `count` is capped at 200 by default "for fast testing").
- `environment.yml` is a minimal environment description, not an exact CUDA lock: `requirements-lock.txt` records the server snapshot, including `torch==2.4.1+cu124` and `xarray`.

## trainer.py is a template — edit placeholders before running

These strings must be replaced or the run crashes/fails silently:
- `np.load('your dataset')` → data path (data layout `bs nt x y z c`, channels are the 3 velocity components)
- `info_folder_path = "max_min_sigma info of your dataset"` → dir where the normalization cache is stored/loaded
- `parent_dir = "your directory for saving files"` → checkpoint output dir

## Data & normalization gotchas

- Model tensors are `bs c x y z`; raw data is `bs x y z c` — converted with einops `rearrange` in trainer.py. Keep this ordering in any new data-loading code.
- Input/output are min-max normalized per channel using cached stats; cache filename encodes hyperparams: `ts{trainset_num}_c{count}_iw{InferenceWidth}_ii{InitialInterval}.npy` (also stores `sigma` used as `sigma_data` in the diffusion model). Deleting the cache triggers recomputation.
- Local data lives in `~/datasets/` (symlinks to `/data/{copernicus_uv_data,era_data/raw_data,PRE_ocean_data,sst_data/...}`) and `~/data_processed/{Copernicus,ERA5,PRE,SST}`. These dirs are outside the repo.

## Architecture notes

- `diffusion.py`: EDM (`ElucidatedDiffusion`) — training loss in `forward()`, inference via `sample()` (Heun sampler); do not "fix" the commented-out self-conditioning code, it is intentionally disabled. `forward(images, self_cond, mask=None)` accepts an optional broadcastable ocean mask (1=valid); when given, the denoising MSE is a per-sample mean over valid elements only.
- `IAFNO.py`: AFNO token mixer (FFT-based) + patch embedding. Grid is 64x65x32; y-axis is zero-padded to 66 for even patching (`dim` vs `dim_f`). `IAFNODiff(..., cond_chans=None)`: noisy-target channels (`in_chans`) and external-condition channels (`cond_chans`) are decoupled — patch-embed input is `in_chans + cond_chans`; the default `cond_chans=None` reproduces the legacy doubling (`in_chans*2`).

## PRE_ocean_data forecast task (pre_*.py)

7-day condition (14 ch, day-major u/v interleaved) → next-day u/v (2 ch), single-step conditional diffusion; 15-day forecasts via autoregressive rollout. Plan A: raw staggered u/v collocated onto the rho grid (no rotation to east/north).

- Pipeline files: `scripts/preprocess_align_uv.py` (one-time colocation, ~15-60 min, outputs `~/data_processed/PRE/aligned/{u_rho,v_rho}.npy` + `mask_uv.npy`; masks are authoritative — NaN at a mask==1 ocean cell fails hard, values at mask==0 cells (45 static boundary u-faces in this dataset) are discarded to NaN and counted), `pre_config.py` (side-effect-free presets `surface_smoke`/`full3d`), `pre_dataset.py` (contiguous time splits train/val/test = [0,8401)/[8401,9496)/[9496,10591); percentile-clip min-max normalization on train ocean points, cached in `~/data_processed/PRE/norm/`), `pre_trainer.py` (training entrypoint), `pre_evaluate.py` (15-step rollout + persistence baseline + masked RMSE/MAE per lead day × variable × sigma layer).
- `pre_trainer.py`/`pre_evaluate.py` are scripts (module top-level, like trainer.py) — never import them; shared config lives in `pre_config.py`. Checkpoints: `~/checkpoints/PRE/<run_tag>/`.
- Grids use exact-division patches so IAFNO padding never triggers: surface 400x441x1 patch (4,3,1); full3d 400x441x30 patch (4,3,2). 441 = 3²·7² constrains y-patch choices.
- Full step-by-step run instructions and verification points: `docs/PRE_runbook.md`.
- `utilities3.py`: shared FNO utilities (`LpLoss`, `count_params`, normalizers).
- Core IAFNO device handling follows the model/input device; the constructor no longer forces `.cuda()`, and padding uses `x.new_zeros`. The legacy reader/normalizer helper methods in `utilities3.py` still expose optional `.cuda()` convenience methods, but they are not called by the main training path.
- `loss.dat` (train/test/real loss per epoch) is written to the CWD; checkpoints are `test_Ep{n}.pth` per epoch (`.gitignore`d).
- `checkpoint_path` can load a model state dict. Current saves contain only `model.state_dict()`, so optimizer/scheduler/scaler state and the completed epoch are not resumed.

## Conventions

- Global seeds are fixed at module import (`torch.manual_seed(123)` in trainer.py/IAFNO.py).
- Hyperparameters are module-level constants in trainer.py (batch_size, embed_dim, implicit/explicit_layer, sampling_steps, ...) — not CLI args.
- Do not commit checkpoints, runs/, outputs/, *.pt/*.pth (all gitignored).
- Upstream repo: https://github.com/yuchi-richard-jiang/DiAFNO (adds dataset link + citation); origin is this adaptation fork.
