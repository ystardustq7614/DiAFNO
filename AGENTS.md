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

- `diffusion.py`: EDM (`ElucidatedDiffusion`) — training loss in `forward()`, inference via `sample()` (Heun sampler); do not "fix" the commented-out self-conditioning code, it is intentionally disabled.
- `IAFNO.py`: AFNO token mixer (FFT-based) + patch embedding. Grid is 64x65x32; y-axis is zero-padded to 66 for even patching (`dim` vs `dim_f`).
- `utilities3.py`: shared FNO utilities (`LpLoss`, `count_params`, normalizers).
- Core IAFNO device handling follows the model/input device; the constructor no longer forces `.cuda()`, and padding uses `x.new_zeros`. The legacy reader/normalizer helper methods in `utilities3.py` still expose optional `.cuda()` convenience methods, but they are not called by the main training path.
- `loss.dat` (train/test/real loss per epoch) is written to the CWD; checkpoints are `test_Ep{n}.pth` per epoch (`.gitignore`d).
- `checkpoint_path` can load a model state dict. Current saves contain only `model.state_dict()`, so optimizer/scheduler/scaler state and the completed epoch are not resumed.

## Conventions

- Global seeds are fixed at module import (`torch.manual_seed(123)` in trainer.py/IAFNO.py).
- Hyperparameters are module-level constants in trainer.py (batch_size, embed_dim, implicit/explicit_layer, sampling_steps, ...) — not CLI args.
- Do not commit checkpoints, runs/, outputs/, *.pt/*.pth (all gitignored).
- Upstream repo: https://github.com/yuchi-richard-jiang/DiAFNO (adds dataset link + citation); origin is this adaptation fork.
