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

- Pipeline files: `scripts/preprocess_align_uv.py` (one-time colocation, ~15-60 min, outputs `~/data_processed/PRE/aligned/{u_rho,v_rho}.npy` + `mask_uv.npy`; masks are authoritative — NaN at a mask==1 ocean cell fails hard, values at mask==0 cells (45 static boundary u-faces in this dataset) are discarded to NaN and counted), `pre_config.py` (side-effect-free presets `surface_smoke`/`full3d` + shared sigma_data scale), `pre_dataset.py` (contiguous time splits train/val/test = [0,8401)/[8401,9496)/[9496,10591); percentile-clip min-max normalization on train ocean points, cached in `~/data_processed/PRE/norm/`), `pre_rollout.py` (side-effect-free autoregressive ensemble rollout, no data imports), `pre_trainer.py` (training entrypoint), `pre_evaluate.py` (15-step rollout + persistence/zero/rho-oracle baselines + masked RMSE/MAE per lead day × variable × sigma layer).
- **sigma_data scale**: the stats cache stores the pooled std of the [0,1]-normalized data (0.08560 for surface); diffusion.py internally normalizes images with `images*2-1`, so the EDM sigma_data is `2.0 * stats["sigma"]` (0.17120). The shared conversion lives in `pre_config.py` (`SIGMA_DATA_SCALE`, `sigma_data_from_stats`, `sigma_data_from_checkpoint`); training and evaluation MUST call the same implementation. New checkpoints store `config.{stats_sigma,sigma_data_scale,sigma_data}`; evaluation prefers the checkpoint value and falls back to the legacy stats-only scale for old checkpoints with an explicit warning.
- `pre_trainer.py`/`pre_evaluate.py` are scripts (module top-level, like trainer.py) — never import them; shared config lives in `pre_config.py`. Checkpoints: `~/checkpoints/PRE/<run_tag>/` where `run_tag_for(preset)` appends `_SD2` to the legacy tag (fixed-scale runs never share a directory with sd1 runs).
- `pre_trainer.py`: per-preset `EPOCH_OVERRIDES` (surface_smoke=4 for the short retrain; other presets keep their own num_epochs), new AMP API (`torch.amp.GradScaler/autocast`), `scheduler.step()` only after a real optimizer update (skipped-update counts reported per epoch), non-finite loss aborts with epoch/batch/scale, early stop after 2 consecutive worsening val epochs. Resume scale policy `RESUME_SIGMA_POLICY`: `"error"` (default — refuses to resume a checkpoint whose sigma_data differs from the current SD2 scale), `"migrate"` (explicit scale migration, keeps SD2), `"adopt"` (explicit legacy continuation: uses the checkpoint's old sigma_data, writes outputs into a dedicated `legacy_resume/` subdirectory of the checkpoint's dir — history read from the ORIGINAL loss.dat, so the continuation's loss.dat holds the FULL history; the ACTUAL scale 1.0 is recorded in config, and a resumed run can never be mistaken for SD2). Pre-flight checks before the training loop refuse any `Ep{n}.pth` collision or `loss.dat` truncation (per-epoch guard re-checks truncation before saving anything).
- `pre_evaluate.py`: config constants `ROLLOUT_DAYS`/`ENSEMBLE_SIZE`/`SAMPLER_S_CHURN`/`EVAL_SEED`/`OUTPUT_TAG`; per-WINDOW seeding (`EVAL_SEED + start_day`) so trajectories are independent of batch size/batching (`BATCH_SIZE` recorded in metadata); outputs are tagged `eval_<split>_h{rd}_ch{churn}_e{es}_s{seed}_ckpt{stem}[_tag].npz` + `figures_<tag>/` and existing outputs are REFUSED, never overwritten; rollout runs under autocast (AMP).
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
