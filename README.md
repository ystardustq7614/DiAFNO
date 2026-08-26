# DiAFNO
# Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence

Code accompanying the manuscript titled ["Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence"](https://arxiv.org/abs/2512.12628), authored by Yuchi Jiang, Yunpeng Wang, Huiyu Yang and Jianchun Wang.

## Abstract

Accurately autoregressive prediction of three-dimensional (3D) turbulence has been one of the most challenging problems for machine learning approaches. Diffusion models have demonstrated high accuracy in predicting two-dimensional (2D) turbulence, but their applications in 3D turbulence are relatively limited. To achieve reliable autoregressive predictions of 3D turbulence, we propose the DiAFNO model which integrates the implicit adaptive Fourier neural operator (IAFNO) with diffusion model. IAFNO can effectively capture the global frequency and structural features, which is crucial for global consistent reconstructions of the denoising process in diffusion models. Furthermore, based on conditional generation from diffusion models, we design an autoregressive framework in DiAFNO to achieve long-term stable predictions of 3D turbulence. The proposed DiAFNO model is systematically trained and tested separately with fixed hyperparameters in several types of 3D turbulence, including forced homogeneous isotropic turbulence (HIT) at Taylor Reynolds number $Re_{\lambda}\approx100$, decaying HIT at initial Taylor Reynolds number at $Re_{\lambda}\approx100$ and turbulent channel flow at friction Reynolds numbers $Re_{\tau}\approx395$ and $Re_{\tau}\approx590$ with case-specific training at each Reynolds number. The results in the \textit{a posteriori} tests demonstrate that DiAFNO exhibits a significantly higher prediction accuracy in most of the analyzed statistics (such as the velocity spectra, the root-mean-square (RMS) values of both velocity and vorticity, and Reynolds stresses), as compared to the elucidated diffusion model (EDM) and the traditional large-eddy simulation (LES) using dynamic Smagorinsky model (DSM). Although DiAFNO is not optimal in certain statistics, its overall performance is substantially better than all baseline models (EDM and DSM). Meanwhile, we record the time taken by machine learning models and DSM during the inference stage. Ignoring training costs, the well-trained DiAFNO achieves higher inference efficiency than EDM and LES with DSM.

## Dataset

The datasets can be downloaded at [IAFNO_fDNS_kaggle](https://www.kaggle.com/datasets/yuchirichardjiang/coarsened-fdns-data-iafno).

## Citation

arXiv version:
```
@misc{jiang2026integratingfourierneuraloperator,
      title={Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence}, 
      author={Yuchi Jiang and Yunpeng Wang and Huiyu Yang and Jianchun Wang},
      year={2026},
      eprint={2512.12628},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
      url={https://arxiv.org/abs/2512.12628}, 
}
```

This manuscript has been accepted by Acta Mechanica Sinica with citing inform: Acta Mech. Sin. 43, 360674 (2027), DOI: 10.1007/s10409-026-60674-x. When the final version of the article provided by the journal becomes retrievable, please cite it using the information of the final version. Many thanks.

---

## PRE_ocean_data forecast task (this fork, branch `adapt-weather-ocean`)

Task: given 7 consecutive days of the raw 3D `u/v` ocean-current fields, predict
the next day, then autoregressively roll out 15 days. See
`docs/PRE_runbook.md` for the full step-by-step runbook.

### Grids, masks, interpolation (no rotation)

| grid | u | v | mask |
|---|---|---|---|
| native (ROMS C-grid) | `(T, 30, 400, 440)` | `(T, 30, 399, 441)` | `mask_u` (400,440), `mask_v` (399,441) |
| rho (model grid) | `u_rho.npy` `(10591, 30, 400, 441)` | `v_rho.npy` same | `mask_u_rho` / `mask_v_rho` (400,441) |

- Colocation to the rho grid keeps the raw grid-xi/eta component semantics —
  **no rotation** to east/north. `u_rho[r,c]` = NaN-aware mean of the adjacent
  u faces `u[r,c-1]`, `u[r,c]` (one-sided at boundaries); `v_rho` analogous
  along eta. Land stays NaN.
- Bivariate validity masks `mask_u_rho` / `mask_v_rho` are derived from
  `mask_u` / `mask_v` with the same stencil (a rho point is valid iff at least
  one adjacent face is valid). `mask_uv.npy` = intersection (compat only).
  Training statistics, the masked diffusion loss and validation use the
  **bivariate** masks. The provided masks are authoritative in preprocessing:
  a NaN at a `mask==1` cell (dynamic missing data) fails hard on any
  day/layer, while values found at `mask==0` cells (this dataset has 45
  static land-boundary u-faces) are discarded to NaN and counted.
- Formal evaluation maps rho predictions back to the native grids with a fixed
  rule: rho u -> native u by averaging adjacent rho points along xi -> (400,
  440); rho v -> native v by averaging adjacent rho points along eta -> (399,
  441).

### Normalization & clipping

- Per-variable min-max to `[0,1]` over **train** ocean points of that variable
  (u uses `mask_u_rho`, v uses `mask_v_rho`); land filled with 0 after
  normalization; loss/metrics always masked.
- Percentile clipping is **disabled by default** (`clip_pct = None`) and must
  be configured explicitly; the stats cache records the clipping policy, the
  depth preset, the split boundaries and a mask hash (stale caches, including
  changed splits or a missing `splits` field, are recomputed). `sigma_data` =
  true pooled std of the CLIPPED, min-max normalized u+v concatenation
  (includes the u/v mean-difference term) — values are clipped to the
  per-variable range before normalization and pooling, exactly like the
  dataset normalization.
- Formal metrics use the **unclipped raw native truth** (`NativeUVReader` on
  the original `u.npy`/`v.npy`); normalized targets are never denormalized to
  stand in for raw truth.

### Time & splits

- Contiguous daily timestamps verified from the authoritative `ocean_time`
  metadata (10591 strictly increasing, exactly 24 h apart; checked at
  `datetime64[s]` precision so 23/25-hour gaps fail). Cached as
  `aligned/ocean_time_seconds.npy` (precise `datetime64[s]`) and
  `aligned/ocean_time.npy` (date view `datetime64[D]`, compat).
- Contiguous splits: train `[0, 8401)`, val `[8401, 9496)`, test
  `[9496, 10591)`; sliding windows never cross a split boundary. The stats
  cache records the split boundaries and is recomputed when they change.

### Run commands (from repo root, env `diafno`)

```bash
python scripts/preprocess_align_uv.py     # one-time colocation + masks + time
python pre_trainer.py                     # PRESET='surface_smoke' | 'full3d'
python pre_evaluate.py                    # 15-step rollout + persistence + figures
python smoke_test.py && python pre_smoke_test.py   # minimal regression tests
```

- Presets live in `pre_config.py`; training/eval presets must match.
- Outputs: `<checkpoint_dir>/PRE/<run_tag>/{Ep{n}.pth, best.pth, loss.dat}`,
  `eval_<split>.npz` (native-grid `rmse_model/mae_model/rmse_persistence/
  mae_persistence/valid_count` shape `(15, 2, Z)` — `Z=1` for `surface_smoke`,
  `Z=30` for `full3d` — plus full reproduction metadata: checkpoint, epoch,
  preset, seed, sampling_steps, stride, window starts, norm stats, grid mapping
  rule), and `figures/d{1,3,5,7,10,15}_s{layer}_{u|v}.png`
  (truth/prediction/error). Only native-grid (formal) metrics are saved; there
  are no rho-grid supplementary arrays.
- Overall RMSE = `sqrt(sum(squared_error)/sum(valid_count))` — never the
  arithmetic mean of per-layer RMSEs. The console summary pools the same way
  (`pre_metrics.pooled_rmse`), per lead day and per variable.
- `pre_metrics.py` holds the shared metric implementations (`rho_to_native`,
  `masked_error_sums`, `pooled_rmse`, `masked_rel_l2`) used by training,
  evaluation and the smoke test — formulas are never re-implemented in tests.
- `NativeUVReader.get()` returns a unified layout with the sigma axis last:
  u `(days, H, W-1, Z)` / v `(days, H-1, W, Z)` for both presets (surface
  `Z=1`, full3d `Z=30`); evaluation never transposes it.
- Validation diffusion sampling runs inside `torch.random.fork_rng()` with the
  fixed `VAL_SEED`, so validation RNG is isolated and training RNG state is
  restored afterwards.
- Reproduction: seed 123 (training) / 1234 (validation sampling), the chosen
  checkpoint and `sampling_steps` are recorded in the eval metadata.
