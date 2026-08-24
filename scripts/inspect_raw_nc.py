#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect a single raw COAWST/ROMS NetCDF average file:
dimensions, coordinates, variables, dtypes, and the static grid file.

Usage:
    python scripts/inspect_raw_nc.py /data/PRE_ocean_data/raw/dyn/coawst_avg_00001.nc
    python scripts/inspect_raw_nc.py /data/PRE_ocean_data/raw/PRE-90921-V2.nc
"""

import sys
import numpy as np
import xarray as xr


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    fp = sys.argv[1]
    ds = xr.open_dataset(fp)
    print(f"== {fp} ==")
    print(f"dims: {dict(ds.sizes)}")
    print(f"coords: {list(ds.coords)}")
    print(f"global attrs: {dict(list(ds.attrs.items())[:12])}")
    print(f"\nn_vars: {len(ds.data_vars)}")
    for name, da in ds.data_vars.items():
        enc = da.encoding.get("dtype", da.dtype)
        shape = tuple(da.shape)
        ndim = da.ndim
        tag = "FIELD" if ndim >= 3 else "scalar"
        line = f"  {name:<22} {str(shape):<28} {str(enc):<12} {tag}"
        if ndim == 1 and name in ("s_rho", "s_w", "Cs_r", "Cs_w"):
            line += f"  {da.values[:5]}"
        if ndim == 2 and name in ("mask_rho", "mask_u", "mask_v", "h"):
            line += f"  min={float(np.nanmin(da.values)):.4g} max={float(np.nanmax(da.values)):.4g}"
        if ndim == 4:
            v = da.values
            line += f"  min={float(np.nanmin(v)):.4g} max={float(np.nanmax(v)):.4g}"
            line += f"  NaN={float(np.isnan(v).mean())*100:.2f}%"
        print(line)
    ds.close()


if __name__ == "__main__":
    main()
