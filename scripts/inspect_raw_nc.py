#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块职责：检查单个原始 COAWST/ROMS NetCDF 平均文件：维度、坐标、
变量、dtype（含 encoding），以及静态网格文件。

不负责：纯只读检查 —— 只 xr.open_dataset 后逐项打印，不写盘、不改数据；
不依赖任何正式 PRE 模块。

关键约束：
- ndim>=3 的变量标记为 FIELD，其余记为标量/配置变量；
- 对 4 维动态场与 2 维网格变量会物化整个变量（da.values）来计算
  min/max/NaN 比例 —— 单文件全量读入内存，注意大文件的内存占用。

用法：
    python scripts/inspect_raw_nc.py /data/PRE_ocean_data/raw/dyn/coawst_avg_00001.nc
    python scripts/inspect_raw_nc.py /data/PRE_ocean_data/raw/PRE-90921-V2.nc
"""

import sys
import numpy as np
import xarray as xr


def main():
    """打印目标 NetCDF 的维度/坐标/属性与逐变量概要；缺省参数时打印
    用法并以退出码 1 退出。"""
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
