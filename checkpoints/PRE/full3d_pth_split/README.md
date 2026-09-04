# full3d 大 checkpoint 分卷说明

`full3d_*` 训练产生的 4 个模型权重各约 342 MB，超过 GitHub 单文件 100 MB 硬限制
（Git LFS 免费额度也不足），因此以 `split -b 90M` 切成 <100 MB 分卷入库。
权威原件始终在服务器磁盘 `~/checkpoints/PRE/<run_dir>/` 下。

## 分卷清单与校验

| 原件（磁盘路径） | 大小 (B) | md5 |
|---|---|---|
| `full3d_BS1_EMD128_I2_E4_S32_C7_SD2_RES/best.pth` | 358,803,854 | `768c12c72c1ba3258a76c1b0f1b942f2` |
| `full3d_BS1_EMD128_I2_E4_S32_C7_SD2_RES/Ep1.pth` | 358,803,538 | `61a8ea29e3d1806e49c992ab1acaacfc` |
| `full3d_BS1_EMD128_I2_E4_S4_C7_SD2_RES_SMOKE/best.pth` | 358,803,790 | `a34ededb95059964162235fd72443ee0` |
| `full3d_BS1_EMD128_I2_E4_S4_C7_SD2_RES_SMOKE/Ep1.pth` | 358,803,474 | `f678aa4ac04579cccf149793914a2d56` |

分卷命名规则：`<run_dir>_<file>.part-{00..03}`（`/` 替换为 `_`）。

## 重组方法

在本目录执行（逐文件）：

```bash
OUT=~/checkpoints/PRE   # 或任意目标目录
cat full3d_BS1_EMD128_I2_E4_S32_C7_SD2_RES_best.pth.part-* > "$OUT/full3d_BS1_EMD128_I2_E4_S32_C7_SD2_RES/best.pth"
cat full3d_BS1_EMD128_I2_E4_S32_C7_SD2_RES_Ep1.pth.part-*  > "$OUT/full3d_BS1_EMD128_I2_E4_S32_C7_SD2_RES/Ep1.pth"
cat full3d_BS1_EMD128_I2_E4_S4_C7_SD2_RES_SMOKE_best.pth.part-* > "$OUT/full3d_BS1_EMD128_I2_E4_S4_C7_SD2_RES_SMOKE/best.pth"
cat full3d_BS1_EMD128_I2_E4_S4_C7_SD2_RES_SMOKE_Ep1.pth.part-*  > "$OUT/full3d_BS1_EMD128_I2_E4_S4_C7_SD2_RES_SMOKE/Ep1.pth"
```

重组后用上表 md5 校验（`md5sum <file>`）。

## 约定

- 这些分卷只是大权重的**搬运副本**；训练/评估直接使用磁盘原件，勿在仓库目录内重组。
- `torch.load` 前必须先重组并通过 md5 校验。
- 新增超过 100 MB 的产物沿用本模式：`split -b 90M -d` + 更新本 README。
