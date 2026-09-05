"""最小 CPU smoke 测试：设备处理与 checkpoint 恢复。

模块职责：验证三件事在纯 CPU 下成立——IAFNODiff 按输入张量的设备运行（构造函数
不再强制 .cuda()）；GaussianNormalizer 的 encode/decode 互逆；裸 state_dict
checkpoint 经 load_checkpoint（weights_only=True 默认）恢复到指定设备。

不负责：PRE 协议回归（由 pre_smoke_test.py 承担）；真实数据与 GPU 路径。

关键约束：模型网格取最小可整除尺寸，几秒内跑完；load_checkpoint 的裸 state_dict
回退分支即 legacy 信任加载路径，weights_only=True 只还原张量字典。

依赖关系：IAFNO.IAFNODiff、utilities3.GaussianNormalizer/load_checkpoint。
"""

from pathlib import Path
import tempfile

import torch

from IAFNO import IAFNODiff
from utilities3 import GaussianNormalizer, load_checkpoint


def main():
    # 最小 fixture：dim=(2,2,2) 为补零后网格，dim_f=(2,1,2) 为真实网格（y=1）；
    # embed_dim=2、单层隐式/显式块，只为跑通前向
    model = IAFNODiff(
        dim=(2, 2, 2), patch_size=(1, 1, 1), embed_dim=2, num_blocks=1,
        in_chans=2, out_chans=2, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(2, 1, 2), self_condition=True,
    ).cpu()
    # 输入 (B, C, X, Y, Z) = (1, 2, 2, 1, 2)；时间步 t 用零张量占位；
    # 条件传 None：self_condition=True 时 IAFNODiff 内部以零张量回退
    output = model(torch.zeros(1, 2, 2, 1, 2), torch.zeros(1), None)
    # 失败=设备处理回归：输出必须裁掉 y 填充后与输入同形，且留在 CPU
    assert output.shape == (1, 2, 2, 1, 2)
    assert output.device.type == 'cpu'

    # 全标量归一化的 encode/decode 互逆性
    normalizer = GaussianNormalizer(torch.tensor([1.0, 2.0]))
    assert torch.allclose(normalizer.decode(normalizer.encode(torch.tensor([1.5]))), torch.tensor([1.5]))

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'model.pth'
        # 落盘裸 state_dict（无 model_state_dict 包装），模拟 legacy checkpoint 形态
        torch.save(model.state_dict(), checkpoint)
        restored = IAFNODiff(
            dim=(2, 2, 2), patch_size=(1, 1, 1), embed_dim=2, num_blocks=1,
            in_chans=2, out_chans=2, ex_layer=1, nlayer=1,
            hidden_size_factor=1, dim_f=(2, 1, 2), self_condition=True,
        )
        # weights_only=True 只还原张量字典；裸 dict 经 load_checkpoint 的
        # 'model_state_dict' 缺省回退分支读取（legacy 信任加载路径）。
        # 失败=checkpoint 恢复或 map_location 语义回归
        load_checkpoint(checkpoint, restored, map_location='cpu')
        assert next(restored.parameters()).device.type == 'cpu'

    print('CPU smoke test passed')


if __name__ == '__main__':
    main()
