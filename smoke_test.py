"""Minimal CPU smoke test for device handling and checkpoint restoration."""

from pathlib import Path
import tempfile

import torch

from IAFNO import IAFNODiff
from utilities3 import GaussianNormalizer, load_checkpoint


def main():
    model = IAFNODiff(
        dim=(2, 2, 2), patch_size=(1, 1, 1), embed_dim=2, num_blocks=1,
        in_chans=2, out_chans=2, ex_layer=1, nlayer=1,
        hidden_size_factor=1, dim_f=(2, 1, 2), self_condition=True,
    ).cpu()
    output = model(torch.zeros(1, 2, 2, 1, 2), torch.zeros(1), None)
    assert output.shape == (1, 2, 2, 1, 2)
    assert output.device.type == 'cpu'

    normalizer = GaussianNormalizer(torch.tensor([1.0, 2.0]))
    assert torch.allclose(normalizer.decode(normalizer.encode(torch.tensor([1.5]))), torch.tensor([1.5]))

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'model.pth'
        torch.save(model.state_dict(), checkpoint)
        restored = IAFNODiff(
            dim=(2, 2, 2), patch_size=(1, 1, 1), embed_dim=2, num_blocks=1,
            in_chans=2, out_chans=2, ex_layer=1, nlayer=1,
            hidden_size_factor=1, dim_f=(2, 1, 2), self_condition=True,
        )
        load_checkpoint(checkpoint, restored, map_location='cpu')
        assert next(restored.parameters()).device.type == 'cpu'

    print('CPU smoke test passed')


if __name__ == '__main__':
    main()
