import torch

import continual as co


def test_reshape():
    x = torch.randn((1, 2, 3, 4, 5))

    r1 = co.Reshape((2, 20))
    r2 = co.Reshape(2, 20)

    assert r1.shape == r2.shape

    # Temporal axis is always in position 2
    r1.forward(x).shape == (2, 20, 3)
    r1.forward_steps(x).shape == (2, 20, 3)
    r1.forward_step(x[:, :, 0]).shape == (2, 20)
