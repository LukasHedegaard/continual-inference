import torch

from continual.closure import Add, Constant, Identity, Lambda, Multiply, One, Zero


def test_add():
    x = torch.ones((1, 1, 2, 2))

    # int
    add_5 = Add(5)
    assert torch.equal(add_5.forward(x), x + 5)
    assert add_5.delay == 0

    # float
    add_pi = Add(3.14)
    assert torch.equal(add_pi.forward_steps(x), x + 3.14)

    # Tensor
    constants = torch.tensor([[[[2.0, 3.0]]]])
    add_constants = Add(constants)
    assert torch.equal(
        add_constants.forward_step(x[:, :, 0]), torch.tensor([[[3.0, 4.0]]])
    )


def test_multiply():
    x = torch.ones((1, 1, 2, 2))

    # int
    mul_5 = Multiply(5)
    assert torch.equal(mul_5.forward(x), x * 5)
    assert mul_5.delay == 0

    # float
    mul_pi = Multiply(3.14)
    assert torch.equal(mul_pi.forward_steps(x), x * 3.14)

    # Tensor
    constants = torch.tensor([[[[2.0, 3.0]]]])
    mul_constants = Multiply(constants)
    assert torch.equal(
        mul_constants.forward_step(x[:, :, 0]), torch.tensor([[[2.0, 3.0]]])
    )


def global_always42(x):
    return torch.ones_like(x) * 42


def test_lambda():
    x = torch.ones((1, 1, 2, 2))
    target = torch.ones_like(x) * 42

    def local_always42(x):
        return torch.ones_like(x) * 42

    # Test if layer works in different scopes
    # Global
    assert torch.equal(target, Lambda(global_always42)(x))

    # Local
    assert torch.equal(target, Lambda(local_always42)(x))

    # Anonymous
    mod = Lambda.build_from(lambda x: torch.ones_like(x) * 42)
    assert torch.equal(target, mod.forward_steps(x))

    # Functor
    functor = torch.nn.Sigmoid()
    assert torch.equal(functor(x), Lambda(functor)(x))

    mod = Lambda.build_from(lambda x: torch.ones_like(x) * 42, takes_time=True)
    assert torch.equal(target, mod(x))

    # __repr__
    assert (
        mod.__repr__() == "Lambda(lambda x: torch.ones_like(x) * 42, takes_time=True)"
    )

    modules = []
    modules.append(("view", Lambda(lambda x: x.view(x.shape[0], -1))))
    assert modules[0][1].__repr__() == "Lambda(lambda x: x.view(x.shape[0], -1))"


def test_lambda_call_specific_fns():
    x = torch.ones((1, 1, 2, 2))

    mod = Lambda(
        fn=lambda x: x + 1,
        forward_only_fn=lambda x: x + 2,
        forward_step_only_fn=lambda x: x + 3,
        forward_steps_only_fn=lambda x: x + 4,
    )

    assert torch.equal(x + 2, mod.forward(x))
    assert torch.equal(x.squeeze(2) + 3, mod.forward_step(x))
    assert torch.equal(x + 4, mod.forward_steps(x))

    # __repr__
    assert (
        mod.__repr__()
        == "Lambda(lambda x: x + 1, lambda x: x + 2, lambda x: x + 3, lambda x: x + 4)"
    )


def test_identity():
    x = torch.ones((1, 1, 2, 2))
    assert torch.equal(x, Identity()(x))


def test_constant():
    x = torch.randn((1, 1, 2, 2))
    const = 42
    assert torch.equal(const * torch.ones_like(x), Constant(const)(x))


def test_zero():
    x = torch.randn((1, 1, 2, 2))
    assert torch.equal(torch.zeros_like(x), Zero()(x))


def test_one():
    x = torch.randn((1, 1, 2, 2))
    assert torch.equal(torch.ones_like(x), One()(x))
