from continual.utils import temporary_parameter


def test_temporary_parameter():
    class MyClass:
        def __init__(self) -> None:
            self.x = 0

    c = MyClass()
    assert c.x == 0

    # Existing param
    with temporary_parameter(c, "x", 42):
        assert c.x == 42

    assert c.x == 0

    # Non-existing param
    with temporary_parameter(c, "y", 42):
        assert c.y == 42

    assert not hasattr(c, "y")
