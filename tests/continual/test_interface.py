from continual import TensorPlaceholder


def test_TensorPlaceholder():
    # No shape
    tp = TensorPlaceholder()
    assert tp.size() == tuple()
    assert len(tp) == 0

    # Shape
    tp = TensorPlaceholder(shape=(1, 2, 3))
    assert tp.size() == (1, 2, 3)
    assert len(tp) == 0
