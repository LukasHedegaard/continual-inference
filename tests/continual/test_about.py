import continual.__about__ as about


def test_info():
    assert type(about.__version__) == str
    assert type(about.__author__) == str
    assert type(about.__author_email__) == str
    assert type(about.__license__) == str
    assert type(about.__copyright__) == str
    assert type(about.__homepage__) == str
    assert type(about.__docs__) == str
