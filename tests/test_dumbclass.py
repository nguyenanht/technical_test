def hello(name):
    return 'Hello ' + name


def test_useless_assert():
    assert hello('John') == 'Hello John'
