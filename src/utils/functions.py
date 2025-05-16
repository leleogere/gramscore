import functools


def compose2(f, g):
    """
    Composes two functions f and g such that compose2(f, g)(x) = f(g(x)).
    """
    return lambda x: f(g(x))


def compose(*funcs):
    """
    Composes multiple functions such that compose(f, g, h)(x) = f(g(h(x))).
    """
    if not funcs:
        raise ValueError("At least one function must be provided")
    return functools.reduce(compose2, funcs)


def rcompose(*funcs):
    """
    Composes multiple functions in reverse order such that rcompose(f, g, h)(x) = h(g(f(x))).
    """
    if not funcs:
        raise ValueError("At least one function must be provided")
    return functools.reduce(compose2, reversed(funcs))
