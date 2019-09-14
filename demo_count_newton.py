from scipy.optimize import newton


def f(arg):
    return -2.0 + arg / 2.0


if __name__ == '__main__':
    x0_ = 17.0
    g = newton(f, x0=x0_)
    print(g)
    print(f(g))
