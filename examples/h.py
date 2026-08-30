import numpy as np
from menedpy.newton import newton_system


def f(X):
    x, y = X
    return np.array([x**2 + y**2 - 1, x**2 - y**2 - 0.5])

def df(X):
    x, y = X
    return np.array([[2*x, 2*y], [2*x, -2*y]])

x0 = np.array([0.5, 0.5])
sol = newton_system(f, df, x0)
print("Solution:", sol)
