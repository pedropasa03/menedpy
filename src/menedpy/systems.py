"""Bare Newton solver for nonlinear systems of equations."""

from typing import Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

System = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
NonlinearSolver = Callable[
    [System, np.ndarray], tuple[np.ndarray, list[float], list[float]]
]

norm2 = lambda x: np.dot(x, x)


def newton(
    F: System,
    x0: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Solve F(x) = 0 with Newton's method.

    Returns
    -------
    x
        Approximate root.
    residual
        Final residual norm, ``||F(x)||``.
    step_error
        Norm of the last Newton update,
        ``||x_{n+1} - x_n||``.
    """
    
    x = x0.copy()

    Fx, Jx = F(x)

    residuals: list[float] = []
    errors: list[float] = []
    for _ in range(max_iter):

        delta = np.linalg.solve(Jx, Fx)
        
        err = np.linalg.norm(delta)
        errors.append(err)

        x -= delta

        Fx, Jx = F(x)

        res = np.linalg.norm(Fx)
        residuals.append(res)
        
        if res < tol:
            return x, residuals, errors

    raise RuntimeError(
        "Newton solver failed after "
        f"{max_iter} iterations; residual={float(residuals[-1]):.3e}"
    )


def KLAM(
    F: System,
    x0: np.ndarray,
    *,
    lamb: float = -5.0,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Solve the nonlinear system F(x) = 0 using the KLAM method.

    Parameters
    ----------
    F
        Callable returning ``(F(x), J(x))``, where ``J(x)`` is the
        Jacobian matrix of ``F`` at ``x``.
    x0
        Initial guess.
    tol
        Convergence tolerance on the residual norm.
    max_iter
        Maximum number of iterations.
    l
        KLAM method parameter.

    Returns
    -------
    x
        Approximate solution.
    residual
        Final residual norm, ``||F(x)||``.
    step_error
        Norm of the last update, ``||x_{n+1} - x_n||``.

    Raises
    ------
    RuntimeError
        If the method fails to converge within ``max_iter`` iterations.
    """

    x = x0.copy()
    
    Fx, Jx = F(x)

    residuals: list[float] = []
    errors: list[float] = []
    for _ in range(max_iter):

        lu_piv = lu_factor(Jx)
        Jx_inv_Fx = lu_solve(lu_piv, Fx)
        
        y = x - Jx_inv_Fx

        Fy, _ = F(y)

        v = norm2(Fy) / norm2(Fx)
        K = 1.0 / (1.0 + lamb * v)

        delta = lu_solve(lu_piv, Fy + 2.0 * v * Fx)

        x_new = y - K * delta

        err = np.linalg.norm(x_new - x)
        errors.append(err)
        x = x_new

        Fx, Jx = F(x)
        res = np.linalg.norm(Fx)
        residuals.append(res)

        if res < tol:
            return x, residuals, errors

    raise RuntimeError(
        "KLAM solver failed after "
        f"{max_iter} iterations; residual={float(residuals[-1]):.3e}"
    )


def super_ostrowski(
    F: System,
    x0: np.ndarray,
    *,
    m: int = 4,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Solve F(x) = 0 using the Super-Ostrowski multi-step method.

    Parameters
    ----------
    F
        Callable that returns the residual and Jacobian: F(x) -> (Fx, Jx)
    x0
        Initial guess.
    m
        Number of steps for the Super-Ostrowski scheme (m >= 2).
    tol
        Tolerance for the residual norm.
    max_iter
        Maximum number of iterations.

    Returns
    -------
    x
        Approximate root.
    residual
        Final residual norm, ``||F(x)||``.
    step_error
        Norm of the last Super-Ostrowski update,
        ``||x_{n+1} - x_n||``.
    """
    if m < 2:
        raise ValueError(f"The Super-Ostrowski scheme requires at least m=2 steps. Got m={m}")

    x = x0.copy()
    
    Fx, Jx = F(x)

    residuals: list[float] = []
    errors: list[float] = []

    for _ in range(max_iter):

        lu_piv = lu_factor(Jx)
        delta_1 = lu_solve(lu_piv, Fx)

        S = x - delta_1

        FS, _ = F(S)

        num = 2.0 * np.dot(Fx, FS)
        den = np.dot(Fx, Fx)
        alpha_k = 1.0 - (num / den)

        for i in range(2, m + 1):
            delta_i = lu_solve(lu_piv, FS)
        
            S -= delta_i / alpha_k

            if i < m:
                FS, _ = F(S)

        err = np.linalg.norm(x - S)
        errors.append(err)
        x = S

        Fx, Jx = F(x)
        res = np.linalg.norm(Fx)
        residuals.append(res)

        if res < tol:
            return x, residuals, errors

    raise RuntimeError(
        "Super-Ostrowski solver failed after "
        f"{max_iter} iterations; residual={float(residuals[-1]):.3e}"
    )