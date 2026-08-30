"""Quadrature rules and shape functions for triangular P1 elements."""

import numpy as np


def triangle_quadrature(order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-triangle quadrature points and weights.

    The reference triangle is ``{(s, t): s >= 0, t >= 0,
    s + t <= 1}``. The weights integrate over that triangle, so they sum to
    ``1/2``.

    Parameters
    ----------
    order:
        Rule selector. ``1`` gives the centroid rule, ``3`` gives the midpoint
        edge rule, and ``4`` gives the four-point Hammer rule used by default.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Quadrature points with shape ``(n_points, 2)`` and weights with shape
        ``(n_points,)``.

    Raises
    ------
    ValueError
        If ``order`` is not one of ``1``, ``3``, or ``4``.
    """

    if order == 1:
        return (
            np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float),
            np.array([0.5], dtype=float),
        )
    if order == 3:
        return (
            np.array(
                [[0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
                dtype=float,
            ),
            np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float),
        )
    if order == 4:
        return (
            np.array(
                [
                    [1.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 5.0, 3.0 / 5.0],
                    [1.0 / 5.0, 1.0 / 5.0],
                    [3.0 / 5.0, 1.0 / 5.0],
                ],
                dtype=float,
            ),
            np.array(
                [-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0],
                dtype=float,
            ),
        )
    raise ValueError(f"unsupported triangle quadrature order: {order}")


def shape_p1(s_t: np.ndarray) -> np.ndarray:
    """Evaluate the three P1 shape functions on the reference triangle.

    Parameters
    ----------
    s_t:
        Reference coordinates ``(s, t)``.

    Returns
    -------
    numpy.ndarray
        Values ``[1 - s - t, s, t]`` of the three linear basis
        functions.
    """

    s, t = s_t
    return np.array([1.0 - s - t, s, t], dtype=float)
