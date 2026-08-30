"""Local finite-element geometry utilities."""

import numpy as np


def signed_area(vertices: np.ndarray) -> float:
    """Compute the signed area of a triangle.

    Parameters
    ----------
    vertices:
        Array with shape ``(3, 2)`` containing the triangle vertices in order.

    Returns
    -------
    float
        Positive area for counterclockwise vertices, negative area for
        clockwise vertices, and zero for collinear vertices.
    """

    return 0.5 * (
        (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1])
        - (vertices[2, 0] - vertices[0, 0]) * (vertices[1, 1] - vertices[0, 1])
    )


def area(vertices: np.ndarray) -> float:
    """Compute the geometric area of a triangle.

    Parameters
    ----------
    vertices:
        Array with shape ``(3, 2)`` containing the triangle vertices.

    Returns
    -------
    float
        Nonnegative triangle area.
    """

    return abs(signed_area(vertices))


def p1_gradients(vertices: np.ndarray) -> np.ndarray:
    """Compute gradients of the three linear basis functions on a triangle.

    Parameters
    ----------
    vertices:
        Array with shape ``(3, 2)`` containing a nondegenerate triangle.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(3, 2)``. Row ``i`` stores the constant physical
        gradient of the ``i``-th P1 basis function on the triangle.

    Raises
    ------
    ValueError
        If the triangle area is numerically zero.
    """

    x = vertices[:, 0]
    y = vertices[:, 1]
    two_area = (
        x[0] * (y[1] - y[2])
        + x[1] * (y[2] - y[0])
        + x[2] * (y[0] - y[1])
    )
    if abs(two_area) < 1e-15:
        raise ValueError("degenerate triangle with near-zero area")
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=float)
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=float)
    return np.column_stack((b, c)) / two_area


def map_to_physical(vertices: np.ndarray, s_t: np.ndarray) -> np.ndarray:
    """Map a reference-triangle point to a physical triangle.

    Parameters
    ----------
    vertices:
        Array with shape ``(3, 2)`` containing the physical triangle vertices.
    s_t:
        Reference coordinates ``(s, t)`` in the triangle with vertices
        ``(0, 0)``, ``(1, 0)``, and ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Physical coordinate obtained by the affine P1 map.
    """

    s, t = s_t
    phi = np.array([1.0 - s - t, s, t], dtype=float)
    return phi @ vertices


def local_mass_matrix(triangle_area: float) -> np.ndarray:
    """Return the local P1 mass matrix for one triangle.

    Parameters
    ----------
    triangle_area:
        Physical area of the triangle.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(3, 3)`` matrix with entries
        ``integral_T phi_i phi_j dx`` for linear basis functions.
    """

    return (triangle_area / 12.0) * np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=float,
    )
