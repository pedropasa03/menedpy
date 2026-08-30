"""Finite-element assembly routines for the dense minimal solver."""

from typing import Callable

import numpy as np

from .elements import area, local_mass_matrix, map_to_physical, p1_gradients
from .mesh import Mesh
from .quadrature import shape_p1, triangle_quadrature


def assemble_mass_stiffness(
    mesh: Mesh,
    diffusion: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    quadrature_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the global P1 mass and diffusion stiffness matrices.

    Parameters
    ----------
    mesh:
        Triangular mesh on which the P1 finite-element space is defined.
    diffusion:
        Function ``alpha(x, y)`` evaluated at NumPy arrays of coordinates and
        returning the scalar diffusion coefficient.
    quadrature_order:
        Reference-triangle quadrature rule passed to
        :func:`fem2d.quadrature.triangle_quadrature`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Dense mass matrix ``M`` and stiffness matrix ``K`` with shape
        ``(mesh.n_nodes, mesh.n_nodes)``.
    """

    quad_points, quad_weights = triangle_quadrature(quadrature_order)
    rows: list[int] = []
    cols: list[int] = []
    mass_data: list[float] = []
    stiffness_data: list[float] = []

    for tri in mesh.triangles:
        vertices = mesh.points[tri]
        tri_area = area(vertices)
        jacobian_abs = 2.0 * tri_area
        grads = p1_gradients(vertices)
        local_mass = local_mass_matrix(tri_area)
        local_stiffness = np.zeros((3, 3), dtype=float)
        grad_dot = grads @ grads.T

        for qp, weight in zip(quad_points, quad_weights):
            physical = map_to_physical(vertices, qp)
            alpha_value = _scalar_value(diffusion, physical[0], physical[1])
            local_stiffness += weight * jacobian_abs * alpha_value * grad_dot

        for i_local, i_global in enumerate(tri):
            for j_local, j_global in enumerate(tri):
                rows.append(int(i_global))
                cols.append(int(j_global))
                mass_data.append(float(local_mass[i_local, j_local]))
                stiffness_data.append(float(local_stiffness[i_local, j_local]))

    mass = _assemble_matrix(mesh.n_nodes, rows, cols, mass_data)
    stiffness = _assemble_matrix(mesh.n_nodes, rows, cols, stiffness_data)
    return mass, stiffness


def assemble_load_vector(
    mesh: Mesh,
    source: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    t: float,
    *,
    quadrature_order: int = 4,
) -> np.ndarray:
    """Assemble the load vector at one time value.

    Parameters
    ----------
    mesh:
        Triangular mesh on which the P1 finite-element space is defined.
    source:
        Function ``f(x, y, t)`` evaluated at NumPy arrays of coordinates and a
        scalar time.
    t:
        Time at which the source term is evaluated.
    quadrature_order:
        Reference-triangle quadrature rule used for element integration.

    Returns
    -------
    numpy.ndarray
        Dense vector with one entry per mesh node.
    """

    quad_points, quad_weights = triangle_quadrature(quadrature_order)
    load = np.zeros(mesh.n_nodes, dtype=float)

    for tri in mesh.triangles:
        vertices = mesh.points[tri]
        tri_area = area(vertices)
        jacobian_abs = 2.0 * tri_area
        local = np.zeros(3, dtype=float)
        for qp, weight in zip(quad_points, quad_weights):
            phi = shape_p1(qp)
            physical = map_to_physical(vertices, qp)
            value = _time_scalar_value(source, physical[0], physical[1], t)
            local += weight * jacobian_abs * value * phi
        np.add.at(load, tri, local)

    return load


def assemble_reaction(
    mesh: Mesh,
    u: np.ndarray,
    *,
    quadrature_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the nonlinear reaction vector and Jacobian.

    The reaction term is ``integral_T u_h^2 v_h dx``. Its Jacobian with respect
    to nodal coefficients has element entries
    ``integral_T 2 u_h phi_i phi_j dx``.

    Parameters
    ----------
    mesh:
        Triangular mesh on which the P1 finite-element space is defined.
    u:
        Nodal coefficient vector for the current finite-element solution.
    quadrature_order:
        Reference-triangle quadrature rule used for element integration.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Dense reaction vector and dense reaction Jacobian matrix.
    """

    quad_points, quad_weights = triangle_quadrature(quadrature_order)
    vector = np.zeros(mesh.n_nodes, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for tri in mesh.triangles:
        vertices = mesh.points[tri]
        tri_area = area(vertices)
        jacobian_abs = 2.0 * tri_area
        u_local = u[tri]
        local_vector = np.zeros(3, dtype=float)
        local_jacobian = np.zeros((3, 3), dtype=float)

        for qp, weight in zip(quad_points, quad_weights):
            phi = shape_p1(qp)
            uh = float(phi @ u_local)
            scaled_weight = weight * jacobian_abs
            local_vector += scaled_weight * uh * uh * phi
            local_jacobian += scaled_weight * 2.0 * uh * np.outer(phi, phi)

        np.add.at(vector, tri, local_vector)
        for i_local, i_global in enumerate(tri):
            for j_local, j_global in enumerate(tri):
                rows.append(int(i_global))
                cols.append(int(j_global))
                data.append(float(local_jacobian[i_local, j_local]))

    jacobian = _assemble_matrix(mesh.n_nodes, rows, cols, data)
    return vector, jacobian


def _assemble_matrix(
    n: int,
    rows: list[int],
    cols: list[int],
    data: list[float],
) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    np.add.at(matrix, (rows, cols), data)
    return matrix


def _scalar_value(
    field: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: float,
    y: float,
) -> float:
    """Evaluate a vectorized spatial field at one point and return a scalar."""

    value = field(np.array([x], dtype=float), np.array([y], dtype=float))
    return float(value.reshape(-1)[0])


def _time_scalar_value(
    field: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    x: float,
    y: float,
    t: float,
) -> float:
    """Evaluate a vectorized time-dependent field at one point and time."""

    value = field(np.array([x], dtype=float), np.array([y], dtype=float), t)
    return float(value.reshape(-1)[0])
