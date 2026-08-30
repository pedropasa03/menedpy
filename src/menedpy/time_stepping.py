"""Time-discretization schemes for the nonlinear FEM system.

All steppers share one calling convention.  ``previous`` is the solution at
``t_n``; ``previous_load`` and ``load`` are the assembled load vectors at
``t_n`` and ``t_{n+1}``, respectively.  The solver always supplies both load
vectors, which lets a stepper select the time level required by its scheme.
"""

from collections.abc import Callable
from time import perf_counter

import numpy as np

from .assembly import assemble_reaction
from .mesh import Mesh
from .systems import NonlinearSolver, System, newton

StepperInfo = dict[str, float | int | list[float]]
TimeStepper = Callable[..., tuple[np.ndarray, StepperInfo]]


def _nodal_vector(name: str, values: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Convert ``values`` to a finite nodal vector with the mesh's size."""

    vector = np.asarray(values, dtype=float)
    expected_shape = (mesh.n_nodes,)
    if vector.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _prepare_previous(
    mesh: Mesh,
    previous: np.ndarray,
    boundary_values: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one step and impose the prescribed boundary values."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    previous = _nodal_vector("previous", previous, mesh)
    boundary_values = _nodal_vector("boundary_values", boundary_values, mesh)

    previous = previous.copy()
    boundary = mesh.boundary_nodes
    previous[boundary] = boundary_values[boundary]
    return previous, boundary_values, mesh.free_nodes, boundary


def _step_info(
    residuals: list[float] | None = None,
    errors: list[float] | None = None,
    *,
    elapsed: float = 0.0,
) -> StepperInfo:
    """Return the common diagnostics payload used by every stepper."""

    residuals = [] if residuals is None else residuals
    errors = [] if errors is None else errors
    return {
        "residuals": residuals,
        "errors": errors,
        "iterations": len(residuals),
        "nss_time": elapsed,
    }


def _solve_free_system(
    mesh: Mesh,
    boundary_values: np.ndarray,
    initial_guess: np.ndarray,
    residual_and_jacobian: System,
    nonlinear_solver: NonlinearSolver,
) -> tuple[np.ndarray, StepperInfo]:
    """Solve a nonlinear system on free nodes and reconstruct its full vector."""

    solution = np.asarray(boundary_values, dtype=float).copy()
    free = mesh.free_nodes
    if free.size == 0:
        return solution, _step_info()

    start = perf_counter()
    solution[free], residuals, errors = nonlinear_solver(
        residual_and_jacobian, initial_guess
    )
    return solution, _step_info(residuals, errors, elapsed=perf_counter() - start)


def explicit_euler_step(
    mesh: Mesh,
    mass: np.ndarray,
    stiffness: np.ndarray,
    previous: np.ndarray,
    load: np.ndarray,
    boundary_values: np.ndarray,
    dt: float,
    *,
    previous_load: np.ndarray | None = None,
    quadrature_order: int = 4,
    nonlinear_solver: NonlinearSolver = newton,
) -> tuple[np.ndarray, StepperInfo]:
    """Solve one forward-Euler step.

    Forward Euler uses the known right-hand side at ``t_n``.  For direct use,
    omitting ``previous_load`` preserves the convenient convention that
    ``load`` itself is the load at ``t_n``.  ``nonlinear_solver`` is accepted
    solely to keep the shared time-stepper interface; it is not used here.
    """

    del nonlinear_solver
    previous, boundary_values, free, boundary = _prepare_previous(
        mesh, previous, boundary_values, dt
    )
    load_at_previous_time = _nodal_vector(
        "load" if previous_load is None else "previous_load",
        load if previous_load is None else previous_load,
        mesh,
    )

    reaction, _ = assemble_reaction(mesh, previous, quadrature_order=quadrature_order)
    rhs = np.asarray(load_at_previous_time, dtype=float) - stiffness @ previous - reaction

    solution = previous.copy()
    if free.size:
        solution[free] += dt * np.linalg.solve(mass[np.ix_(free, free)], rhs[free])
    solution[boundary] = boundary_values[boundary]
    return solution, _step_info()


def implicit_euler_step(
    mesh: Mesh,
    mass: np.ndarray,
    stiffness: np.ndarray,
    previous: np.ndarray,
    load: np.ndarray,
    boundary_values: np.ndarray,
    dt: float,
    *,
    previous_load: np.ndarray | None = None,
    quadrature_order: int = 4,
    nonlinear_solver: NonlinearSolver = newton,
) -> tuple[np.ndarray, StepperInfo]:
    """Solve one nonlinear backward-Euler step using the load at ``t_{n+1}``.

    ``previous_load`` is accepted but intentionally unused so this function
    has the same interface as the other time steppers.
    """

    del previous_load
    previous, boundary_values, free, boundary = _prepare_previous(
        mesh, previous, boundary_values, dt
    )
    u_boundary = boundary_values[boundary]
    initial_guess = previous[free].copy()

    linear_matrix = mass / dt + stiffness
    linear_ff = linear_matrix[np.ix_(free, free)]
    linear_fb = linear_matrix[np.ix_(free, boundary)]
    rhs = mass @ previous / dt + _nodal_vector("load", load, mesh)
    rhs_free = rhs[free] - linear_fb @ u_boundary

    def residual_and_jacobian(v_free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_full = np.asarray(boundary_values, dtype=float).copy()
        u_full[free] = v_free
        reaction, reaction_jacobian = assemble_reaction(
            mesh, u_full, quadrature_order=quadrature_order
        )
        return (
            linear_ff @ v_free + reaction[free] - rhs_free,
            linear_ff + reaction_jacobian[np.ix_(free, free)],
        )

    return _solve_free_system(
        mesh, boundary_values, initial_guess, residual_and_jacobian, nonlinear_solver
    )


def crank_nicolson_step(
    mesh: Mesh,
    mass: np.ndarray,
    stiffness: np.ndarray,
    previous: np.ndarray,
    load: np.ndarray,
    boundary_values: np.ndarray,
    dt: float,
    *,
    previous_load: np.ndarray | None = None,
    quadrature_order: int = 4,
    nonlinear_solver: NonlinearSolver = newton,
) -> tuple[np.ndarray, StepperInfo]:
    """Solve one nonlinear Crank--Nicolson step.

    ``previous_load`` and ``load`` must be the load vectors at ``t_n`` and
    ``t_{n+1}``, respectively, because this scheme uses their trapezoidal
    average.
    """

    if previous_load is None:
        raise ValueError("Crank--Nicolson requires previous_load at t_n")

    previous, boundary_values, free, boundary = _prepare_previous(
        mesh, previous, boundary_values, dt
    )
    u_boundary = boundary_values[boundary]
    initial_guess = previous[free].copy()

    previous_reaction, _ = assemble_reaction(
        mesh, previous, quadrature_order=quadrature_order
    )
    left_matrix = mass / dt + stiffness / 2.0
    right_matrix = mass / dt - stiffness / 2.0
    left_ff = left_matrix[np.ix_(free, free)]
    left_fb = left_matrix[np.ix_(free, boundary)]
    averaged_load = (
        _nodal_vector("load", load, mesh)
        + _nodal_vector("previous_load", previous_load, mesh)
    ) / 2.0
    rhs = (
        averaged_load
        + right_matrix @ previous
        - previous_reaction / 2.0
    )
    rhs_free = rhs[free] - left_fb @ u_boundary

    def residual_and_jacobian(v_free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_full = np.asarray(boundary_values, dtype=float).copy()
        u_full[free] = v_free
        reaction, reaction_jacobian = assemble_reaction(
            mesh, u_full, quadrature_order=quadrature_order
        )
        return (
            left_ff @ v_free + reaction[free] / 2.0 - rhs_free,
            left_ff + reaction_jacobian[np.ix_(free, free)] / 2.0,
        )

    return _solve_free_system(
        mesh, boundary_values, initial_guess, residual_and_jacobian, nonlinear_solver
    )
