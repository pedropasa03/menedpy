"""Time integration driver for the nonlinear FEM system."""

import numpy as np

from .assembly import assemble_load_vector, assemble_mass_stiffness
from .mesh import Mesh
from .problem import Problem
from .systems import NonlinearSolver, newton
from .time_stepping import StepperInfo, TimeStepper, implicit_euler_step


def boundary_values(problem: Problem, mesh: Mesh) -> np.ndarray:
    """Evaluate Dirichlet boundary data at the boundary nodes of ``mesh``."""

    values = np.zeros(mesh.n_nodes, dtype=float)
    boundary = mesh.boundary_nodes
    values[boundary] = np.asarray(
        problem.boundary_conditions(mesh.points[boundary, 0], mesh.points[boundary, 1]),
        dtype=float,
    )
    return values


def initial_condition(problem: Problem, mesh: Mesh) -> np.ndarray:
    """Evaluate the initial condition and impose the Dirichlet boundary data."""

    values = np.asarray(
        problem.initial_condition(mesh.points[:, 0], mesh.points[:, 1]), dtype=float
    ).copy()
    if values.shape != (mesh.n_nodes,):
        raise ValueError(
            "initial_condition must return one value per mesh node; "
            f"got shape {values.shape} for {mesh.n_nodes} nodes"
        )

    boundary = mesh.boundary_nodes
    values[boundary] = boundary_values(problem, mesh)[boundary]
    return values


def solve(
    problem: Problem,
    mesh: Mesh,
    *,
    n_steps: int = 100,
    store_every: int = 1,
    quadrature_order: int = 4,
    time_stepper: TimeStepper = implicit_euler_step,
    nonlinear_solver: NonlinearSolver = newton,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[StepperInfo]]:
    """Solve the nonlinear diffusion--reaction problem in time.

    Every time stepper receives both endpoint loads using the shared contract:
    ``load`` is the vector at ``t_{n+1}`` and ``previous_load`` is the vector
    at ``t_n``.  Backward Euler uses only ``load``; Crank--Nicolson uses both;
    and forward Euler uses ``previous_load``.  Consequently any supplied
    stepper can be selected directly through ``time_stepper``.

    Parameters
    ----------
    n_steps:
        The number of time steps performed by the solver.
    store_every:
        Store every ``store_every``-th solution and always store the final
        solution.
    return_info:
        If true, also return one diagnostics dictionary per time step.  The
        default two-value return keeps the public API used by the examples.
    """

    mass, stiffness = assemble_mass_stiffness(
        mesh, problem.diffusion_coefficient, quadrature_order=quadrature_order
    )
    boundary = boundary_values(problem, mesh)
    u = initial_condition(problem, mesh)

    times = [0.0]
    snapshots = [u.copy()]
    step_info: list[StepperInfo] = []
    previous_load = assemble_load_vector(
        mesh, problem.source, 0.0, quadrature_order=quadrature_order
    )

    time_grid = np.linspace(0.0, problem.final_time, n_steps + 1)
    for step in range(n_steps):
        t = time_grid[step]
        next_time = time_grid[step + 1]
        dt_step = next_time - t


        load = assemble_load_vector(
            mesh, problem.source, next_time, quadrature_order=quadrature_order
        )

        u, info = time_stepper(
            mesh,
            mass,
            stiffness,
            u,
            load,
            boundary,
            dt_step,
            previous_load=previous_load,
            quadrature_order=quadrature_order,
            nonlinear_solver=nonlinear_solver,
        )
        if not np.all(np.isfinite(u)):
            raise FloatingPointError(
                "time step produced non-finite solution values; "
                "reduce dt or use an implicit time stepper"
            )
        step_info.append(info)
        previous_load = load

        completed_steps = step + 1
        if completed_steps % store_every == 0 or completed_steps == n_steps:
            times.append(next_time)
            snapshots.append(u.copy())

    result = (np.asarray(times, dtype=float), np.asarray(snapshots, dtype=float))
    if return_info:
        return (*result, step_info)
    return result
