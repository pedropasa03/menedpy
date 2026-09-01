"""Finite element solver for the proposal nonlinear diffusion-reaction PDE."""

from .domain import Domain
from .mesh import Mesh, generate_mesh
from .problem import Problem
from .solver import boundary_values, initial_condition, solve
from .systems import newton, KLAM, super_ostrowski
from .visualize import graph_solution
from .time_stepping import explicit_euler_step, implicit_euler_step, crank_nicolson_step

__all__ = [
    "Domain",
    "Mesh",
    "Problem",
    "boundary_values",
    "generate_mesh",
    "initial_condition",
    "solve",
    "newton",
    "KLAM",
    "super_ostrowski",
    "graph_solution",
    "explicit_euler_step",
    "implicit_euler_step",
    "crank_nicolson_step"
]
