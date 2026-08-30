"""Finite element solver for the proposal nonlinear diffusion-reaction PDE."""

from .domain import Domain
from .mesh import Mesh, generate_mesh
from .problem import Problem
from .solver import boundary_values, initial_condition, solve
from .systems import newton, KLAM

__all__ = [
    "Domain",
    "Mesh",
    "Problem",
    "boundary_values",
    "generate_mesh",
    "initial_condition",
    "solve",
    "newton",
    "KLAM"
]
