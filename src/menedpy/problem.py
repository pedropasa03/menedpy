"""Problem data for the nonlinear diffusion-reaction equation."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .domain import Domain

ScalarField = Callable[[np.ndarray, np.ndarray], np.ndarray]
TimeScalarField = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def proposal_diffusion(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the discontinuous diffusion coefficient from the proposal.

    Parameters
    ----------
    x:
        Array of x-coordinates.
    y:
        Array of y-coordinates. The proposal coefficient is independent of
        ``y``; the argument is accepted to match the scalar-field interface.

    Returns
    -------
    numpy.ndarray
        Values equal to ``1`` for ``x < 0.5`` and ``5`` for ``x >= 0.5``.
    """

    del y
    return np.where(x < 0.5, 1.0, 5.0)


def proposal_source(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Evaluate the source term used in the proposal example.

    Parameters
    ----------
    x:
        Array of x-coordinates.
    y:
        Array of y-coordinates.
    t:
        Time value. The source is time-independent, but the argument is
        accepted to match the time-dependent scalar-field interface.

    Returns
    -------
    numpy.ndarray
        Values ``sin(pi*x) * sin(pi*y)``.
    """

    del t
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def proposal_initial(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the initial condition used in the proposal example.

    Parameters
    ----------
    x:
        Array of x-coordinates.
    y:
        Array of y-coordinates.

    Returns
    -------
    numpy.ndarray
        Values ``sin(pi*x) * sin(pi*y)``.
    """

    return np.sin(np.pi * x) * np.sin(np.pi * y)


def homogeneous_boundary_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the default homogeneous Dirichlet boundary condition."""

    del y
    return np.zeros_like(x, dtype=float)


@dataclass(frozen=True)
class Problem:
    """Data needed to solve the nonlinear diffusion-reaction equation."""

    domain: Domain = field(default_factory=Domain)
    diffusion_coefficient: ScalarField = proposal_diffusion
    source: TimeScalarField = proposal_source
    initial_condition: ScalarField = proposal_initial
    boundary_conditions: ScalarField = homogeneous_boundary_condition
    final_time: float = 0.1
