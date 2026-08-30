"""Structured triangular mesh generation for the irregular domain."""

from dataclasses import dataclass

import numpy as np

from .domain import Domain
from .elements import signed_area, area


@dataclass(frozen=True)
class Mesh:
    """Triangular P1 mesh with a homogeneous Dirichlet boundary mask."""

    points: np.ndarray
    triangles: np.ndarray
    boundary_nodes: np.ndarray

    @property
    def n_nodes(self) -> int:
        """Return the number of mesh vertices."""

        return int(self.points.shape[0])

    @property
    def n_elements(self) -> int:
        """Return the number of triangular elements."""

        return int(self.triangles.shape[0])

    @property
    def free_nodes(self) -> np.ndarray:
        """Return the indices of vertices that are not on the boundary."""

        return np.flatnonzero(~self.boundary_nodes)


def _node_id(i: int, j: int, ny: int) -> int:
    """Return the flattened vertex index for grid coordinates ``(i, j)``."""

    return i * (ny + 1) + j


def generate_mesh(domain: Domain, *, nx: int = 10, ny: int = 10) -> Mesh:
    """Generate a mapped triangular mesh for the proposal domain.

    The mesh starts from a tensor grid on ``[0, 1] x [0, 1]`` and maps each
    vertical line onto the physical interval between the irregular lower
    boundary and ``y = 1``. Each quadrilateral cell is split into two
    positively oriented triangles.

    Parameters
    ----------
    domain:
        Domain object that supplies the x-range and vertical boundaries.
    nx:
        Number of triangles in the x-axis.
    ny: 
        Number of triangles in the y-axis.
        
    Returns
    -------
    Mesh
        The generated mesh coordinates, element connectivity, and boolean
        boundary-node mask.

    Raises
    ------
    ValueError
        If ``nx`` is not greater than 1.
    ValueError
        If ``ny`` is not greater than 1.
    """

    if nx <= 0:
        raise ValueError("nx must be greater or equal than 1")
    if ny <= 0:
        raise ValueError("ny must be greater or equal than 1")

    n_points = (nx+1)*(ny+1)
    points = np.zeros(shape=(n_points, 2), dtype=float)

    for i, x in enumerate(np.linspace(*domain.x_range, nx+1)):
        for j, y in enumerate(np.linspace(domain.lower_boundary(x), domain.upper_boundary(x), ny+1)):
            points[_node_id(i, j, ny)] = [x, y]

    triangles: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            bl = _node_id(i, j, ny)
            br = _node_id(i+1, j, ny)
            tl = _node_id(i, j+1, ny)
            tr = _node_id(i+1, j+1, ny)
            if (i + j) % 2 == 0:
                candidates = [(bl, br, tr), (bl, tr, tl)]
            else:
                candidates = [(bl, br, tl), (br, tr, tl)]
            
            for triangle in candidates:
                if area(points[np.array(triangle)]) > 1e-10:
                    triangles.append(triangle)

    boundary_nodes = np.zeros(shape=(nx+1, ny+1), dtype=bool)
    boundary_nodes[ 0,:] = True
    boundary_nodes[-1,:] = True
    boundary_nodes[:, 0] = True
    boundary_nodes[:,-1] = True

    return Mesh(
        points=points,
        triangles=np.array(triangles, dtype=int),
        boundary_nodes=boundary_nodes.flatten()
    )
