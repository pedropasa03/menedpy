"""Module for visualizing the solution of a problem defined on a 2D domain."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np

def graph_solution(domain, mesh, solution, res=500):
    triangulation = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
    interpolator = mtri.LinearTriInterpolator(triangulation, solution)

    x = np.linspace(*domain.x_range, res)
    y = np.linspace(*domain.x_range, res)
    X, Y = np.meshgrid(x, y)

    mask = (Y > np.vectorize(domain.lower_boundary)(X)) * (Y < np.vectorize(domain.upper_boundary)(X))

    fig, ax = plt.subplots()

    U = np.ma.filled(interpolator(X, Y), np.nan)
    U_masked = np.where(mask, U, np.nan)

    im = ax.pcolormesh(
        X, Y, U_masked,
        cmap="viridis",
        rasterized=True,
        norm=matplotlib.colors.PowerNorm(gamma=1.5)
    )

    upper = np.vectorize(domain.upper_boundary)(x)
    lower = np.vectorize(domain.lower_boundary)(x)

    plt.plot(x, upper, 'k', linewidth=0.75)
    plt.plot(x, lower, 'k', linewidth=0.75)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


    plt.colorbar(mappable=im)
    ax.set_aspect('equal')
    plt.tight_layout(pad=0.0)

    return fig, ax