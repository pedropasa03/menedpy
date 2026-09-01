from menedpy import *

import matplotlib.pyplot as plt


def main():
    domain = Domain()
    mesh = generate_mesh(domain)

    problem = Problem(domain)

    time, solution = solve(
        problem,
        mesh,
        n_steps=100,
        time_stepper=crank_nicolson_step,
        nonlinear_solver=super_ostrowski
    )

    # Get the return values of the functions
    print("mesh=", mesh)
    print("time=", time)
    print("solution=", solution)

    # Graph the solution
    fig, ax = graph_solution(domain, mesh, solution[-1])
    plt.savefig("examples/solution.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
