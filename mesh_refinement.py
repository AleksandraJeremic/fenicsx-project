import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from ufl import (TestFunction, TrialFunction, Identity,
                 tr, sym, grad, dx, inner, dot, sqrt)

# Materijalni parametri
E   = 210e9
nu  = 0.3
rho = 7850.0
g   = 9.81

def rijesi_gredu(nx, ny, nz, naziv):
    print(f"\n── {naziv} mreža ({nx}x{ny}x{nz}) ──")

    # Mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [5.0, 0.3, 0.2]],
        [nx, ny, nz],
        mesh.CellType.tetrahedron
    )

    # Funkcionalni prostor
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    # Materijalni parametri
    mu      = fem.Constant(domain, np.float64(E / (2 * (1 + nu))))
    lambda_ = fem.Constant(domain, np.float64(E * nu / ((1 + nu) * (1 - 2 * nu))))
    f       = fem.Constant(domain, np.array([0.0, -rho*g, 0.0], dtype=np.float64))

    # Rubni uslovi
    def lijeva_strana(x):
        return np.isclose(x[0], 0.0)

    dofs = locate_dofs_geometrical(V, lijeva_strana)
    bc   = dirichletbc(np.zeros(3, dtype=np.float64), dofs, V)

    # Varijaciona formulacija
    u = TrialFunction(V)
    v = TestFunction(V)

    def epsilon(u):
        return sym(grad(u))

    def sigma(u):
        return lambda_ * tr(epsilon(u)) * Identity(len(u)) + \
               2 * mu * epsilon(u)

    a = inner(sigma(u), epsilon(v)) * dx
    L = dot(f, v) * dx

    # Mjerimo vrijeme rješavanja
    start = time.time()
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="solve_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()
    kraj = time.time()

    # Von Mises
    def sigma_vm(u):
        s = sigma(u) - (1/3) * tr(sigma(u)) * Identity(len(u))
        return sqrt(3/2 * inner(s, s))

    V_scalar = fem.functionspace(domain, ("Lagrange", 1))
    vm_expr  = fem.Expression(sigma_vm(uh), V_scalar.element.interpolation_points)
    vm       = fem.Function(V_scalar)
    vm.interpolate(vm_expr)

    # Rezultati
    print(f"  Broj čvorova:          {domain.topology.index_map(0).size_global}")
    print(f"  Maksimalni pomak:      {abs(uh.x.array.min()):.6f} m")
    print(f"  Maks. von Mises:       {vm.x.array.max()/1e6:.4f} MPa")
    print(f"  Vrijeme rješavanja:    {kraj - start:.2f} s")

# Pokrenemo tri mreže
rijesi_gredu(10,  3, 2, "Gruba")
rijesi_gredu(25,  5, 4, "Srednja")
rijesi_gredu(50, 10, 8, "Fina")