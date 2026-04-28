import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from ufl import (TestFunction, TrialFunction, Identity,
                 tr, sym, grad, dx, ds, inner, dot, sqrt)
import pyvista
from dolfinx.plot import vtk_mesh

# ── 1. MESH ──────────────────────────────────────────────
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [5.0, 0.3, 0.2]],
    [25, 5, 4],
    mesh.CellType.tetrahedron
)

# ── 2. FUNKCIONALNI PROSTOR ───────────────────────────────
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

# ── 3. MATERIJALNI PARAMETRI ──────────────────────────────
E  = 210e9
nu = 0.3
rho = 7850.0  # gustina čelika [kg/m³]
g   = 9.81    # gravitaciono ubrzanje [m/s²]

mu      = fem.Constant(domain, np.float64(E / (2 * (1 + nu))))
lambda_ = fem.Constant(domain, np.float64(E * nu / ((1 + nu) * (1 - 2 * nu))))

# Gravitacijska sila po jedinici zapremine [N/m³]
f = fem.Constant(domain, np.array([0.0, -rho*g, 0.0], dtype=np.float64))

# ── 4. RUBNI USLOVI ───────────────────────────────────────
def lijeva_strana(x):
    return np.isclose(x[0], 0.0)

dofs = locate_dofs_geometrical(V, lijeva_strana)
bc = dirichletbc(np.zeros(3, dtype=np.float64), dofs, V)

# ── 5. VARIJACIONA FORMULACIJA ────────────────────────────
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lambda_ * tr(epsilon(u)) * Identity(len(u)) + \
           2 * mu * epsilon(u)

a = inner(sigma(u), epsilon(v)) * dx

# Gravitacija djeluje po cijeloj zapremini (dx), ne samo po granici (ds)
L = dot(f, v) * dx

# ── 6. RJEŠAVANJE ─────────────────────────────────────────
problem = LinearProblem(
    a, L,
    bcs=[bc],
    petsc_options_prefix="solve_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

uh = problem.solve()

print("Maksimalni pomak [m]:", uh.x.array.max())
print("Minimalni pomak [m]:", uh.x.array.min())

# ── 7. VIZUALIZACIJA POMAKA ───────────────────────────────
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

pomaci_3d = uh.x.array.reshape((geometry.shape[0], 3))
grid.point_data["pomak"] = pomaci_3d

warped = grid.warp_by_vector("pomak", factor=500)

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(warped, scalars=pomaci_3d[:, 1],
                 cmap="coolwarm", show_edges=True)
plotter.add_title("Deformacija od gravitacije (y pomak)")
plotter.camera_position = [(2.5, -8, 4), (2.5, 0.15, 0.1), (0, 1, 0)]
plotter.show(screenshot="gravitacija_pomak.png")
print("Slika sačuvana kao gravitacija_pomak.png")

# ── 8. VON MISES NAPON ────────────────────────────────────
def sigma_vm(u):
    s = sigma(u) - (1/3) * tr(sigma(u)) * Identity(len(u))
    return sqrt(3/2 * inner(s, s))

V_scalar = fem.functionspace(domain, ("Lagrange", 1))
vm_expr = fem.Expression(sigma_vm(uh), V_scalar.element.interpolation_points)
vm = fem.Function(V_scalar)
vm.interpolate(vm_expr)

print("Maksimalni von Mises napon [Pa]:", vm.x.array.max())

topology2, cell_types2, geometry2 = vtk_mesh(V_scalar)
grid2 = pyvista.UnstructuredGrid(topology2, cell_types2, geometry2)
grid2.point_data["von_mises"] = vm.x.array

plotter2 = pyvista.Plotter(off_screen=True)
plotter2.add_mesh(grid2, scalars="von_mises",
                  cmap="jet", show_edges=True)
plotter2.add_title("Von Mises napon - gravitacija [Pa]")
plotter2.camera_position = [(2.5, -8, 4), (2.5, 0.15, 0.1), (0, 1, 0)]
plotter2.show(screenshot="gravitacija_mises.png")
print("Slika sačuvana kao gravitacija_mises.png")