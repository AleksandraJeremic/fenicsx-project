import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from ufl import (TestFunction, TrialFunction, Identity, 
                 tr, sym, grad, dx, ds, inner, dot, sqrt, dev)
import pyvista
from dolfinx.plot import vtk_mesh

# ── 1. MESH ──────────────────────────────────────────────
# Pravougaonik: dužina 5m, visina 0.2m
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [5.0, 0.2]],
    [50, 10],
    mesh.CellType.triangle
)

# ── 2. FUNKCIONALNI PROSTOR ───────────────────────────────
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

# ── 3. MATERIJALNI PARAMETRI ──────────────────────────────
E  = 210e9   # čelik
nu = 0.3

mu      = fem.Constant(domain, np.float64(E / (2 * (1 + nu))))
lambda_ = fem.Constant(domain, np.float64(E * nu / ((1 + nu) * (1 - 2 * nu))))

# ── 4. RUBNI USLOVI ───────────────────────────────────────
# Lijeva strana (x=0) je uklještena
def lijeva_strana(x):
    return np.isclose(x[0], 0.0)

dofs = locate_dofs_geometrical(V, lijeva_strana)
bc = dirichletbc(np.zeros(2, dtype=np.float64), dofs, V)

# ── 5. VARIJACIONA FORMULACIJA ────────────────────────────
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lambda_ * tr(epsilon(u)) * Identity(len(u)) + \
           2 * mu * epsilon(u)

# Sila – djeluje prema dolje na desnoj strani (x=10)
T = fem.Constant(domain, np.array([0.0, -10000.0], dtype=np.float64))

a = inner(sigma(u), epsilon(v)) * dx
L = dot(T, v) * ds

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

pomaci_3d = np.zeros((geometry.shape[0], 3))
pomaci_3d[:, :2] = uh.x.array.reshape((geometry.shape[0], 2))
grid.point_data["pomak"] = pomaci_3d

warped = grid.warp_by_vector("pomak", factor=100)

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(warped, scalars=pomaci_3d[:, 1],
                 cmap="coolwarm", show_edges=True)
plotter.add_title("Deformacija grede (y pomak)")
plotter.camera_position = [(2.5, -15, 1), (2.5, 0.1, 0), (0, 1, 0)]
plotter.show(screenshot="greda_pomak.png")
print("Slika sačuvana kao greda_pomak.png")

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
plotter2.add_title("Von Mises napon - greda [Pa]")
plotter2.view_xy()
plotter2.show(screenshot="greda_mises.png")
print("Slika sačuvana kao greda_mises.png")
