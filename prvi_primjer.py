import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from ufl import (TestFunction, TrialFunction,
                 Identity, tr, sym, grad, dx, ds, inner, dot)
import ufl

# ── 1. MESH ──────────────────────────────────────────────
domain = mesh.create_unit_square(
    MPI.COMM_WORLD, 10, 10, mesh.CellType.triangle
)

# ── 2. FUNKCIONALNI PROSTOR ───────────────────────────────
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

# ── 3. MATERIJALNI PARAMETRI ──────────────────────────────
E  = 210e9
nu = 0.3

mu      = fem.Constant(domain, np.float64(E / (2 * (1 + nu))))
lambda_ = fem.Constant(domain, np.float64(E * nu / ((1 + nu) * (1 - 2 * nu))))

# ── 4. RUBNI USLOVI ───────────────────────────────────────
def lijeva_strana(x):
    return np.isclose(x[0], 0)

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

T = fem.Constant(domain, np.array([0.0, -1e6], dtype=np.float64))

a = inner(sigma(u), epsilon(v)) * dx
L = dot(T, v) * ds

# ── 6. RJEŠAVANJE ─────────────────────────────────────────
problem = LinearProblem(
    a, L,
    bcs=[bc],
    petsc_options_prefix="solve_",
    petsc_options={"ksp_type": "preonly",
                   "pc_type": "lu"}
)

uh = problem.solve()

print("Maksimalni pomak [m]:", uh.x.array.max())
print("Minimalni pomak [m]:", uh.x.array.min())
print("Gotovo!")
print("Veličina rješenja:", uh.x.array.shape)
print("Maksimalni pomak [m]:", uh.x.array.max())
print("Minimalni pomak [m]:", uh.x.array.min())
import pyvista
from dolfinx.plot import vtk_mesh

# Pripremi mesh za vizualizaciju
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Dodaj pomake na mesh - moraju biti 3D za pyvista (dodaj nulu za z)
pomaci_3d = np.zeros((geometry.shape[0], 3))
pomaci_3d[:, :2] = uh.x.array.reshape((geometry.shape[0], 2))
grid.point_data["pomak"] = pomaci_3d

# Warp – prikaži deformisanu ploču (uvećano 1000x da se vidi)
warped = grid.warp_by_vector("pomak", factor=1000)

# Crtaj
plotter = pyvista.Plotter()
plotter.add_mesh(warped, scalars=pomaci_3d[:, 1],
                 cmap="coolwarm",
                 show_edges=True)
plotter.add_title("Deformacija ploče (y pomak)")
plotter.show_axes()
plotter.off_screen = True
plotter.show(screenshot="deformacija.png", auto_close=True)

print("Slika sačuvana kao deformacija.png")
import dolfinx.fem as fem
from dolfinx.fem.petsc import LinearProblem
from ufl import sqrt, dev, tr, inner

# ── VON MISES NAPON ───────────────────────────────────────

# Napon u svakoj tački
def sigma_vm(u):
    s = sigma(u) - (1/3) * tr(sigma(u)) * Identity(len(u))  # devijatorski napon
    return sqrt(3/2 * inner(s, s))

# Projekcija na skalarni funkcionalni prostor
V_scalar = fem.functionspace(domain, ("Lagrange", 1))
vm_expr = fem.Expression(sigma_vm(uh), V_scalar.element.interpolation_points)
vm = fem.Function(V_scalar)
vm.interpolate(vm_expr)

print("Maksimalni von Mises napon [Pa]:", vm.x.array.max())
print("Minimalni von Mises napon [Pa]:", vm.x.array.min())

# ── VIZUALIZACIJA VON MISES NAPONA ────────────────────────
topology2, cell_types2, geometry2 = vtk_mesh(V_scalar)
grid2 = pyvista.UnstructuredGrid(topology2, cell_types2, geometry2)
grid2.point_data["von_mises"] = vm.x.array

plotter2 = pyvista.Plotter(off_screen=True)
plotter2.add_mesh(grid2, scalars="von_mises",
                  cmap="jet",
                  show_edges=True)
plotter2.add_title("Von Mises napon [Pa]")
plotter2.show_axes()
plotter2.view_xy()
plotter2.show(screenshot="von_mises.png")

print("Slika sačuvana kao von_mises.png")