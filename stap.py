import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from ufl import (TestFunction, TrialFunction,
                 grad, dx, inner, dot)
import pyvista
from dolfinx.plot import vtk_mesh
print("Početak...")

# ── 1. MESH – 1D linija ───────────────────────────────────
# Štap dužine 5m, podijeljen na 20 elemenata
domain = mesh.create_interval(
    MPI.COMM_WORLD,
    20,           # broj elemenata
    [0.0, 5.0]    # od 0 do 5m
)

# ── 2. FUNKCIONALNI PROSTOR ───────────────────────────────
# Skalarni prostor – samo aksijalni pomak u x smjeru
V = fem.functionspace(domain, ("Lagrange", 1))

# ── 3. MATERIJALNI PARAMETRI ──────────────────────────────
E = 210e9    # modul elastičnosti čelik [Pa]
A = 0.06     # površina presjeka 0.3m x 0.2m [m²]

EA = fem.Constant(domain, np.float64(E * A))

# ── 4. RUBNI USLOVI ───────────────────────────────────────
# Lijevi kraj (x=0) je uklješten – pomak = 0
def lijevi_kraj(x):
    return np.isclose(x[0], 0.0)

dofs = locate_dofs_geometrical(V, lijevi_kraj)
bc = dirichletbc(np.float64(0.0), dofs, V)

# ── 5. VARIJACIONA FORMULACIJA ────────────────────────────
u = TrialFunction(V)
v = TestFunction(V)

# Aksijalna krutost – EA * du/dx * dv/dx
a = EA * inner(grad(u), grad(v)) * dx

# Aksijalna sila na desnom kraju – 100 kN
F = fem.Constant(domain, np.float64(100000.0))

# Površinska sila na desnom kraju
from ufl import ds
L = F * v * ds

# ── 6. OZNACAVANJE DESNOG KRAJA ───────────────────────────
from dolfinx.mesh import locate_entities_boundary, meshtags

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

def desni_kraj(x):
    return np.isclose(x[0], 5.0)

desni_facets  = locate_entities_boundary(domain, fdim, desni_kraj)
facet_indices = np.array(desni_facets, dtype=np.int32)
facet_markers = np.full_like(facet_indices, 1)
facet_tags    = meshtags(domain, fdim, facet_indices, facet_markers)
ds_desno      = ds(domain=domain, subdomain_data=facet_tags, subdomain_id=1)

# Ponovo definiši L sa oznacenom granicom
L = F * v * ds_desno

# ── 7. RJEŠAVANJE ─────────────────────────────────────────
problem = LinearProblem(
    a, L, bcs=[bc],
    petsc_options_prefix="solve_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()

print("Maksimalni pomak [m]:", uh.x.array.max())

# ── 8. ANALITIČKO RJEŠENJE ────────────────────────────────
# Za štap: u = F*L / (E*A)
u_analiticki = 100000.0 * 5.0 / (E * 0.06)
print("Analitičko rješenje [m]:", u_analiticki)
print("Greška [%]:", abs(uh.x.array.max() - u_analiticki) / u_analiticki * 100)

# ── 9. VIZUALIZACIJA ──────────────────────────────────────
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["pomak"] = uh.x.array

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(grid, scalars="pomak",
                 cmap="coolwarm", 
                 show_edges=True,
                 line_width=10)
plotter.add_title("Štapni element - aksijalni pomak")
plotter.view_xy()
plotter.show(screenshot="stap_pomak.png")
print("Slika sačuvana kao stap_pomak.png")
