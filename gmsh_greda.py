import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import dirichletbc
from dolfinx.mesh import meshtags
from ufl import (TestFunction, TrialFunction, Identity,
                 tr, sym, grad, dx, ds, inner, dot, sqrt)
import pyvista
from dolfinx.plot import vtk_mesh
import gmsh

# ── 1. GMSH GEOMETRIJA ────────────────────────────────────
gmsh.initialize()
gmsh.model.add("greda")

# Dimenzije grede
L  = 5.0   # dužina
H  = 0.3   # visina
W  = 0.2   # širina

# Napravi kutiju (box)
greda = gmsh.model.occ.addBox(0, 0, 0, L, H, W)
gmsh.model.occ.synchronize()

# ── 2. OZNACAVANJE GRANICA ────────────────────────────────
# Pronađi sve površine
surfaces = gmsh.model.getEntities(dim=2)

lijeva  = []
desna   = []
ostale  = []

for surf in surfaces:
    com = gmsh.model.occ.getCenterOfMass(2, surf[1])
    if abs(com[0] - 0.0) < 1e-6:        # x = 0, lijeva strana
        lijeva.append(surf[1])
    elif abs(com[0] - L) < 1e-6:        # x = L, desna strana
        desna.append(surf[1])
    else:
        ostale.append(surf[1])

# Dodaj fizičke grupe sa oznakama
gmsh.model.addPhysicalGroup(2, lijeva, tag=1)
gmsh.model.setPhysicalName(2, 1, "lijeva")
gmsh.model.addPhysicalGroup(2, desna, tag=2)
gmsh.model.setPhysicalName(2, 2, "desna")
gmsh.model.addPhysicalGroup(3, [greda], tag=1)
gmsh.model.setPhysicalName(3, 1, "greda")

# ── 3. PRAVLJENJE MREŽE ───────────────────────────────────
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
gmsh.model.mesh.generate(3)
gmsh.write("greda.msh")
gmsh.finalize()
print("Mesh napravljen i sačuvan kao greda.msh")

# ── 4. UCITAVANJE U FENICSX ───────────────────────────────
from dolfinx.io import gmsh as gmshio
result = gmshio.read_from_msh(
    "greda.msh", MPI.COMM_WORLD, gdim=3
)
domain = result.mesh
cell_tags = result.cell_tags
facet_tags = result.facet_tags
print("Mesh učitan u FEniCSx")
print("Broj čvorova:", domain.topology.index_map(0).size_global)

# ── 5. FUNKCIONALNI PROSTOR ───────────────────────────────
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

# ── 6. MATERIJALNI PARAMETRI ──────────────────────────────
E   = 210e9
nu  = 0.3
rho = 7850.0
g   = 9.81

mu      = fem.Constant(domain, np.float64(E / (2 * (1 + nu))))
lambda_ = fem.Constant(domain, np.float64(E * nu / ((1 + nu) * (1 - 2 * nu))))
f       = fem.Constant(domain, np.array([0.0, -rho*g, 0.0], dtype=np.float64))
T       = fem.Constant(domain, np.array([0.0, -50000.0, 0.0], dtype=np.float64))

# ── 7. RUBNI USLOVI ───────────────────────────────────────
# Koristimo facet_tags koje smo definisali u Gmsh-u
lijeva_dofs = fem.locate_dofs_topological(V, 2, facet_tags.find(1))
bc = dirichletbc(np.zeros(3, dtype=np.float64), lijeva_dofs, V)

# ── 8. VARIJACIONA FORMULACIJA ────────────────────────────
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lambda_ * tr(epsilon(u)) * Identity(len(u)) + \
           2 * mu * epsilon(u)

ds_desno = ds(domain=domain, subdomain_data=facet_tags, subdomain_id=2)

a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds_desno

# ── 9. RJEŠAVANJE ─────────────────────────────────────────
problem = LinearProblem(
    a, L, bcs=[bc],
    petsc_options_prefix="solve_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()
print("Maksimalni pomak [m]:", abs(uh.x.array.min()))

# ── 10. VIZUALIZACIJA ─────────────────────────────────────
pyvista.start_xvfb()
topology, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

pomaci_3d = uh.x.array.reshape((geometry.shape[0], 3))
grid.point_data["pomak"] = pomaci_3d
warped = grid.warp_by_vector("pomak", factor=50)

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(warped, scalars=pomaci_3d[:, 1],
                 cmap="coolwarm", show_edges=True)
plotter.add_title("Gmsh greda - pomak")
plotter.camera_position = [(2.5, -8, 4), (2.5, 0.15, 0.1), (0, 1, 0)]
plotter.show(screenshot="gmsh_pomak.png")
print("Slika sačuvana kao gmsh_pomak.png")