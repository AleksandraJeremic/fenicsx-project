import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import (TestFunction, TrialFunction, Identity,
                 tr, sym, grad, dx, ds, inner, dot, sqrt)
import pyvista
from dolfinx.plot import vtk_mesh

# ── 1. PARAMETRI ──────────────────────────────────────────
L  = 5.0
H  = 0.5
t  = 0.3
E  = 210e9
nu = 0.3
q  = -10000.0

# ── 2. MESH ───────────────────────────────────────────────
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [L, H]],
    [50, 10],
    mesh.CellType.triangle
)

# ── 3. FUNKCIONALNI PROSTOR ───────────────────────────────
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

# ── 4. MATERIJALNI PARAMETRI ──────────────────────────────
mu      = fem.Constant(domain, np.float64(E / (2 * (1 + nu))))
lambda_ = fem.Constant(domain, np.float64(E * nu / ((1 + nu) * (1 - 2 * nu))))

# ── 5. RUBNI USLOVI ───────────────────────────────────────
# Collapse subprostora
V0, _ = V.sub(0).collapse()
V1, _ = V.sub(1).collapse()

# Funkcije za pronalazak oslonaca
def lijevi_oslonac(x):
    return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

def desni_oslonac(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0.0)

# Pronađi dofs – uzimamo [0] jer locate vraća [parent_dofs, sub_dofs]
lijevi_dofs_x = locate_dofs_geometrical((V.sub(0), V0), lijevi_oslonac)[0]
lijevi_dofs_y = locate_dofs_geometrical((V.sub(1), V1), lijevi_oslonac)[0]
desni_dofs_y  = locate_dofs_geometrical((V.sub(1), V1), desni_oslonac)[0]

# Postavi rubne uslove
bc_lijevi_x = dirichletbc(np.float64(0.0), lijevi_dofs_x, V.sub(0))
bc_lijevi_y = dirichletbc(np.float64(0.0), lijevi_dofs_y, V.sub(1))
bc_desni_y  = dirichletbc(np.float64(0.0), desni_dofs_y,  V.sub(1))

bcs = [bc_lijevi_x, bc_lijevi_y, bc_desni_y]

# ── 6. OZNACAVANJE GORNJE GRANICE ─────────────────────────
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

def gornja_granica(x):
    return np.isclose(x[1], H)

gornji_facets = locate_entities_boundary(domain, fdim, gornja_granica)
facet_indices = np.array(gornji_facets, dtype=np.int32)
facet_markers = np.full_like(facet_indices, 1)
facet_tags    = meshtags(domain, fdim, facet_indices, facet_markers)
ds_gore       = ds(domain=domain, subdomain_data=facet_tags, subdomain_id=1)

# ── 7. VARIJACIONA FORMULACIJA ────────────────────────────
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lambda_ * tr(epsilon(u)) * Identity(2) + \
           2 * mu * epsilon(u)

T      = fem.Constant(domain, np.array([0.0, q], dtype=np.float64))
a      = inner(sigma(u), epsilon(v)) * dx
L_form = dot(T, v) * ds_gore

# ── 8. RJEŠAVANJE ─────────────────────────────────────────
problem = LinearProblem(
    a, L_form, bcs=bcs,
    petsc_options_prefix="solve_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()

print("Maksimalni pomak u y [m]:", abs(uh.x.array.min()))

# Analitičko rješenje
I            = t * H**3 / 12
w_analiticki = 5 * abs(q) * t * L**4 / (384 * E * I)
print("Analitičko rješenje [m]:", w_analiticki)
print("Greška [%]:", abs(abs(uh.x.array.min()) - w_analiticki) / w_analiticki)