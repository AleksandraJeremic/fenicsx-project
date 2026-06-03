"""
PROSTA GREDA - ANALIZA GRANICNIH USLOVA (BC)
=============================================
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
import ufl
from ufl import dx, inner, grad, sym, Identity, tr

# ============================================================
# MATERIJALNE KARAKTERISTIKE (celik)
# ============================================================
E  = 210e9
nu = 0.3

lam    = E * nu / ((1 + nu) * (1 - 2*nu))
mu     = E / (2 * (1 + nu))
lam_ps = 2 * lam * mu / (lam + 2 * mu)


def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lam_ps * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)


# ============================================================
# POMOCNA: dofs na ivici (za subspace)
# ============================================================
def dofs_ivica(domain, V_sub, marker_fn):
    fdim   = domain.topology.dim - 1
    facets = locate_entities_boundary(domain, fdim, marker_fn)
    return fem.locate_dofs_topological(V_sub, fdim, facets)


# ============================================================
# POMOCNA: dofs u tacki - koristimo kolaps subspace-a
# ============================================================
def dofs_tacka(domain, V_sub, marker_fn):
    """
    Pronalazi DOF-ove u tackama (vertices) koristeci
    locate_dofs_topological na dim=0 (cvorovi).
    """
    vertices = locate_entities_boundary(domain, 0, marker_fn)
    return fem.locate_dofs_topological(V_sub, 0, vertices)


# ============================================================
# FUNKCIJA ZA MODELIRANJE
# ============================================================
def modeliraj_gredu(L, H, nx, ny, tip_bc, opterecenje=-1e6):
    domain = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, H])],
        [nx, ny],
        CellType.triangle
    )

    V   = functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    bcs = []
    tol = 1e-10

    def lijeva_ivica(x):
        return np.abs(x[0]) < tol

    def desna_ivica(x):
        return np.abs(x[0] - L) < tol

    def gornja_ivica(x):
        return np.abs(x[1] - H) < tol

    # Ugaone tacke
    def lijevi_ugao(x):
        return np.logical_and(np.abs(x[0]) < tol, np.abs(x[1]) < tol)

    def desni_ugao(x):
        return np.logical_and(np.abs(x[0] - L) < tol, np.abs(x[1]) < tol)

    zero = fem.Constant(domain, 0.0)

    if tip_bc == 'prosta':
        # Lijevi ugao (pin):   u_x = 0, u_y = 0
        # Desni ugao (roller): u_y = 0
        bcs.append(dirichletbc(zero, dofs_tacka(domain, V.sub(0), lijevi_ugao), V.sub(0)))
        bcs.append(dirichletbc(zero, dofs_tacka(domain, V.sub(1), lijevi_ugao), V.sub(1)))
        bcs.append(dirichletbc(zero, dofs_tacka(domain, V.sub(1), desni_ugao),  V.sub(1)))

    elif tip_bc == 'konzola':
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(0), lijeva_ivica), V.sub(0)))
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(1), lijeva_ivica), V.sub(1)))

    elif tip_bc == 'slobodna_os':
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(1), lijeva_ivica), V.sub(1)))
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(1), desna_ivica),  V.sub(1)))

    # --- Opterecenje na gornjoj ivici ---
    fdim       = domain.topology.dim - 1
    facets_top = locate_entities_boundary(domain, fdim, gornja_ivica)
    facet_tags = mesh.meshtags(
        domain, fdim,
        facets_top, np.full(len(facets_top), 1, dtype=np.int32)
    )
    ds_top = ufl.Measure("ds", domain=domain,
                         subdomain_data=facet_tags, subdomain_id=1)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_vol = fem.Constant(domain, np.array([0.0, 0.0]))
    T     = fem.Constant(domain, np.array([0.0, opterecenje]))

    a      = inner(sigma(u), epsilon(v)) * dx
    L_form = inner(f_vol, v) * dx + inner(T, v) * ds_top

    problem = LinearProblem(a, L_form, bcs=bcs,
                            petsc_options_prefix="greda_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    u_vals = uh.x.array.reshape(-1, 2)
    max_uy = np.min(u_vals[:, 1])

    return uh, domain, V, max_uy


# ============================================================
# POKRETANJE
# ============================================================
print("=" * 60)
print("ANALIZA GRANICNIH USLOVA - PROSTA GREDA")
print("=" * 60)
print(f"  Raspon:   L = 5.0 m")
print(f"  Visina:   H = 0.5 m  (L/H = 10)")
print(f"  Mreza:    50 x 10 elemenata")
print(f"  Pritisak: q = 1.0 MPa (na gornjoj ivici)")
print("=" * 60)

L, H   = 5.0, 0.5
nx, ny = 50, 10

tipovi = {
    'prosta':      'Pin-roller (klasicna prosta greda)',
    'konzola':     'Uklještenje s lijeve strane',
    'slobodna_os': 'Klizaci (sprijeceno samo u y)'
}

for tip, opis in tipovi.items():
    uh, domain, V, max_uy = modeliraj_gredu(L, H, nx, ny, tip)
    print(f"\n  [{tip}]")
    print(f"    Opis:       {opis}")
    print(f"    max |u_y| = {abs(max_uy)*1000:.4f} mm")

print("\n" + "=" * 60)
print("OBJASNJENJE GRANICNIH USLOVA:")
print("=" * 60)
print("""
  PROSTA GREDA (pin-roller):
    - Lijevi ugao (pin):    u_x = 0,  u_y = 0
    - Desni ugao (roller):  u_y = 0   (sloboda u x!)
    * Blokiramo samo dvije tacke, ne cijelu ivicu,
      kako bi greda mogla slobodno da se savija.

  KONZOLA (uklještenje):
    - Cijela lijeva ivica:  u_x = 0,  u_y = 0
    * Progib konzole L=5m je ~10x veci od proste grede.

  SLOBODNA OS (klizaci):
    - Lijeva ivica:  u_y = 0
    - Desna ivica:   u_y = 0
    * Simulira gredu izmedju dva paralelna zida.
""")