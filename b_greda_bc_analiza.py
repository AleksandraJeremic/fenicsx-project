"""
ARMIRANOBETONSKA GREDA - ANALIZA GRANICNIH USLOVA
==================================================
Realni parametri za medjuspratnu gredu stambene zgrade:
  - Beton C25/30
  - Raspon L = 6.0 m
  - Presjek 30 x 45 cm
  - Stalno + korisno opterecenje
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, dirichletbc, Expression, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
from dolfinx.io import XDMFFile
import ufl
from ufl import dx, inner, grad, sym, Identity, tr, sqrt

# ============================================================
# MATERIJALNE KARAKTERISTIKE - Beton C25/30
# ============================================================
E   = 31e9    # Modul elasticnosti [Pa]
nu  = 0.2     # Poissonov koeficijent [-]
rho = 2500.0  # Gustina betona [kg/m3]
g   = 9.81    # Gravitacijsko ubrzanje [m/s2]

lam    = E * nu / ((1 + nu) * (1 - 2*nu))
mu     = E / (2 * (1 + nu))
lam_ps = 2 * lam * mu / (lam + 2 * mu)  # korekcija plane stress

# ============================================================
# GEOMETRIJA
# ============================================================
L  = 6.0    # Raspon [m]
H  = 0.45   # Visina presjeka [m]
b  = 0.30   # Sirina presjeka [m]  - koristi se za opterecenje
nx = 60     # Broj elemenata u x pravcu
ny = 10     # Broj elemenata u y pravcu

print("=" * 60)
print("ARMIRANOBETONSKA GREDA - STAMBENA ZGRADA")
print("=" * 60)
print(f"  Materijal:  Beton C25/30  (E = {E/1e9} GPa, nu = {nu})")
print(f"  Raspon:     L = {L} m")
print(f"  Presjek:    b x h = {b*100:.0f} x {H*100:.0f} cm  (L/H = {L/H:.1f})")

# ============================================================
# OPTERECENJE
# ============================================================
# Vlastita težina grede [N/m2] - zapreminska sila
g_vlastita = rho * g   # N/m3  -> djeluje kao f_vol

# Stalno opterecenje (pod, obloga i sl.) [kN/m2]
g_k = 2.0e3   # Pa

# Korisno opterecenje (stanovi prema EC1) [kN/m2]
q_k = 2.0e3   # Pa

# Ukupno povrsинsko opterecenje na gornjoj ivici [Pa]
# (stalno + korisno, bez vlastite tezine koja ide kao f_vol)
q_ukupno = g_k + q_k   # Pa

# Ekvivalentna linijska sila za provjeru [kN/m]
q_linijska        = q_ukupno * b / 1000
g_vlastita_lin    = rho * g * b * H / 1000

print(f"\n  OPTERECENJE:")
print(f"    Vlastita tezina:     {g_vlastita_lin:.2f} kN/m")
print(f"    Stalno (pod+obloga): {g_k/1000:.1f} kN/m2 × {b} m = {g_k*b/1000:.2f} kN/m")
print(f"    Korisno (stanovi):   {q_k/1000:.1f} kN/m2 × {b} m = {q_k*b/1000:.2f} kN/m")
print(f"    Ukupno linijsko:     ~{q_linijska + g_vlastita_lin:.2f} kN/m")
print("=" * 60)


# ============================================================
# TENZORI
# ============================================================
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lam_ps * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)

def sigma_vm(u):
    s = sigma(u) - (1/3) * tr(sigma(u)) * Identity(len(u))
    return sqrt(3/2 * inner(s, s))


# ============================================================
# POMOCNE FUNKCIJE ZA BC
# ============================================================
def dofs_ivica(domain, V_sub, marker_fn):
    fdim   = domain.topology.dim - 1
    facets = locate_entities_boundary(domain, fdim, marker_fn)
    return fem.locate_dofs_topological(V_sub, fdim, facets)

def dofs_tacka(domain, V_sub, marker_fn):
    vertices = locate_entities_boundary(domain, 0, marker_fn)
    return fem.locate_dofs_topological(V_sub, 0, vertices)


# ============================================================
# FUNKCIJA ZA MODELIRANJE
# ============================================================
def modeliraj_gredu(L, H, nx, ny, tip_bc):
    domain = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, H])],
        [nx, ny],
        CellType.triangle
    )

    V   = functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    bcs = []
    tol = 1e-10

    def lijeva_ivica(x):  return np.abs(x[0]) < tol
    def desna_ivica(x):   return np.abs(x[0] - L) < tol
    def gornja_ivica(x):  return np.abs(x[1] - H) < tol

    def lijevi_ugao(x):
        return np.logical_and(np.abs(x[0]) < tol, np.abs(x[1]) < tol)
    def desni_ugao(x):
        return np.logical_and(np.abs(x[0] - L) < tol, np.abs(x[1]) < tol)

    zero = fem.Constant(domain, 0.0)

    if tip_bc == 'prosta':
        bcs.append(dirichletbc(zero, dofs_tacka(domain, V.sub(0), lijevi_ugao), V.sub(0)))
        bcs.append(dirichletbc(zero, dofs_tacka(domain, V.sub(1), lijevi_ugao), V.sub(1)))
        bcs.append(dirichletbc(zero, dofs_tacka(domain, V.sub(1), desni_ugao),  V.sub(1)))

    elif tip_bc == 'konzola':
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(0), lijeva_ivica), V.sub(0)))
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(1), lijeva_ivica), V.sub(1)))

    elif tip_bc == 'slobodna_os':
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(1), lijeva_ivica), V.sub(1)))
        bcs.append(dirichletbc(zero, dofs_ivica(domain, V.sub(1), desna_ivica),  V.sub(1)))

    # --- Opterecenje ---
    fdim       = domain.topology.dim - 1
    facets_top = np.sort(locate_entities_boundary(domain, fdim, gornja_ivica))
    facet_tags = mesh.meshtags(
        domain, fdim,
        facets_top, np.full(len(facets_top), 1, dtype=np.int32)
    )
    ds_top = ufl.Measure("ds", domain=domain,
                         subdomain_data=facet_tags, subdomain_id=1)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Vlastita tezina kao zapreminska sila [N/m3]
    f_vol = fem.Constant(domain, np.array([0.0, -rho * g]))

    # Stalno + korisno opterecenje na gornjoj ivici [Pa]
    T = fem.Constant(domain, np.array([0.0, -q_ukupno]))

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
# POKRETANJE ZA SVA TRI TIPA BC
# ============================================================
tipovi = {
    'prosta':      'Pin-roller (prosta greda)',
    'konzola':     'Uklještenje s lijeve strane',
    'slobodna_os': 'Klizaci (sprijeceno samo u y)'
}

# Euler-Bernoulli provjera za prostu gredu
I      = b * H**3 / 12
q_lin  = q_ukupno * b + rho * g * b * H   # N/m
delta_EB = 5 * q_lin * L**4 / (384 * E * I)

print(f"\n  Provjera Euler-Bernoulli (prosta greda):")
print(f"    Moment inercije:  I = {I*1e6:.2f} cm4  ({I:.6f} m4)")
print(f"    Ukupna lin. sila: q = {q_lin/1000:.2f} kN/m")
print(f"    EB progib:        δ = {delta_EB*1000:.2f} mm")
print("=" * 60)

for tip, opis in tipovi.items():
    uh, domain, V, max_uy = modeliraj_gredu(L, H, nx, ny, tip)
    print(f"\n  [{tip}]")
    print(f"    Opis:          {opis}")
    print(f"    max |u_y|    = {abs(max_uy)*1000:.2f} mm")


# ============================================================
# IZVOZ ZA PARAVIEW
# ============================================================
print("\n\nIzvoz .xdmf fajlova za ParaView...")

for tip in ['prosta', 'konzola', 'slobodna_os']:
    uh, domain, V, _ = modeliraj_gredu(L, H, nx, ny, tip)

    # Interpolacija na Lagrange 1 za XDMF
    V1  = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    uh1 = Function(V1)
    uh1.interpolate(uh)
    uh1.name = "pomjeranja"

    with XDMFFile(MPI.COMM_WORLD, f"ab_greda_{tip}_pomjeranja.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh1)

    # Von Mises napon
    V_scalar = functionspace(domain, ("Lagrange", 1))
    vm_expr  = Expression(sigma_vm(uh), V_scalar.element.interpolation_points)
    vm_h     = Function(V_scalar)
    vm_h.interpolate(vm_expr)
    vm_h.name = "von_mises"

    with XDMFFile(MPI.COMM_WORLD, f"ab_greda_{tip}_naponi.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(vm_h)

    print(f"  Sacuvano: ab_greda_{tip}_pomjeranja.xdmf")
    print(f"  Sacuvano: ab_greda_{tip}_naponi.xdmf")

print("\nSvi fajlovi spremni za ParaView!")