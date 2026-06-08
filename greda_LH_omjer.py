"""
PROSTA GREDA - ANALIZA UTICAJA ODNOSA L/H
==========================================
Ispituje kako odnos raspon/visina (L/H) utice na:
  - Maksimalni progib (FEM vs Euler-Bernoulli)
  - Znacajnost smicajnih deformacija (Timoshenko efekat)

Geometrija: 2D ravninski napon (plane stress)
Materijal:  Beton C25/30
Oslanjanje: Prosta greda (pin-roller)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
import ufl
from ufl import dx, inner, grad, sym, Identity, tr

# ============================================================
# MATERIJALNE KARAKTERISTIKE - Beton C25/30
# ============================================================
E   = 31e9   # Modul elasticnosti [Pa]
nu  = 0.2    # Poissonov koeficijent [-]
rho = 2500.0 # Gustina betona [kg/m3]
g   = 9.81   # Gravitacijsko ubrzanje [m/s2]

lam    = E * nu / ((1 + nu) * (1 - 2*nu))
mu     = E / (2 * (1 + nu))
lam_ps = 2 * lam * mu / (lam + 2 * mu)  # korekcija plane stress


# ============================================================
# TENZORI
# ============================================================
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lam_ps * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)


# ============================================================
# POMOCNE FUNKCIJE ZA BC
# ============================================================
def dofs_tacka(domain, V_sub, marker_fn):
    vertices = locate_entities_boundary(domain, 0, marker_fn)
    return fem.locate_dofs_topological(V_sub, 0, vertices)


# ============================================================
# FUNKCIJA ZA MODELIRANJE
# ============================================================
def modeliraj_prostu_gredu(L, H, nx, ny):
    """
    Modelira prostu gredu sa:
      - Vlastitom tezinom (zapreminska sila)
      - Stalnim + korisnim opterecenjem na gornjoj ivici
    Vraca maksimalni vertikalni pomak.
    """
    domain = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, H])],
        [nx, ny],
        CellType.triangle
    )

    V   = functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    tol = 1e-10

    def lijevi_ugao(x):
        return np.logical_and(np.abs(x[0]) < tol, np.abs(x[1]) < tol)

    def desni_ugao(x):
        return np.logical_and(np.abs(x[0] - L) < tol, np.abs(x[1]) < tol)

    def gornja_ivica(x):
        return np.abs(x[1] - H) < tol

    zero = fem.Constant(domain, 0.0)
    bcs  = [
        dirichletbc(zero, dofs_tacka(domain, V.sub(0), lijevi_ugao), V.sub(0)),
        dirichletbc(zero, dofs_tacka(domain, V.sub(1), lijevi_ugao), V.sub(1)),
        dirichletbc(zero, dofs_tacka(domain, V.sub(1), desni_ugao),  V.sub(1)),
    ]

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

    # Vlastita tezina kao zapreminska sila
    f_vol = fem.Constant(domain, np.array([0.0, -rho * g]))

    # Stalno (2 kN/m2) + korisno (2 kN/m2) na gornjoj ivici
    q_ukupno = 4000.0   # Pa
    T = fem.Constant(domain, np.array([0.0, -q_ukupno]))

    a      = inner(sigma(u), epsilon(v)) * dx
    L_form = inner(f_vol, v) * dx + inner(T, v) * ds_top

    problem = LinearProblem(a, L_form, bcs=bcs,
                            petsc_options_prefix="lh_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    u_vals = uh.x.array.reshape(-1, 2)
    max_uy = np.min(u_vals[:, 1])

    return abs(max_uy)


# ============================================================
# ANALIZA ZA RAZLICITE L/H ODNOSE
# ============================================================
print("=" * 65)
print("ANALIZA UTICAJA ODNOSA L/H - PROSTA AB GREDA")
print("=" * 65)
print(f"  Materijal: Beton C25/30  (E = {E/1e9} GPa)")
print(f"  Fiksni raspon: L = 6.0 m")
print(f"  Opterecenje:   vlastita tezina + 4.0 kN/m2")
print("=" * 65)
print(f"  {'L/H':>4}  {'H [cm]':>7}  {'FEM [mm]':>10}  {'EB [mm]':>10}  {'FEM/EB':>7}")
print("-" * 65)

L_fiksno = 6.0        # Fiksni raspon [m]
b        = 0.30       # Sirina presjeka [m]
odnosi   = [4, 6, 8, 10, 13, 15, 20]  # L/H odnosi

rezultati_fem = []
rezultati_EB  = []

q_ukupno = 4000.0   # Pa (stalno + korisno)

for r in odnosi:
    H_i  = L_fiksno / r
    nx_i = max(30, int(60 * r / 10))
    ny_i = 10

    # FEM rezultat
    max_uy = modeliraj_prostu_gredu(L_fiksno, H_i, nx_i, ny_i)
    rezultati_fem.append(max_uy * 1000)   # mm

    # Euler-Bernoulli analiticka vrijednost:
    # delta = 5 * q_line * L^4 / (384 * E * I)
    I        = b * H_i**3 / 12.0
    q_line   = q_ukupno * b + rho * g * b * H_i   # N/m
    delta_EB = 5 * q_line * L_fiksno**4 / (384 * E * I)
    rezultati_EB.append(delta_EB * 1000)   # mm

    print(f"  {r:>4}  {H_i*100:>7.1f}  {max_uy*1000:>10.3f}  "
          f"{delta_EB*1000:>10.3f}  {max_uy/delta_EB:>7.3f}")

print("=" * 65)


# ============================================================
# GRAFICKI PRIKAZ
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graf 1: Progib
ax1.plot(odnosi, rezultati_fem, 'bo-',  linewidth=2, markersize=8,
         label='FEniCSx (FEM)')
ax1.plot(odnosi, rezultati_EB,  'r^--', linewidth=2, markersize=8,
         label='Euler-Bernoulli (analiticko)')
ax1.set_xlabel('Odnos L/H [-]', fontsize=12)
ax1.set_ylabel('Maksimalni progib [mm]', fontsize=12)
ax1.set_title('Uticaj odnosa L/H na progib\nAB grede (beton C25/30)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Graf 2: Omjer FEM/EB
omjer = [f / e for f, e in zip(rezultati_fem, rezultati_EB)]
ax2.plot(odnosi, omjer, 'gs-', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
            label='EB teorija (= 1.0)')
ax2.set_xlabel('Odnos L/H [-]', fontsize=12)
ax2.set_ylabel('FEM / EB [-]', fontsize=12)
ax2.set_title('Odnos FEM i EB rjesenja\n(Timoshenko efekat smicanja)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.8, 1.5])

plt.tight_layout()
plt.savefig('greda_LH_omjer.png', dpi=150, bbox_inches='tight')
print("\nGrafik sacuvan: greda_LH_omjer.png")
plt.show()

print("""
ZAKLJUCAK:
  L/H < 6  -> Debela greda: smicanje bitno, FEM > EB
  L/H 6-10 -> Prelazna zona
  L/H > 10 -> Tanka greda: FEM ≈ EB (smicanje zanemarivo)

  Za nasu AB gredu (L/H = 13.3):
  FEM i EB se dobro slazu -> smicanje nije dominantno.
  FEniCSx automatski ukljucuje smicajne deformacije!
""")