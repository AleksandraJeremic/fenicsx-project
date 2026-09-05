# ============================================================
# PRIMJER: IZBOR FAKTORA KAZNE
#
# Ista greda 30/30 rjesava se za niz vrijednosti faktora kazne k,
# a rezultat se poredi sa Dirihleovim uslovom, koji uslov uy = 0
# ispunjava tacno.
#
# Kazneni clan djeluje kao elasticna podloga krutosti k pod krajnjom
# ivicom. Reakcija R = qL/2 = 50 kN prenosi se preko povrsine
# A = h*b = 0,09 m2, pa je pritisak p = R/A = 556 kPa, a uleknuce
# oslonca u = p/k. Faktor k se bira tako da to uleknuce bude
# zanemarivo u odnosu na ugib koji se mjeri.
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0
h  = 0.30
b  = 0.30

E  = 31.0e9
nu = 0.2

q   = 20.0e3
t_q = q / b

Nx, Ny = 40, 4

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

I_ef = h**3 / 12.0
w_EB = 5.0 * t_q * L**4 / (384.0 * E * I_ef)

# eksponenti faktora kazne: k = 10^n
eksponenti = [8, 9, 10, 11, 12, 13, 14, 15]

# ============================================================
# 2) MREZA, PROSTOR, OZNAKE
# ============================================================
domen = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)
V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

fdim = domen.topology.dim - 1

f_gore   = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h))
f_lijevo = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], 0.0))
f_desno  = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], L))
f_osl    = np.concatenate([f_lijevo, f_desno])

svi  = np.concatenate([f_gore, f_osl])
vrij = np.concatenate([np.full(len(f_gore), 1, dtype=np.int32),
                       np.full(len(f_osl),  2, dtype=np.int32)])
poredak = np.argsort(svi)
oznake = meshtags(domen, fdim, svi[poredak], vrij[poredak])
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)

T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
nula_x = fem.Function(Vx)
nula_y = fem.Function(Vy)

dofs_ux = fem.locate_dofs_geometrical(
    (V.sub(0), Vx),
    lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], h/2.0)))
bc_ux = fem.dirichletbc(nula_x, dofs_ux, V.sub(0))

dofs_uy = fem.locate_dofs_topological((V.sub(1), Vy), fdim, np.sort(f_osl))
bc_uy = fem.dirichletbc(nula_y, dofs_uy, V.sub(1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def epsilon(w):
    return ufl.sym(ufl.grad(w))

def sigma(w):
    return 2.0*mu*epsilon(w) + lam*ufl.tr(epsilon(w))*ufl.Identity(2)

stablo = geometry.bb_tree(domen, domen.topology.dim)

def vrijednost_u_tacki(funkcija, x, y):
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    kand = geometry.compute_collisions_points(stablo, tacka)
    cel = geometry.compute_colliding_cells(domen, kand, tacka)
    return funkcija.eval(tacka, np.array([cel.links(0)[0]], dtype=np.int32))

# ============================================================
# 3) FUNKCIJA: jedno rjesenje
# ============================================================
def rijesi(k=None):
    """k = None -> Dirihleov uslov;  inace kazneni pristup."""
    a_forma = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    bcs = [bc_ux]

    if k is None:
        bcs.append(bc_uy)
        prefiks = "dirihle_"
    else:
        kazna = fem.Constant(domen, np.float64(k))
        a_forma += kazna * u[1] * v[1] * ds(2)
        prefiks = f"kazna_{int(np.log10(k))}_"

    Lf = ufl.dot(T, v) * ds(1)
    problem = LinearProblem(
        a_forma, Lf, bcs=bcs,
        petsc_options_prefix=prefiks,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez

    w      = abs(vrijednost_u_tacki(uh, L/2.0, h/2.0)[1])
    uy_osl = abs(vrijednost_u_tacki(uh, 0.0, h/2.0)[1])
    return w, uy_osl

# ============================================================
# 4) PRORACUN I TABELA
# ============================================================
p_osl = (q*L/2.0) / (h*b)          # pritisak na osloncu [Pa]

print("=" * 78)
print(f"IZBOR FAKTORA KAZNE   (greda {b*100:.0f}/{h*100:.0f}, L = {L:.1f} m, "
      f"mreza {Nx}x{Ny})")
print("=" * 78)
print(f"Reakcija R = {q*L/2/1e3:.1f} kN,  povrsina oslonca A = {h*b:.3f} m2,  "
      f"pritisak p = {p_osl/1e3:.1f} kPa")
print(f"Ocekivano slijeganje oslonca:  u = p/k\n")

w_dir, uy_dir = rijesi(None)
print(f"Dirihleov uslov : w = {w_dir*1000:.5f} mm   "
      f"(slijeganje oslonca = {uy_dir*1000:.2e} mm)\n")

print(f"{'k':>10} {'w [mm]':>11} {'razlika od Dirihlea [%]':>25} "
      f"{'slijeganje [mm]':>16} {'p/k [mm]':>12}")
print("-" * 78)

n_niz, w_niz, u_niz = [], [], []

for n in eksponenti:
    k = 10.0**n
    w, uy = rijesi(k)
    n_niz.append(n); w_niz.append(w); u_niz.append(uy)
    print(f"{f'10^{n}':>10} {w*1000:11.5f} {(w-w_dir)/w_dir*100:25.4f} "
          f"{uy*1000:16.3e} {p_osl/k*1000:12.3e}")

# ============================================================
# 5) GRAFIK
# ============================================================
n_niz = np.array(n_niz); w_niz = np.array(w_niz)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.plot(n_niz, w_niz*1000, "o-", color="tab:blue", lw=2, ms=7,
        label="kazneni pristup")
ax.axhline(w_dir*1000, color="tab:red", ls="--", lw=1.8,
           label=f"Dirihleov uslov ({w_dir*1000:.4f} mm)")

ax.set_xlabel("eksponent faktora kazne   (k = 10 na n)", fontsize=11)
ax.set_ylabel("ugib u sredini raspona [mm]", fontsize=11)
ax.set_title("Konvergencija kaznenog pristupa ka Dirihleovom rjesenju",
             fontsize=12)
ax.set_xticks(eksponenti)
ax.set_ylim(7.7, 9.5)
ax.grid(alpha=0.35)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("faktor_kazne.png", dpi=150)
print("\nGrafik sacuvan: faktor_kazne.png")