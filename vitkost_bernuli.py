# ============================================================
# PRIMJER: UTICAJ VITKOSTI GREDE NA TACNOST BERNOULLIJEVE TEORIJE
#
# Prosta greda raspona L = 5 m, opterecena ravnomjerno q = 20 kN/m.
# Oslonci su zadati duz cijele krajnje ivice (postavka provjerena u
# prethodnom primjeru), pa nema singularnosti i ugib konvergira.
#
# Mijenja se SAMO visina presjeka h, cime se mijenja vitkost L/h.
# Za svaku vitkost se racuna koliko rjesenje ravanskog modela odstupa
# od Bernoulli-Ojlerovog, kako bi se odredilo od koje vitkosti se
# smicuca deformacija moze zanemariti.
#
# Ocekivanje: razlika opada s KVADRATOM vitkosti, po zakonu
#   razlika = (12/5)*(h/L)^2*(4/5 + nu/2) = 2,16*(h/L)^2   za nu = 0,2
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
L  = 5.0             # raspon [m]  - ostaje nepromijenjen
b  = 0.30            # sirina presjeka [m]

E  = 31.0e9
nu = 0.2

q   = 20.0e3
t_q = q / b

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

G   = E / (2.0 * (1.0 + nu))
k_s = 5.0 / 6.0

# visine presjeka koje se ispituju
h_lista = [1.00, 0.75, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15]

# broj elemenata po visini; po duzini se bira tako da elementi budu kvadratni
Ny = 4

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(h, Nx, Ny):
    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

    fdim = domen.topology.dim - 1

    # --- uy = 0 duz CIJELE lijeve i desne krajnje ivice ---
    facets_L = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], 0.0))
    facets_D = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], L))
    dofs_L_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_L)
    dofs_D_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_D)

    # --- ux = 0 u jednoj tacki, da presjek moze da se obrce ---
    def tacka_ux(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], h/2.0))
    dofs_ux = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_ux)

    bcs = [
        fem.dirichletbc(nula_y, dofs_L_y, V.sub(1)),
        fem.dirichletbc(nula_y, dofs_D_y, V.sub(1)),
        fem.dirichletbc(nula_x, dofs_ux,  V.sub(0)),
    ]

    # --- opterecenje ---
    facets_gore = np.sort(locate_entities_boundary(
        domen, fdim, lambda x: np.isclose(x[1], h)))
    oznake = meshtags(domen, fdim, facets_gore,
                      np.full(len(facets_gore), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)
    T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

    # --- formulacija ---
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a  = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    Lf = ufl.dot(T, v) * ds(1)

    problem = LinearProblem(
        a, Lf, bcs=bcs,
        petsc_options_prefix=f"vitkost_{int(h*100)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez

    stablo = geometry.bb_tree(domen, domen.topology.dim)

    def vrijednost_u_tacki(funkcija, x, y):
        tacka = np.array([[x, y, 0.0]], dtype=np.float64)
        kandidati = geometry.compute_collisions_points(stablo, tacka)
        celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
        celija = np.array([celije.links(0)[0]], dtype=np.int32)
        return funkcija.eval(tacka, celija)

    w = abs(vrijednost_u_tacki(uh, L/2.0, h/2.0)[1])
    n_el = domen.topology.index_map(domen.topology.dim).size_local
    return w, n_el

# ============================================================
# 3) PETLJA PO VISINAMA PRESJEKA
# ============================================================
print("=" * 100)
print(f"UTICAJ VITKOSTI NA TACNOST BERNOULLIJEVE TEORIJE   "
      f"(L = {L:.1f} m, b = {b*100:.0f} cm, q = {q/1e3:.0f} kN/m)")
print("=" * 100)
print(f"{'h [m]':>7} {'L/h':>7} {'mreza':>10} {'elem.':>7} "
      f"{'w MKE [mm]':>12} {'w Bernoulli':>13} {'razlika [%]':>13} "
      f"{'Timosenko [%]':>15}")
print("-" * 100)

vitkosti, razlike, razlike_TIM = [], [], []

for h in h_lista:
    Nx = max(20, int(round(L / h * Ny)))
    Nx += Nx % 2                       # da postoji cvor tacno na sredini raspona

    I_ef = h**3 / 12.0
    w_EB = 5.0 * t_q * L**4 / (384.0 * E * I_ef)

    # predvidjanja teorija
    w_smik = t_q * L**2 / (8.0 * k_s * G * h)
    r_TIM  = w_smik / w_EB * 100.0

    w, n_el = rijesi(h, Nx, Ny)
    r_mke = (w - w_EB) / w_EB * 100.0

    vitkosti.append(L/h)
    razlike.append(r_mke)
    razlike_TIM.append(r_TIM)

    print(f"{h:7.2f} {L/h:7.2f} {f'{Nx}x{Ny}':>10} {n_el:>7} "
          f"{w*1000:12.5f} {w_EB*1000:13.5f} {r_mke:13.3f} "
          f"{r_TIM:15.3f}")

# ============================================================
# 5) GRAFIK
# ============================================================
vit = np.array(vitkosti)
raz = np.array(razlike)
raz_T = np.array(razlike_TIM)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.plot(vit, raz, "o-", color="tab:blue", lw=1.8, ms=7,
        label="MKE u odnosu na Bernoullija")
ax.plot(vit, raz_T, "s--", color="tab:green", lw=1.5, ms=5,
        label="Timosenko (doprinos smicanja)")

for prag, boja in [(5.0, "tab:red"), (1.0, "tab:orange")]:
    ax.axhline(prag, color=boja, ls=":", lw=1.5)
    ax.text(33, prag + 0.15, f"{prag:.0f} %", color=boja, fontsize=9, ha="right")

ax.set_xlabel("vitkost L/h")
ax.set_ylabel("razlika u odnosu na Bernoullija [%]")
ax.set_title("Razlika u zavisnosti od vitkosti grede", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("vitkost_bernoulli.png", dpi=150)
print("\nGrafik sacuvan: vitkost_bernoulli.png")