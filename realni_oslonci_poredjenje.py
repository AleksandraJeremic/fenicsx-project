# ============================================================
# PRIMJER: LEZISTE KONACNE SIRINE NA DONJOJ IVICI
#
# Greda 30/30, razmak sredina lezista Lr = 5,00 m.
# Leziste sirine c nalazi se na DONJOJ ivici, na oba kraja, pa je
# ukupna duzina grede Lt = Lr + c (preko sredina lezista ostaju
# prepusti od po c/2).
#
#            q = 20 kN/m
#   | | | | | | | | | | | | | |
#   v v v v v v v v v v v v v v
#   +==========================+
#   |                          |
#   +##====================##==+
#    ^^                    ^^
#    c                      c        ## = uy = 0 na donjoj ivici
#   |<-- Lr = 5,00 m (sredine lezista) -->|
#
# Ispituje se kako sirina lezista utice na ugib, tj. koliko takav
# oslonac odstupa od zgloba.
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dolfinx import fem, geometry
from dolfinx.io import VTXWriter
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
Lr = 5.00            # razmak sredina lezista (racunski raspon) [m]
h  = 0.30            # visina presjeka [m]
b  = 0.30            # sirina presjeka [m]

E  = 31.0e9
nu = 0.2

q   = 20.0e3
t_q = q / b

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

I_ef = h**3 / 12.0

# --- granicni slucajevi po tehnickoj teoriji ---
w_prosta = 5.0 * t_q * Lr**4 / (384.0 * E * I_ef)   # prosta greda (zglobovi)
w_ukljes = 1.0 * t_q * Lr**4 / (384.0 * E * I_ef)   # obostrano ukljestena

# --- sirine lezista koje se ispituju ---
# velicina elementa 2,5 cm, pa sve sirine moraju biti umnozak toga
dx = 0.025
c_lista = [0.025, 0.05, 0.10, 0.20, 0.30, 0.40]

IZVOZ_C = 0.20       # za koju sirinu se pravi ParaView izvoz

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(c, izvoz=False):
    Lt = Lr + c                       # ukupna duzina grede
    Nx = int(round(Lt / dx))
    Ny = int(round(h / dx))

    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lt, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

    fdim = domen.topology.dim - 1

    # --- uy = 0 na donjoj ivici, duz lezista sirine c na oba kraja ---
    def lezite_lijevo(x):
        return np.logical_and(np.isclose(x[1], 0.0), x[0] <= c + 1e-9)

    def lezite_desno(x):
        return np.logical_and(np.isclose(x[1], 0.0), x[0] >= Lt - c - 1e-9)

    facets_L = locate_entities_boundary(domen, fdim, lezite_lijevo)
    facets_D = locate_entities_boundary(domen, fdim, lezite_desno)

    dofs_L = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_L)
    dofs_D = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_D)

    # --- ux = 0 u jednoj tacki (uklanja pomjeranje krutog tijela) ---
    def tacka_ux(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    dofs_ux = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_ux)

    for ime, d in [("lijevo", dofs_L), ("desno", dofs_D), ("ux", dofs_ux)]:
        if len(d[0]) == 0:
            raise RuntimeError(f"Granicni uslov '{ime}' nije pronadjen (c = {c})")

    bcs = [
        fem.dirichletbc(nula_y, dofs_L, V.sub(1)),
        fem.dirichletbc(nula_y, dofs_D, V.sub(1)),
        fem.dirichletbc(nula_x, dofs_ux, V.sub(0)),
    ]

    # --- opterecenje po cijeloj gornjoj ivici ---
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
        petsc_options_prefix=f"lez_{int(c*1000)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez
    uh.name = "pomjeranje"

    S1 = fem.functionspace(domen, ("Lagrange", 1))
    sxx = fem.Function(S1, name="sigma_xx")
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

    # --- ocitavanje ---
    stablo = geometry.bb_tree(domen, domen.topology.dim)

    def vrijednost_u_tacki(funkcija, x, y):
        tacka = np.array([[x, y, 0.0]], dtype=np.float64)
        kandidati = geometry.compute_collisions_points(stablo, tacka)
        celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
        celija = np.array([celije.links(0)[0]], dtype=np.int32)
        return funkcija.eval(tacka, celija)

    w   = abs(vrijednost_u_tacki(uh,  Lt/2.0, h/2.0)[1])   # ugib na sredini
    s_d = vrijednost_u_tacki(sxx, Lt/2.0, 1e-9)[0]         # donje vlakno, sredina
    # napon u gornjem vlaknu nad osloncem - pokazuje da li se javlja
    # negativan moment, sto je znak ukljestenja
    s_o = vrijednost_u_tacki(sxx, c/2.0, h - 1e-9)[0]

    n_el = domen.topology.index_map(domen.topology.dim).size_local

    if izvoz:
        V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
        u1 = fem.Function(V1, name="pomjeranje")
        u1.interpolate(uh)
        with VTXWriter(domen.comm, f"leziste_c{int(c*100)}.bp",
                       [u1, sxx], engine="BP4") as vtx:
            vtx.write(0.0)

    return n_el, w, s_d, s_o

# ============================================================
# 3) PETLJA PO SIRINAMA LEZISTA
# ============================================================
print("=" * 92)
print(f"UTICAJ SIRINE LEZISTA   (greda 30/{h*100:.0f}, razmak sredina "
      f"lezista {Lr:.2f} m, q = {q/1e3:.0f} kN/m)")
print("=" * 92)
print(f"\nProsta greda (zglobovi)  : w = {w_prosta*1000:.4f} mm")
print(f"Obostrano ukljestena     : w = {w_ukljes*1000:.4f} mm\n")
print(f"{'c [cm]':>8} {'c/h':>7} {'elem.':>7} {'w [mm]':>10} "
      f"{'% proste grede':>16} {'stepen ukljest. [%]':>21} {'sxx sredina':>13} "
      f"{'sxx nad osl.':>14}")
print("-" * 92)

sirine, ugibi, stepeni = [], [], []

for c in c_lista:
    izvoz = abs(c - IZVOZ_C) < 1e-9
    n_el, w, s_d, s_o = rijesi(c, izvoz)

    procenat = w / w_prosta * 100.0
    # stepen ukljestenja: 0 % = zglob, 100 % = potpuno ukljestenje
    stepen = (w_prosta - w) / (w_prosta - w_ukljes) * 100.0

    sirine.append(c)
    ugibi.append(w)
    stepeni.append(stepen)

    print(f"{c*100:8.1f} {c/h:7.2f} {n_el:7d} {w*1000:10.4f} "
          f"{procenat:16.1f} {stepen:21.1f} {s_d/1e6:13.4f} {s_o/1e6:14.4f}")

# ============================================================
# 4) GRAFIK
# ============================================================
c_cm = np.array(sirine) * 100.0
w_mm = np.array(ugibi) * 1000.0

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# --- (a) ugib u zavisnosti od sirine lezista ---
ax[0].plot(c_cm, w_mm, "o-", color="tab:blue", lw=1.8, ms=7, label="MKE")
ax[0].axhline(w_prosta*1000, color="tab:green", ls="--", lw=1.5,
              label=f"prosta greda ({w_prosta*1000:.2f} mm)")
ax[0].axhline(w_ukljes*1000, color="tab:red", ls="--", lw=1.5,
              label=f"obostrano ukljestena ({w_ukljes*1000:.2f} mm)")
ax[0].set_xlabel("sirina lezista c [cm]")
ax[0].set_ylabel("ugib u sredini raspona [mm]")
ax[0].set_title("(a) Ugib u zavisnosti od sirine lezista", fontsize=11)
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=9)

# --- (b) stepen ukljestenja ---
ax[1].plot(np.array(sirine)/h, stepeni, "o-", color="tab:purple", lw=1.8, ms=7)
ax[1].axhline(0.0, color="tab:green", ls="--", lw=1.5)
ax[1].axhline(100.0, color="tab:red", ls="--", lw=1.5)
ax[1].text(0.05, 4, "zglob", color="tab:green", fontsize=9)
ax[1].text(0.05, 94, "ukljestenje", color="tab:red", fontsize=9)
ax[1].set_xlabel("odnos sirine lezista i visine presjeka  c/h")
ax[1].set_ylabel("stepen ukljestenja [%]")
ax[1].set_title("(b) Koliko se oslonac ponasa kao ukljestenje", fontsize=11)
ax[1].set_ylim(-5, 105)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("leziste_sirina.png", dpi=150)
print("\nGrafik sacuvan: leziste_sirina.png")
print(f"ParaView izvoz: leziste_c{int(IZVOZ_C*100)}.bp")