# ============================================================
# PRIMJER: UKLANJANJE SINGULARNOSTI - oslonac duz cijele ivice
#
# Ista greda kao u primjerima 1-4 (L = 5 m, presjek 30/50 cm),
# isto opterecenje, ista mreza. Mijenja se SAMO nacin zadavanja
# oslonca, i to se poredi sa prvobitnom postavkom:
#
#   "tacka" - uy blokirano u JEDNOM cvoru (prvobitno)
#   "ivica" - uy blokirano duz CIJELE krajnje vertikalne ivice
#
# U oba slucaja ux je blokirano samo u jednoj tacki, da bi krajnji
# presjek mogao slobodno da se obrce - inace bi oslonac postao
# ukljestenje i sistem vise ne bi bio prosta greda.
#
# Referenca: Bernoulli-Ojler, w = 5*q*L^4/(384*E*I)
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
L  = 5.0             # raspon [m]
h  = 0.50            # visina presjeka [m]
b  = 0.30            # sirina presjeka [m]

E  = 31.0e9          # modul elasticnosti [Pa]
nu = 0.2             # Poissonov koeficijent

q   = 20.0e3         # linijsko opterecenje [N/m]
t_q = q / b          # povrsinski pritisak za plane stress [Pa]

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

I_ef = h**3 / 12.0

# --- referentne vrijednosti po tehnickoj teoriji (Bernoulli-Ojler) ---
w_EB     = 5.0 * t_q * L**4 / (384.0 * E * I_ef)
M_sr     = t_q * L**2 / 8.0
sxx_EB   = M_sr * (h/2.0) / I_ef

mreze = [(10, 1), (20, 2), (40, 4), (80, 8), (160, 16)]

# geometrija/mreza za koju se radi ParaView izvoz
IZVOZ_MREZA = (40, 4)

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(Nx, Ny, tip_oslonca, izvoz=False):
    """tip_oslonca: 'tacka' ili 'ivica'"""

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

    if tip_oslonca == "tacka":
        # --- PRVOBITNA POSTAVKA: uy blokirano u jednom cvoru ---
        def tacka_A(x):
            return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

        def tacka_B(x):
            return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

        bcs = [
            fem.dirichletbc(nula_x,
                fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A), V.sub(0)),
            fem.dirichletbc(nula_y,
                fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A), V.sub(1)),
            fem.dirichletbc(nula_y,
                fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B), V.sub(1)),
        ]

    else:
        # --- NOVA POSTAVKA: uy blokirano duz CIJELE krajnje ivice ---
        # Marker funkcija ima samo JEDAN uslov (x = 0, odnosno x = L),
        # pa obuhvata cijelu vertikalnu ivicu, od donjeg do gornjeg vlakna.
        # Dofovi se nalaze topoloski, preko granicnih strana elemenata.
        facets_L = locate_entities_boundary(domen, fdim,
                                            lambda x: np.isclose(x[0], 0.0))
        facets_D = locate_entities_boundary(domen, fdim,
                                            lambda x: np.isclose(x[0], L))

        dofs_L_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_L)
        dofs_D_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_D)

        # ux ostaje blokirano u SAMO JEDNOJ tacki - u tezistu lijevog
        # presjeka. Da je blokirano duz cijele ivice, presjek se ne bi
        # mogao obrtati i oslonac bi postao ukljestenje.
        def tacka_ux(x):
            return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], h/2.0))

        dofs_ux = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_ux)

        bcs = [
            fem.dirichletbc(nula_y, dofs_L_y, V.sub(1)),
            fem.dirichletbc(nula_y, dofs_D_y, V.sub(1)),
            fem.dirichletbc(nula_x, dofs_ux,  V.sub(0)),
        ]

    # --- OPTERECENJE po gornjoj ivici ---
    facets_gore = np.sort(locate_entities_boundary(
        domen, fdim, lambda x: np.isclose(x[1], h)))
    oznake = meshtags(domen, fdim, facets_gore,
                      np.full(len(facets_gore), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)
    T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

    # --- VARIJACIONA FORMULACIJA ---
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
        petsc_options_prefix=f"osl_{tip_oslonca}_{Nx}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez
    uh.name = "pomjeranje"

    # --- NAPON ---
    S1 = fem.functionspace(domen, ("Lagrange", 1))
    sxx = fem.Function(S1, name="sigma_xx")
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

    # --- OCITAVANJE ---
    stablo = geometry.bb_tree(domen, domen.topology.dim)

    def vrijednost_u_tacki(funkcija, x, y):
        tacka = np.array([[x, y, 0.0]], dtype=np.float64)
        kandidati = geometry.compute_collisions_points(stablo, tacka)
        celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
        celija = np.array([celije.links(0)[0]], dtype=np.int32)
        return funkcija.eval(tacka, celija)

    # APSOLUTNI ugib - bez ikakve korekcije
    w = abs(vrijednost_u_tacki(uh, L/2.0, h/2.0)[1])
    s = vrijednost_u_tacki(sxx, L/2.0, 1e-9)[0]

    n_el  = domen.topology.index_map(domen.topology.dim).size_local
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    if izvoz:
        V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
        u1 = fem.Function(V1, name="pomjeranje")
        u1.interpolate(uh)
        with VTXWriter(domen.comm, f"oslonac_{tip_oslonca}.bp",
                       [u1, sxx], engine="BP4") as vtx:
            vtx.write(0.0)

    return n_el, n_dof, w, s

# ============================================================
# 3) PETLJA PO OBA TIPA OSLONCA
# ============================================================
print("=" * 88)
print("UTICAJ NACINA ZADAVANJA OSLONCA NA KONVERGENCIJU")
print(f"Greda L = {L:.1f} m, presjek {b*100:.0f}/{h*100:.0f} cm, "
      f"cetvorougaoni elementi, Lagranz 2")
print("=" * 88)
print(f"\nBernoulli-Ojler:  w = {w_EB*1000:.4f} mm,  "
      f"sigma_xx = {sxx_EB/1e6:.4f} MPa")

rezultati = {}

for tip, opis in [("tacka", "PRVOBITNO: uy blokirano u jednoj tacki"),
                  ("ivica", "NOVO: uy blokirano duz cijele krajnje ivice")]:

    print(f"\n--- {opis} ---")
    print(f"{'Nx x Ny':>10} {'elem.':>7} {'DOF':>7} {'w [mm]':>11} "
          f"{'prirastaj':>11} {'razlika EB [%]':>16} {'sxx [MPa]':>11} "
          f"{'razlika [%]':>13}")

    niz = []
    w_prethodno = None

    for (Nx, Ny) in mreze:
        izvoz = ((Nx, Ny) == IZVOZ_MREZA)
        n_el, n_dof, w, s = rijesi(Nx, Ny, tip, izvoz)

        prirastaj = "-" if w_prethodno is None else f"{(w - w_prethodno)*1000:.5f}"
        w_prethodno = w

        r_w = (w - w_EB) / w_EB * 100.0
        r_s = (s - sxx_EB) / sxx_EB * 100.0

        niz.append((Nx, Ny, n_el, n_dof, w, r_w, s, r_s))
        print(f"{Nx:>5} x{Ny:>3} {n_el:>7} {n_dof:>7} {w*1000:>11.5f} "
              f"{prirastaj:>11} {r_w:>16.3f} {s/1e6:>11.4f} {r_s:>13.3f}")

    rezultati[tip] = niz

# ============================================================
# 4) GRAFIK
# ============================================================
poz = np.arange(len(mreze))
oznake_x = [f"{Nx}x{Ny}" for (Nx, Ny) in mreze]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

stil = {"tacka": ("o--", "tab:gray",  "oslonac u tacki (prvobitno)"),
        "ivica": ("o-",  "tab:blue",  "oslonac duz cijele ivice")}

# --- (a) apsolutni ugib ---
for tip, niz in rezultati.items():
    m, c, lab = stil[tip]
    ax[0].plot(poz, [r[4]*1000 for r in niz], m, color=c, lw=1.8, ms=5, label=lab)
ax[0].axhline(w_EB*1000, color="r", ls="--", lw=1.5, label="Bernoulli-Ojler")
ax[0].set_xticks(poz); ax[0].set_xticklabels(oznake_x)
ax[0].set_xlabel("mreza konacnih elemenata")
ax[0].set_ylabel("apsolutni ugib u sredini raspona [mm]")
ax[0].set_title("(a) Konvergencija ugiba", fontsize=11)
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)

# --- (b) relativna razlika prema TTS ---
for tip, niz in rezultati.items():
    m, c, lab = stil[tip]
    ax[1].plot(poz, [r[5] for r in niz], m, color=c, lw=1.8, ms=5, label=lab)
ax[1].set_xticks(poz); ax[1].set_xticklabels(oznake_x)
ax[1].set_xlabel("mreza konacnih elemenata")
ax[1].set_ylabel("razlika u odnosu na Bernoulli-Ojlera [%]")
ax[1].set_title("(b) Relativna razlika", fontsize=11)
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("oslonac_ivica_konvergencija.png", dpi=150)
print("\nGrafik sacuvan: oslonac_ivica_konvergencija.png")
print("ParaView izvozi: oslonac_tacka.bp  i  oslonac_ivica.bp  (mreza 40x4)")