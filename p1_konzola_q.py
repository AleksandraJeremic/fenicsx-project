# ============================================================
# PRIMJER P1: KONZOLA opterecena jednako podijeljenim opterecenjem
#
#   q = 20 kN/m
#   | | | | | | | | | |
#   v v v v v v v v v v
#   #====================
#   #                    |
#   #====================
#   ^ ukljestenje (x = 0)     slobodan kraj (x = L)
#
# Za svaku kombinaciju raspona L i visine presjeka h proracun se
# ponavlja za vise gustina mreze, pa se rezultat poredi sa
# tehnickom teorijom savijanja (TTS):
#
#   ugib slobodnog kraja :  w = q*L^4 / (8*E*I)
#   napon               :  sigma = M*(h/2)/I,  M = q*(L-x)^2/2
#
# Ocekivanje: sto je greda zdepastija (manji odnos L/h), to je
# razlika prema TTS veca, jer TTS zanemaruje smicucu deformaciju.
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
b  = 0.3             # sirina presjeka (debljina 2D modela) [m]
E  = 31.0e9          # modul elasticnosti [Pa]
nu = 0.2             # Poissonov koeficijent

q   = 20.0e3         # linijsko opterecenje [N/m]
t_q = q / b          # povrsinski pritisak za plane stress [Pa]

mu  = E / (2.0 * (1.0 + nu))     # Lameove konstante - plane stress
lam = E * nu / (1.0 - nu**2)

# --- geometrije koje poredimo ---
L_lista  = [5.0, 10.0]                   # rasponi [m]
h_lista  = [0.5, 0.75, 1.0, 1.25]        # visine presjeka [m]

# --- gustine mreze (broj elemenata po duzini grede) ---
Nx_lista = [10, 20, 40, 80, 160]

# --- geometrija za koju se pravi ParaView izvoz ---
IZVOZ_L, IZVOZ_H, IZVOZ_NX = 5.0, 0.5, 40

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(L, h, Nx, izvoz=False):
    """Rjesava konzolu za zadate L, h i gustinu mreze.
       Vraca broj elemenata, ugib slobodnog kraja i sigma_xx
       u kontrolnom presjeku."""

    # --- mreza: Ny biramo tako da elementi budu priblizno kvadratni ---
    Ny = max(1, int(round(Nx * h / L)))

    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

    # --- GRANICNI USLOV: ukljestenje cijele lijeve ivice (x = 0) ---
    # Blokiraju se OBJE komponente pomjeranja duz cijele ivice, pa se
    # koristi locate_dofs_topological nad punim prostorom V.
    fdim = domen.topology.dim - 1
    facets_ukljestenje = locate_entities_boundary(
        domen, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_ukljestenje = fem.locate_dofs_topological(V, fdim, facets_ukljestenje)

    nula_vek = fem.Function(V)            # vektorska nula (ux = uy = 0)
    bcs = [fem.dirichletbc(nula_vek, dofs_ukljestenje)]

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
        petsc_options_prefix=f"P1_{int(L)}_{int(h*100)}_{Nx}_",
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

    # ugib slobodnog kraja, na tezisnoj osi
    # (ukljestenje je zadato duz cijele ivice, pa nema singularnosti
    #  kao kod tackastog oslonca - apsolutna vrijednost je upotrebljiva)
    w = abs(vrijednost_u_tacki(uh, L - 1e-9, h / 2.0)[1])

    # napon u kontrolnom presjeku x = h/2 od ukljestenja
    # (u samom uglu ukljestenja postoji singularnost, pa se napon
    #  ocitava na maloj udaljenosti od njega, izvan poremecene zone)
    x_k = h / 2.0
    s_k = vrijednost_u_tacki(sxx, x_k, h - 1e-9)[0]

    # napon u samom ukljestenju - samo informativno, u tabeli
    s_u = vrijednost_u_tacki(sxx, 1e-9, h - 1e-9)[0]

    n_el = domen.topology.index_map(domen.topology.dim).size_local

    # --- izvoz za ParaView, samo za odabranu geometriju ---
    if izvoz:
        V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
        u1 = fem.Function(V1, name="pomjeranje")
        u1.interpolate(uh)
        with VTXWriter(domen.comm, "P1_konzola_q.bp",
                       [u1, sxx], engine="BP4") as vtx:
            vtx.write(0.0)

    return n_el, Ny, w, s_k, s_u, x_k

# ============================================================
# 3) PETLJA PO GEOMETRIJAMA I MREZAMA
# ============================================================
rezultati = {}      # kljuc: (L, h) -> lista po mrezama

print("=" * 78)
print("PRIMJER P1: KONZOLA, jednako podijeljeno opterecenje q = 20 kN/m")
print("=" * 78)

for L in L_lista:
    for h in h_lista:
        I_ef = h**3 / 12.0                      # moment inercije po jed. debljine

        # --- TEHNICKA TEORIJA SAVIJANJA ---
        w_tts = t_q * L**4 / (8.0 * E * I_ef)   # ugib slobodnog kraja
        x_k = h / 2.0
        M_k  = t_q * (L - x_k)**2 / 2.0         # moment u kontrolnom presjeku
        s_tts = M_k * (h/2.0) / I_ef            # napon u donjem vlaknu
        M_u   = t_q * L**2 / 2.0                # moment u ukljestenju
        s_tts_u = M_u * (h/2.0) / I_ef

        print(f"\n--- L = {L:.0f} m,  h = {h:.2f} m,  L/h = {L/h:.1f} ---")
        print(f"TTS:  w = {w_tts*1000:.4f} mm   "
              f"sigma(x=h/2) = {s_tts/1e6:.4f} MPa   "
              f"sigma(ukljestenje) = {s_tts_u/1e6:.4f} MPa")
        print(f"{'Nx':>6} {'Ny':>4} {'elem.':>7} {'w [mm]':>11} {'razlika w [%]':>15} "
              f"{'sxx [MPa]':>11} {'razlika s [%]':>15} {'sxx uklj. [MPa]':>17}")

        niz = []
        for Nx in Nx_lista:
            izvoz = (L == IZVOZ_L and h == IZVOZ_H and Nx == IZVOZ_NX)
            n_el, Ny, w, s_k, s_u, _ = rijesi(L, h, Nx, izvoz)

            r_w = (w - w_tts) / w_tts * 100.0
            r_s = (s_k - s_tts) / s_tts * 100.0

            niz.append((Nx, n_el, w, r_w, s_k, r_s))
            print(f"{Nx:>6} {Ny:>4} {n_el:>7} {w*1000:>11.5f} {r_w:>15.3f} "
                  f"{s_k/1e6:>11.4f} {r_s:>15.3f} {s_u/1e6:>17.4f}")

        rezultati[(L, h)] = niz

# ============================================================
# 4) GRAFICI
# ============================================================
# Raspored: gornji red = ugib, donji red = napon
#           lijevo = L = 5 m, desno = L = 10 m
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
poz = np.arange(len(Nx_lista))          # kategorijske pozicije na x-osi
boje = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

for j, L in enumerate(L_lista):
    for i, h in enumerate(h_lista):
        niz = rezultati[(L, h)]
        r_w = [r[3] for r in niz]
        r_s = [r[5] for r in niz]
        oznaka = f"h = {h:.2f} m  (L/h = {L/h:.0f})"

        ax[0, j].plot(poz, r_w, "o-", color=boje[i], lw=1.6, ms=4, label=oznaka)
        ax[1, j].plot(poz, r_s, "o-", color=boje[i], lw=1.6, ms=4, label=oznaka)

    for red, naslov in [(0, "Ugib slobodnog kraja"), (1, "Normalni napon")]:
        ax[red, j].set_xticks(poz)
        ax[red, j].set_xticklabels([str(n) for n in Nx_lista])
        ax[red, j].set_xlabel("broj konacnih elemenata po duzini grede")
        ax[red, j].set_ylabel("relativna razlika u odnosu na TTS [%]")
        ax[red, j].set_title(f"{naslov} - konzola, L = {L:.0f} m", fontsize=11)
        ax[red, j].axhline(0.0, color="k", lw=1, ls=":")
        ax[red, j].grid(alpha=0.3)
        ax[red, j].legend(fontsize=8)

plt.tight_layout()
plt.savefig("P1_konzola_q.png", dpi=150)
print("\nGrafik sacuvan: P1_konzola_q.png")
print("ParaView izvoz: P1_konzola_q.bp  (L = 5 m, h = 0.5 m, mreza 40)")