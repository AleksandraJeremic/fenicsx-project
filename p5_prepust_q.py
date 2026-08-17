# ============================================================
# PRIMJER P5: GREDA SA PREPUSTOM, jednako podijeljeno opterecenje
#
#              q = 20 kN/m  (po cijeloj duzini, i preko prepusta)
#   | | | | | | | | | | | | | | | | | | | |
#   v v v v v v v v v v v v v v v v v v v v
#   +=================================+=====+
#   |                                 |     |
#   +=================================+=====+
#   ^                                 o
#   A (0,0)                        B (L,0)   slobodan kraj (L+c)
#   |<---------- raspon L ----------->|<-c->|      c = L/4
#
# TTS (reakcije):  R_B = q*(L+c)^2/(2L),   R_A = q*(L+c) - R_B
#     moment u rasponu:  M(x) = R_A*x - q*x^2/2
#     ugib na sredini raspona:
#         w = 5*q*L^4/(384*E*I) - M0*L^2/(16*E*I),   M0 = q*c^2/2
#       (prvi clan je od opterecenja u rasponu, drugi je uticaj
#        negativnog momenta koji prepust unosi nad osloncem B)
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

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

L_lista  = [5.0, 10.0]                # RASPON izmedju oslonaca [m]
h_lista  = [0.5, 0.75, 1.0, 1.25]
Nx_lista = [10, 20, 40, 80, 160]      # svi djeljivi sa 10 -> cvorovi
                                      # postoje tacno na x = L i x = L/2

IZVOZ_L, IZVOZ_H, IZVOZ_NX = 5.0, 0.5, 40

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(L, h, Nx, izvoz=False):
    c  = L / 4.0                 # duzina prepusta
    Lt = L + c                   # ukupna duzina grede

    Ny = max(1, int(round(Nx * h / Lt)))

    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lt, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

    # --- GRANICNI USLOVI: tackasti oslonci A (x=0) i B (x=L) ---
    # Prepust od x = L do x = Lt ostaje slobodan.
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

    def tacka_A(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    def tacka_B(x):
        return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

    dofs_A_x = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A)
    dofs_A_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A)
    dofs_B_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B)

    for ime, d in [("A-ux", dofs_A_x), ("A-uy", dofs_A_y), ("B-uy", dofs_B_y)]:
        if len(d[0]) == 0:
            raise RuntimeError(f"Oslonac {ime} nije pronadjen (L={L}, Nx={Nx})")

    bcs = [
        fem.dirichletbc(nula_x, dofs_A_x, V.sub(0)),
        fem.dirichletbc(nula_y, dofs_A_y, V.sub(1)),
        fem.dirichletbc(nula_y, dofs_B_y, V.sub(1)),
    ]

    # --- OPTERECENJE po cijeloj gornjoj ivici (ukljucujuci prepust) ---
    fdim = domen.topology.dim - 1
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
        petsc_options_prefix=f"P5_{int(L)}_{int(h*100)}_{Nx}_",
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

    # Datum: srednje uy u presjecima nad osloncima (x = 0 i x = L),
    # samo unutrasnje tacke, da se izbjegnu singularne tacke oslanjanja
    y_uz = np.linspace(0.0, h, 9)[1:-1]
    uy_A = np.mean([vrijednost_u_tacki(uh, 0.0, yy)[1] for yy in y_uz])
    uy_B = np.mean([vrijednost_u_tacki(uh, L,   yy)[1] for yy in y_uz])
    datum = 0.5 * (uy_A + uy_B)

    # ugib na sredini raspona
    uy_mid = vrijednost_u_tacki(uh, L/2.0, h/2.0)[1]
    w_rel = abs(uy_mid - datum)

    # ugib slobodnog kraja prepusta (moze biti i naviše - informativno)
    uy_kraj = vrijednost_u_tacki(uh, Lt - 1e-9, h/2.0)[1] - datum

    # napon: donje vlakno na sredini raspona (pozitivan moment)
    s_mid = vrijednost_u_tacki(sxx, L/2.0, 1e-9)[0]

    # napon: gornje vlakno blizu oslonca B (negativan moment)
    # ocitava se na h/2 od oslonca, izvan njegove poremecene zone
    x_B = L - h/2.0
    s_B = vrijednost_u_tacki(sxx, x_B, h - 1e-9)[0]

    n_el = domen.topology.index_map(domen.topology.dim).size_local

    if izvoz:
        V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
        u1 = fem.Function(V1, name="pomjeranje")
        u1.interpolate(uh)
        with VTXWriter(domen.comm, "P5_prepust_q.bp", [u1, sxx], engine="BP4") as vtx:
            vtx.write(0.0)

    return n_el, Ny, w_rel, uy_kraj, s_mid, s_B

# ============================================================
# 3) PETLJA PO GEOMETRIJAMA I MREZAMA
# ============================================================
rezultati = {}

print("=" * 94)
print("PRIMJER P5: GREDA SA PREPUSTOM (c = L/4), jednako podijeljeno "
      "opterecenje q = 20 kN/m")
print("=" * 94)

for L in L_lista:
    c  = L / 4.0
    Lt = L + c

    # --- reakcije po TTS ---
    R_B = t_q * Lt**2 / (2.0 * L)
    R_A = t_q * Lt - R_B

    for h in h_lista:
        I_ef = h**3 / 12.0
        ch   = h / 2.0

        # --- TTS: ugib na sredini raspona ---
        M0    = t_q * c**2 / 2.0                   # negativan moment nad B
        w_tts = (5.0 * t_q * L**4 / (384.0 * E * I_ef)
                 - M0 * L**2 / (16.0 * E * I_ef))

        # --- TTS: naponi ---
        M_mid  = R_A * (L/2.0) - t_q * (L/2.0)**2 / 2.0     # moment na L/2
        s_tts  = M_mid * ch / I_ef

        x_B    = L - h/2.0
        M_B    = R_A * x_B - t_q * x_B**2 / 2.0             # moment blizu B
        s_tts_B = -M_B * ch / I_ef      # gornje vlakno: obrnut znak od momenta

        # --- TTS: ugib kraja prepusta (informativno) ---
        w_tts_kraj = (t_q * c / (E * I_ef)) * (c**3/8.0 + c**2*L/6.0 - L**3/24.0)

        print(f"\n--- L = {L:.0f} m,  c = {c:.2f} m,  h = {h:.2f} m,  "
              f"L/h = {L/h:.1f} ---")
        print(f"TTS:  w(L/2) = {w_tts*1000:.4f} mm   "
              f"sigma(L/2) = {s_tts/1e6:.4f} MPa   "
              f"sigma(uz B) = {s_tts_B/1e6:.4f} MPa   "
              f"w(kraj prepusta) = {w_tts_kraj*1000:+.4f} mm")
        print(f"{'Nx':>6} {'Ny':>4} {'elem.':>7} {'w [mm]':>11} {'razl. w [%]':>13} "
              f"{'sxx(L/2)':>11} {'razl. s [%]':>13} {'sxx(uz B)':>11} "
              f"{'w kraj [mm]':>13}")

        niz = []
        for Nx in Nx_lista:
            izvoz = (L == IZVOZ_L and h == IZVOZ_H and Nx == IZVOZ_NX)
            n_el, Ny, w_rel, w_kraj, s_mid, s_B = rijesi(L, h, Nx, izvoz)

            r_w = (w_rel - w_tts) / w_tts * 100.0
            r_s = (s_mid - s_tts) / s_tts * 100.0

            niz.append((Nx, n_el, w_rel, r_w, s_mid, r_s))
            print(f"{Nx:>6} {Ny:>4} {n_el:>7} {w_rel*1000:>11.5f} {r_w:>13.3f} "
                  f"{s_mid/1e6:>11.4f} {r_s:>13.3f} {s_B/1e6:>11.4f} "
                  f"{w_kraj*1000:>+13.5f}")

        rezultati[(L, h)] = niz

# ============================================================
# 4) GRAFICI
# ============================================================
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
poz = np.arange(len(Nx_lista))
boje = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

for j, L in enumerate(L_lista):
    for i, h in enumerate(h_lista):
        niz = rezultati[(L, h)]
        r_w = [r[3] for r in niz]
        r_s = [r[5] for r in niz]
        oznaka = f"h = {h:.2f} m  (L/h = {L/h:.0f})"
        ax[0, j].plot(poz, r_w, "o-", color=boje[i], lw=1.6, ms=4, label=oznaka)
        ax[1, j].plot(poz, r_s, "o-", color=boje[i], lw=1.6, ms=4, label=oznaka)

    for red, naslov in [(0, "Ugib na sredini raspona"), (1, "Normalni napon")]:
        ax[red, j].set_xticks(poz)
        ax[red, j].set_xticklabels([str(n) for n in Nx_lista])
        ax[red, j].set_xlabel("broj konacnih elemenata po duzini grede")
        ax[red, j].set_ylabel("relativna razlika u odnosu na TTS [%]")
        ax[red, j].set_title(f"{naslov} - greda sa prepustom, L = {L:.0f} m",
                             fontsize=11)
        ax[red, j].axhline(0.0, color="k", lw=1, ls=":")
        ax[red, j].grid(alpha=0.3)
        ax[red, j].legend(fontsize=8)

plt.tight_layout()
plt.savefig("P5_prepust_q.png", dpi=150)
print("\nGrafik sacuvan: P5_prepust_q.png")
print("ParaView izvoz: P5_prepust_q.bp  (L = 5 m, h = 0.5 m, mreza 40)")