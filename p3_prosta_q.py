# ============================================================
# PRIMJER P3: PROSTA GREDA opterecena jednako podijeljenim opterecenjem
#
#            q = 20 kN/m
#   | | | | | | | | | | | | | |
#   v v v v v v v v v v v v v v
#   +==========================+
#   |                          |
#   +==========================+
#   ^                          o
#   A (0,0)                  B (L,0)
#   nepomicni zglob          pomicni zglob
#
# Oslonci su tackasti, na donjoj ivici. Zbog singularnosti u tacki
# oslanjanja apsolutni ugib ne konvergira, pa se ugib mjeri RELATIVNO -
# u odnosu na srednje vertikalno pomjeranje krajnjih presjeka.
#
# TTS:  w = 5*q*L^4 / (384*E*I)
#       sigma = M*(h/2)/I,  M = q*L^2/8   (donje vlakno zategnuto)
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

L_lista  = [5.0, 10.0]
h_lista  = [0.5, 0.75, 1.0, 1.25]
Nx_lista = [10, 20, 40, 80, 160]

IZVOZ_L, IZVOZ_H, IZVOZ_NX = 5.0, 0.5, 40

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(L, h, Nx, izvoz=False):
    # Nx mora biti paran da cvor postoji tacno na sredini raspona
    Ny = max(1, int(round(Nx * h / L)))

    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

    # --- GRANICNI USLOVI: tackasti oslonci na donjoj ivici ---
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

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

    # --- OPTERECENJE po gornjoj ivici ---
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
        petsc_options_prefix=f"P3_{int(L)}_{int(h*100)}_{Nx}_",
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

    # Datum za relativni ugib: srednje uy krajnjih presjeka.
    # Uzorkuju se SAMO unutrasnje tacke presjeka (bez y = 0 i y = h),
    # da se izbjegne tacka oslanjanja u kojoj je polje singularno.
    y_uz = np.linspace(0.0, h, 9)[1:-1]
    uy_A = np.mean([vrijednost_u_tacki(uh, 0.0, yy)[1] for yy in y_uz])
    uy_B = np.mean([vrijednost_u_tacki(uh, L,   yy)[1] for yy in y_uz])
    datum = 0.5 * (uy_A + uy_B)

    uy_mid = vrijednost_u_tacki(uh, L/2.0, h/2.0)[1]
    w_rel = abs(uy_mid - datum)          # konvergira
    w_aps = abs(uy_mid)                  # ne konvergira (informativno)

    # Napon: donje vlakno na sredini raspona (zategnuto)
    s_k = vrijednost_u_tacki(sxx, L/2.0, 1e-9)[0]

    n_el = domen.topology.index_map(domen.topology.dim).size_local

    if izvoz:
        V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
        u1 = fem.Function(V1, name="pomjeranje")
        u1.interpolate(uh)
        with VTXWriter(domen.comm, "P3_prosta_q.bp", [u1, sxx], engine="BP4") as vtx:
            vtx.write(0.0)

    return n_el, Ny, w_rel, w_aps, s_k

# ============================================================
# 3) PETLJA PO GEOMETRIJAMA I MREZAMA
# ============================================================
rezultati = {}

print("=" * 86)
print("PRIMJER P3: PROSTA GREDA, jednako podijeljeno opterecenje q = 20 kN/m")
print("=" * 86)

for L in L_lista:
    for h in h_lista:
        I_ef = h**3 / 12.0

        # --- TEHNICKA TEORIJA SAVIJANJA ---
        w_tts = 5.0 * t_q * L**4 / (384.0 * E * I_ef)
        M_sr  = t_q * L**2 / 8.0
        s_tts = M_sr * (h/2.0) / I_ef

        print(f"\n--- L = {L:.0f} m,  h = {h:.2f} m,  L/h = {L/h:.1f} ---")
        print(f"TTS:  w = {w_tts*1000:.4f} mm   sigma = {s_tts/1e6:.4f} MPa")
        print(f"{'Nx':>6} {'Ny':>4} {'elem.':>7} {'w_rel [mm]':>12} {'razlika w [%]':>15} "
              f"{'w_aps [mm]':>12} {'sxx [MPa]':>11} {'razlika s [%]':>15}")

        niz = []
        for Nx in Nx_lista:
            izvoz = (L == IZVOZ_L and h == IZVOZ_H and Nx == IZVOZ_NX)
            n_el, Ny, w_rel, w_aps, s_k = rijesi(L, h, Nx, izvoz)

            r_w = (w_rel - w_tts) / w_tts * 100.0
            r_s = (s_k - s_tts) / s_tts * 100.0

            niz.append((Nx, n_el, w_rel, r_w, s_k, r_s))
            print(f"{Nx:>6} {Ny:>4} {n_el:>7} {w_rel*1000:>12.5f} {r_w:>15.3f} "
                  f"{w_aps*1000:>12.5f} {s_k/1e6:>11.4f} {r_s:>15.3f}")

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
        ax[red, j].set_title(f"{naslov} - prosta greda, q, L = {L:.0f} m", fontsize=11)
        ax[red, j].axhline(0.0, color="k", lw=1, ls=":")
        ax[red, j].grid(alpha=0.3)
        ax[red, j].legend(fontsize=8)

plt.tight_layout()
plt.savefig("P3_prosta_q.png", dpi=150)
print("\nGrafik sacuvan: P3_prosta_q.png")
print("ParaView izvoz: P3_prosta_q.bp  (L = 5 m, h = 0.5 m, mreza 40)")