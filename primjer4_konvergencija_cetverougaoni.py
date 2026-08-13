# ============================================================
# PRIMJER 4: Analiza konvergencije rjesenja u zavisnosti od
#            broja konacnih elemenata - CETVOROUGAONI elementi
#
# Isti postupak kao u Primjeru 3, samo sa quadrilateral mrezom.
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")          # crtanje u fajl, bez prozora
import matplotlib.pyplot as plt

from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0            # raspon [m]
h  = 0.5            # visina presjeka [m]
b  = 0.3            # sirina presjeka [m]

E  = 31.0e9         # modul elasticnosti [Pa]
nu = 0.2

q   = 20.0e3        # linijsko opterecenje [N/m]
t_q = q / b         # povrsinski pritisak [Pa]

mu  = E / (2.0 * (1.0 + nu))     # Lame - plane stress
lam = E * nu / (1.0 - nu**2)

# --- ANALITICKO RJESENJE (referenca) ---
I_ef      = h**3 / 12.0                               # moment inercije po jed. debljine
w_teor    = 5.0 * t_q * L**4 / (384.0 * E * I_ef)     # maks. ugib
M_teor    = t_q * L**2 / 8.0                          # maks. moment
sxx_teor  = M_teor * (h/2.0) / I_ef                   # napon u krajnjem vlaknu

# --- NIZ MREZA ---
mreze = [(10, 1), (20, 2), (40, 4), (80, 8), (160, 16)]

# ============================================================
# 2) FUNKCIJA ZA JEDAN PRORACUN
# ============================================================
def rijesi(Nx, Ny):
    """Rjesava gredu za mrezu Nx x Ny (cetvorougaoni elementi) i vraca
       (broj celija, broj DOF-ova, ugib u sredini, sigma_xx u donjem vlaknu)."""

    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,     # <<< CETVOROUGAONI elementi
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

    # --- oslonci ---
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

    def tacka_A(x):    # nepomicni zglob (0,0)
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    def tacka_B(x):    # pomicni zglob (L,0)
        return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

    bcs = [
        fem.dirichletbc(nula_x, fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A), V.sub(0)),
        fem.dirichletbc(nula_y, fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A), V.sub(1)),
        fem.dirichletbc(nula_y, fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B), V.sub(1)),
    ]

    # --- opterecenje po gornjoj ivici ---
    fdim = domen.topology.dim - 1
    facets_gore = np.sort(
        locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h))
    )
    oznake = meshtags(domen, fdim, facets_gore,
                      np.full(len(facets_gore), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)
    T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

    # --- varijaciona formulacija ---
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a  = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    Lf = ufl.dot(T, v) * ds(1)

    # --- solver ---
    problem = LinearProblem(
        a, Lf, bcs=bcs,
        petsc_options_prefix=f"konv_quad_{Nx}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez

    # --- napon ---
    S1 = fem.functionspace(domen, ("Lagrange", 1))
    sxx = fem.Function(S1)
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

    # --- ocitavanje u tacki ---
    def vrijednost_u_tacki(funkcija, x, y):
        tacka = np.array([[x, y, 0.0]], dtype=np.float64)
        stablo = geometry.bb_tree(domen, domen.topology.dim)
        kandidati = geometry.compute_collisions_points(stablo, tacka)
        celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
        celija = np.array([celije.links(0)[0]], dtype=np.int32)
        return funkcija.eval(tacka, celija)

    n_cel = domen.topology.index_map(domen.topology.dim).size_local
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    w  = vrijednost_u_tacki(uh,  L/2.0, h/2.0)[1]     # ugib na sredini
    sx = vrijednost_u_tacki(sxx, L/2.0, 0.0)[0]       # napon u donjem vlaknu

    return n_cel, n_dof, abs(w), sx

# ============================================================
# 3) PETLJA PO MREZAMA + TABELA
# ============================================================
broj_cel, broj_dof, ugibi, naponi = [], [], [], []

print("CETVOROUGAONI ELEMENTI (Lagrange 2)")
print(f"Analiticki (Euler-Bernoulli): w = {w_teor*1000:.4f} mm, "
      f"sigma_xx = {sxx_teor/1e6:.4f} MPa\n")
print(f"{'Nx x Ny':>10} {'celija':>8} {'DOF':>8} {'w [mm]':>12} {'greska w [%]':>14} "
      f"{'sxx [MPa]':>12} {'greska s [%]':>14}")
print("-" * 84)

for (Nx, Ny) in mreze:
    n_cel, n_dof, w, sx = rijesi(Nx, Ny)

    gr_w = (w - w_teor) / w_teor * 100.0        # relativna greska ugiba
    gr_s = (sx - sxx_teor) / sxx_teor * 100.0   # relativna greska napona

    broj_cel.append(n_cel)
    broj_dof.append(n_dof)
    ugibi.append(w)
    naponi.append(sx)

    print(f"{Nx:>5} x{Ny:>3} {n_cel:>8} {n_dof:>8} {w*1000:>12.5f} {gr_w:>14.3f} "
          f"{sx/1e6:>12.4f} {gr_s:>14.3f}")

# ============================================================
# 4) GRAFICI KONVERGENCIJE
# ============================================================
ugibi    = np.array(ugibi)
naponi   = np.array(naponi)
broj_cel = np.array(broj_cel)

greska_w = np.abs(ugibi - w_teor) / w_teor * 100.0
greska_s = np.abs(naponi - sxx_teor) / sxx_teor * 100.0

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# Lijevo: vrijednost ugiba prema broju elemenata
ax[0].plot(broj_cel, ugibi * 1000, "o-", color="tab:green", label="MKE (cetvorougaoni)")
ax[0].axhline(w_teor * 1000, color="r", ls="--", label="Analiticki (Euler-Bernoulli)")
ax[0].set_xscale("log")
ax[0].set_xlabel("Broj konacnih elemenata")
ax[0].set_ylabel("Ugib u sredini raspona [mm]")
ax[0].set_title("Konvergencija ugiba - cetvorougaoni elementi")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

# Desno: relativna greska u log-log razmjeri
ax[1].loglog(broj_cel, greska_w, "o-", color="tab:green", label="greska ugiba")
ax[1].loglog(broj_cel, greska_s, "s-", color="tab:orange", label="greska sigma_xx")
ax[1].set_xlabel("Broj konacnih elemenata")
ax[1].set_ylabel("Relativna greska [%]")
ax[1].set_title("Relativna greska u odnosu na analiticko rjesenje")
ax[1].grid(True, which="both", alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.savefig("primjer4_konvergencija_cetvorougaoni.png", dpi=150)
print("\nGrafik sacuvan: primjer4_konvergencija_cetvorougaoni.png")