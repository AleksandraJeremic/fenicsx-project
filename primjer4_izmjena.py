# ============================================================
# PRIMJER 4: Konvergencija rjesenja - CETVOROUGAONI elementi
#
# Oslonci ostaju TACKASTI. Prati se i apsolutni i relativni ugib.
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
h  = 0.5
b  = 0.3

E  = 31.0e9
nu = 0.2

q   = 20.0e3
t_q = q / b

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

# ============================================================
# 2) REFERENTNA RJESENJA
# ============================================================
I_ef  = h**3 / 12.0
A_ef  = h
G     = E / (2.0 * (1.0 + nu))
k_s   = 5.0 / 6.0

w_EB   = 5.0 * t_q * L**4 / (384.0 * E * I_ef)
w_smik = t_q * L**2 / (8.0 * k_s * G * A_ef)
w_teor = w_EB + w_smik

M_teor   = t_q * L**2 / 8.0
sxx_teor = M_teor * (h/2.0) / I_ef

mreze = [(10, 1), (20, 2), (40, 4), (80, 8), (160, 16)]

# ============================================================
# 3) FUNKCIJA ZA JEDAN PRORACUN
# ============================================================
def rijesi(Nx, Ny):
    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,     # <<< CETVOROUGAONI elementi
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

    # --- granicni uslovi: TACKASTI oslonci ---
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

    def tacka_A(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    def tacka_B(x):
        return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

    bcs = [
        fem.dirichletbc(nula_x, fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A), V.sub(0)),
        fem.dirichletbc(nula_y, fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A), V.sub(1)),
        fem.dirichletbc(nula_y, fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B), V.sub(1)),
    ]

    # --- opterecenje ---
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

    # --- ocitavanje ---
    def vrijednost_u_tacki(funkcija, x, y):
        tacka = np.array([[x, y, 0.0]], dtype=np.float64)
        stablo = geometry.bb_tree(domen, domen.topology.dim)
        kandidati = geometry.compute_collisions_points(stablo, tacka)
        celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
        celija = np.array([celije.links(0)[0]], dtype=np.int32)
        return funkcija.eval(tacka, celija)

    n_cel = domen.topology.index_map(domen.topology.dim).size_local
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # --- apsolutni i relativni ugib (vidi komentar u Primjeru 3) ---
    w_sred = vrijednost_u_tacki(uh, L/2.0, h/2.0)[1]
    w_A    = vrijednost_u_tacki(uh, 0.0,   h/2.0)[1]
    w_B    = vrijednost_u_tacki(uh, L,     h/2.0)[1]

    w_aps = abs(w_sred)
    w_rel = abs(w_sred - 0.5 * (w_A + w_B))

    sx = vrijednost_u_tacki(sxx, L/2.0, 0.0)[0]

    return n_cel, n_dof, w_aps, w_rel, sx

# ============================================================
# 4) PETLJA PO MREZAMA + TABELA
# ============================================================
broj_cel, ugibi_aps, ugibi_rel, naponi = [], [], [], []

print("CETVOROUGAONI ELEMENTI (Lagrange 2), tackasti oslonci\n")
print(f"Euler-Bernoulli    : w = {w_EB*1000:.4f} mm")
print(f"Doprinos smicanja  : w = {w_smik*1000:.4f} mm")
print(f"Timosenko (referenca): w = {w_teor*1000:.4f} mm")
print(f"Napon (referenca)  : sigma_xx = {sxx_teor/1e6:.4f} MPa\n")

print(f"{'Nx x Ny':>10} {'celija':>8} {'DOF':>8} "
      f"{'w_aps [mm]':>12} {'gr. [%]':>9} "
      f"{'w_rel [mm]':>12} {'gr. [%]':>9} "
      f"{'sxx [MPa]':>11} {'gr. [%]':>9}")
print("-" * 96)

for (Nx, Ny) in mreze:
    n_cel, n_dof, w_aps, w_rel, sx = rijesi(Nx, Ny)

    gr_aps = (w_aps - w_teor) / w_teor * 100.0
    gr_rel = (w_rel - w_teor) / w_teor * 100.0
    gr_s   = (sx - sxx_teor) / sxx_teor * 100.0

    broj_cel.append(n_cel)
    ugibi_aps.append(w_aps)
    ugibi_rel.append(w_rel)
    naponi.append(sx)

    print(f"{Nx:>5} x{Ny:>3} {n_cel:>8} {n_dof:>8} "
          f"{w_aps*1000:>12.5f} {gr_aps:>9.3f} "
          f"{w_rel*1000:>12.5f} {gr_rel:>9.3f} "
          f"{sx/1e6:>11.4f} {gr_s:>9.3f}")

# ============================================================
# 5) GRAFICI
# ============================================================
broj_cel  = np.array(broj_cel)
ugibi_aps = np.array(ugibi_aps)
ugibi_rel = np.array(ugibi_rel)
naponi    = np.array(naponi)

greska_rel = np.abs(ugibi_rel - w_teor) / w_teor * 100.0
greska_s   = np.abs(naponi - sxx_teor) / sxx_teor * 100.0

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].plot(broj_cel, ugibi_aps * 1000, "o--", color="tab:gray",
           label="apsolutni ugib (divergira)")
ax[0].plot(broj_cel, ugibi_rel * 1000, "o-", color="tab:green",
           label="relativni ugib (konvergira)")
ax[0].axhline(w_teor * 1000, color="r", ls="--", label="Timosenko")
ax[0].axhline(w_EB * 1000, color="k", ls=":", label="Euler-Bernoulli")
ax[0].set_xscale("log")
ax[0].set_xlabel("Broj konacnih elemenata")
ax[0].set_ylabel("Ugib u sredini raspona [mm]")
ax[0].set_title("Konvergencija ugiba - cetvorougaoni elementi")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

ax[1].loglog(broj_cel, greska_rel, "o-", color="tab:green",
             label="greska relativnog ugiba")
ax[1].loglog(broj_cel, greska_s, "s-", color="tab:orange",
             label="greska sigma_xx")
ax[1].set_xlabel("Broj konacnih elemenata")
ax[1].set_ylabel("Relativna greska [%]")
ax[1].set_title("Greska u odnosu na referentno rjesenje")
ax[1].grid(True, which="both", alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.savefig("primjer4_izmjena.png", dpi=150)
print("\nGrafik sacuvan: primjer4_izmjena.png")