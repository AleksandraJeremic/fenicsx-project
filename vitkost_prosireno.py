# ============================================================
# PRIMJER: UTICAJ VITKOSTI - prosirena analiza
#
# Prosta greda raspona 5 m, opterecena ravnomjerno q = 20 kN/m,
# rjesava se za niz visina presjeka, od vitke do vrlo zdepaste
# grede (L/h od 20 do 1,5).
#
# Rezultat se poredi sa tri teorijska rjesenja:
#   Bernoulli-Ojler  - bez smicanja
#   Timosenko        - smicanje preko G i k = 5/6
#   Tacno 2D rjesenje - Timoshenko, Theory of Elasticity, jed. (34)
#
# Cilj je pokazati gdje gredne teorije prestaju da vaze.
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
b  = 0.30
E  = 31.0e9
nu = 0.2

q   = 20.0e3
t_q = q / b

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

G   = E / (2.0 * (1.0 + nu))
k_s = 5.0 / 6.0

# visine presjeka - od vitke do vrlo zdepaste grede
h_lista = [0.25, 0.30, 0.50, 0.75, 1.00, 1.25, 1.67, 2.00, 2.50, 3.33]

Ny = 8                      # elemenata po visini; Nx se bira tako da
                            # elementi ostanu priblizno kvadratni

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
    f_L = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], 0.0))
    f_D = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], L))
    dofs_L = fem.locate_dofs_topological((V.sub(1), Vy), fdim, f_L)
    dofs_D = fem.locate_dofs_topological((V.sub(1), Vy), fdim, f_D)
    dofs_ux = fem.locate_dofs_geometrical(
        (V.sub(0), Vx),
        lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], h/2.0)))

    bcs = [fem.dirichletbc(nula_y, dofs_L, V.sub(1)),
           fem.dirichletbc(nula_y, dofs_D, V.sub(1)),
           fem.dirichletbc(nula_x, dofs_ux, V.sub(0))]

    f_gore = np.sort(locate_entities_boundary(
        domen, fdim, lambda x: np.isclose(x[1], h)))
    oznake = meshtags(domen, fdim, f_gore,
                      np.full(len(f_gore), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)
    T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0*mu*epsilon(w) + lam*ufl.tr(epsilon(w))*ufl.Identity(2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a  = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    Lf = ufl.dot(T, v) * ds(1)

    problem = LinearProblem(
        a, Lf, bcs=bcs,
        petsc_options_prefix=f"vit_{int(h*100)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez

    stablo = geometry.bb_tree(domen, domen.topology.dim)
    tacka = np.array([[L/2.0, h/2.0, 0.0]], dtype=np.float64)
    kand = geometry.compute_collisions_points(stablo, tacka)
    cel = geometry.compute_colliding_cells(domen, kand, tacka)
    w = uh.eval(tacka, np.array([cel.links(0)[0]], dtype=np.int32))[1]

    n_el = domen.topology.index_map(domen.topology.dim).size_local
    return abs(w), n_el

# ============================================================
# 3) PRORACUN
# ============================================================
print("=" * 92)
print(f"UTICAJ VITKOSTI   (L = {L:.1f} m, b = {b*100:.0f} cm, "
      f"q = {q/1e3:.0f} kN/m, nu = {nu})")
print("=" * 92)
print("Greske su date u odnosu na tacno 2D rjesenje "
      "(Timoshenko, Theory of Elasticity, jed. 34)\n")
print(f"{'h [m]':>7} {'L/h':>7} {'mreza':>10} {'w MKE [mm]':>12} "
      f"{'w tacno [mm]':>13} {'gr. Bernoulli':>15} {'gr. Timosenko':>15} "
      f"{'gr. MKE':>10}")
print("-" * 92)

vit, r_mke, r_tim, r_2d = [], [], [], []
e_eb, e_tim, e_mke = [], [], []

for h in h_lista:
    I_ef = h**3 / 12.0
    w_EB  = 5.0 * t_q * L**4 / (384.0 * E * I_ef)
    w_TIM = w_EB + t_q * L**2 / (8.0 * k_s * G * h)
    w_2D  = w_EB * (1.0 + (12.0/5.0)*(h/L)**2*(4.0/5.0 + nu/2.0))

    Nx = max(8, int(round(L/h*Ny)))
    Nx += Nx % 2
    w, n_el = rijesi(h, Nx, Ny)

    vit.append(L/h)
    # razlike u odnosu na Bernoullija - koliko smicanje utice
    r_mke.append((w - w_EB)/w_EB*100.0)
    r_tim.append((w_TIM - w_EB)/w_EB*100.0)
    r_2d.append((w_2D - w_EB)/w_EB*100.0)
    # greske u odnosu na tacno rjesenje - koliko je koji model tacan
    e_eb.append((w_EB - w_2D)/w_2D*100.0)
    e_tim.append((w_TIM - w_2D)/w_2D*100.0)
    e_mke.append((w - w_2D)/w_2D*100.0)

    print(f"{h:7.2f} {L/h:7.2f} {f'{Nx}x{Ny}':>10} {w*1000:12.5f} "
          f"{w_2D*1000:13.5f} {e_eb[-1]:15.3f} {e_tim[-1]:15.3f} "
          f"{e_mke[-1]:10.3f}")
# ============================================================
# 4) GRAFIK
# ============================================================
from matplotlib.ticker import FixedLocator, FixedFormatter

vit = np.array(vit)
# greske u odnosu na tacno rjesenje, apsolutne vrijednosti
e_EB_niz  = np.abs((np.array(r_mke)*0 + 1.0/(1.0 + np.array(r_2d)/100.0) - 1.0)*100.0)
e_TIM_niz = np.abs(np.array(e_tim))
e_MKE_niz = np.abs(np.array(e_mke))

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(vit, e_EB_niz,  "o-", color="tab:red",   lw=2, ms=7, label="Bernoulli-Ojler")
ax.plot(vit, e_TIM_niz, "s-", color="tab:green", lw=2, ms=7, label="Timosenko")
ax.plot(vit, e_MKE_niz, "^-", color="tab:blue",  lw=2, ms=7, label="MKE (ravanski model)")

ax.set_xscale("log")
ax.set_yscale("log")

xt = [1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_major_formatter(FixedFormatter(["1,5", "2", "2,5", "3", "4", "5",
                                             "7", "10", "15", "20"]))
ax.xaxis.set_minor_locator(FixedLocator([]))

yt = [0.001, 0.01, 0.1, 1, 10, 100]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.yaxis.set_major_formatter(FixedFormatter(["0,001", "0,01", "0,1", "1", "10", "100"]))
ax.yaxis.set_minor_locator(FixedLocator([]))

ax.axhline(1.0, color="gray", ls=":", lw=1.5)
ax.text(19, 1.15, "greska 1 %", color="gray", fontsize=9, ha="right")

ax.set_xlabel("vitkost  L / h", fontsize=11)
ax.set_ylabel("greska u odnosu na tacno rjesenje  [%]", fontsize=11)
ax.set_title("Tacnost teorija u zavisnosti od vitkosti grede", fontsize=12)
ax.grid(alpha=0.35, which="major")
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(1.4, 22)
ax.set_ylim(0.0003, 100)

plt.tight_layout()
plt.savefig("vitkost_prosireno.png", dpi=150)
print("\nGrafik sacuvan: vitkost_prosireno.png")