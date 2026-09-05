# ============================================================
# PRIMJER: OBLIK KONACNOG ELEMENTA - trougaoni ili cetvorougaoni
#
# Ista greda, isti granicni uslovi i isti niz mreza rjesavaju se
# trougaonim i cetvorougaonim elementima, oba Lagranzovog tipa
# drugog stepena.
#
# Poredjenje se radi u odnosu na BROJ STEPENI SLOBODE, a ne broj
# elemenata, jer broj nepoznatih odredjuje velicinu matrice krutosti
# i vrijeme proracuna. Trougaona mreza ima dvostruko vise elemenata
# od cetvorougaone pri istom broju cvorova, pa bi poredjenje po broju
# elemenata bilo neposteno.
#
# Referenca je tacno 2D rjesenje (Timoshenko, Theory of Elasticity,
# jed. 34, str. 43).
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0
h  = 0.50
b  = 0.30

E  = 31.0e9
nu = 0.2

q   = 20.0e3
t_q = q / b

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

I_ef = h**3 / 12.0
w_EB = 5.0 * t_q * L**4 / (384.0 * E * I_ef)
w_2D = w_EB * (1.0 + (12.0/5.0)*(h/L)**2*(4.0/5.0 + nu/2.0))   # tacno rjesenje

mreze = [(10, 1), (20, 2), (40, 4), (80, 8), (160, 16)]

tipovi = [(CellType.triangle,      "trougaoni",     "o-", "tab:blue"),
          (CellType.quadrilateral, "cetvorougaoni", "s-", "tab:green")]

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(tip, Nx, Ny):
    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=tip,
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

    # ux se blokira u donjem lijevom uglu - taj cvor postoji u svakoj mrezi
    dofs_ux = fem.locate_dofs_geometrical(
        (V.sub(0), Vx),
        lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)))

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
        petsc_options_prefix=f"oblik_{Nx}_{int(tip == CellType.triangle)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez

    stablo = geometry.bb_tree(domen, domen.topology.dim)
    tacka = np.array([[L/2.0, h/2.0, 0.0]], dtype=np.float64)
    kand = geometry.compute_collisions_points(stablo, tacka)
    cel = geometry.compute_colliding_cells(domen, kand, tacka)
    w = abs(uh.eval(tacka, np.array([cel.links(0)[0]], dtype=np.int32))[1])

    n_el  = domen.topology.index_map(domen.topology.dim).size_local
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    return w, n_el, n_dof

# ============================================================
# 3) PRORACUN
# ============================================================
print("=" * 86)
print(f"OBLIK KONACNOG ELEMENTA   (greda {b*100:.0f}/{h*100:.0f}, L = {L:.1f} m, "
      f"Lagranz 2)")
print("=" * 86)
print(f"Tacno 2D rjesenje: w = {w_2D*1000:.5f} mm\n")

rezultati = {}

for tip, ime, stil, boja in tipovi:
    print(f"--- {ime.upper()} ELEMENTI ---")
    print(f"{'mreza':>10} {'elem.':>8} {'DOF':>8} {'w [mm]':>12} {'greska [%]':>12}")
    print("-" * 56)
    dofovi, greske = [], []
    for (Nx, Ny) in mreze:
        w, n_el, n_dof = rijesi(tip, Nx, Ny)
        gr = (w - w_2D)/w_2D*100.0
        dofovi.append(n_dof); greske.append(gr)
        print(f"{f'{Nx}x{Ny}':>10} {n_el:>8} {n_dof:>8} {w*1000:>12.5f} {gr:>12.4f}")
    rezultati[ime] = (dofovi, greske, stil, boja)
    print()

# ============================================================
# 4) GRAFIK
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

for ime, (dofovi, greske, stil, boja) in rezultati.items():
    ax.plot(dofovi, np.abs(greske), stil, color=boja, lw=2, ms=8, label=ime)

ax.set_xscale("log")
ax.set_yscale("log")

xt = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_major_formatter(FixedFormatter(
    ["100", "200", "500", "1000", "2000", "5000", "10000", "20000"]))
ax.xaxis.set_minor_locator(FixedLocator([]))

yt = [0.001, 0.01, 0.1, 1]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.yaxis.set_major_formatter(FixedFormatter(["0,001", "0,01", "0,1", "1"]))
ax.yaxis.set_minor_locator(FixedLocator([]))

ax.set_xlabel("broj stepeni slobode", fontsize=11)
ax.set_ylabel("greska u odnosu na tacno rjesenje  [%]", fontsize=11)
ax.set_title("Uticaj oblika konacnog elementa", fontsize=12)
ax.grid(alpha=0.35, which="major")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("oblik_elementa.png", dpi=150)
print("Grafik sacuvan: oblik_elementa.png")