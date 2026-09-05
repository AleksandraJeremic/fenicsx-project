# ============================================================
# PRIMJER: IZBOR KONACNOG ELEMENTA
#
# Ista greda se rjesava sa cetiri vrste konacnih elemenata:
#   - trougaoni,     Lagranz 1. stepena
#   - cetvorougaoni, Lagranz 1. stepena
#   - trougaoni,     Lagranz 2. stepena
#   - cetvorougaoni, Lagranz 2. stepena
#
# Svaka vrsta se rjesava kroz niz mreza, a greska se prikazuje u
# zavisnosti od BROJA STEPENI SLOBODE. Broj nepoznatih odredjuje
# velicinu matrice krutosti i vrijeme proracuna, dok trougaona mreza
# pri istom broju cvorova ima dvostruko vise elemenata - pa bi
# poredjenje po broju elemenata bilo neopravdano.
#
# Referenca je tacno 2D rjesenje (Timoshenko, Theory of Elasticity,
# jed. 34), w = 1,71640 mm.
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
w_2D = w_EB * (1.0 + (12.0/5.0)*(h/L)**2*(4.0/5.0 + nu/2.0))

# elementi prvog stepena traze gusce mreze, pa im se dodaje jos jedna
mreze_1 = [(10, 1), (20, 2), (40, 4), (80, 8), (160, 16), (320, 32)]
mreze_2 = [(10, 1), (20, 2), (40, 4), (80, 8), (160, 16)]

# (tip celije, stepen, mreze, naziv, boja, marker)
slucajevi = [
    (CellType.triangle,      1, mreze_1, "trougaoni, Lagranz 1",     "tab:red",   "o"),
    (CellType.quadrilateral, 1, mreze_1, "cetvorougaoni, Lagranz 1", "tab:orange","s"),
    (CellType.triangle,      2, mreze_2, "trougaoni, Lagranz 2",     "tab:blue",  "o"),
    (CellType.quadrilateral, 2, mreze_2, "cetvorougaoni, Lagranz 2", "tab:green", "s"),
]

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(tip, stepen, Nx, Ny):
    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=tip,
    )
    V = fem.functionspace(domen, ("Lagrange", stepen, (domen.geometry.dim,)))

    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

    fdim = domen.topology.dim - 1
    f_L = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], 0.0))
    f_D = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], L))
    dofs_L = fem.locate_dofs_topological((V.sub(1), Vy), fdim, f_L)
    dofs_D = fem.locate_dofs_topological((V.sub(1), Vy), fdim, f_D)

    # ux se sprjecava u donjem lijevom uglu - taj cvor postoji u svakoj mrezi
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
        petsc_options_prefix=f"el_{stepen}_{Nx}_{int(tip == CellType.triangle)}_",
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
print("=" * 74)
print(f"IZBOR KONACNOG ELEMENTA   (greda {b*100:.0f}/{h*100:.0f}, L = {L:.1f} m)")
print("=" * 74)
print(f"Tacno 2D rjesenje: w = {w_2D*1000:.5f} mm\n")

rezultati = {}

for tip, stepen, mreze, naziv, boja, marker in slucajevi:
    print(f"--- {naziv.upper()} ---")
    print(f"{'mreza':>10} {'elem.':>8} {'DOF':>8} {'w [mm]':>12} {'greska [%]':>12}")
    print("-" * 56)
    dofovi, greske = [], []
    for (Nx, Ny) in mreze:
        w, n_el, n_dof = rijesi(tip, stepen, Nx, Ny)
        gr = (w - w_2D)/w_2D*100.0
        dofovi.append(n_dof); greske.append(gr)
        print(f"{f'{Nx}x{Ny}':>10} {n_el:>8} {n_dof:>8} {w*1000:>12.5f} {gr:>12.4f}")
    rezultati[naziv] = (dofovi, greske, boja, marker)
    print()

# ============================================================
# 4) GRAFIK
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 6.5))

for naziv, (dofovi, greske, boja, marker) in rezultati.items():
    ax.plot(dofovi, np.abs(greske), marker + "-", color=boja, lw=2, ms=7,
            label=naziv)

ax.set_xscale("log")
ax.set_yscale("log")

xt = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
ax.xaxis.set_major_locator(FixedLocator(xt))
ax.xaxis.set_major_formatter(FixedFormatter(
    ["50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000"]))
ax.xaxis.set_minor_locator(FixedLocator([]))

yt = [0.001, 0.01, 0.1, 1, 10, 100]
ax.yaxis.set_major_locator(FixedLocator(yt))
ax.yaxis.set_major_formatter(FixedFormatter(["0,001", "0,01", "0,1", "1", "10", "100"]))
ax.yaxis.set_minor_locator(FixedLocator([]))

ax.axhline(1.0, color="gray", ls=":", lw=1.5)
ax.text(19000, 1.15, "greska 1 %", color="gray", fontsize=9, ha="right")

ax.set_xlabel("broj stepeni slobode", fontsize=11)
ax.set_ylabel("greska u odnosu na tacno rjesenje  [%]", fontsize=11)
ax.set_title("Tacnost u zavisnosti od vrste konacnog elementa", fontsize=12)
ax.grid(alpha=0.35, which="major")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("izbor_elementa.png", dpi=150)
print("Grafik sacuvan: izbor_elementa.png")