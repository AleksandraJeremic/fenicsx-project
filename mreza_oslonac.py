# ============================================================
# PROVJERA: zavisi li rezultat od gustine mreze i od nacina
#           zadavanja uslova uy = 0 duz krajnje ivice
#
# Ista greda se rjesava kroz niz mreza, i to dva puta:
#   - Dirihleovim uslovom, koji uy = 0 ispunjava tacno
#   - kaznenim pristupom sa k = 1e13
#
# Cilj je provjeriti trazi li neki od ta dva pristupa gusću mrezu.
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl

from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0
h  = 0.30
b  = 0.30

E  = 31.0e9
nu = 0.2

q   = 20.0e3
t_q = q / b

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

KAZNA = 1.0e13
mreze = [(20, 2), (40, 4), (80, 8), (160, 16)]

# ============================================================
# 2) FUNKCIJA: jedan proracun
# ============================================================
def rijesi(Nx, Ny, kazna=None):
    """kazna = None -> Dirihleov uslov;  inace kazneni pristup."""
    domen = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, h])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )
    V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))
    S1 = fem.functionspace(domen, ("Lagrange", 1))

    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    nula_x = fem.Function(Vx)
    nula_y = fem.Function(Vy)

    fdim = domen.topology.dim - 1
    f_gore = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h))
    f_L    = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], 0.0))
    f_D    = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], L))
    f_osl  = np.concatenate([f_L, f_D])

    svi  = np.concatenate([f_gore, f_osl])
    vrij = np.concatenate([np.full(len(f_gore), 1, dtype=np.int32),
                           np.full(len(f_osl),  2, dtype=np.int32)])
    p = np.argsort(svi)
    oznake = meshtags(domen, fdim, svi[p], vrij[p])
    ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)
    T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0*mu*epsilon(w) + lam*ufl.tr(epsilon(w))*ufl.Identity(2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dofs_ux = fem.locate_dofs_geometrical(
        (V.sub(0), Vx),
        lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], h/2.0)))
    bcs = [fem.dirichletbc(nula_x, dofs_ux, V.sub(0))]

    a_forma = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    if kazna is None:
        dofs_uy = fem.locate_dofs_topological((V.sub(1), Vy), fdim, np.sort(f_osl))
        bcs.append(fem.dirichletbc(nula_y, dofs_uy, V.sub(1)))
        prefiks = f"dir_{Nx}_"
    else:
        kk = fem.Constant(domen, np.float64(kazna))
        a_forma += kk * u[1] * v[1] * ds(2)
        prefiks = f"kaz_{Nx}_"

    Lf = ufl.dot(T, v) * ds(1)
    problem = LinearProblem(
        a_forma, Lf, bcs=bcs,
        petsc_options_prefix=prefiks,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez

    # napon u uglu oslonca - provjera ima li tamo koncentracije
    sxx = fem.Function(S1)
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

    stablo = geometry.bb_tree(domen, domen.topology.dim)

    def u_tacki(f, x, y):
        t = np.array([[x, y, 0.0]], dtype=np.float64)
        kand = geometry.compute_collisions_points(stablo, t)
        cel = geometry.compute_colliding_cells(domen, kand, t)
        return f.eval(t, np.array([cel.links(0)[0]], dtype=np.int32))

    w     = abs(u_tacki(uh, L/2.0, h/2.0)[1])
    s_ugao = u_tacki(sxx, 1e-9, 1e-9)[0]
    n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    return w, s_ugao, n_dof

# ============================================================
# 3) PRORACUN
# ============================================================
print("=" * 88)
print(f"ZAVISNOST OD GUSTINE MREZE   (greda {b*100:.0f}/{h*100:.0f}, L = {L:.1f} m)")
print("=" * 88)
print(f"{'mreza':>9} {'DOF':>8} | {'Dirihle w [mm]':>16} {'sxx ugao [MPa]':>16} "
      f"| {'kazna w [mm]':>14} {'sxx ugao [MPa]':>16}")
print("-" * 88)

for (Nx, Ny) in mreze:
    w1, s1, nd = rijesi(Nx, Ny, None)
    w2, s2, _  = rijesi(Nx, Ny, KAZNA)
    print(f"{f'{Nx}x{Ny}':>9} {nd:>8} | {w1*1000:16.5f} {s1/1e6:16.4f} "
          f"| {w2*1000:14.5f} {s2/1e6:16.4f}")