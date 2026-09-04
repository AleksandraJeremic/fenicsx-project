# ============================================================
# PRIMJER: PORTALNI RAM - ravanski model i uticaj graničnih uslova
#
#            q = 20 kN/m
#   | | | | | | | | | | | | | |
#   v v v v v v v v v v v v v v
#   +==========================+   <- rigla, presjek 30/30
#   |#|                      |#|
#   |#|                      |#|   <- stubovi, presjek 30/30
#   |#|                      |#|
#   ###                      ###   <- stope
#   |<---- raspon 5,00 m ---->|       (razmak osa stubova)
#
# Ram se rjesava kao ravanski problem, istim postupkom kao i grede.
# Dodatno se ispituje kako se ponasaju tri nacina zadavanja oslonca
# u stopi, i rezultati se porede sa klasicnim stapnim modelom.
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl

from dolfinx import fem, geometry
from dolfinx.io import VTXWriter
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import (create_rectangle, create_submesh, locate_entities,
                          locate_entities_boundary, meshtags, CellType)

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
Lr = 5.00            # raspon rama - razmak osa stubova [m]
Hr = 3.00            # visina rama - od stope do ose rigle [m]
t  = 0.30            # debljina stubova i rigle (visina presjeka) [m]
b  = 0.30            # sirina presjeka (debljina 2D modela) [m]

E  = 31.0e9
nu = 0.2

q   = 20.0e3         # opterecenje na rigli [N/m]
t_q = q / b          # povrsinski pritisak [Pa]

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

dx_el = 0.05         # velicina konacnog elementa [m] -> 6 elemenata po presjeku

# --- izvedene dimenzije okvirnog pravougaonika ---
Bx = Lr + t          # ukupna sirina  = 5,30 m
By = Hr + t/2.0      # ukupna visina  = 3,15 m

x_os_L = t/2.0       # osa lijevog stuba  = 0,15 m
x_os_D = Bx - t/2.0  # osa desnog stuba   = 5,15 m
y_don  = Hr - t/2.0  # donje vlakno rigle = 2,85 m
x_lice = t           # unutrasnje lice lijevog stuba = 0,30 m

tol = 1e-9

# ============================================================
# 2) STAPNI (1D) MODEL - referentne vrijednosti
# ============================================================
def stapni_ram(ukljestenje=True):
    """Metoda deformacija za portalni ram. Vraca moment nad cvorom,
       moment u sredini rigle i ugib sredine rigle."""
    A = b * t
    I = b * t**3 / 12.0
    cvor = np.array([[0.0, 0.0], [0.0, Hr], [Lr, Hr], [Lr, 0.0]])
    stap = [(0, 1), (1, 2), (3, 2)]

    def ke_lok(Le):
        k = np.zeros((6, 6))
        k[0, 0] = k[3, 3] = E*A/Le
        k[0, 3] = k[3, 0] = -E*A/Le
        c = E*I
        k[1, 1] = k[4, 4] = 12*c/Le**3
        k[1, 4] = k[4, 1] = -12*c/Le**3
        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = 6*c/Le**2
        k[2, 4] = k[4, 2] = k[4, 5] = k[5, 4] = -6*c/Le**2
        k[2, 2] = k[5, 5] = 4*c/Le
        k[2, 5] = k[5, 2] = 2*c/Le
        return k

    K = np.zeros((12, 12))
    F = np.zeros(12)
    for (i, j) in stap:
        dxx, dyy = cvor[j] - cvor[i]
        Le = np.hypot(dxx, dyy)
        cc, ss = dxx/Le, dyy/Le
        T = np.zeros((6, 6))
        R = np.array([[cc, ss, 0], [-ss, cc, 0], [0, 0, 1]])
        T[:3, :3] = R
        T[3:, 3:] = R
        d = np.array([3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2])
        K[np.ix_(d, d)] += T.T @ ke_lok(Le) @ T
        if (i, j) == (1, 2):
            F[d] += np.array([0.0, -q*Le/2, -q*Le**2/12,
                              0.0, -q*Le/2,  q*Le**2/12])

    fiksni = [0, 1, 2, 9, 10, 11] if ukljestenje else [0, 1, 9, 10]
    slob = np.setdiff1d(np.arange(12), fiksni)
    u = np.zeros(12)
    u[slob] = np.linalg.solve(K[np.ix_(slob, slob)], F[slob])

    d = np.array([3, 4, 5, 6, 7, 8])
    fe = np.array([0.0, -q*Lr/2, -q*Lr**2/12, 0.0, -q*Lr/2, q*Lr**2/12])
    s = ke_lok(Lr) @ u[d] - fe
    M_cvor = -s[2]
    M_sred = (M_cvor + s[5])/2 + q*Lr**2/8
    w_sred = (5*q*Lr**4/(384*E*I)
              - abs(M_cvor) * Lr**2/(8*E*I))       # oba kraja
    return M_cvor, M_sred, w_sred

M_cvor_u, M_sred_u, w_u = stapni_ram(True)
M_cvor_z, M_sred_z, w_z = stapni_ram(False)

# ============================================================
# 3) MREZA U OBLIKU RAMA
# ============================================================
# Prvo se pravi mreza preko okvirnog pravougaonika, pa se iz nje
# izdvajaju samo elementi koji leze unutar rama.
Nx = int(round(Bx / dx_el))
Ny = int(round(By / dx_el))

domen_pun = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([Bx, By])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)

def u_ramu(x):
    """True za tacke koje pripadaju ramu (dva stuba + rigla)."""
    stub_L = x[0] <= t + tol
    stub_D = x[0] >= Bx - t - tol
    rigla  = x[1] >= y_don - tol
    return np.logical_or(np.logical_or(stub_L, stub_D), rigla)

tdim = domen_pun.topology.dim
celije = locate_entities(domen_pun, tdim, u_ramu)
domen = create_submesh(domen_pun, tdim, celije)[0]

fdim = domen.topology.dim - 1
domen.topology.create_connectivity(fdim, domen.topology.dim)

V  = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))
S1 = fem.functionspace(domen, ("Lagrange", 1))
V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))

print("=" * 78)
print("PORTALNI RAM - ravanski model")
print("=" * 78)
print(f"Raspon {Lr:.2f} m, visina {Hr:.2f} m, presjeci {b*100:.0f}/{t*100:.0f} cm")
print(f"Velicina elementa {dx_el*100:.0f} cm  ->  {int(round(t/dx_el))} elemenata po presjeku")
print(f"Broj elemenata : {domen.topology.index_map(tdim).size_local}")
print(f"Broj DOF-ova   : {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

# ============================================================
# 4) OPTERECENJE NA GORNJOJ IVICI RIGLE
# ============================================================
facets_gore = np.sort(locate_entities_boundary(
    domen, fdim, lambda x: np.isclose(x[1], By)))
oznake = meshtags(domen, fdim, facets_gore,
                  np.full(len(facets_gore), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)
T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

# ============================================================
# 5) FORMULACIJA
# ============================================================
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(2)

u_tr = ufl.TrialFunction(V)
v_te = ufl.TestFunction(V)
a  = ufl.inner(sigma(u_tr), epsilon(v_te)) * ufl.dx
Lf = ufl.dot(T, v_te) * ds(1)

Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
nula_x = fem.Function(Vx)
nula_y = fem.Function(Vy)
nula_v = fem.Function(V)

stablo = geometry.bb_tree(domen, tdim)

def vrijednost_u_tacki(funkcija, x, y):
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    kand = geometry.compute_collisions_points(stablo, tacka)
    cel = geometry.compute_colliding_cells(domen, kand, tacka)
    return funkcija.eval(tacka, np.array([cel.links(0)[0]], dtype=np.int32))

# ============================================================
# 6) TRI NACINA ZADAVANJA OSLONCA U STOPI
# ============================================================
def granicni_uslovi(tip):
    stope = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], 0.0))

    if tip == "ukljestenje":
        # ux = uy = 0 duz cijele donje ivice oba stuba
        dofs = fem.locate_dofs_topological(V, fdim, stope)
        return [fem.dirichletbc(nula_v, dofs)]

    def t_L(x):
        return np.logical_and(np.isclose(x[0], x_os_L), np.isclose(x[1], 0.0))
    def t_D(x):
        return np.logical_and(np.isclose(x[0], x_os_D), np.isclose(x[1], 0.0))

    if tip == "ivica_uy":
        # uy = 0 duz cijele stope; ux blokirano u osi OBJE stope
        dofs_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, stope)
        dL_x = fem.locate_dofs_geometrical((V.sub(0), Vx), t_L)
        dD_x = fem.locate_dofs_geometrical((V.sub(0), Vx), t_D)
        return [fem.dirichletbc(nula_y, dofs_y, V.sub(1)),
                fem.dirichletbc(nula_x, dL_x, V.sub(0)),
                fem.dirichletbc(nula_x, dD_x, V.sub(0))]

    if tip == "tacka":
        # pravi zglob: uy i ux blokirani u osi OBJE stope
        dL_y = fem.locate_dofs_geometrical((V.sub(1), Vy), t_L)
        dD_y = fem.locate_dofs_geometrical((V.sub(1), Vy), t_D)
        dL_x = fem.locate_dofs_geometrical((V.sub(0), Vx), t_L)
        dD_x = fem.locate_dofs_geometrical((V.sub(0), Vx), t_D)
        return [fem.dirichletbc(nula_y, dL_y, V.sub(1)),
                fem.dirichletbc(nula_y, dD_y, V.sub(1)),
                fem.dirichletbc(nula_x, dL_x, V.sub(0)),
                fem.dirichletbc(nula_x, dD_x, V.sub(0))]

    raise ValueError(tip)

opisi = {"ukljestenje": "UKLJESTENJE  (ux = uy = 0 duz cijele stope)",
         "ivica_uy":    "SAMO uy = 0 duz cijele stope (pokusaj zgloba)",
         "tacka":       "TACKASTI ZGLOB (uy = 0 u osi stuba)"}

print(f"\nStapni model, ukljestene stope : M cvor = {M_cvor_u/1e3:7.2f} kNm, "
      f"M sredina = {M_sred_u/1e3:6.2f} kNm, w = {w_u*1000:.4f} mm")
print(f"Stapni model, zglobovi u stopi : M cvor = {M_cvor_z/1e3:7.2f} kNm, "
      f"M sredina = {M_sred_z/1e3:6.2f} kNm, w = {w_z*1000:.4f} mm")

rezultati = {}

for tip in ["ukljestenje", "ivica_uy", "tacka"]:
    bcs = granicni_uslovi(tip)

    problem = LinearProblem(
        a, Lf, bcs=bcs,
        petsc_options_prefix=f"ram_{tip}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez
    uh.name = "pomjeranje"

    sxx = fem.Function(S1, name="sigma_xx")
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))
    syy = fem.Function(S1, name="sigma_yy")
    syy.interpolate(fem.Expression(sigma(uh)[1, 1], S1.element.interpolation_points))

    # --- ocitavanja ---
    w_sred = abs(vrijednost_u_tacki(uh, Bx/2.0, Hr)[1])           # ugib sredine rigle
    ux_cvor = vrijednost_u_tacki(uh, x_os_L, Hr)[0]               # vodoravno, vrh stuba
    s_sred = vrijednost_u_tacki(sxx, Bx/2.0, y_don + tol)[0]      # donje vlakno rigle
    s_cvor = vrijednost_u_tacki(sxx, x_lice + t, By - tol)[0]     # gornje vlakno, uklonjeno od cvora

    rezultati[tip] = (w_sred, ux_cvor, s_sred, s_cvor)

    print(f"\n--- {opisi[tip]} ---")
    print(f"  ugib sredine rigle          : {w_sred*1000:9.4f} mm")
    print(f"  vodoravno pomjeranje cvora  : {ux_cvor*1000:9.4f} mm")
    print(f"  sigma_xx sredina rigle, dno : {s_sred/1e6:9.4f} MPa")
    print(f"  sigma_xx uz cvor, gore      : {s_cvor/1e6:9.4f} MPa")

    u1 = fem.Function(V1, name="pomjeranje")
    u1.interpolate(uh)
    with VTXWriter(domen.comm, f"ram_{tip}.bp", [u1, sxx, syy], engine="BP4") as vtx:
        vtx.write(0.0)

print("\nParaView izvozi: ram_ukljestenje.bp, ram_ivica_uy.bp, ram_tacka.bp")