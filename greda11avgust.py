"""
PROSTA GREDA - 2D model, ravansko stanje napona
================================================
Greda raspona L = 5,0 m, presjek b/h = 30/50 cm, beton C30 (E = 30 GPa, nu = 0,2),
opterecena jednakopodijeljenim opterecenjem q = 20 kN/m.

Greda se modelira kao pravougaona oblast 5,00 x 0,50 m u ravni x-y.
Debljina b = 0,30 m ne pojavljuje se u geometriji, vec se uzima u obzir
pri preracunavanju opterecenja u povrsinsko (vidjeti nize).

"""

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh, geometry, io
from dolfinx.fem.petsc import LinearProblem

# =====================================================================
# 1. ULAZNI PODACI
# =====================================================================

L = 5.0             # raspon grede                          [m]
h = 0.5             # visina presjeka                       [m]
b = 0.3             # sirina presjeka (debljina modela)     [m]

E = 30.0e9          # modul elasticnosti                    [Pa]
nu = 0.2            # Poasonov koeficijent                  [-]

q = 20.0e3          # jednakopodijeljeno opterecenje        [N/m]

nx, ny = 100, 10    # broj podjela po duzini i po visini
stepen = 2          # stepen interpolacionog polinoma

# =====================================================================
# 2. MREZA KONACNIH ELEMENATA
# =====================================================================
# Pravougaona oblast [0, L] x [0, h] dijeli se na nx * ny cetvorouglova.

domen = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],
    [nx, ny],
    cell_type=mesh.CellType.quadrilateral,
)

# =====================================================================
# 3. PROSTOR KONACNIH ELEMENATA
# =====================================================================
# Trazena velicina je pomjeranje - vektorska velicina sa dvije komponente
# (u_x, u_y), pa se koristi vektorski prostor. Zadnji clan (2,) oznacava
# broj komponenti.
#
# NAPOMENA: koristi se stepen 2 (kvadratni element). Linearni cetvorougao
# u savijanju pokazuje pojavu tzv. shear locking-a i daje znatno manje
# ugibe od stvarnih, sto je i samo po sebi zanimljivo za analizu.

V = fem.functionspace(domen, ("Lagrange", stepen, (2,)))

# =====================================================================
# 4. GRANICNI USLOVI
# =====================================================================
# Prosta greda: nepomicni oslonac u tacki A (0, 0) sprjecava pomjeranje
# u oba pravca, pomicni oslonac u tacki B (L, 0) sprjecava samo vertikalno
# pomjeranje. Time je sistem staticki odredjen, a sprijeceno je i pomjeranje
# grede kao krutog tijela.

def tacka_A(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

def tacka_B(x):
    return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

# Za zadavanje uslova na pojedinacnoj komponenti pomjeranja potrebni su
# tzv. "sabijeni" (collapsed) potprostori.
Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()

nula_x = fem.Function(Vx)      # funkcija cije su sve vrijednosti nula
nula_y = fem.Function(Vy)

dofs_A_x = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A)
dofs_A_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A)
dofs_B_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B)

bcs = [
    fem.dirichletbc(nula_x, dofs_A_x, V.sub(0)),   # u_x = 0 u tacki A
    fem.dirichletbc(nula_y, dofs_A_y, V.sub(1)),   # u_y = 0 u tacki A
    fem.dirichletbc(nula_y, dofs_B_y, V.sub(1)),   # u_y = 0 u tacki B
]

# =====================================================================
# 5. OPTERECENJE
# =====================================================================
# Opterecenje q [N/m] djeluje na gornjoj ivici grede. U ravanskom modelu
# ono se zadaje kao povrsinsko opterecenje (napon) na toj ivici:
#
#       p = q / b = 20 000 / 0,30 = 66 667 Pa
#
# Prvo se pronalaze ivice (entiteti dimenzije 1) na gornjoj granici i
# oznacavaju brojem 1, kako bi se integracija vrsila samo po njima.

gornja_ivica = mesh.locate_entities_boundary(
    domen, domen.topology.dim - 1, lambda x: np.isclose(x[1], h)
)
oznake = mesh.meshtags(
    domen,
    domen.topology.dim - 1,
    gornja_ivica,
    np.full(len(gornja_ivica), 1, dtype=np.int32),
)
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)

p = fem.Constant(domen, np.array([0.0, -q / b]))   # smjer nanize

# =====================================================================
# 6. VARIJACIONA FORMULACIJA
# =====================================================================
# Lameovi koeficijenti za RAVANSKO STANJE NAPONA:
#       mu     = E / (2 (1 + nu))
#       lambda = E nu / (1 - nu^2)

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

def deformacija(u):
    """Tenzor deformacije: eps = 1/2 (grad u + grad u^T)"""
    return ufl.sym(ufl.grad(u))

def napon(u):
    """Tenzor napona: sigma = 2 mu eps + lambda tr(eps) I"""
    return 2.0 * mu * deformacija(u) + lam * ufl.tr(deformacija(u)) * ufl.Identity(2)

u = ufl.TrialFunction(V)     # probna funkcija (nepoznato pomjeranje)
v = ufl.TestFunction(V)      # test funkcija

# Bilinearna forma - virtuelni rad unutrasnjih sila
a = ufl.inner(napon(u), deformacija(v)) * ufl.dx

# Linearna forma - virtuelni rad spoljasnjeg opterecenja po gornjoj ivici
Lf = ufl.dot(p, v) * ds(1)

# =====================================================================
# 7. RJESAVANJE
# =====================================================================

problem = LinearProblem(
    a, Lf, bcs=bcs,
    petsc_options_prefix="greda_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
uh = problem.solve()

# =====================================================================
# 8. OBRADA REZULTATA
# =====================================================================

def vrijednost_u_tacki(funkcija, x, y):
    """Vraca vrijednost funkcije u zadatoj tacki (x, y)."""
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    stablo = geometry.bb_tree(domen, domen.topology.dim)
    kandidati = geometry.compute_collisions_points(stablo, tacka)
    celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
    celija = np.array([celije.links(0)[0]], dtype=np.int32)
    return funkcija.eval(tacka, celija)

# --- ugib u sredini raspona, na tezisnoj osi ---
ugib = vrijednost_u_tacki(uh, L / 2.0, h / 2.0)[1]

# --- normalni napon sigma_xx u donjoj ivici, u sredini raspona ---
S = fem.functionspace(domen, ("Lagrange", 1))
izraz = fem.Expression(napon(uh)[0, 0], S.element.interpolation_points)
sigma_xx = fem.Function(S)
sigma_xx.interpolate(izraz)

napon_donja = vrijednost_u_tacki(sigma_xx, L / 2.0, 0.0)[0]

# --- analiticko rjesenje prema teoriji Ojler-Bernulijeve grede ---
I = b * h**3 / 12.0
ugib_teor = 5.0 * q * L**4 / (384.0 * E * I)      # 5qL^4 / 384EI
M_teor = q * L**2 / 8.0                            # qL^2 / 8
napon_teor = M_teor * (h / 2.0) / I                # M z / I

# =====================================================================
# 9. ISPIS
# =====================================================================

if domen.comm.rank == 0:
    print()
    print("=" * 58)
    print("  PROSTA GREDA  L = {:.2f} m,  b/h = {:.0f}/{:.0f} cm".format(L, b * 100, h * 100))
    print("  Mreza: {} x {} elemenata,  stepen elementa: {}".format(nx, ny, stepen))
    print("  Broj stepeni slobode: {}".format(V.dofmap.index_map.size_global * 2))
    print("=" * 58)
    print()
    print("  UGIB U SREDINI RASPONA")
    print("    MKE (FEniCSx)   : {:8.3f} mm".format(abs(ugib) * 1000))
    print("    Analiticki      : {:8.3f} mm".format(ugib_teor * 1000))
    print("    Odstupanje      : {:8.2f} %".format(
        (abs(ugib) - ugib_teor) / ugib_teor * 100))
    print()
    print("  NORMALNI NAPON U DONJOJ IVICI, SREDINA RASPONA")
    print("    MKE (FEniCSx)   : {:8.3f} MPa".format(napon_donja / 1e6))
    print("    Analiticki      : {:8.3f} MPa".format(napon_teor / 1e6))
    print("    Odstupanje      : {:8.2f} %".format(
        (napon_donja - napon_teor) / napon_teor * 100))
    print()
    print("=" * 58)

# =====================================================================
# 10. ZAPIS ZA GRAFICKI PRIKAZ (ParaView)
# =====================================================================
# Datoteke se zapisuju u folder "rezultati" i otvaraju u programu ParaView.

uh.name = "pomjeranje"
sigma_xx.name = "sigma_xx"

with io.XDMFFile(domen.comm, "rezultati/greda.xdmf", "w") as f:
    f.write_mesh(domen)
    f.write_function(sigma_xx)