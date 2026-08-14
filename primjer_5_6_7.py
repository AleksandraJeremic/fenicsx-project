# ============================================================
# PRIMJERI 5, 6 i 7: Prosta greda - uticaj POLOZAJA OSLONCA
#
#   PRIMJER 5 -> oslonci na DONJOJ ivici      (y = 0)
#   PRIMJER 6 -> oslonci u TEZISTU presjeka   (y = h/2)
#   PRIMJER 7 -> oslonci na GORNJOJ ivici     (y = h)
#
# Sve ostalo je nepromijenjeno: geometrija, materijal, opterecenje,
# mreza (40x4, cetvorougaoni elementi, Lagrange 2).
#
#   PRIMJER 5              PRIMJER 6              PRIMJER 7
#   +-----------+          +-----------+          ^-----------o
#   |           |          o           |          |           |
#   |           |         >|           |o         |           |
#   ^           o          |           |          |           |
#   +-----------+          +-----------+          +-----------+
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, geometry, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 0) IZBOR PRIMJERA  <<< JEDINO OVO MIJENJAS >>>
# ============================================================
PRIMJER = 5          # 5, 6 ili 7

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0             # raspon [m]
h  = 0.5             # visina presjeka [m]
b  = 0.3             # sirina presjeka [m]

E  = 31.0e9          # modul elasticnosti [Pa]
nu = 0.2             # Poissonov koeficijent

q   = 20.0e3         # linijsko opterecenje [N/m]
t_q = q / b          # povrsinski pritisak za plane stress [Pa]

Nx, Ny = 40, 4       # gustina mreze

mu  = E / (2.0 * (1.0 + nu))     # Lameove konstante - plane stress
lam = E * nu / (1.0 - nu**2)

# --- polozaj oslonca u zavisnosti od izabranog primjera ---
if PRIMJER == 5:
    y_osl, opis, kratko = 0.0,     "donja ivica",       "donja_ivica"
elif PRIMJER == 6:
    y_osl, opis, kratko = h / 2.0, "teziste presjeka",  "teziste"
elif PRIMJER == 7:
    y_osl, opis, kratko = h,       "gornja ivica",      "gornja_ivica"
else:
    raise ValueError("PRIMJER mora biti 5, 6 ili 7")

naziv_izlaza = f"primjer{PRIMJER}_oslonci_{kratko}"

# ============================================================
# 2) GEOMETRIJA I MREZA
# ============================================================
domen = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)

V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

print("=" * 60)
print(f"PRIMJER {PRIMJER}: oslonci na visini y = {y_osl:.3f} m ({opis})")
print("=" * 60)
print(f"Tip elementa : cetvorougaoni (quadrilateral), Lagrange 2")
print(f"Mreza        : {Nx} x {Ny}")
print(f"Broj celija  : {domen.topology.index_map(domen.topology.dim).size_local}")
print(f"Broj DOF-ova : {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

# ============================================================
# 3) GRANICNI USLOVI - oslonci na visini y_osl
# ============================================================
# A (x = 0)  : nepomicni zglob, ux = uy = 0
# B (x = L)  : pomicni zglob (klizac), uy = 0
# Oba su TACKASTI oslonci, samo se mijenja njihova visina.

Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()

nula_x = fem.Function(Vx)
nula_y = fem.Function(Vy)

def tacka_A(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], y_osl))

def tacka_B(x):
    return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], y_osl))

dofs_A_x = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A)
dofs_A_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A)
dofs_B_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B)

# kontrola: ako je neki od nizova prazan, oslonac nije pronadjen
for naziv_bc, dofs in [("A-ux", dofs_A_x), ("A-uy", dofs_A_y), ("B-uy", dofs_B_y)]:
    if len(dofs[0]) == 0:
        raise RuntimeError(f"Oslonac {naziv_bc} nije pronadjen na y = {y_osl}")

bcs = [
    fem.dirichletbc(nula_x, dofs_A_x, V.sub(0)),
    fem.dirichletbc(nula_y, dofs_A_y, V.sub(1)),
    fem.dirichletbc(nula_y, dofs_B_y, V.sub(1)),
]

# ============================================================
# 4) OPTERECENJE NA GORNJOJ IVICI
# ============================================================
fdim = domen.topology.dim - 1
facets_gore = np.sort(
    locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h))
)
oznake = meshtags(domen, fdim, facets_gore,
                  np.full(len(facets_gore), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)

T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

# ============================================================
# 5) VARIJACIONA FORMULACIJA I RJESAVANJE
# ============================================================
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
    petsc_options_prefix=f"greda_p{PRIMJER}_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
rez = problem.solve()
uh = rez[0] if isinstance(rez, tuple) else rez
uh.name = "pomjeranje"

# ============================================================
# 6) NAPONI
# ============================================================
S1 = fem.functionspace(domen, ("Lagrange", 1))

sxx = fem.Function(S1, name="sigma_xx")      # normalni napon duz ose grede
sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

syy = fem.Function(S1, name="sigma_yy")      # normalni napon upravno na osu
syy.interpolate(fem.Expression(sigma(uh)[1, 1], S1.element.interpolation_points))

# ============================================================
# 7) OCITAVANJE REZULTATA
# ============================================================
def vrijednost_u_tacki(funkcija, x, y):
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    stablo = geometry.bb_tree(domen, domen.topology.dim)
    kandidati = geometry.compute_collisions_points(stablo, tacka)
    celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
    celija = np.array([celije.links(0)[0]], dtype=np.int32)
    return funkcija.eval(tacka, celija)

# --- ugib ---
# Apsolutni ugib zavisi od singularnosti u tackastom osloncu, pa uz njega
# racunamo i relativni: u odnosu na SREDNJE vertikalno pomjeranje krajnjih
# presjeka. Taj postupak je isti za sva tri primjera, pa su rezultati
# medjusobno uporedivi bez obzira gdje je oslonac.
y_uzorci = np.clip(np.linspace(0.0, h, 5), 1e-9, h - 1e-9)

uy_kraj_A = np.mean([vrijednost_u_tacki(uh, 0.0, yy)[1] for yy in y_uzorci])
uy_kraj_B = np.mean([vrijednost_u_tacki(uh, L,   yy)[1] for yy in y_uzorci])
datum = 0.5 * (uy_kraj_A + uy_kraj_B)

u_sredina = vrijednost_u_tacki(uh, L/2.0, h/2.0)
w_aps = abs(u_sredina[1])
w_rel = abs(u_sredina[1] - datum)

# --- naponi na sredini raspona ---
s_donje  = vrijednost_u_tacki(sxx, L/2.0, 0.0)[0]
s_osa    = vrijednost_u_tacki(sxx, L/2.0, h/2.0)[0]
s_gornje = vrijednost_u_tacki(sxx, L/2.0, h)[0]

# --- vertikalni napon u presjeku iznad/ispod oslonca ---
# Pokazuje kako se reakcija prenosi kroz presjek: pritisak ako je oslonac
# ispod opterecenja, zatezanje ako greda "visi" o gornji oslonac.
x_kontrola = 0.25                                   # blizu lijevog oslonca
sy_dolje = vrijednost_u_tacki(syy, x_kontrola, 0.05)[0]
sy_osa   = vrijednost_u_tacki(syy, x_kontrola, h/2.0)[0]
sy_gore  = vrijednost_u_tacki(syy, x_kontrola, h - 0.05)[0]

print(f"\n--- POMJERANJA (sredina raspona, tezisna osa) ---")
print(f"Apsolutni ugib      : {w_aps*1000:10.4f} mm")
print(f"Relativni ugib      : {w_rel*1000:10.4f} mm")

print(f"\n--- NAPON sigma_xx (sredina raspona) ---")
print(f"y = 0    (donja ivica)  : {s_donje/1e6:10.4f} MPa")
print(f"y = h/2  (tezisna osa)  : {s_osa/1e6:10.4f} MPa")
print(f"y = h    (gornja ivica) : {s_gornje/1e6:10.4f} MPa")

print(f"\n--- NAPON sigma_yy (presjek na x = {x_kontrola} m, blizu oslonca) ---")
print(f"y = 0.05      : {sy_dolje/1e6:10.4f} MPa")
print(f"y = h/2       : {sy_osa/1e6:10.4f} MPa")
print(f"y = h - 0.05  : {sy_gore/1e6:10.4f} MPa")
print("   (pozitivno = zatezanje, negativno = pritisak)")

# ============================================================
# 8) IZVOZ ZA PARAVIEW
# ============================================================
V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
u1 = fem.Function(V1, name="pomjeranje")
u1.interpolate(uh)

with io.XDMFFile(domen.comm, f"{naziv_izlaza}.xdmf", "w") as xdmf:
    xdmf.write_mesh(domen)
    xdmf.write_function(u1)
    xdmf.write_function(sxx)
    xdmf.write_function(syy)

print(f"\nRezultati izvezeni: {naziv_izlaza}.xdmf / .h5")