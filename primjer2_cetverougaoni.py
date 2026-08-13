# ============================================================
# PRIMJER 2: Prosta greda, ravnomjerno raspodijeljeno opterecenje
#            CETVOROUGAONI konacni elementi, Lagrange stepen 2
#
# Isti staticki sistem kao u Primjeru 1; jedina razlika je tip
# konacnog elementa (quadrilateral umjesto triangle).
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, geometry, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0            # raspon [m]
h  = 0.5            # visina presjeka [m]
b  = 0.3            # sirina presjeka [m]

E  = 31.0e9         # modul elasticnosti [Pa]
nu = 0.2            # Poissonov koeficijent

q   = 20.0e3        # linijsko opterecenje [N/m]
t_q = q / b         # povrsinski pritisak za 2D model [Pa]

Nx, Ny = 40, 4      # ista gustina mreze kao u Primjeru 1

# Lameove konstante - plane stress
mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

# ============================================================
# 2) GEOMETRIJA I MREZA
# ============================================================
domen = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,     # <<< CETVOROUGAONI elementi
)

# Za kvadrilaterale Lagrange 2 znaci 9-cvorni element (bikvadratna interpolacija)
V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

print(f"Tip elementa : cetvorougaoni (quadrilateral), Lagrange 2")
print(f"Broj celija  : {domen.topology.index_map(domen.topology.dim).size_local}")
print(f"Broj DOF-ova : {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

# ============================================================
# 3) GRANICNI USLOVI (oslonci)
# ============================================================
Vx, _ = V.sub(0).collapse()      # podprostor za ux
Vy, _ = V.sub(1).collapse()      # podprostor za uy

nula_x = fem.Function(Vx)        # vrijednost 0 za ux
nula_y = fem.Function(Vy)        # vrijednost 0 za uy

def tacka_A(x):
    # lijevi oslonac (0, 0) - nepomicni zglob
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

def tacka_B(x):
    # desni oslonac (L, 0) - pomicni zglob
    return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

dofs_A_x = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A)
dofs_A_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A)
dofs_B_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B)

bcs = [
    fem.dirichletbc(nula_x, dofs_A_x, V.sub(0)),   # ux = 0 u A
    fem.dirichletbc(nula_y, dofs_A_y, V.sub(1)),   # uy = 0 u A
    fem.dirichletbc(nula_y, dofs_B_y, V.sub(1)),   # uy = 0 u B
]

# ============================================================
# 4) OPTERECENJE NA GORNJOJ IVICI
# ============================================================
fdim = domen.topology.dim - 1                       # dimenzija ivice
facets_gore = np.sort(                              # sve granicne ivice na y = h
    locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h))
)
oznake = meshtags(domen, fdim, facets_gore,         # oznaka 1 za gornju ivicu
                  np.full(len(facets_gore), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)

T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))   # smjer nanize

# ============================================================
# 5) VARIJACIONA FORMULACIJA
# ============================================================
def epsilon(u):
    return ufl.sym(ufl.grad(u))                      # tenzor deformacije

def sigma(u):
    return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(2)   # Hooke

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a  = ufl.inner(sigma(u), epsilon(v)) * ufl.dx        # matrica krutosti
Lf = ufl.dot(T, v) * ds(1)                           # vektor opterecenja

# ============================================================
# 6) RJESAVANJE
# ============================================================
problem = LinearProblem(
    a, Lf, bcs=bcs,
    petsc_options_prefix="greda_quad_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},   # direktni solver
)
rez = problem.solve()
uh = rez[0] if isinstance(rez, tuple) else rez
uh.name = "pomjeranje"

# ============================================================
# 7) NAPONI
# ============================================================
S1 = fem.functionspace(domen, ("Lagrange", 1))

sxx = fem.Function(S1, name="sigma_xx")     # normalni napon duz ose grede
sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

syy = fem.Function(S1, name="sigma_yy")     # normalni napon upravno na osu
syy.interpolate(fem.Expression(sigma(uh)[1, 1], S1.element.interpolation_points))

# ============================================================
# 8) OCITAVANJE U TACKI (bb_tree)
# ============================================================
def vrijednost_u_tacki(funkcija, x, y):
    """Vrijednost funkcije u tacki (x, y) - pretraga preko stabla granicnih kutija."""
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    stablo = geometry.bb_tree(domen, domen.topology.dim)
    kandidati = geometry.compute_collisions_points(stablo, tacka)
    celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
    celija = np.array([celije.links(0)[0]], dtype=np.int32)
    return funkcija.eval(tacka, celija)

u_sredina = vrijednost_u_tacki(uh,  L/2.0, h/2.0)    # ugib na sredini raspona
s_donje   = vrijednost_u_tacki(sxx, L/2.0, 0.0)      # zategnuto donje vlakno
s_osa     = vrijednost_u_tacki(sxx, L/2.0, h/2.0)    # tezisna osa
s_gornje  = vrijednost_u_tacki(sxx, L/2.0, h)        # pritisnuto gornje vlakno

print("\n--- REZULTATI (sredina raspona, x = L/2) ---")
print(f"Ugib  uy            : {u_sredina[1]*1000:10.4f} mm")
print(f"Pomjeranje ux       : {u_sredina[0]*1000:10.4f} mm")
print(f"sigma_xx  (y = 0)   : {s_donje[0]/1e6:10.4f} MPa   (donja ivica)")
print(f"sigma_xx  (y = h/2) : {s_osa[0]/1e6:10.4f} MPa   (tezisna osa)")
print(f"sigma_xx  (y = h)   : {s_gornje[0]/1e6:10.4f} MPa   (gornja ivica)")

# ============================================================
# 9) IZVOZ ZA PARAVIEW
# ============================================================
V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
u1 = fem.Function(V1, name="pomjeranje")
u1.interpolate(uh)                          # Lagrange 2 -> Lagrange 1 zbog XDMF-a

with io.XDMFFile(domen.comm, "primjer2_cetvorougaoni.xdmf", "w") as xdmf:
    xdmf.write_mesh(domen)
    xdmf.write_function(u1)
    xdmf.write_function(sxx)
    xdmf.write_function(syy)

print("\nRezultati izvezeni: primjer2_cetvorougaoni.xdmf / .h5")