# ============================================================
# PRIMJER 1: Prosta greda, ravnomjerno raspodijeljeno opterecenje
#            TROUGAONI konacni elementi, Lagrange stepen 2
#
# Staticki sistem:
#
#        q = 20 kN/m  (ravnomjerno po gornjoj ivici)
#     | | | | | | | | | | | | | | | |
#     v v v v v v v v v v v v v v v v
#    +-------------------------------+  <- gornja ivica (y = h)
#    |                               |
#    +-------------------------------+  <- donja ivica (y = 0)
#    ^                               o
#   A (0,0)                        B (L,0)
#   nepomicni zglob               pomicni zglob
#   ux = uy = 0                   uy = 0
# ============================================================

# --- BIBLIOTEKE ---
from mpi4py import MPI                       # paralelni komunikator (i za 1 procesor je obavezan)
import numpy as np                           # numericki niz, isclose, itd.
import ufl                                   # jezik za zapis varijacione formulacije
from dolfinx import fem, geometry, io        # konacni elementi, bb_tree pretraga, izvoz
from dolfinx.fem.petsc import LinearProblem  # omotac oko PETSc linearnog solvera
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0            # raspon grede [m]
h  = 0.5            # visina poprecnog presjeka [m]
b  = 0.3            # sirina poprecnog presjeka [m] (debljina 2D modela)

E  = 31.0e9         # modul elasticnosti betona C25/30 [Pa]
nu = 0.2            # Poissonov koeficijent [-]

q   = 20.0e3        # linijsko opterecenje [N/m]
# 2D plane stress model racuna "po jedinici debljine", pa linijsko
# opterecenje q [N/m] moramo pretvoriti u povrsinski pritisak [N/m2]:
t_q = q / b         # = 20000 / 0.3 = 66 666.7 Pa

Nx, Ny = 40, 4      # broj elemenata: 40 po duzini, 4 po visini
                    # (dx = 5/40 = 0.125 m, dy = 0.5/4 = 0.125 m -> pravilan oblik elementa)

# --- Lameove konstante za RAVNINSKO STANJE NAPONA (plane stress) ---
# Paznja: lam se za plane stress racuna DRUGACIJE nego za plane strain!
mu  = E / (2.0 * (1.0 + nu))     # smicuci modul G
lam = E * nu / (1.0 - nu**2)     # plane stress verzija (plane strain bi bio E*nu/((1+nu)(1-2nu)))

# ============================================================
# 2) GEOMETRIJA I MREZA
# ============================================================
# create_rectangle: pravougaonik zadan donjim lijevim i gornjim desnim tjemenom
domen = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],   # (x_min,y_min) i (x_max,y_max)
    [Nx, Ny],                                   # podjela mreze
    cell_type=CellType.triangle,                # <<< TROUGAONI elementi
)

# Prostor funkcija: vektorsko polje pomjeranja (ux, uy), Lagrange stepena 2.
# Stepen 2 znaci kvadratna interpolacija pomjeranja unutar elementa ->
# deformacije/naponi su linearni po elementu, sto je bitno za savijanje.
V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

# Kontrolni ispis velicine modela
print(f"Tip elementa : trougaoni (triangle), Lagrange 2")
print(f"Broj celija  : {domen.topology.index_map(domen.topology.dim).size_local}")
print(f"Broj DOF-ova : {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

# ============================================================
# 3) GRANICNI USLOVI (oslonci)
# ============================================================
# Zelimo blokirati POJEDINACNE komponente pomjeranja (ux odvojeno od uy),
# pa moramo "collapse()"-ovati podprostore V.sub(0) -> Vx i V.sub(1) -> Vy.
Vx, _ = V.sub(0).collapse()      # skalarni prostor koji odgovara komponenti ux
Vy, _ = V.sub(1).collapse()      # skalarni prostor koji odgovara komponenti uy

# Funkcije koje nose zadatu vrijednost pomjeranja - podrazumijevano su nule
nula_x = fem.Function(Vx)
nula_y = fem.Function(Vy)

# Geometrijski markeri: vracaju True u cvorovima koji pripadaju osloncu
def tacka_A(x):
    # lijevi oslonac: x = 0 I y = 0
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

def tacka_B(x):
    # desni oslonac: x = L I y = 0
    return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

# Pronalazimo stepene slobode (DOF) na tim lokacijama.
# Prosljedjujemo par (V.sub(i), Vsub) jer BC pisemo na podprostor.
dofs_A_x = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A)   # ux u tacki A
dofs_A_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A)   # uy u tacki A
dofs_B_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B)   # uy u tacki B

# Lista Dirichletovih granicnih uslova:
#   A: ux = 0 i uy = 0  (nepomicni zglob)
#   B: uy = 0           (pomicni zglob - klizac, ux je slobodno)
bcs = [
    fem.dirichletbc(nula_x, dofs_A_x, V.sub(0)),
    fem.dirichletbc(nula_y, dofs_A_y, V.sub(1)),
    fem.dirichletbc(nula_y, dofs_B_y, V.sub(1)),
]

# ============================================================
# 4) OPTERECENJE NA GORNJOJ IVICI (Neumannov uslov)
# ============================================================
fdim = domen.topology.dim - 1    # dimenzija ivice = 2 - 1 = 1 (linija)

# Nalazimo sve granicne ivice (facete) koje leze na y = h
facets_gore = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h))
facets_gore = np.sort(facets_gore)          # meshtags zahtijeva sortiran niz

# Oznacavamo te ivice brojem 1, da bismo integralili bas po njima
oznake = meshtags(domen, fdim, facets_gore,
                  np.full(len(facets_gore), 1, dtype=np.int32))

# Mjera povrsinskog integrala vezana za nase oznake -> ds(1) = integral po gornjoj ivici
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)

# Vektor opterecenja: djeluje nanize (negativan y smjer)
T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

# ============================================================
# 5) VARIJACIONA FORMULACIJA
# ============================================================
# Tenzor deformacije: eps = 1/2 (grad u + grad u^T)
def epsilon(u):
    return ufl.sym(ufl.grad(u))

# Hookeov zakon: sigma = 2*mu*eps + lam*tr(eps)*I
def sigma(u):
    return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(2)

u = ufl.TrialFunction(V)     # nepoznato polje pomjeranja
v = ufl.TestFunction(V)      # test (virtuelna) funkcija

# Bilinearna forma = unutrasnji rad (krutost)
a  = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
# Linearna forma = rad spoljasnjih sila po gornjoj ivici
Lf = ufl.dot(T, v) * ds(1)

# ============================================================
# 6) RJESAVANJE SISTEMA K*u = F
# ============================================================
problem = LinearProblem(
    a, Lf, bcs=bcs,
    petsc_options_prefix="greda_tri_",
    petsc_options={"ksp_type": "preonly",   # bez iterativnog solvera
                   "pc_type": "lu"},        # direktna LU dekompozicija
)
rez = problem.solve()
# u nekim verzijama solve() vraca tuple (u, razlog, br_iteracija) - hvatamo oba slucaja
uh = rez[0] if isinstance(rez, tuple) else rez
uh.name = "pomjeranje"

# ============================================================
# 7) RACUNANJE NAPONA IZ POLJA POMJERANJA
# ============================================================
# Naponi su izvodi pomjeranja, pa ih moramo "projektovati" u svoj prostor.
S1 = fem.functionspace(domen, ("Lagrange", 1))   # skalarni prostor za komponente napona

sxx = fem.Function(S1, name="sigma_xx")          # normalni napon u pravcu ose grede
sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

syy = fem.Function(S1, name="sigma_yy")          # normalni napon upravno na osu
syy.interpolate(fem.Expression(sigma(uh)[1, 1], S1.element.interpolation_points))

# ============================================================
# 8) OCITAVANJE VRIJEDNOSTI U ZADATOJ TACKI (bb_tree)
# ============================================================
def vrijednost_u_tacki(funkcija, x, y):
    """Vraca vrijednost proizvoljne fem.Function u tacki (x, y)."""
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)          # dolfinx uvijek ocekuje 3D koordinatu
    stablo = geometry.bb_tree(domen, domen.topology.dim)       # stablo granicnih kutija svih celija
    kandidati = geometry.compute_collisions_points(stablo, tacka)   # celije koje MOZDA sadrze tacku
    celije = geometry.compute_colliding_cells(domen, kandidati, tacka)  # celije koje STVARNO sadrze tacku
    celija = np.array([celije.links(0)[0]], dtype=np.int32)    # uzimamo prvu pronadjenu
    return funkcija.eval(tacka, celija)

u_sredina  = vrijednost_u_tacki(uh,  L/2.0, h/2.0)   # pomjeranje u tezistu presjeka na sredini
s_donje    = vrijednost_u_tacki(sxx, L/2.0, 0.0)     # napon u donjem vlaknu (zatezanje)
s_osa      = vrijednost_u_tacki(sxx, L/2.0, h/2.0)   # napon u tezisnoj osi (treba biti ~0)
s_gornje   = vrijednost_u_tacki(sxx, L/2.0, h)       # napon u gornjem vlaknu (pritisak)

print("\n--- REZULTATI (sredina raspona, x = L/2) ---")
print(f"Ugib  uy            : {u_sredina[1]*1000:10.4f} mm")
print(f"Pomjeranje ux       : {u_sredina[0]*1000:10.4f} mm")
print(f"sigma_xx  (y = 0)   : {s_donje[0]/1e6:10.4f} MPa   (donja ivica)")
print(f"sigma_xx  (y = h/2) : {s_osa[0]/1e6:10.4f} MPa   (tezisna osa)")
print(f"sigma_xx  (y = h)   : {s_gornje[0]/1e6:10.4f} MPa   (gornja ivica)")

# ============================================================
# 9) IZVOZ REZULTATA ZA PARAVIEW
# ============================================================
# XDMF format ne podrzava Lagrange 2, pa pomjeranja interpoliramo na Lagrange 1.
V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))
u1 = fem.Function(V1, name="pomjeranje")
u1.interpolate(uh)

with io.XDMFFile(domen.comm, "primjer1_trougaoni.xdmf", "w") as xdmf:
    xdmf.write_mesh(domen)      # prvo mreza
    xdmf.write_function(u1)     # pa polja
    xdmf.write_function(sxx)
    xdmf.write_function(syy)

print("\nRezultati izvezeni: primjer1_trougaoni.xdmf / .h5")
# U ParaView-u: otvori .xdmf -> Apply -> filter "Warp By Vector" (pomjeranje)
# za deformisani oblik, a boju postavi na sigma_xx.