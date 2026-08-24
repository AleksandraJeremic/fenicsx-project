# ============================================================
# NOVI PRIMJER 1: PROSTA GREDA - oslonci duz CIJELE krajnje ivice
#
#            q = 20 kN/m
#   | | | | | | | | | | | | | |
#   v v v v v v v v v v v v v v
#   #==========================#
#   #                          #     # = cijela vertikalna ivica,
#   #==========================#         uy = 0 duz cijele visine
#   ^                          ^
#   x = 0                    x = L
#
# IZMJENA U ODNOSU NA RANIJE PRIMJERE:
# Umjesto blokiranja uy u JEDNOJ TACKI, uy se blokira duz CIJELE
# krajnje (vertikalne) ivice. Reakcija se time prenosi preko konacne
# povrsine, pa nema koncentracije sile u tacki - a time ni logaritamske
# singularnosti zbog koje apsolutni ugib ranije nije konvergirao.
#
# VAZNO: duz ivice se blokira SAMO uy. Horizontalno pomjeranje ux
# blokira se u samo JEDNOJ tacki (tezistu lijevog presjeka), jer bi
# blokiranje ux duz cijele ivice sprijecilo obrtanje krajnjeg presjeka
# i pretvorilo oslonac u uklještenje - dobili bismo obostrano
# ukljestenu gredu umjesto proste.
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl

from dolfinx import fem, geometry
from dolfinx.io import VTXWriter
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0             # raspon [m]
h  = 0.30            # visina presjeka [m]
b  = 0.30            # sirina presjeka [m]   -> kvadratni presjek 30/30

E  = 31.0e9          # modul elasticnosti [Pa]  (C25/30)
nu = 0.2             # Poissonov koeficijent

q   = 20.0e3         # linijsko opterecenje [N/m]
t_q = q / b          # povrsinski pritisak za plane stress [Pa]

Nx, Ny = 40, 4       # gustina mreze

# Lameove konstante - PLANE STRESS
mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

# ============================================================
# 2) REFERENTNA RJESENJA (tehnicka teorija)
# ============================================================
I_ef = h**3 / 12.0                       # moment inercije po jed. debljine
A_ef = h                                 # povrsina po jed. debljine
G    = E / (2.0 * (1.0 + nu))
k_s  = 5.0 / 6.0                         # koeficijent smicanja, pravougaonik

w_EB   = 5.0 * t_q * L**4 / (384.0 * E * I_ef)      # Bernoulli-Euler
w_smik = t_q * L**2 / (8.0 * k_s * G * A_ef)        # doprinos smicanja
w_TIM  = w_EB + w_smik                               # Timosenko

M_sr     = t_q * L**2 / 8.0
sxx_teor = M_sr * (h/2.0) / I_ef                     # tehnicka teorija
sxx_2D   = sxx_teor + t_q / 5.0                      # tacno 2D rjesenje

# ============================================================
# 3) GEOMETRIJA I MREZA
# ============================================================
domen = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)

V = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))

print("=" * 66)
print("PROSTA GREDA - oslonci zadati duz cijele krajnje ivice")
print("=" * 66)
print(f"Raspon L     : {L:.2f} m")
print(f"Presjek b/h  : {b*100:.0f}/{h*100:.0f} cm   (L/h = {L/h:.1f})")
print(f"Mreza        : {Nx} x {Ny}   ({Nx*Ny} elemenata)")
print(f"Broj DOF-ova : {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

# ============================================================
# 4) GRANICNI USLOVI
# ============================================================
Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
nula_x = fem.Function(Vx)
nula_y = fem.Function(Vy)

fdim = domen.topology.dim - 1

# --- uy = 0 duz CIJELE lijeve i CIJELE desne vertikalne ivice ---
def ivica_lijevo(x):
    return np.isclose(x[0], 0.0)

def ivica_desno(x):
    return np.isclose(x[0], L)

facets_lijevo = locate_entities_boundary(domen, fdim, ivica_lijevo)
facets_desno  = locate_entities_boundary(domen, fdim, ivica_desno)

dofs_L_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_lijevo)
dofs_D_y = fem.locate_dofs_topological((V.sub(1), Vy), fdim, facets_desno)

# --- ux = 0 u JEDNOJ tacki: teziste lijevog presjeka (0, h/2) ---
# U toj tacki je ux u tacnom rjesenju ionako nula (nema uzduzne sile,
# a obrtanje presjeka se odvija oko tezista), pa ovaj uslov ne unosi
# nikakvo dodatno ogranicenje - samo uklanja pomjeranje krutog tijela.
def tacka_ux(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], h/2.0))

dofs_ux = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_ux)

for ime, d in [("uy lijevo", dofs_L_y), ("uy desno", dofs_D_y), ("ux tacka", dofs_ux)]:
    if len(d[0]) == 0:
        raise RuntimeError(f"Granicni uslov '{ime}' nije pronadjen")

bcs = [
    fem.dirichletbc(nula_y, dofs_L_y, V.sub(1)),
    fem.dirichletbc(nula_y, dofs_D_y, V.sub(1)),
    fem.dirichletbc(nula_x, dofs_ux,  V.sub(0)),
]

print(f"\nBlokirano uy  : {len(dofs_L_y[0])} DOF-ova lijevo, "
      f"{len(dofs_D_y[0])} desno  (cijela ivica)")
print(f"Blokirano ux  : {len(dofs_ux[0])} DOF  (samo tacka (0, h/2))")

# ============================================================
# 5) OPTERECENJE NA GORNJOJ IVICI
# ============================================================
facets_gore = np.sort(locate_entities_boundary(
    domen, fdim, lambda x: np.isclose(x[1], h)))
oznake = meshtags(domen, fdim, facets_gore,
                  np.full(len(facets_gore), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)

T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

# ============================================================
# 6) VARIJACIONA FORMULACIJA I RJESAVANJE
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
    petsc_options_prefix="greda_ivica_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
rez = problem.solve()
uh = rez[0] if isinstance(rez, tuple) else rez
uh.name = "pomjeranje"

# ============================================================
# 7) NAPONI
# ============================================================
S1 = fem.functionspace(domen, ("Lagrange", 1))

sxx = fem.Function(S1, name="sigma_xx")
sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

sxy = fem.Function(S1, name="tau_xy")
sxy.interpolate(fem.Expression(sigma(uh)[0, 1], S1.element.interpolation_points))

# ============================================================
# 8) OCITAVANJE REZULTATA
# ============================================================
stablo = geometry.bb_tree(domen, domen.topology.dim)

def vrijednost_u_tacki(funkcija, x, y):
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    kandidati = geometry.compute_collisions_points(stablo, tacka)
    celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
    celija = np.array([celije.links(0)[0]], dtype=np.int32)
    return funkcija.eval(tacka, celija)

# --- UGIB ---
# Sada se ocitava APSOLUTNA vrijednost, bez ikakve korekcije:
# posto reakcija vise nije koncentrisana u tacki, nema singularnosti
# i vrijednost konvergira sama po sebi.
w_mke = abs(vrijednost_u_tacki(uh, L/2.0, h/2.0)[1])

print("\n--- UGIB NA SREDINI RASPONA ---")
print(f"MKE (apsolutni)          : {w_mke*1000:10.5f} mm")
print(f"Bernoulli-Euler          : {w_EB*1000:10.5f} mm   "
      f"razlika {(w_mke-w_EB)/w_EB*100:+7.3f} %")
print(f"Timosenko (sa smicanjem) : {w_TIM*1000:10.5f} mm   "
      f"razlika {(w_mke-w_TIM)/w_TIM*100:+7.3f} %")
print(f"   (doprinos smicanja u Timosenkovom rjesenju: "
      f"{w_smik/w_EB*100:.2f} % od Bernoullijevog ugiba)")

# --- NORMALNI NAPON PO VISINI, NA SREDINI RASPONA ---
print("\n--- sigma_xx NA SREDINI RASPONA (x = L/2) ---")
print(f"{'y [m]':>8} {'MKE [MPa]':>12} {'teorija [MPa]':>15} {'razlika [%]':>13}")
for yy in np.linspace(0.0, h, 5):
    yc = min(max(yy, 1e-9), h - 1e-9)
    s_mke  = vrijednost_u_tacki(sxx, L/2.0, yc)[0]
    s_teor = M_sr * (h/2.0 - yy) / I_ef
    razl = (s_mke - s_teor) / s_teor * 100.0 if abs(s_teor) > 1.0 else np.nan
    print(f"{yy:8.3f} {s_mke/1e6:12.4f} {s_teor/1e6:15.4f} {razl:13.3f}")

s_dno = vrijednost_u_tacki(sxx, L/2.0, 1e-9)[0]
print(f"\nDonje vlakno: MKE = {s_dno/1e6:.4f} MPa")
print(f"   tehnicka teorija = {sxx_teor/1e6:.4f} MPa   "
      f"(razlika {(s_dno-sxx_teor)/sxx_teor*100:+.3f} %)")
print(f"   tacno 2D rjesenje = {sxx_2D/1e6:.4f} MPa   "
      f"(razlika {(s_dno-sxx_2D)/sxx_2D*100:+.3f} %)")

# --- SMICUCI NAPON UZ OSLONAC ---
# Presjek na jednu visinu grede od kraja - izvan zone poremecaja.
x_p = h
V_p = t_q * (L/2.0 - x_p)
print(f"\n--- tau_xy, presjek x = h = {h:.2f} m ---")
print(f"Transverzalna sila V = {V_p/1e3:.2f} kN/m, "
      f"tau_max po teoriji = {1.5*V_p/h/1e6:.4f} MPa")
print(f"{'y [m]':>8} {'MKE [MPa]':>12} {'teorija [MPa]':>15}")
for yy in np.linspace(0.0, h, 5):
    yc = min(max(yy, 1e-9), h - 1e-9)
    t_mke  = vrijednost_u_tacki(sxy, x_p, yc)[0]
    yb = yy - h/2.0
    t_teor = -V_p * ((h/2.0)**2 - yb**2) / (2.0 * I_ef)
    print(f"{yy:8.3f} {t_mke/1e6:12.4f} {t_teor/1e6domen.geometry.dim,)))
u1 = fem.Function(V1, name="pomjeranje")
u1.interpolate(uh)

with VTXWriter(domen.comm, "N1_oslonac_ivica.bp",
               [u1, sxx, sxy], engine="BP4") as vtx:
    vtx.write(0.0)

print("\nParaView izvoz: N1_oslonac_ivica.bp")