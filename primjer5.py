# ============================================================
# PRIMJER 5: Prosta greda - uticaj POLOZAJA OSLONCA
#
# Jedan te isti model rjesava se tri puta, mijenja se samo visina
# na kojoj su zadati oslonci:
#     (a) donja ivica      y = 0
#     (b) teziste presjeka y = h/2
#     (c) gornja ivica     y = h
#
# Mreza, opterecenje i bilinearna forma prave se SAMO JEDNOM i
# koriste za sva tri slucaja - time je osigurano da je jedina
# razlika zaista polozaj oslonca.
#
# Prate se velicine koje imaju pandan u tehnickoj teoriji:
#     sigma_xx = M*y/I        normalni napon od savijanja
#     tau_xy   = V*S/(I*b)    smicuci napon, parabolicna raspodjela
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dolfinx import fem, geometry, io
from dolfinx.io import VTXWriter
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

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

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / (1.0 - nu**2)

I_ef   = h**3 / 12.0                # moment inercije po jedinici debljine
M_sred = t_q * L**2 / 8.0           # moment na sredini raspona

# --- tri slucaja oslanjanja ---
slucajevi = [
    (0.0,     "donja ivica",      "donja_ivica",  "tab:blue"),
    (h / 2.0, "teziste presjeka", "teziste",      "tab:green"),
    (h,       "gornja ivica",     "gornja_ivica", "tab:red"),
]

# --- presjeci u kojima posmatramo smicuci napon ---
presjeci = [(h,       "x = h  (jedna visina grede od kraja)"),
            (h / 4.0, "x = h/4  (neposredno uz oslonac)")]

# ============================================================
# 2) MREZA I PROSTOR FUNKCIJA  (zajednicki za sva tri slucaja)
# ============================================================
domen = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)

V  = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))
S1 = fem.functionspace(domen, ("Lagrange", 1))
V1 = fem.functionspace(domen, ("Lagrange", 1, (domen.geometry.dim,)))

print("=" * 70)
print("PRIMJER 5: uticaj polozaja oslonca na prostu gredu")
print("=" * 70)
print(f"Tip elementa : cetvorougaoni (quadrilateral), Lagrange 2")
print(f"Mreza        : {Nx} x {Ny}")
print(f"Broj celija  : {domen.topology.index_map(domen.topology.dim).size_local}")
print(f"Broj DOF-ova : {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

# ============================================================
# 3) OPTERECENJE I BILINEARNA FORMA  (zajednicki)
# ============================================================
fdim = domen.topology.dim - 1
facets_gore = np.sort(
    locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h))
)
oznake = meshtags(domen, fdim, facets_gore,
                  np.full(len(facets_gore), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)

T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lam * ufl.tr(epsilon(u)) * ufl.Identity(2)

u_tr = ufl.TrialFunction(V)
v_te = ufl.TestFunction(V)

a  = ufl.inner(sigma(u_tr), epsilon(v_te)) * ufl.dx
Lf = ufl.dot(T, v_te) * ds(1)

# --- podprostori za granicne uslove (zajednicki) ---
Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
nula_x = fem.Function(Vx)
nula_y = fem.Function(Vy)

# --- stablo za ocitavanje u tacki (pravi se jednom) ---
stablo = geometry.bb_tree(domen, domen.topology.dim)

def vrijednost_u_tacki(funkcija, x, y):
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    kandidati = geometry.compute_collisions_points(stablo, tacka)
    celije = geometry.compute_colliding_cells(domen, kandidati, tacka)
    celija = np.array([celije.links(0)[0]], dtype=np.int32)
    return funkcija.eval(tacka, celija)

# ============================================================
# 4) FUNKCIJA: rjesenje za zadatu visinu oslonca
# ============================================================
def rijesi(y_osl, kratko):
    """Rjesava model sa osloncima na visini y_osl i vraca rezultate."""

    def tacka_A(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], y_osl))

    def tacka_B(x):
        return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], y_osl))

    dofs_A_x = fem.locate_dofs_geometrical((V.sub(0), Vx), tacka_A)
    dofs_A_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_A)
    dofs_B_y = fem.locate_dofs_geometrical((V.sub(1), Vy), tacka_B)

    for naziv_bc, dofs in [("A-ux", dofs_A_x), ("A-uy", dofs_A_y), ("B-uy", dofs_B_y)]:
        if len(dofs[0]) == 0:
            raise RuntimeError(f"Oslonac {naziv_bc} nije pronadjen na y = {y_osl}")

    bcs = [
        fem.dirichletbc(nula_x, dofs_A_x, V.sub(0)),   # ux = 0 u A
        fem.dirichletbc(nula_y, dofs_A_y, V.sub(1)),   # uy = 0 u A
        fem.dirichletbc(nula_y, dofs_B_y, V.sub(1)),   # uy = 0 u B
    ]

    problem = LinearProblem(
        a, Lf, bcs=bcs,
        petsc_options_prefix=f"greda_{kratko}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez
    uh.name = "pomjeranje"

    # --- naponi ---
    sxx = fem.Function(S1, name="sigma_xx")
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

    sxy = fem.Function(S1, name="tau_xy")
    sxy.interpolate(fem.Expression(sigma(uh)[0, 1], S1.element.interpolation_points))

    # --- ugib: apsolutni i relativni ---
    # Relativni se mjeri u odnosu na srednje vertikalno pomjeranje krajnjih
    # presjeka; postupak je isti za sva tri slucaja, pa su rezultati uporedivi.
    y_uz = np.clip(np.linspace(0.0, h, 5), 1e-9, h - 1e-9)
    uy_A = np.mean([vrijednost_u_tacki(uh, 0.0, yy)[1] for yy in y_uz])
    uy_B = np.mean([vrijednost_u_tacki(uh, L,   yy)[1] for yy in y_uz])
    datum = 0.5 * (uy_A + uy_B)

    uy_mid = vrijednost_u_tacki(uh, L/2.0, h/2.0)[1]
    w_aps = abs(uy_mid)
    w_rel = abs(uy_mid - datum)

    # --- profil sigma_xx na sredini raspona ---
    y_prof = np.clip(np.linspace(0.0, h, 21), 1e-9, h - 1e-9)
    prof_sxx = np.array([vrijednost_u_tacki(sxx, L/2.0, yy)[0] for yy in y_prof])

    # --- profili tau_xy u dva presjeka ---
    prof_tau = []
    for x_p, _ in presjeci:
        prof_tau.append(np.array([vrijednost_u_tacki(sxy, x_p, yy)[0]
                                  for yy in y_prof]))

    # --- izvoz za ParaView (VTX / ADIOS2) ---
    # Sva tri polja idu na JEDNU mrezu, pa ih ParaView vidi odjednom.
    u1 = fem.Function(V1, name="pomjeranje")
    u1.interpolate(uh)

    with VTXWriter(domen.comm, f"primjer5_{kratko}.bp",
                   [u1, sxx, sxy], engine="BP4") as vtx:
        vtx.write(0.0)

    return w_aps, w_rel, y_prof, prof_sxx, prof_tau

# ============================================================
# 5) PETLJA PO SLUCAJEVIMA + TABELA
# ============================================================
rezultati = []

for (y_osl, opis, kratko, boja) in slucajevi:
    w_aps, w_rel, y_prof, prof_sxx, prof_tau = rijesi(y_osl, kratko)
    rezultati.append(dict(y_osl=y_osl, opis=opis, kratko=kratko, boja=boja,
                          w_aps=w_aps, w_rel=w_rel, y=y_prof,
                          sxx=prof_sxx, tau=prof_tau))

print(f"\n--- UPOREDNI PREGLED ---")
print(f"{'oslonac':>18} {'y_osl [m]':>10} {'w_aps [mm]':>12} {'w_rel [mm]':>12} "
      f"{'sxx dolje [MPa]':>17} {'sxx gore [MPa]':>16}")
print("-" * 92)
for r in rezultati:
    print(f"{r['opis']:>18} {r['y_osl']:10.3f} {r['w_aps']*1000:12.5f} "
          f"{r['w_rel']*1000:12.5f} {r['sxx'][0]/1e6:17.4f} {r['sxx'][-1]/1e6:16.4f}")

print(f"\nTeorija (tehnicka): sigma_xx = +/- {M_sred*(h/2)/I_ef/1e6:.4f} MPa")

# --- smicuci napon po presjecima ---
for k, (x_p, opis_p) in enumerate(presjeci):
    V_p = t_q * (L/2.0 - x_p)          # transverzalna sila u presjeku
    print(f"\n--- tau_xy [MPa], presjek {opis_p} ---")
    print(f"Transverzalna sila V = {V_p/1e3:.2f} kN/m, "
          f"tau_max po teoriji = {1.5*V_p/h/1e6:.4f} MPa")
    print(f"{'y [m]':>8}", end="")
    for r in rezultati:
        print(f"{r['opis'][:12]:>14}", end="")
    print(f"{'teorija':>12}")
    y_prof = rezultati[0]["y"]
    yb = y_prof - h/2.0
    tau_teor = -V_p * ((h/2.0)**2 - yb**2) / (2.0 * I_ef)
    for i in range(0, len(y_prof), 5):
        print(f"{y_prof[i]:8.3f}", end="")
        for r in rezultati:
            print(f"{r['tau'][k][i]/1e6:14.4f}", end="")
        print(f"{tau_teor[i]/1e6:12.4f}")

# ============================================================
# 6) GRAFICI
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
y_prof = rezultati[0]["y"]

# --- (a) sigma_xx na sredini raspona ---
for r in rezultati:
    ax[0].plot(r["sxx"] / 1e6, r["y"], "-", lw=1.6, color=r["boja"],
               label=f"oslonac: {r['opis']}")
s_teor = M_sred * (h/2.0 - y_prof) / I_ef
ax[0].plot(s_teor / 1e6, y_prof, "k--", lw=2, label="tehnicka teorija")
ax[0].set_xlabel("sigma_xx [MPa]")
ax[0].set_ylabel("y - visina presjeka [m]")
ax[0].set_title("(a) sigma_xx na sredini raspona (x = L/2)", fontsize=10)
ax[0].grid(True, alpha=0.3)
ax[0].legend(fontsize=8)

# --- (b) i (c) tau_xy u dva presjeka ---
for k, (x_p, opis_p) in enumerate(presjeci):
    V_p = t_q * (L/2.0 - x_p)
    yb = y_prof - h/2.0
    tau_teor = -V_p * ((h/2.0)**2 - yb**2) / (2.0 * I_ef)

    for r in rezultati:
        ax[k+1].plot(r["tau"][k] / 1e6, r["y"], "-", lw=1.6, color=r["boja"],
                     label=f"oslonac: {r['opis']}")
    ax[k+1].plot(tau_teor / 1e6, y_prof, "k--", lw=2, label="teorija V*S/(I*b)")
    ax[k+1].axvline(0.0, color="gray", ls=":", lw=1)
    ax[k+1].set_xlabel("tau_xy [MPa]")
    ax[k+1].set_title(f"({'bc'[k]}) tau_xy, {opis_p}", fontsize=10)
    ax[k+1].grid(True, alpha=0.3)
    ax[k+1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("primjer5_polozaj_oslonca.png", dpi=150)
print("\nGrafik sacuvan: primjer5_polozaj_oslonca.png")
print("ParaView izvozi: primjer5_donja_ivica / _teziste / _gornja_ivica (.xdmf/.h5)")