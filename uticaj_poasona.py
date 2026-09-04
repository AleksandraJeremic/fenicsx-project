# ============================================================
# PRIMJER: UTICAJ POISSONOVOG KOEFICIJENTA
#
# Prosta greda se rjesava za niz vrijednosti Poissonovog koeficijenta,
# a rezultat se poredi sa tri teorijska rjesenja:
#
#   Bernoulli-Ojler  : w = 5qL^4/(384EI)              - bez smicanja
#   Timosenko        : w = w_EB + qL^2/(8*k*G*A)      - smicanje preko G i k = 5/6
#   Tacno 2D rjesenje: Timoshenko, Theory of Elasticity, jed. (34), str. 43
#                      w = w_EB * [1 + (12/5)(h/L)^2 * (4/5 + nu/2)]
#
# Ocekivanje: pri nu = 0 sva tri rjesenja se poklapaju, cime se
# potvrdjuje da je koeficijent korekcije smicanja k = 5/6 ispravan.
# Sa porastom nu Timosenkova gredna teorija precjenjuje uticaj, jer
# Poissonov efekat hvata samo preko modula smicanja G = E/(2(1+nu)),
# dok koeficijent k ostaje fiksan. Ravanski model to hvata tacno.
#
# Za gredu je usvojen odnos L/h = 5, jer je uticaj smicanja srazmjeran
# kvadratu (h/L) pa je kod zdepastije grede razlika jasno vidljiva.
# ============================================================

from mpi4py import MPI
import numpy as np
import ufl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags

# ============================================================
# 1) ULAZNI PODACI
# ============================================================
L  = 5.0             # raspon [m]
h  = 1.00            # visina presjeka [m]   -> L/h = 5
b  = 0.30            # sirina presjeka [m]

E  = 31.0e9          # modul elasticnosti [Pa]

q   = 20.0e3         # linijsko opterecenje [N/m]
t_q = q / b          # povrsinski pritisak za plane stress [Pa]

Nx, Ny = 80, 16       # mreza (elementi 0,125 x 0,125 m)

# Poissonovi koeficijenti koji se ispituju
nu_lista = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45]

I_ef = h**3 / 12.0
A_ef = h
k_s  = 5.0 / 6.0

# Bernoullijev ugib i napon ne zavise od nu
w_EB  = 5.0 * t_q * L**4 / (384.0 * E * I_ef)
M_sr  = t_q * L**2 / 8.0
s_EB  = M_sr * (h/2.0) / I_ef
s_2D  = s_EB + t_q / 5.0        # tacno 2D rjesenje, jed. (33) - ne zavisi od nu

# ============================================================
# 2) MREZA I PROSTOR (isti za sve nu)
# ============================================================
domen = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, h])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)
V  = fem.functionspace(domen, ("Lagrange", 2, (domen.geometry.dim,)))
S1 = fem.functionspace(domen, ("Lagrange", 1))

fdim = domen.topology.dim - 1

# --- granicni uslovi: uy = 0 duz obje krajnje ivice ---
Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
nula_x = fem.Function(Vx)
nula_y = fem.Function(Vy)

f_L = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], 0.0))
f_D = locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[0], L))
dofs_L = fem.locate_dofs_topological((V.sub(1), Vy), fdim, f_L)
dofs_D = fem.locate_dofs_topological((V.sub(1), Vy), fdim, f_D)

dofs_ux = fem.locate_dofs_geometrical(
    (V.sub(0), Vx),
    lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], h/2.0)))

bcs = [fem.dirichletbc(nula_y, dofs_L, V.sub(1)),
       fem.dirichletbc(nula_y, dofs_D, V.sub(1)),
       fem.dirichletbc(nula_x, dofs_ux, V.sub(0))]

# --- opterecenje na gornjoj ivici ---
f_gore = np.sort(locate_entities_boundary(domen, fdim, lambda x: np.isclose(x[1], h)))
oznake = meshtags(domen, fdim, f_gore, np.full(len(f_gore), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=domen, subdomain_data=oznake)
T = fem.Constant(domen, np.array([0.0, -t_q], dtype=np.float64))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

stablo = geometry.bb_tree(domen, domen.topology.dim)

def vrijednost_u_tacki(funkcija, x, y):
    tacka = np.array([[x, y, 0.0]], dtype=np.float64)
    kand = geometry.compute_collisions_points(stablo, tacka)
    cel = geometry.compute_colliding_cells(domen, kand, tacka)
    return funkcija.eval(tacka, np.array([cel.links(0)[0]], dtype=np.int32))

# ============================================================
# 3) FUNKCIJA: rjesenje za zadato nu
# ============================================================
def rijesi(nu):
    # Lameove konstante se racunaju IZNOVA za svako nu
    mu  = E / (2.0 * (1.0 + nu))
    lam = E * nu / (1.0 - nu**2)

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0*mu*epsilon(w) + lam*ufl.tr(epsilon(w))*ufl.Identity(2)

    a  = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    Lf = ufl.dot(T, v) * ds(1)

    problem = LinearProblem(
        a, Lf, bcs=bcs,
        petsc_options_prefix=f"nu_{int(nu*100)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    rez = problem.solve()
    uh = rez[0] if isinstance(rez, tuple) else rez

    sxx = fem.Function(S1)
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], S1.element.interpolation_points))

    w  = abs(vrijednost_u_tacki(uh, L/2.0, h/2.0)[1])
    sx = vrijednost_u_tacki(sxx, L/2.0, 1e-9)[0]
    return w, sx

# ============================================================
# 4) PRORACUN I TABELA
# ============================================================
print("=" * 104)
print(f"UTICAJ POISSONOVOG KOEFICIJENTA   (greda {b*100:.0f}/{h*100:.0f}, "
      f"L = {L:.1f} m, L/h = {L/h:.0f}, mreza {Nx}x{Ny})")
print("=" * 104)
print(f"\nBernoulli-Ojler:  w = {w_EB*1000:.5f} mm,  sigma = {s_EB/1e6:.4f} MPa")
print(f"Tacno 2D, napon:  sigma = {s_2D/1e6:.4f} MPa   (ne zavisi od nu)\n")

print(f"{'nu':>6} {'w MKE [mm]':>12} {'w Timos.':>11} {'w tacno 2D':>12} "
      f"{'MKE-EB [%]':>11} {'Tim-EB [%]':>11} {'2D-EB [%]':>10} "
      f"{'MKE vs 2D [%]':>14} {'sigma [MPa]':>12}")
print("-" * 104)

nu_niz, r_mke, r_tim, r_2d = [], [], [], []
e_mke, e_tim = [], []          # greske u odnosu na tacno 2D rjesenje

for nu in nu_lista:
    G     = E / (2.0 * (1.0 + nu))
    w_TIM = w_EB + t_q * L**2 / (8.0 * k_s * G * A_ef)
    w_2D  = w_EB * (1.0 + (12.0/5.0)*(h/L)**2*(4.0/5.0 + nu/2.0))

    w, sx = rijesi(nu)

    nu_niz.append(nu)
    r_mke.append((w - w_EB)/w_EB*100.0)
    r_tim.append((w_TIM - w_EB)/w_EB*100.0)
    r_2d.append((w_2D - w_EB)/w_EB*100.0)
    e_mke.append((w - w_2D)/w_2D*100.0)
    e_tim.append((w_TIM - w_2D)/w_2D*100.0)

    print(f"{nu:6.2f} {w*1000:12.5f} {w_TIM*1000:11.5f} {w_2D*1000:12.5f} "
          f"{r_mke[-1]:11.3f} {r_tim[-1]:11.3f} {r_2d[-1]:10.3f} "
          f"{(w-w_2D)/w_2D*100:14.3f} {sx/1e6:12.4f}")
# ============================================================
# 5) GRESKE U ODNOSU NA TACNO 2D RJESENJE
# ============================================================
# Prva tabela pokazuje KOLIKO smicanje utice (osnova je Bernoulli).
# Ova pokazuje KOLIKO JE KOJI MODEL TACAN (osnova je tacno rjesenje).
print("\n" + "=" * 60)
print("GRESKE U ODNOSU NA TACNO 2D RJESENJE (jed. 34)")
print("=" * 60)
print(f"{'nu':>6} {'greska MKE [%]':>17} {'greska Timosenka [%]':>23}")
print("-" * 60)
for i, nu in enumerate(nu_niz):
    print(f"{nu:6.2f} {e_mke[i]:17.3f} {e_tim[i]:23.3f}")


# ============================================================
# 6) GRAFICI
# ============================================================
nu_g = np.linspace(0.0, 0.5, 200)

# krive u odnosu na Bernoullija
kriva_2D  = (12.0/5.0)*(h/L)**2*(4.0/5.0 + nu_g/2.0)*100.0
kriva_TIM = 1.92*(1.0 + nu_g)*(h/L)**2*100.0

# greska Timosenka u odnosu na tacno rjesenje
w_rel_TIM = 1.0 + kriva_TIM/100.0
w_rel_2D  = 1.0 + kriva_2D/100.0
greska_TIM = (w_rel_TIM - w_rel_2D)/w_rel_2D*100.0

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# --- (a) u odnosu na Bernoullija: koliko smicanje utice ---
ax[0].plot(nu_g, kriva_2D,  "k-",  lw=2,
           label="tacno 2D rjesenje (jed. 34)")
ax[0].plot(nu_g, kriva_TIM, "g--", lw=1.8,
           label="Timosenkova gredna teorija")
ax[0].plot(nu_niz, r_mke, "o", color="tab:blue", ms=8,
           label="MKE (ravanski model)")
ax[0].set_xlabel("Poissonov koeficijent  nu")
ax[0].set_ylabel("razlika u odnosu na Bernoullija [%]")
ax[0].set_title("(a) Koliki je uticaj smicanja", fontsize=11)
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=9)

# --- (b) u odnosu na tacno rjesenje: koliko je koji model tacan ---
ax[1].axhline(0.0, color="k", lw=2, label="tacno 2D rjesenje (osnova)")
ax[1].plot(nu_g, greska_TIM, "g--", lw=1.8,
           label="Timosenkova gredna teorija")
ax[1].plot(nu_niz, e_mke, "o", color="tab:blue", ms=8,
           label="MKE (ravanski model)")
ax[1].set_xlabel("Poissonov koeficijent  nu")
ax[1].set_ylabel("greska u odnosu na tacno rjesenje [%]")
ax[1].set_title("(b) Koliko je koji model tacan", fontsize=11)
ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("poissonov_efekat.png", dpi=150)
print("\nGrafik sacuvan: poissonov_efekat.png")