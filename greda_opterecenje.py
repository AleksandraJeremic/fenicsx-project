"""
PROSTA AB GREDA - POREDJENJE MJESTA NANOSENJA OPTERECENJA
==========================================================
Poredi dva slucaja:
  1. Opterecenje na gornjoj ivici (pritisak odozgo)
  2. Opterecenje na donjoj ivici (sila prema dolje)

Posmatramo:
  - Maksimalni progib
  - Naponski dijagram po presjeku (x = L/2)
  - Von Mises napon
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace, dirichletbc, Expression, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary
from dolfinx.io import XDMFFile
import ufl
from ufl import dx, inner, grad, sym, Identity, tr, sqrt

# ============================================================
# MATERIJALNE KARAKTERISTIKE - Beton C25/30
# ============================================================
E   = 31e9
nu  = 0.2
rho = 2500.0
g   = 9.81

lam    = E * nu / ((1 + nu) * (1 - 2*nu))
mu     = E / (2 * (1 + nu))
lam_ps = 2 * lam * mu / (lam + 2 * mu)

# ============================================================
# GEOMETRIJA
# ============================================================
L  = 6.0    # Raspon [m]
H  = 0.45   # Visina presjeka [m]
b  = 0.30   # Sirina presjeka [m]
nx = 60
ny = 10

# Opterecenje
q_ukupno = 4000.0   # Pa (stalno + korisno)


# ============================================================
# TENZORI
# ============================================================
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lam_ps * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)

def sigma_vm(u):
    s = sigma(u) - (1/3) * tr(sigma(u)) * Identity(len(u))
    return sqrt(3/2 * inner(s, s))


# ============================================================
# POMOCNE FUNKCIJE
# ============================================================
def dofs_tacka(domain, V_sub, marker_fn):
    vertices = locate_entities_boundary(domain, 0, marker_fn)
    return fem.locate_dofs_topological(V_sub, 0, vertices)


# ============================================================
# FUNKCIJA ZA MODELIRANJE
# ============================================================
def modeliraj_gredu(tip_opterecenja):
    """
    tip_opterecenja:
      'gornja' - sila djeluje na gornjoj ivici
      'donja'  - sila djeluje na donjoj ivici
    """
    domain = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([L, H])],
        [nx, ny],
        CellType.triangle
    )

    V   = functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    tol = 1e-10

    def lijevi_ugao(x):
        return np.logical_and(np.abs(x[0]) < tol, np.abs(x[1]) < tol)

    def desni_ugao(x):
        return np.logical_and(np.abs(x[0] - L) < tol, np.abs(x[1]) < tol)

    def gornja_ivica(x):
        return np.abs(x[1] - H) < tol

    def donja_ivica(x):
        return np.abs(x[1]) < tol

    zero = fem.Constant(domain, 0.0)
    bcs  = [
        dirichletbc(zero, dofs_tacka(domain, V.sub(0), lijevi_ugao), V.sub(0)),
        dirichletbc(zero, dofs_tacka(domain, V.sub(1), lijevi_ugao), V.sub(1)),
        dirichletbc(zero, dofs_tacka(domain, V.sub(1), desni_ugao),  V.sub(1)),
    ]

    # --- Oznaci ivicu na kojoj djeluje opterecenje ---
    fdim = domain.topology.dim - 1

    if tip_opterecenja == 'gornja':
        ivica_fn = gornja_ivica
    else:
        ivica_fn = donja_ivica

    facets_opt = np.sort(locate_entities_boundary(domain, fdim, ivica_fn))
    facet_tags = mesh.meshtags(
        domain, fdim,
        facets_opt, np.full(len(facets_opt), 1, dtype=np.int32)
    )
    ds_opt = ufl.Measure("ds", domain=domain,
                         subdomain_data=facet_tags, subdomain_id=1)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Vlastita tezina kroz cijeli volumen
    f_vol = fem.Constant(domain, np.array([0.0, -rho * g]))

    # Vanjsko opterecenje na odabranoj ivici (prema dolje)
    T = fem.Constant(domain, np.array([0.0, -q_ukupno]))

    a      = inner(sigma(u), epsilon(v)) * dx
    L_form = inner(f_vol, v) * dx + inner(T, v) * ds_opt

    problem = LinearProblem(a, L_form, bcs=bcs,
                            petsc_options_prefix=f"opt_{tip_opterecenja}_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh, domain, V


# ============================================================
# POKRETANJE ZA OBA SLUCAJA
# ============================================================
print("=" * 65)
print("POREDJENJE MJESTA NANOSENJA OPTERECENJA")
print("=" * 65)
print(f"  Greda:       L={L}m, H={H}m, b={b}m  (beton C25/30)")
print(f"  Opterecenje: vlastita tezina + {q_ukupno/1000} kN/m2")
print("=" * 65)

rezultati = {}
for tip in ['gornja', 'donja']:
    uh, domain, V = modeliraj_gredu(tip)
    u_vals = uh.x.array.reshape(-1, 2)
    max_uy = np.min(u_vals[:, 1])
    rezultati[tip] = {'uh': uh, 'domain': domain, 'V': V, 'max_uy': max_uy}
    print(f"\n  Opterecenje na {tip}j ivici:")
    print(f"    max |u_y| = {abs(max_uy)*1000:.4f} mm")

print("=" * 65)


# ============================================================
# NAPONSKI DIJAGRAM PO PRESJEKU (x = L/2)
# ============================================================

# Pomocna funkcija za ekstrakciju napona po presjeku
def naponi_po_presjeku(uh, domain):
    coords  = domain.geometry.x
    x_n     = coords[:, 0]
    y_n     = coords[:, 1]
    cells_a = domain.geometry.dofmap.reshape(-1, 3)
    n_nodes = len(x_n)

    # DG0 prostor za napone
    W = functionspace(domain, ("DG", 0))

    def sigma_xx_fn(u): return sigma(u)[0, 0]
    def sigma_yy_fn(u): return sigma(u)[1, 1]
    def sigma_xy_fn(u): return sigma(u)[0, 1]

    sxx_h = Function(W)
    syy_h = Function(W)
    sxy_h = Function(W)
    vm_h  = Function(W)

    sxx_h.interpolate(Expression(sigma_xx_fn(uh), W.element.interpolation_points))
    syy_h.interpolate(Expression(sigma_yy_fn(uh), W.element.interpolation_points))
    sxy_h.interpolate(Expression(sigma_xy_fn(uh), W.element.interpolation_points))
    vm_h.interpolate(Expression(sigma_vm(uh),     W.element.interpolation_points))

    # Prosjecavanje DG0 na cvorove
    def elem_to_node(vals):
        out = np.zeros(n_nodes)
        cnt = np.zeros(n_nodes)
        for i, tri in enumerate(cells_a):
            for nd in tri:
                out[nd] += vals[i]
                cnt[nd] += 1
        return out / np.maximum(cnt, 1)

    sxx_n = elem_to_node(sxx_h.x.array)
    syy_n = elem_to_node(syy_h.x.array)
    sxy_n = elem_to_node(sxy_h.x.array)
    vm_n  = elem_to_node(vm_h.x.array)

    # Cvorovi blizu sredine grede (x = L/2)
    tol_x = L / nx * 1.5
    maska  = np.abs(x_n - L/2) < tol_x

    y_mid   = y_n[maska]
    sxx_mid = sxx_n[maska] / 1e6   # MPa
    syy_mid = syy_n[maska] / 1e6
    sxy_mid = sxy_n[maska] / 1e6
    vm_mid  = vm_n[maska]  / 1e6

    idx = np.argsort(y_mid)
    return y_mid[idx], sxx_mid[idx], syy_mid[idx], sxy_mid[idx], vm_mid[idx]


# Ekstrakcija za oba slucaja
y_g, sxx_g, syy_g, sxy_g, vm_g = naponi_po_presjeku(
    rezultati['gornja']['uh'], rezultati['gornja']['domain'])

y_d, sxx_d, syy_d, sxy_d, vm_d = naponi_po_presjeku(
    rezultati['donja']['uh'],  rezultati['donja']['domain'])

# EB analiticka krivulja za sigma_xx
M_max  = abs(-q_ukupno) * b * L**2 / 8 + rho*g*b*H * L**2 / 8
I_val  = b * H**3 / 12
y_EB   = np.linspace(0, H, 100)
sxx_EB = M_max * (y_EB - H/2) / I_val / 1e6   # MPa


# ============================================================
# GRAFICKI PRIKAZ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 7))

# --- sigma_xx ---
ax = axes[0]
ax.plot(sxx_g, y_g, 'b-o', markersize=4, linewidth=2,
        label='Gornja ivica (FEM)')
ax.plot(sxx_d, y_d, 'r-s', markersize=4, linewidth=2,
        label='Donja ivica (FEM)')
ax.plot(sxx_EB, y_EB, 'k--', linewidth=1.5,
        label='Euler-Bernoulli')
ax.axvline(0,   color='k',    linewidth=0.8, linestyle=':')
ax.axhline(H/2, color='gray', linewidth=0.8, linestyle='--',
           label='Neutralna osa')
ax.set_xlabel('σ_xx [MPa]', fontsize=12)
ax.set_ylabel('y [m]', fontsize=12)
ax.set_title('Normalni napon σ_xx\n(x = L/2)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- sigma_xy ---
ax = axes[1]
ax.plot(sxy_g, y_g, 'b-o', markersize=4, linewidth=2,
        label='Gornja ivica (FEM)')
ax.plot(sxy_d, y_d, 'r-s', markersize=4, linewidth=2,
        label='Donja ivica (FEM)')
ax.axvline(0,   color='k',    linewidth=0.8, linestyle=':')
ax.axhline(H/2, color='gray', linewidth=0.8, linestyle='--',
           label='Neutralna osa')
ax.set_xlabel('σ_xy [MPa]', fontsize=12)
ax.set_ylabel('y [m]', fontsize=12)
ax.set_title('Smicajni napon σ_xy\n(x = L/2)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Von Mises ---
ax = axes[2]
ax.plot(vm_g, y_g, 'b-o', markersize=4, linewidth=2,
        label='Gornja ivica (FEM)')
ax.plot(vm_d, y_d, 'r-s', markersize=4, linewidth=2,
        label='Donja ivica (FEM)')
ax.axhline(H/2, color='gray', linewidth=0.8, linestyle='--',
           label='Neutralna osa')
ax.set_xlabel('σ_VM [MPa]', fontsize=12)
ax.set_ylabel('y [m]', fontsize=12)
ax.set_title('Von Mises napon σ_VM\n(x = L/2)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Poredjenje mjesta nanosenja opterecenja  (x = L/2 = {L/2} m)\n'
    f'Plavo = gornja ivica  |  Crveno = donja ivica',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('greda_opterecenje_poredjenje.png', dpi=150, bbox_inches='tight')
print("\nGrafik sacuvan: greda_opterecenje_poredjenje.png")
plt.show()


# ============================================================
# IZVOZ ZA PARAVIEW
# ============================================================
print("\nIzvoz za ParaView...")
for tip in ['gornja', 'donja']:
    uh     = rezultati[tip]['uh']
    domain = rezultati[tip]['domain']
    V      = rezultati[tip]['V']

    V1  = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    uh1 = Function(V1)
    uh1.interpolate(uh)
    uh1.name = "pomjeranja"

    with XDMFFile(MPI.COMM_WORLD, f"greda_opt_{tip}_pomjeranja.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh1)

    V_scalar = functionspace(domain, ("Lagrange", 1))
    vm_h     = Function(V_scalar)
    vm_h.interpolate(Expression(sigma_vm(uh), V_scalar.element.interpolation_points))
    vm_h.name = "von_mises"

    with XDMFFile(MPI.COMM_WORLD, f"greda_opt_{tip}_naponi.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(vm_h)

    print(f"  Sacuvano: greda_opt_{tip}_pomjeranja.xdmf")
    print(f"  Sacuvano: greda_opt_{tip}_naponi.xdmf")

print("\nGotovo!")