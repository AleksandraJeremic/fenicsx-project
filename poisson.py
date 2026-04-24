from dolfinx.fem.petsc import LinearProblem
import os
import numpy
import ufl
import dolfinx
from dolfinx import mesh, fem, io, plot 
from mpi4py import MPI
from petsc4py import PETSc
import pyvista as pv

# clears the terminal and prints dolfinx version
os.system('clear')
# prints dolfinx version
print(f"DOLFINx version: {dolfinx.__version__}")

# no of elements in each direction
NElem   = 8

# create a unit square with 8x8 elements with quad elements and use first order shape function
domain  = mesh.create_unit_square(MPI.COMM_WORLD,NElem,NElem,mesh.CellType.quadrilateral)
V       = fem.functionspace(domain,("CG",1))

## define trial and test functions
u       = ufl.TrialFunction(V)
v       = ufl.TestFunction(V)

# source term of the poisson equation
f       = fem.Constant(domain, PETSc.ScalarType(-6))

## applying boundary conditions
uD      = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
tdim    = domain.topology.dim
fdim    = tdim-1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc      = fem.dirichletbc(uD, boundary_dofs)

# bilinear form
a       = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L       = f * v * ufl.dx

# set PETSc solver options
sol_opts = {"ksp_type": "preonly", "pc_type": "lu"}
# formulate the problem
problem = LinearProblem(
    a, L, 
    bcs=[bc], 
    petsc_options=sol_opts,
    petsc_options_prefix="fenics_"
    )
# solve the problem
uh = problem.solve()

## error calculation
V2          = fem.functionspace(domain, ("CG", 2))
uex         = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error    = fem.form(ufl.inner(uh-uex, uh-uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2    = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max   = numpy.max(numpy.abs(uD.x.array-uh.x.array))

# print the error
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# writing output files in xdmf format
with io.XDMFFile(domain.comm, "output.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# 2D contour plots of the mesh and result using pyvista
#pv.start_xvfb()    # Uncomment the line on WSL
#pv.off_screen = True
#topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
#grid = pv.UnstructuredGrid(topology, cell_types, geometry)

#plotter = pv.Plotter()
#plotter.add_mesh(grid,show_edges=True)
#plotter.view_xy()
#plotter.save_graphic('mesh.pdf')

#u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
#u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
#u_grid.point_data["u"] = uh.x.array.real
#u_grid.set_active_scalars("u")
#u_plotter = pv.Plotter()
#u_plotter.add_mesh(u_grid,show_edges=True)
#u_plotter.view_xy()
#u_plotter.save_graphic('contour.pdf')

import pyvista as pv
from dolfinx.plot import vtk_mesh

pv.off_screen = True

# --- MESH ---
topology, cell_types, geometry = vtk_mesh(domain)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.save_graphic("mesh.pdf")

# --- SOLUTION ---
topology, cell_types, geometry = vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")

plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.save_graphic("contour.pdf")
