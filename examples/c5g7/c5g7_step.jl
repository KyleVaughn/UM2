# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.
using MOCNeutronTransport

filename = "c5g7.step"
add_timestamps_to_logger()

gmsh.initialize()
gmsh.option.set_number("General.NumThreads", 0) # 0 uses system default, i.e. OMP_NUM_THREADS)
gmsh.option.set_number("Geometry.OCCParallel", 1) # use parallel OCC boolean operations
gmsh.option.set_number("General.Verbosity", 2) # 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug

materials = import_model(filename, names = true)

lat_div = Vec{4,Float64}([i*21.42 for i = 0:3])
lattice_grid = RectilinearGrid(lat_div, lat_div)
module_grid = lattice_grid # assembly modular ray tracing
coarse_div = Vec{52,Float64}([1.26*i for i ∈ 0:17*3])
coarse_grid = RectilinearGrid(coarse_div, coarse_div) 
mpact_grid = MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)

# See http://juliagraphics.github.io/Colors.jl/stable/namedcolors/ for named colors
# or use an RGB tuple
push!(materials, Material(name="Moderator", color="sienna", mesh_size=1.0))
overlay_mpact_grid_hierarchy(mpact_grid, materials)

Σₜᵢ_FC  = [1.90730E-01, 4.56520E-01, 6.40700E-01, 6.49840E-01, 6.70630E-01, 8.75060E-01, 1.43450E+00]
Σₜᵢ_GT  = [1.90730E-01, 4.56520E-01, 6.40670E-01, 6.49670E-01, 6.70580E-01, 8.75050E-01, 1.43450E+00]
Σₜᵢ_M43 = [2.11920E-01, 3.55810E-01, 4.88900E-01, 5.71940E-01, 4.32390E-01, 6.84950E-01, 6.88910E-01]
Σₜᵢ_M70 = [2.14540E-01, 3.59350E-01, 4.98910E-01, 5.96220E-01, 4.80350E-01, 8.39360E-01, 8.59480E-01]
Σₜᵢ_M87 = [2.16280E-01, 3.61700E-01, 5.05630E-01, 6.11170E-01, 5.08900E-01, 9.26670E-01, 9.60990E-01]
Σₜᵢ_UO2 = [2.12450E-01, 3.55470E-01, 4.85540E-01, 5.59400E-01, 3.18030E-01, 4.01460E-01, 5.70610E-01]
Σₜᵢ_Mod = [2.30070E-01, 7.76460E-01, 1.48420E+00, 1.50520E+00, 1.55920E+00, 2.02540E+00, 3.30570E+00]

frac_of_MFP = 1//4
materials[1].mesh_size = frac_of_MFP*inv(mean(Σₜᵢ_FC )) # Fission Chamber
materials[2].mesh_size = frac_of_MFP*inv(mean(Σₜᵢ_GT )) # Guide Tube
materials[3].mesh_size = frac_of_MFP*inv(mean(Σₜᵢ_M43)) # MOX-4.3% enriched
materials[4].mesh_size = frac_of_MFP*inv(mean(Σₜᵢ_M70)) # MOX-7.0% enriched
materials[5].mesh_size = frac_of_MFP*inv(mean(Σₜᵢ_M87)) # MOX-8.7% enriched
materials[6].mesh_size = frac_of_MFP*inv(mean(Σₜᵢ_UO2)) # Uranium Oxide
materials[7].mesh_size = frac_of_MFP*inv(mean(Σₜᵢ_Mod)) # Moderator

#gmsh.model.mesh.set_size(gmsh.model.get_entities(0), 0.05)
#gmsh.model.mesh.generate(2)
set_mesh_field_using_materials(materials)
#println("Mesh->Define->Size fields. Then, new view of Min field.")
#gmsh.fltk.run()
##gmsh.view.write(0, filename[1:end-5]*".pos")
#field_data = gmsh.view.get_model_data(0, 0)
generate_mesh(order = 1, faces = "Triangle", opt_iters = 2)
gmsh.write("c5g7.inp")
mesh_error = get_cad_to_mesh_error()
for i in eachindex(mesh_error)
    println(mesh_error[i])
end
gmsh.fltk.run()
gmsh.finalize()

mesh = import_mesh("c5g7.inp")
statistics(mesh)
# Partition mesh according to mpact grid hierarchy, and write as an xdmf file
mpt = partition(mesh)
export_mesh(mpt, "c5g7.xdmf")
