using UM2 

file_prefix = "fnr_slice"
mesh_order = 1
mesh_faces = "Triangle"
lc = 0.2

lc_str = string(round(lc, digits = 4))    
full_file_prefix = file_prefix * "_" * lowercase(mesh_faces) * string(mesh_order) *    
                   "_" * replace(lc_str, "."=>"_")

add_timestamps_to_logger()    
gmsh.initialize()    
gmsh.option.set_number("General.NumThreads", 0) # 0 uses system default, i.e. OMP_NUM_THREADS)   
gmsh.option.set_number("Geometry.OCCParallel", 1) # use parallel OCC boolean operations
gmsh.option.set_number("General.Verbosity", 4) # 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug

# Model
# ----------------------------------------------------------------------------------------
# Import CAD file
gmsh.merge("fnr_core_slice.step")
# Visualize the model
gmsh.fltk.run()

## Shift the model to the origin of the xy-plane
## Get the dim tags of all dimension 2 entites
#dim_tags = gmsh.model.get_entities(2)
## Translate it so that the corner is at the origin
#gmsh.model.occ.translate(dim_tags, 407.51125 + 10 * grid_offset,
#                         307.848 + 10 * grid_offset,
#                         -300)
## Model is in mm. Convert to cm by shrinking everything by 1/10
#gmsh.model.occ.dilate(dim_tags, 0, 0, 0, 1 // 10, 1 // 10, 0)
## Synchronize the CAD and Gmsh models
#gmsh.model.occ.synchronize()
## Verify results
#gmsh.fltk.run()
#
#
#
#
#
## I can't remember if the bottom left corner of the CMFD grid needs to be (0, 0, 0)
## To compensate, we do some extra math to ensure that happens.
#grid_offset = 0.1 # Buffer between boundary of the model and the edge of the CMFD grid (cm)
## Bounding box for the CMFD grid
## xmin, xmax, ymin, ymax
#bb = (0.0, 80.62054 + 2 * grid_offset,
#      0.0, 46.2645 + 2 * grid_offset)
#
#
#
## Assign materials
## ------------------------------------------------------------------------------------------
#all_tags = getindex.(gmsh.model.get_entities(2), 2)
## Assign the materials as physical groups
#p = gmsh.model.add_physical_group(2, all_tags)
#gmsh.model.set_physical_name(2, p, "MATERIAL_UO2")
#
## Overlay CMFD/hierarchical grid
## -------------------------------------------------------------------------------------------
#nx = [10, 1]
#ny = [6, 1]
#grid_tags = gmsh_overlay_rectangular_grid(bb, "MATERIAL_MODERATOR", nx, ny)
## Verify results
#gmsh.fltk.run()
#
## Mesh
## ------------------------------------------------------------------------------------------------
## Set the characteristic edge length of the mesh cells
#lc = 0.3 # cm
#gmsh.model.mesh.set_size(gmsh.model.get_entities(0), lc)
## Optional mesh optimization:
#niter = 2 # The optimization iterations
#
## 2nd order triangles
#gmsh.model.mesh.generate(2) # Triangles first for high order meshes.
#gmsh.model.mesh.set_order(2)
#for () in 1:niter
#    gmsh.model.mesh.optimize("HighOrderElastic")
#    gmsh.model.mesh.optimize("Relocate2D")
#    gmsh.model.mesh.optimize("HighOrderElastic")
#end
#
## gmsh.fltk.run()
#
## Mesh conversion
## ----------------------------------------------------------------------------------------------------
## We want to write an Abaqus file from Gmsh, read the Abaqus file into Julia, and convert that into
## a rectangularly partitioned hierarchical XDMF file, the final input to MPACT
##
## Write Abaqus file
#gmsh.write("fnr_slice.inp")
#gmsh.finalize() # done with Gmsh. Finalize
#mesh = read_abaqus_2d("fnr_slice.inp")
## Convert mesh into Hierarchical Rectangularly Partitioned Mesh 
#HRPM = partition_rectangularly(mesh)
#write_xdmf_2d("fnr_slice.xdmf", HRPM)
##
### Mass conservation
### ----------------------------------------------------------------------------------------------------
### See how well your mesh conserves mass. Analytic solution not available, so fine tri6 mesh used as 
### reference
##fuel_area = area(mesh, "MATERIAL_UO2")
##fuel_area_ref = 6.119632103711788
##println("Fuel area (reference): $fuel_area_ref")
##println("Fuel area ( computed): $fuel_area")
##err = 100*(fuel_area - fuel_area_ref)/fuel_area_ref
##println("Error (relative): $err %") 
##println("")
##
##clad_area = area(mesh, "MATERIAL_CLAD")
##clad_area_ref = 19.884367501087866
##println("Clad area (reference): $clad_area_ref")
##println("Clad area ( computed): $clad_area")
##err = 100*(clad_area - clad_area_ref)/clad_area_ref
##println("Error (relative): $err %") 
