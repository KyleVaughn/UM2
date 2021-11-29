using MOCNeutronTransport
using StaticArrays
using BenchmarkTools

generate_mesh_file = false
# Model
# ----------------------------------------------------------------------------------------------
log_timestamps()
if generate_mesh_file
    gmsh.initialize()
    # Set gmsh log level to warnings and errors only
    gmsh.option.set_number("General.Verbosity", 3)
    uo2_entities = Int32[]
    gap_entities = Int32[]
    clad_entities = Int32[]
    h2o_entities = Int32[]
    bb = (0.0, 2.52, 0.0, 2.52)
    
    # UO2 pins
    # ----------------------------------------------------------------------------------------------
    r_fuel = 0.4096
    r_gap = 0.418
    r_clad = 0.475
    pitch = 1.26
    coords_gt = [ 
        (2, 2) 
    ]
    # Instrument tube locations (pg. 5)
    for i = 1:2
        for j = 1:2
            if (i, j) ∈  coords_gt
              continue
            end
            push!(uo2_entities, 
                  gmsh.model.occ.add_disk(i*pitch - pitch/2, 
                                          j*pitch - pitch/2, 
                                          0, r_fuel, r_fuel))
            push!(gap_entities, 
                  gmsh.model.occ.add_disk(i*pitch - pitch/2, 
                                          j*pitch - pitch/2, 
                                          0, r_gap, r_gap))
            push!(clad_entities, 
                  gmsh.model.occ.add_disk(i*pitch - pitch/2, 
                                          j*pitch - pitch/2, 
                                          0, r_clad, r_clad))
        end
    end
    # Guide tube
    # ----------------------------------------------------------------------------------------------
    r_gt_inner = 0.561
    r_gt_outer = 0.602
    for (i, j) in coords_gt
        push!(h2o_entities,
              gmsh.model.occ.add_disk(i*pitch - pitch/2, 
                                      j*pitch - pitch/2, 
                                      0, r_gt_inner, r_gt_inner))
        push!(clad_entities, 
              gmsh.model.occ.add_disk(i*pitch - pitch/2, 
                                      j*pitch - pitch/2, 
                                      0, r_gt_outer, r_gt_outer))
    end
    gmsh.model.occ.synchronize()
    # Materials
    # ----------------------------------------------------------------------------------------------
    gmsh.model.occ.synchronize()
    p = gmsh.model.add_physical_group(2, uo2_entities)
    gmsh.model.set_physical_name(2, p, "MATERIAL_UO2")
    p = gmsh.model.add_physical_group(2, gap_entities)
    gmsh.model.set_physical_name(2, p, "MATERIAL_GAP")
    p = gmsh.model.add_physical_group(2, h2o_entities)
    gmsh.model.set_physical_name(2, p, "MATERIAL_WATER")
    p = gmsh.model.add_physical_group(2, clad_entities)
    gmsh.model.set_physical_name(2, p, "MATERIAL_CLAD")
    gmsh_group_preserving_fragment(gmsh.model.get_entities(2),
                                   gmsh.model.get_entities(2);
                                   material_hierarchy = ["MATERIAL_UO2", 
                                                         "MATERIAL_GAP",
                                                         "MATERIAL_WATER",
                                                         "MATERIAL_CLAD"])
    
    # Overlay Grid
    # ---------------------------------------------------------------------------------------------
    grid_tags = gmsh_overlay_rectangular_grid(bb, "MATERIAL_WATER", [2], [2]) 
    
    # Mesh
    # ------------------------------------------------------------------------------------------------
    lc = 0.2 # cm
    gmsh.model.mesh.set_size(gmsh.model.get_entities(0), lc)
    # Optional mesh optimization:
    niter = 2 # The optimization iterations
    
    # Triangles
    gmsh.model.mesh.generate(2) # 2 is dimension of mesh
    for () in 1:niter
        gmsh.model.mesh.optimize("Laplace2D")
        gmsh.model.mesh.optimize("Relocate2D")
        gmsh.model.mesh.optimize("Laplace2D")
    end
    
    # Quadrilaterals
    # gmsh.option.set_number("Mesh.RecombineAll", 1) # recombine all triangles
    # gmsh.option.set_number("Mesh.Algorithm", 8) # Frontal-Delaunay for quads. Better 2D algorithm
    # gmsh.option.set_number("Mesh.RecombinationAlgorithm", 1)
    # gmsh.model.mesh.generate(2)
    # for () in 1:niter                            
    #     gmsh.model.mesh.optimize("Laplace2D")
    #     gmsh.model.mesh.optimize("Relocate2D")
    #     gmsh.model.mesh.optimize("Laplace2D")
    # end
    
    # 2nd order triangles
    # gmsh.option.set_number("Mesh.HighOrderOptimize", 2)
    # gmsh.model.mesh.generate(2) # Triangles first for high order meshes.
    # gmsh.model.mesh.set_order(2)
    # for () in 1:niter
    #     gmsh.model.mesh.optimize("HighOrderElastic")
    #     gmsh.model.mesh.optimize("Relocate2D")
    #     gmsh.model.mesh.optimize("HighOrderElastic")
    # end
    
    # 2nd order quadrilaterals
    # gmsh.option.set_number("Mesh.RecombineAll", 1) # recombine all triangles
    # gmsh.option.set_number("Mesh.Algorithm", 8) # Frontal-Delaunay for quads. Better 2D algorithm
    # gmsh.option.set_number("Mesh.RecombinationAlgorithm", 1)
    # gmsh.option.set_number("Mesh.HighOrderOptimize", 2)
    # gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.set_order(2)
    
    
    # Mesh conversion
    #---------------------------------------------------------------------------------------------------
    gmsh.write("2x2.inp")
    gmsh.finalize() # done with Gmsh. Finalize
end    

# Ray tracing
#---------------------------------------------------------------------------------------------------
F = Float64
U = UInt16
mesh = read_abaqus_2d("2x2.inp", F=F)
mesh = add_everything(mesh)
#HRPM = partition_rectangularly(mesh)
tₛ =  F(0.007)
ang_quad = generate_angular_quadrature("Chebyshev-Chebyshev", 32, 3; F=F)
#template_vec = MVector{2, I}(zeros(I, 2))
tracks = generate_tracks(tₛ, ang_quad, mesh, boundary_shape = "Rectangle");
#the_points_E, the_faces_E = ray_trace_edge_to_edge(the_tracks, mesh)
