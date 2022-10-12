#export MPACTMeshHierarchy
#
#struct MPACTMeshHierarchy{M <: AbstractMesh}
#    #      y    
#    #      ^    
#    # j = 3|    
#    # j = 2|    
#    # j = 1|    
#    #       _____________> x    
#    #       i=1  i=2  i=3  
#    grid_hierarchy::MPACTGridHierarchy
#    mesh_ids::Matrix{Matrix{Matrix{UM2_MESH_INT_TYPE}}}
#    pin_meshes::Vector{M}
#end
#
#function MPACTMeshHierarchy(
#        mesh::M, 
#        elsets::Dict{String, Set{I}},
#        grid_hierarchy::MPACTGridHierarchy
#    ) where {M <: AbstractMesh, I <: Integer}
#    npins = 0
#    # Loop through the RT modules to determine the total number of pins
#    # Get the core size in terms of M by N lattices   
#    core_size = size(grid_hierarchy.lattice_grid)
#    # For each lattice
#    for lj in core_size[2], li in core_size[1]
#        # Get the size of the lattice in terms of M by N modules
#        lattice_size = size(grid_hierarchy.lattice_grid[li, lj])
#        # For each module in the lattice
#        for mj in lattice_size[2], mi in lattice_size[1]
#            # Get the size of the module in terms of M by N pins
#            module_size = size(grid_hierarchy.lattice_grid[li, lj][mi, mj])
#            # Module size is the number of divisions in the x and y directions
#            # We need to subtract 1 from each dimension to get the number of pins
#            npins += (module_size[1] - 1) * (module_size[2] - 1)
#        end
#    end
#    # Allocate the pin meshes
#    pin_meshes = Vector{M}(undef, npins)
#
#    # Allocate the lattice meshes
#    mesh_ids = Matrix{Matrix{Matrix{I}}}(undef, core_size[1], core_size[2])
#    # Get the names of each lattice elset
#    lattice_names = String[]
#    for elset_name in keys(elsets)
#        if startswith(elset_name, "Lattice")
#            push!(lattice_names, elset_name)
#        end
#    end
#    # Determine the locations of each lattice in the matrix using the centroid of
#    # one of the faces in that submesh
#    lattice_locations = Matrix{String}(undef, core_size[1], core_size[2])
#    for lat_name in lattice_names
#        face_id = first(elsets[lat_name])
#        C = centroid(face(face_id, mesh))
#        for lj in 1:core_size[2], li in 1:core_size[1]
#            lattice_size = size(grid_hierarchy.lattice_grid[li, lj])
#            for mj in 1:lattice_size[2], mi in 1:lattice_size[1]
#                if C in grid_hierarchy.lattice_grid[li, lj][mi, mj]
#                    lattice_locations[li, lj] = lat_name
#                end
#            end
#        end
#    end
#    # For each lattice
#    for lj in 1:core_size[2], li in 1:core_size[1]
#        # Create the lattice submesh
#        lattice_elsets, lattice_mesh = submesh(lattice_locations[li, lj], elsets, mesh)
#        # Get the size of the lattice in terms of M by N modules
#        lattice_size = size(grid_hierarchy.lattice_grid[li, lj])
#        # Allocate the module meshes
#        mesh_ids[li, lj] = Matrix{Matrix{I}}(undef, lattice_size[1], lattice_size[2])
#        # Get the names of each module elset
#        module_names = String[]
#        for elset_name in keys(lattice_elsets)
#            if startswith(elset_name, "Module")
#                push!(module_names, elset_name)
#            end
#        end
#        # Determine the locations of each module in the matrix using the centroid of
#        # one of the faces in that submesh
#        module_locations = Matrix{String}(undef, lattice_size[1], lattice_size[2])
#        for mod_name in module_names
#            face_id = first(lattice_elsets[mod_name])
#            C = centroid(face(face_id, lattice_mesh))
#            for mj in 1:lattice_size[2], mi in 1:lattice_size[1]
#                if C in grid_hierarchy.lattice_grid[li, lj][mi, mj]
#                    module_locations[mi, mj] = mod_name
#                end
#            end
#        end
#        # For each module
#        for mj in 1:lattice_size[2], mi in 1:lattice_size[1]
#            # Create the module submesh
#            module_elsets, module_mesh = submesh(module_locations[mi, mj],
#                                                 lattice_elsets, 
#                                                 lattice_mesh)
#            # Get the size of the module in terms of M by N cells
#            module_size = size(grid_hierarchy.lattice_grid[li, lj][mi, mj])
#            # This module size is from the rectilinear grid, so we need to decrement
#            # each dimension by 1 to get the number of cells in each dimension
#            module_size = module_size .- 1
#            # Allocate the pin meshes
#            mesh_ids[li, lj][mi, mj] = Matrix{I}(undef, module_size[1], module_size[2])
#            # Get the names of each pin elset
#            pin_names = String[]
#            for elset_name in keys(module_elsets)
#                if startswith(elset_name, "Cell")
#                    push!(pin_names, elset_name)
#                end
#            end
#            # Determine the locations of each pin in the matrix using the centroid of
#            # one of the faces in that submesh
#            pin_locations = Matrix{String}(undef, module_size[1], module_size[2])
#            for pin_name in pin_names
#                face_id = first(module_elsets[pin_name])
#                C = centroid(face(face_id, module_mesh))
#                for cj in 1:module_size[2], ci in 1:module_size[1]
#                    if C in get_box(grid_hierarchy.lattice_grid[li, lj][mi, mj], ci, cj)
#                        pin_locations[ci, cj] = pin_name
#                    end
#                end
#            end
#            # For each pin
#            for cj in 1:module_size[2], ci in 1:module_size[1]
#                # Create the pin submesh
#                pin_elsets, pin_mesh = submesh(pin_locations[ci, cj],
#                                               module_elsets, 
#                                               module_mesh)
#                # Get the integer ID of the pin mesh
#                pin_id = parse(I, pin_mesh.name[6:end])
#                mesh_ids[li, lj][mi, mj][ci, cj] = pin_id
#                # Store the pin mesh
#                pin_meshes[pin_id] = pin_mesh
#            end
#        end
#    end
#    return MPACTMeshHierarchy(mesh_ids, pin_meshes)
#end
#
#function Base.show(io::IO, mmh::MPACTMeshHierarchy{M}) where {M}
#    println("MPACTMeshHierarchy{", M, "}")
#    nlattices = length(mmh.mesh_ids)
#    nmodules = 0
#    npins = 0
#    for lattice in mmh.mesh_ids
#        nmodules += length(lattice)
#        for rt_module in lattice
#            npins += length(rt_module)
#        end
#    end
#    println(" Number of lattices: ", nlattices)
#    println(" Number of modules: ", nmodules)
#    println(" Number of pins: ", npins)
#    bbox = mapreduce(bounding_box, Base.union, mmh.pin_meshes)
#    println(" Bounding box: ", bbox)
#    return nothing
#end
