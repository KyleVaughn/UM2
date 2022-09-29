export MPACTMeshHierarchy

struct MPACTMeshHierarchy{M <: AbstractMesh}
    #      y    
    #      ^    
    # j = 3|    
    # j = 2|    
    # j = 1|    
    #       _____________> x    
    #       i=1  i=2  i=3  
    lattice_meshes::Matrix{Matrix{Matrix{M}}}
end

function MPACTMeshHierarchy(
        mesh::M, 
        elsets::Dict{String, Set{I}},
        grid_hierarchy::MPACTGridHierarchy
    ) where {M <: AbstractMesh, I <: Integer}
    # Get the core size in terms of M by N lattices
    core_size = size(grid_hierarchy.lattice_grid)
    # Allocate the lattice meshes
    lattice_meshes = Matrix{Matrix{Matrix{M}}}(undef, core_size[1], core_size[2])
    # Get the names of each lattice elset
    lattice_names = String[]
    for elset_name in keys(elsets)
        if startswith(elset_name, "Lattice")
            push!(lattice_names, elset_name)
        end
    end
    # Determine the locations of each lattice in the matrix using the centroid of
    # one of the faces in that submesh
    lattice_locations = Matrix{String}(undef, core_size[1], core_size[2])
    for lat_name in lattice_names
        face_id = first(elsets[lat_name])
        C = centroid(face(face_id, mesh))
        for lj in 1:core_size[2], li in 1:core_size[1]
            lattice_size = size(grid_hierarchy.lattice_grid[li, lj])
            for mj in 1:lattice_size[2], mi in 1:lattice_size[1]
                if C in grid_hierarchy.lattice_grid[li, lj][mi, mj]
                    lattice_locations[li, lj] = lat_name
                end
            end
        end
    end
    # For each lattice
    for lj in core_size[2], li in core_size[1]
        # Create the lattice submesh
        lattice_elsets, lattice_mesh = submesh(lattice_locations[li, lj], elsets, mesh)
        # Get the size of the lattice in terms of M by N modules
        lattice_size = size(grid_hierarchy.lattice_grid[li, lj])
        # Allocate the module meshes
        lattice_meshes[li, lj] = Matrix{Matrix{M}}(undef, lattice_size[1], lattice_size[2])
        # For each module
        for mj in lattice_size[2], mi in lattice_size[1]
            # Create the module submesh
            module_elsets, module_mesh = submesh(grid_hierarchy.lattice_grid[li, lj][mi, mj], 
                                                 lattice_elsets, 
                                                 lattice_mesh)
            # Get the size of the module in terms of M by N cells
            module_size = size(grid_hierarchy.lattice_grid[li, lj][mi, mj])
            # This module size is from the rectilinear grid, so we need to decrement
            # each dimension by 1 to get the number of cells in each dimension
            module_size = module_size .- 1







           lattice_meshes[li, lj][mi, mj] = Matrix{M}(undef, module_size[1], module_size[2])
           for cj in module_size[2], ci in module_size[1]
#                lattice_meshes[li, lj][mi, mj][ci, cj] = mesh
#           end
#        end
    end
#    leaf_meshes = _create_leaf_meshes(mesh, root)
    return MeshPartitionTree(root, leaf_meshes)
end

#function _create_leaf_meshes(mesh::AbstractMesh)
#    leaf_nodes = sort!(leaves(root), by = x -> x.data)
#    leaf_meshes = Vector{typeof(mesh)}(undef, length(leaf_nodes))
#    leaf_ctr = 1
#    node_children = children(root)
#    if !isnothing(node_children)
#        for child in sort(node_children, by = x -> x.data)
#            child_mesh = submesh(mesh, data(child)[2])
#            leaf_ctr = _create_leaf_meshes!(child_mesh, child, leaf_meshes, leaf_ctr)
#        end
#    else
#        # Remove any Lattice, Module, or Coarse_Cell groups, since this info should now
#        # be encoded in the tree
#        mpact_groups = filter(x -> isa_MPACT_partition_name(x), keys(groups(mesh)))
#        for grp in mpact_groups
#            pop!(groups(mesh), grp)
#        end
#        name = data(root)[2]
#        root.data = (leaf_ctr, name)
#        leaf_meshes[leaf_ctr] = mesh
#    end
#    return leaf_meshes
#end
#
#function _create_leaf_meshes!(mesh::AbstractMesh,
#                              node::Tree,
#                              leaf_meshes,
#                              leaf_ctr::Int64 = 1)
#    node_children = children(node)
#    if !isnothing(node_children)
#        for child in sort(node_children, by = x -> x.data)
#            child_mesh = submesh(mesh, data(child)[2])
#            leaf_ctr = _create_leaf_meshes!(child_mesh, child, leaf_meshes, leaf_ctr)
#        end
#    else
#        # Remove any Lattice, Module, or Coarse_Cell groups, since this info should now
#        # be encoded in the tree
#        mpact_groups = filter(x -> isa_MPACT_partition_name(x), keys(groups(mesh)))
#        for grp in mpact_groups
#            pop!(mesh.groups, grp)
#        end
#        leaf_meshes[leaf_ctr] = mesh
#        name = data(node)[2]
#        node.data = (leaf_ctr, name)
#        leaf_ctr += 1
#    end
#    return leaf_ctr
#end
#
## Create a tree to store grid relationships.
#function _create_mesh_partition_tree(mesh::AbstractMesh, partition_names::Vector{String})
#    root = Tree((1, name(mesh)))
#    parents = [root]
#    tree_type = typeof(root)
#    next_parents = tree_type[]
#    remaining_names = copy(partition_names)
#    # Extract the sets that are not a subset of any other set.
#    this_level = Int64[]
#    mesh_groups = groups(mesh)
#    while length(remaining_names) > 0
#        for i in eachindex(remaining_names)
#            name_i = remaining_names[i]
#            i_isa_lattice = startswith(name_i, "Lattice")
#            i_isa_module = startswith(name_i, "Module")
#            isa_subset = false
#            for j in eachindex(remaining_names)
#                if i === j
#                    continue
#                end
#                name_j = remaining_names[j]
#                j_isa_coarsecell = startswith(name_j, "Coarse")
#                if j_isa_coarsecell && (i_isa_module || i_isa_lattice)
#                    continue
#                end
#                j_isa_module = startswith(name_j, "Module")
#                if j_isa_module && i_isa_lattice
#                    continue
#                end
#                if mesh_groups[name_i] ⊆ mesh_groups[name_j]
#                    isa_subset = true
#                    break
#                end
#            end
#            if !isa_subset
#                push!(this_level, i)
#            end
#        end
#        # Add the groups that are not a subset to the tree
#        # and next_parents
#        node_id = 1 
#        for id in this_level
#            name = remaining_names[id]
#            for parent in parents
#                if isroot(parent)
#                    node = Tree((node_id, name), parent)
#                    node_id += 1
#                    push!(next_parents, node)
#                else
#                    parent_name = data(parent)[2]
#                    if mesh_groups[name] ⊆ mesh_groups[parent_name]
#                        node = Tree((node_id, name), parent)
#                        node_id += 1
#                        push!(next_parents, node)
#                        break
#                    end
#                end
#            end
#        end
#        parents = next_parents
#        next_parents = tree_type[]
#        # Remove the groups added to the tree from the remaining names
#        deleteat!(remaining_names, this_level)
#        this_level = Int64[]
#    end
#    return root
#end
#
#function isa_MPACT_partition_name(x::String)
#    return startswith(x, "Coarse_Cell_(") ||
#           startswith(x, "Module_(") ||
#           startswith(x, "Lattice_(")
#end
## Extract partition names
#function _get_partition_names(mesh::AbstractMesh, by::String)
#    # If by === MPACT, then we will partition by Lattice, Module, and Coarse Cell
#    # Otherwise, partition by the string contained in by
#    partition_names = collect(keys(groups(mesh)))
#    if length(partition_names) === 0
#        error("The mesh does not have any groups.")
#    end
#    if by === "MPACT"
#        filter!(x -> isa_MPACT_partition_name(x), partition_names)
#        if length(partition_names) === 0
#            error("The mesh does not have any MPACT grid hierarchy groups.")
#        end
#    else
#        filter!(x -> occursin(by, x), partition_names)
#        if length(partition_names) === 0
#            error("The mesh does not have any groups containing '", by, "'.")
#        end
#    end
#    return sort!(partition_names)
#end
#
#function Base.show(io::IO, mpt::MeshPartitionTree{M}) where {M}
#    println("MeshPartitionTree{", M, "}")
#    println(tree(mpt))
#    return nothing
#end
