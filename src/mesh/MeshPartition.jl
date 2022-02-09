# A data structure to hold a hierarchical partition of a mesh.
# Since the mesh is partitioned, we only need to store the leaves of the partition
# tree to reconstruct the mesh.
#   - partition_tree is a tree with String type data at all nodes except the leaves,
#       denoting the name of the partition. At the leaves, the tree has Int64 data,
#       denoting the index of leaf_meshes in which the leaf mesh may be found.
#   - leaf_meshes is vector of meshes
struct MeshPartition{M <: UnstructuredMesh}
    partition_tree::Tree
    leaf_meshes::Vector{M}
end

# Assumes all leaf nodes of the hierarchical partition occur in the same level
# For a hierarchical partition, all partitions in the hierarchy must contain face sets 
# of the form "name_LN" where name is a string, and N is an integer.
# N is the level of the node
function partition_mesh(mesh::UnstructuredMesh2D; by::String="GRID_L")
    @info "Partitioning mesh"
    by = uppercase(by)
    # Extract set names, partition names, and max level
    set_names, partition_names, max_level = _process_partition_mesh_input(mesh, by)

    # Create a tree to store the partition hierarchy.
    root, leaf_meshes = _create_partition_tree_and_leaf_meshes(mesh, by, partition_names, max_level)

#    # Construct the leaf meshes
#    leaf_meshes = _create_leaf_meshes(mesh, by, partition_names, max_level)

    return MeshPartition(root, leaf_meshes)
end

function _create_leaf_meshes(mesh::M, by::String, 
                             partition_names::Vector{String}, 
                             max_level::Int64) where {M<:UnstructuredMesh2D}
    leaf_meshes = M[]
    leaf_names = String[]
    for name in partition_names
        level = 1
        try
            level = parse(Int64, name[length(by) + 1])
        catch
            level = 1
        end
        if level == max_level
            push!(leaf_names, name)
        end
    end
    for name in leaf_names
        push!(leaf_meshes, submesh(name, mesh))
    end
    for leaf_mesh in leaf_meshes
        for name in keys(leaf_mesh.face_sets)
            if occursin(by, uppercase(name))
                delete!(leaf_mesh.face_sets, name)
            end
        end
    end
    return leaf_meshes
end

# Create a tree to store grid relationships.
function _create_partition_tree_and_leaf_meshes(mesh::M, 
                                                by::String, 
                                                partition_names::Vector{String}, 
                                                max_level::Int64) where {M<:UnstructuredMesh2D}
    root = Tree(mesh.name)
    leaf_meshes = M[]
    current_nodes = Tree[]
    next_nodes = Tree[]
    old_partition_names = copy(partition_names)
    new_partition_names = copy(partition_names)
    # Do first level
    for partition_name in old_partition_names
        partition_level = 1
        try
            partition_level = parse(Int64, partition_name[length(by) + 1])
        catch
            partition_level = 1
        end
        if partition_level === 1
            if partition_level == max_level # is a leaf    
                push!(leaf_meshes, submesh(partition_name, mesh))
                push!(next_nodes, Tree((partition_name, length(leaf_meshes)), root))
            else
                push!(next_nodes, Tree(partition_name, root))
            end
            filter!(x->x ≠ partition_name, new_partition_names)
        end
    end
    # Do all other levels:
    for level in 2:max_level
        old_partition_names = copy(new_partition_names)
        current_nodes = next_nodes
        next_nodes = Tree[]
        for partition_name in old_partition_names
            partition_level = parse(Int64, partition_name[length(by) + 1])
            if partition_level == level
                # find the parent for this partition
                partition_faces = mesh.face_sets[partition_name]
                for node in current_nodes
                    node_faces = mesh.face_sets[node.data]
                    if partition_faces ⊆ node_faces
                        if level == max_level # is a leaf    
                            push!(leaf_meshes, submesh(partition_name, mesh))
                            push!(next_nodes, Tree((partition_name, length(leaf_meshes)), node))
                        else
                            push!(next_nodes, Tree(partition_name, node))
                        end
                        filter!(x->x ≠ partition_name, new_partition_names)
                        break
                    end
                end
            end
        end
    end
    for leaf_mesh in leaf_meshes
        for name in keys(leaf_mesh.face_sets)
            if occursin(by, uppercase(name))
                delete!(leaf_mesh.face_sets, name)
            end
        end
    end
    return root, leaf_meshes
end

# Extract set names, partition names, and max partition level
function _process_partition_mesh_input(mesh::UnstructuredMesh2D, by::String)
    set_names = collect(keys(mesh.face_sets))
    partition_names = copy(set_names)
    for set_name in set_names
        if !occursin(by, uppercase(set_name))
            filter!(x->x ≠ set_name, partition_names)
        end
    end
    if length(partition_names) === 0
        @error "No partition face sets in mesh"
    end

    # Get the number of partition levels
    max_level = 1
    try
        for partition_name in partition_names
            # No 10+ level hierarchy
            level = parse(Int64, partition_name[length(by) + 1])
            if max_level < level
                max_level = level
            end
        end
    catch
        max_level = 1
    end

    return set_names, partition_names, max_level
end
