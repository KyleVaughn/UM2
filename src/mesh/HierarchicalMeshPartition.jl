# A tree representing a hierarchical mesh partition.
#
# Each node has the name of the partition and an ID.
# If the node is an internal node, it has id = 0.
# If the node is a leaf node, it has a non-zero id, corresponding to the index by 
# which the mesh may be found in the HierarchicalMeshPartition type's leaf_meshes.
mutable struct MeshPartitionTree <: Tree
    name::String
    id::Int64
    parent::Union{Nothing, MeshPartitionTree}
    children::Union{Nothing, Vector{MeshPartitionTree}}
    MeshPartitionTree(name) = new(name, 0, nothing, nothing)
    function MeshPartitionTree(name, parent::MeshPartitionTree)
        this = new(data, 0, parent, nothing)
        if isnothing(parent.children)
            parent.children = [this]
        else
            push!(parent.children, this)
        end 
        return this
    end
end

function Base.show(io::IO, tree::MeshPartitionTree, predecessor_string::String)
    if is_root(tree)
        next_predecessor_string = ""
    elseif !is_root(tree) || next_predecessor_string != ""
        last_child = is_parents_last_child(tree)
        if last_child
            print(io, predecessor_string * "└─ ")
            next_predecessor_string = predecessor_string * "   "
        else
            print(io, predecessor_string * "├─ ")
            next_predecessor_string = predecessor_string * "│  "
        end 
    else
        next_predecessor_string = "   "
    end
    if id === 0 # non-leaf node
        println(io, tree.name)
    else
        println(io, tree.name*", ID: "*string(tree.id) )
    end 
    if !isnothing(tree.children)
        for child in tree.children
            show(io, child, next_predecessor_string)
        end 
    end 
end

# A data structure to hold a hierarchical partition of a mesh.
# Since the mesh is partitioned, we only need to store the leaves of the partition
# tree to reconstruct the mesh.
struct HierarchicalMeshPartition{M <: UnstructuredMesh}
    partition_tree::MeshPartitionTree
    leaf_meshes::Vector{M}
end

# Partitions a mesh based upon the names of its face sets
# For a hierarchical partition, all partitions in the hierarchy must contain face sets 
# of the form "<name>_L<N>" where name is a string, and N is an integer.
# N is the level of the node in the tree: 1, 2, 3, etc...
# Example: "Grid_L1_triangle", or "Partition_L3"
function partition_mesh(mesh::UnstructuredMesh2D; by::String="GRID_L")
    @info "Partitioning mesh"
    by = uppercase(by)
    # Extract set names, partition names, and max level
    partition_names, max_level = _get_partition_names(mesh, by)

    # Create a tree to store the partition hierarchy.
    root, leaf_meshes = _create_partition_tree_and_leaf_meshes(mesh, by, partition_names, max_level)

#    # Construct the leaf meshes
#    leaf_meshes = _create_leaf_meshes(mesh, by, partition_names, max_level)

    return HierarchicalMeshPartition(root, leaf_meshes)
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
    root = MeshPartitionTree(mesh.name)
    leaf_meshes = M[]
    current_nodes = MeshPartitionTree[]
    next_nodes = MeshPartitionTree[]
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
                push!(next_nodes, MeshParitionTree((partition_name, 
                                                    length(leaf_meshes)), root))
            else
                push!(next_nodes, MeshPartitionTree(partition_name, root))
            end
            filter!(x->x ≠ partition_name, new_partition_names)
        end
    end
    # Do all other levels, determining the face set's position in the tree by
    # determining for which face sets it is a subset.
    for level in 2:max_level
        old_partition_names = copy(new_partition_names)
        current_nodes = next_nodes
        next_nodes = MeshPartitionTree[]
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
                            push!(next_nodes, MeshPartitionTree((partition_name, 
                                                                 length(leaf_meshes)), node))
                        else
                            push!(next_nodes, MeshPartitionTree(partition_name, node))
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

    return partition_names, max_level
end
