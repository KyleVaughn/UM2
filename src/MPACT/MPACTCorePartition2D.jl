      # MPACT stores pin geometry in arrays with 
      #        x 1   2   3   4   5
      #      y ---------------------
      #      1 | 11| 12| 13| 14| 15|
      #        ---------------------
      #      2 | 6 | 7 | 8 | 9 | 10|
      #        ---------------------
      #      3 | 1 | 2 | 3 | 4 | 5 |
      #        *--------------------  * is where (0.0,0.0,0.0) is
      # so we need to flip the indicies of the array


# Classify level and tell if core, assembly/lattice, RT mod, coarse mesh, pinmesh


# A tree representing a hierarchical mesh partition. 
# Used in the HierarchicalMeshPartition type
#
# Each node has the name of the partition and an ID.
# If the node is an internal node, it has id = 0.
# If the node is a leaf node, it has a non-zero id, corresponding to the index by 
# which the mesh may be found in the HierarchicalMeshPartition type's leaf_meshes.
mutable struct MeshPartitionTree <: Tree
    name::String
    id::Int64
    # use an AABB and put children into a matrix
    parent::Union{Nothing, MeshPartitionTree}
    children::Union{Nothing, Vector{MeshPartitionTree}}
    MeshPartitionTree(name) = new(name, 0, nothing, nothing)
    function MeshPartitionTree(name, parent::MeshPartitionTree)
        this = new(name, 0, parent, nothing)
        if isnothing(parent.children)
            parent.children = [this]
        else
            push!(parent.children, this)
        end 
        return this
    end
end

function Base.show(io::IO, tree::MeshPartitionTree, predecessor_string::String)
    if isroot(tree)
        next_predecessor_string = ""
    elseif !isroot(tree) || next_predecessor_string != ""
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
    if tree.id === 0 # internal node
        println(io, tree.name)
    else
        println(io, tree.name*", ID: "*string(tree.id) )
    end 
    if !isnothing(tree.children)
        for child ∈ tree.children
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
    # Extract the names of all face sets that contain 'by' (the variable)
    partition_names = _get_partition_names(mesh, by)

    # Create a tree to store the partition hierarchy.
    root = _create_mesh_partition_tree(mesh, partition_names)

    # Construct the leaf meshes
    leaf_meshes = _create_leaf_meshes(mesh, by, root)

    return HierarchicalMeshPartition(root, leaf_meshes)
end

function _create_leaf_meshes(mesh::M, by::String, root::MeshPartitionTree) where {M<:UnstructuredMesh2D}
    leaf_nodes = leaves(root)
    mesh_counter = 0
    leaf_meshes = M[]
    for node ∈ leaf_nodes
        mesh_counter += 1
        node.id = mesh_counter
        push!(leaf_meshes, submesh(node.name, mesh))
    end
    for leaf_mesh ∈ leaf_meshes
        for name ∈ keys(leaf_mesh.face_sets)
            if occursin(by, uppercase(name))
                delete!(leaf_mesh.face_sets, name)
            end
        end
    end
    return leaf_meshes
end

# Create a tree to store grid relationships.
function _create_mesh_partition_tree(mesh::UnstructuredMesh, partition_names::Vector{String}) 
    root = MeshPartitionTree(mesh.name)
    remaining_names = copy(partition_names)
    LN_partition_names = copy(partition_names)
    LN⁺_partition_names = String[]
    LN⁻_nodes = MeshPartitionTree[root]
    LN_nodes = MeshPartitionTree[]
    while length(LN_partition_names) > 0
        # Determine the names of nodes at this level
        N = length(remaining_names)
        for i ∈ 1:N
            name_i = remaining_names[i] 
            for j ∈ 1:N
                if i === j
                    continue
                end
                name_j = remaining_names[j]
                if mesh.face_sets[name_i] ⊆ mesh.face_sets[name_j]
                    if name_i ∈ LN_partition_names
                        filter!(x->x ≠ name_i, LN_partition_names)
                        push!(LN⁺_partition_names, name_i)
                    end
                end
            end
        end
        # For each name at level N, attach it to its level N-1 parent
        for parent_node ∈ LN⁻_nodes
            if isroot(parent_node) # All partition names are a subset
                for child_name ∈ LN_partition_names
                    push!(LN_nodes, MeshPartitionTree(child_name, parent_node))
                end
            else
                for child_name ∈ LN_partition_names
                    if mesh.face_sets[child_name] ⊆ mesh.face_sets[parent_node.name]
                        push!(LN_nodes, MeshPartitionTree(child_name, parent_node))
                    end
                end
            end
        end
        # Swap the old with the new
        LN_partition_names = LN⁺_partition_names
        LN⁺_partition_names = String[]
        LN⁻_nodes = LN_nodes
        LN_nodes = MeshPartitionTree[]
    end
    return root
end

# Extract partition names
function _get_partition_names(mesh::UnstructuredMesh2D, by::String)
    set_names = collect(keys(mesh.face_sets))
    partition_names = copy(set_names)
    for set_name ∈ set_names
        if !occursin(by, uppercase(set_name))
            filter!(x->x ≠ set_name, partition_names)
        end
    end
    if length(partition_names) === 0
        error("No partition face sets in mesh")
    end
    return sort!(partition_names)
end
