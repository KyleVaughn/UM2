export MeshPartitionTree
export partition
# A data structure to hold a hierarchical partition of a mesh.
# Since the mesh is partitioned, we only need to store the leaves of the partition
# tree to reconstruct the mesh.
#
# Each node has the name of the partition and an ID.
# If the node is an internal node, it has id = 0.
# If the node is a leaf node, it has a non-zero id, corresponding to the index by 
# which the mesh may be found in the MeshPartitionTree type's leaf_meshes.
struct MeshPartitionTree{M <: PolytopeVertexMesh}
    partition_tree::Tree{Tuple{Int64, String}}
    leaf_meshes::Vector{M}
end
#
## Partitions a mesh based upon the names of its groups 
## For a hierarchical partition, all partitions in the hierarchy must contain face sets 
## of the form "<name>_L<N>" where name is a string, and N is an integer.
## N is the level of the node in the tree: 1, 2, 3, etc...
## Example: "Grid_L1_triangle", or "Partition_L3"
function partition(mesh::PolytopeVertexMesh; by::String="MPACT")
    @info "Partitioning mesh"
    by = uppercase(by)
    # Extract the names of all face sets that contain 'by' (the variable)
    partition_names = _get_partition_names(mesh, by)

    # Create a tree to store the partition hierarchy.
    root = _create_mesh_partition_tree(mesh, partition_names)
#
#    # Construct the leaf meshes
#    leaf_meshes = _create_leaf_meshes(mesh, by, root)
#
#    return HierarchicalMeshPartition(root, leaf_meshes)
end

# function _create_leaf_meshes(mesh::M, by::String, root::MeshPartitionTree) where {M<:UnstructuredMesh2D}
#     leaf_nodes = leaves(root)
#     mesh_counter = 0
#     leaf_meshes = M[]
#     for node ∈ leaf_nodes
#         mesh_counter += 1
#         node.id = mesh_counter
#         push!(leaf_meshes, submesh(node.name, mesh))
#     end
#     for leaf_mesh ∈ leaf_meshes
#         for name ∈ keys(leaf_mesh.face_sets)
#             if occursin(by, uppercase(name))
#                 delete!(leaf_mesh.face_sets, name)
#             end
#         end
#     end
#     return leaf_meshes
# end
# 
# Create a tree to store grid relationships.
function _create_mesh_partition_tree(mesh::PolytopeVertexMesh, partition_names::Vector{String}) 
    root = Tree((0, mesh.name))
    parents = [root]
    next_parents = [root]
    remaining_names = copy(partition_names)
    # Extract the sets that are not a subset of any other set.
    this_level = Int64[] 
    next_level = Int64[]
    while length(remaining_names) > 0
        for i in eachindex(remaining_names)
            name_i = remaining_names[i] 
            isa_subset = false
            for j in eachindex(remaining_names)
                if i === j
                    continue
                end
                name_j = remaining_names[j]
                # Is a subset of a nother remaining group
                if mesh.groups[name_i] ⊆ mesh.groups[name_j]
                    push!(next_level, i)
                    isa_subset = true
                    break
                end
            end
            if !isa_subset
                push!(this_level, i)
            end
        end
        for id in this_level
            name = remaining_names[id]
            for parent in paren
            node = Tree((0, remaining_names[id]))
         
    end
#3    LN_partition_names = copy(partition_names)
#3    LN⁺_partition_names = String[]
#3    LN⁻_nodes = [root]
#3    LN_nodes = Tree{Tuple{Int64,String}}[]
#3    while length(LN_partition_names) > 0
#3        # Determine the names of nodes at this level
#3        for i in eachindex(remaining_names)
#3            name_i = remaining_names[i] 
#3            for j in eachindex(remaining_names)
#3                if i === j
#3                    continue
#3                end
#3                name_j = remaining_names[j]
#3                if mesh.groups[name_i] ⊆ mesh.groups[name_j]
#3                    if name_i ∈ LN_partition_names
#3                        filter!(x->x ≠ name_i, LN_partition_names)
#3                        push!(LN⁺_partition_names, name_i)
#3                    end
#3                end
#3            end
#3        end
#3        # For each name at level N, attach it to its level N-1 parent
#3        for parent_node ∈ LN⁻_nodes
#3            if isroot(parent_node) # All partition names are a subset
#3                for child_name ∈ LN_partition_names
#3                    push!(LN_nodes, Tree((0,child_name), parent_node))
#3                end
#3            else
#3                for child_name ∈ LN_partition_names
#3                    if mesh.groups[child_name] ⊆ mesh.groups[parent_node.data[2]]
#3                        push!(LN_nodes, Tree((0,child_name), parent_node))
#3                    end
#3                end
#3            end
#3        end
#3        # Swap the old with the new
#3        LN_partition_names = LN⁺_partition_names
#3        LN⁺_partition_names = String[]
#3        LN⁻_nodes = LN_nodes
#3        LN_nodes = Tree{Tuple{Int64,String}}[] 
#3    end
    return root
end
function isa_MPACT_partition_name(x::String)
    return occursin("COARSE_CELL_(", x) ||
           occursin("MODULE_(",      x) ||
           occursin("LATTICE_(",     x) 
end
# Extract partition names
function _get_partition_names(mesh::PolytopeVertexMesh, by::String)
    # If by === MPACT, then we will partition by Lattice, Module, and Coarse Cell
    # Otherwise, partition by the string contained in by
    partition_names = collect(keys(mesh.groups))
    if length(partition_names) === 0
        error("The mesh does not have any groups.")
    end
    if by === "MPACT"
        filter!(x->isa_MPACT_partition_name(uppercase(x)), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any MPACT grid hierarchy groups.")
        end
    else
        filter!(x->occursin(by, uppercase(x)), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any groups containing '", by, "'.")
        end
    end
    return sort!(partition_names)
end
