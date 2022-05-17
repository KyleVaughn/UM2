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
    # Extract the names of all face sets that contain 'by' (the variable)
    partition_names = _get_partition_names(mesh, by)

    # Create a tree to store the partition hierarchy.
    root = _create_mesh_partition_tree(mesh, partition_names)

    # Construct the leaf meshes
    leaf_meshes = _create_leaf_meshes(mesh, root)

    return MeshPartitionTree(root, leaf_meshes)
end

function _create_leaf_meshes(mesh::PolytopeVertexMesh, root::Tree)
    leaf_nodes = sort!(leaves(root), by=x->x.data)
    leaf_meshes = Vector{typeof(mesh)}(undef, length(leaf_nodes))
    for (i, node) in enumerate(leaf_nodes)
        name = node.data[2]
        node.data = (i, name)
        leaf_meshes[i] = submesh(mesh, name)
    end
    return leaf_meshes
end
 
# Create a tree to store grid relationships.
function _create_mesh_partition_tree(mesh::PolytopeVertexMesh, partition_names::Vector{String}) 
    root = Tree((0, mesh.name))
    parents = [root]
    tree_type = typeof(root)
    next_parents = tree_type[]
    remaining_names = copy(partition_names)
    # Extract the sets that are not a subset of any other set.
    this_level = Int64[] 
    while length(remaining_names) > 0
        for i in eachindex(remaining_names)
            name_i = remaining_names[i] 
            i_isa_lattice = startswith(name_i, "Lattice")
            i_isa_module = startswith(name_i, "Module")
            isa_subset = false
            for j in eachindex(remaining_names)
                if i === j
                    continue
                end
                name_j = remaining_names[j]
                j_isa_coarsecell = startswith(name_j, "Coarse")
                if j_isa_coarsecell && (i_isa_module || i_isa_lattice)
                    continue
                end
                j_isa_module = startswith(name_j, "Module")
                if j_isa_module && i_isa_lattice
                    continue
                end
                if mesh.groups[name_i] ⊆ mesh.groups[name_j]
                    isa_subset = true
                    break
                end
            end
            if !isa_subset
                push!(this_level, i)
            end
        end
        # Add the groups that are not a subset to the tree
        # and next_parents
        for id in this_level
            name = remaining_names[id]
            for parent in parents
                if isroot(parent)
                    node = Tree((0, name), parent)
                    push!(next_parents, node)
                else
                    parent_name = parent.data[2]
                    if mesh.groups[name] ⊆ mesh.groups[parent_name]
                        node = Tree((0, name), parent)
                        push!(next_parents, node)
                        break
                    end
                end
            end
        end
        parents = next_parents
        next_parents = tree_type[]
        # Remove the groups added to the tree from the remaining names
        deleteat!(remaining_names, this_level)
        this_level = Int64[]
    end
    return root
end

function isa_MPACT_partition_name(x::String)
    return startswith(x, "Coarse_Cell_(") ||
           startswith(x, "Module_("     ) ||
           startswith(x, "Lattice_("    ) 
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
        filter!(x->isa_MPACT_partition_name(x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any MPACT grid hierarchy groups.")
        end
    else
        filter!(x->occursin(by, x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any groups containing '", by, "'.")
        end
    end
    return sort!(partition_names)
end
