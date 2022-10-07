export HierarchicalMesh

export tree, leaf_meshes, getleaf, num_leaves

# A hierarchical mesh partition
struct HierarchicalMesh{M <: AbstractMesh{T <: AbstractFloat, I <: Integer}}
    partition_tree::Tree{Tuple{I, String}}
    leaf_meshes::Vector{M}
end

tree(mpt::HierarchicalMesh) = mpt.tree
leaf_meshes(mpt::HierarchicalMesh) = mpt.leaf_meshes
getleaf(mpt::HierarchicalMesh, id::Integer) = mpt.leaf_meshes[id]
num_leaves(mpt::HierarchicalMesh) = length(mpt.leaf_meshes)

# Convert a mesh to a hierarchical mesh

#function _create_leaf_meshes(mesh::AbstractMesh, root::Tree)
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

function is_MPACT_partition_name(x::String)
    return startswith(x, "Cell_") ||
           startswith(x, "Module_") ||
           startswith(x, "Lattice_")
end

# Extract partition names
function _get_partition_names(elsets::Dict{String, Set{I}}, 
                              by::String) where {I <: Integer}
    # If by == "MPACT", then we will partition by Lattice, Module, and Coarse Cell
    # Otherwise, partition by the string contained in by
    partition_names = collect(keys(elsets))
    if length(partition_names) === 0
        error("The mesh does not have any groups.")
    end
    if by == "MPACT"
        filter!(x -> is_MPACT_partition_name(x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any MPACT spatial hierarchy groups.")
        end
    else
        filter!(x -> occursin(by, x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any groups containing '", by, "'.")
        end
    end
    return sort!(partition_names)
end

# Create a tree to store grid relationships.
function _create_tree(mesh::AbstractMesh, 
                      elsets::Dict{String, Set{I}},
                      partition_names::Vector{String}) where {I <: Integer}
    root = Tree((I(1), name(mesh)))
    parents = [root]
    tree_type = typeof(root)
    next_parents = tree_type[]
    remaining_names = copy(partition_names)
    # Extract the sets that are not a subset of any other set.
    this_level = Int64[]
    mesh_groups = groups(mesh)
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
                if mesh_groups[name_i] ⊆ mesh_groups[name_j]
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
        node_id = 1 
        for id in this_level
            name = remaining_names[id]
            for parent in parents
                if isroot(parent)
                    node = Tree((node_id, name), parent)
                    node_id += 1
                    push!(next_parents, node)
                else
                    parent_name = data(parent)[2]
                    if mesh_groups[name] ⊆ mesh_groups[parent_name]
                        node = Tree((node_id, name), parent)
                        node_id += 1
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

# Partitions a mesh based upon the names of its groups 
# If by = "MPACT", partitions based upon:
#   Cell ⊆ Module ⊆ Lattice
function HierarchicalMesh(mesh::AbstractMesh, 
                          elsets::Dict{String, Set{I}};
                          by::String = "MPACT") where {I <: Integer}
    @info "Partitioning mesh: " * name(mesh)
    # Extract the names of all face sets that contain 'by' (the variable)
    partition_names = _get_partition_names(elsets, by)
    # Create a tree to store the partition hierarchy.
    root = _create_tree(mesh, elsets, partition_names)
#    # Construct the leaf meshes
#    leaf_meshes = _create_leaf_meshes(mesh, root)
#    return HierarchicalMesh(root, leaf_meshes)
end

function Base.show(io::IO, mpt::HierarchicalMesh{M}) where {M}
    println("HierarchicalMesh{", M, "}")
    println(tree(mpt))
    return nothing
end
