export HierarchicalMesh

export tree, leaf_meshes, getleaf, num_leaves, partition_mesh

# A hierarchical mesh partition
struct HierarchicalMesh{M <: AbstractMesh}
    partition_tree::Tree{Tuple{UM_I, String}}
    leaf_meshes::Vector{M}
end

tree(hm::HierarchicalMesh) = hm.partition_tree
leaf_meshes(hm::HierarchicalMesh) = hm.leaf_meshes
getleaf(hm::HierarchicalMesh, id::Integer) = hm.leaf_meshes[id]
num_leaves(hm::HierarchicalMesh) = length(hm.leaf_meshes)

# Convert a mesh to a hierarchical mesh

# Extract partition names
function _get_partition_names(elsets::Dict{String, Set{UM_I}}, by::String)
    # If by == "MPACT", then we will partition by Lattice, Module, and Coarse Cell
    # Otherwise, partition by the string contained in by
    partition_names = collect(keys(elsets))
    if length(partition_names) === 0
        error("The mesh does not have any groups.")
    end
    if by == "MPACT"
        filter!(x -> is_MPACT_partition(x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any MPACT spatial hierarchy groups.")
        end
    else
        filter!(x -> occursin(by, x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any groups containing '", by, "'.")
        end
    end
    sort!(partition_names)
    return partition_names
end

# Number the nodes in each level of the partition tree to allow for easy
# multidimensional indexing
function _number_tree_nodes!(node::Tree{Tuple{UM_I, String}},
                            node_ctr::Vector{UM_I},
                            level::UM_I)
    level += UM_I(1)
    if length(node_ctr) < level
        push!(node_ctr, UM_I(0))
    end
    node_ctr[level] += UM_I(1)
    node.data = (node_ctr[level], node.data[2])
    if !isnothing(node.children)
        for child in node.children
            _number_tree_nodes!(child, node_ctr, level)
        end
    end
    return nothing
end

# Create a tree to store grid relationships.
function _create_tree(mesh::AbstractMesh, 
                      elsets::Dict{String, Set{UM_I}},
                      partition_names::Vector{String})
    root = Tree((UM_I(0), name(mesh)))
    # Each set will be tested to see if it is a subset of any other set.
    # If it is, then it will be added as a child of the node containing that set.
    tree_nodes = Vector{Tree{Tuple{UM_I, String}}}(undef, length(partition_names))
    for i in eachindex(partition_names)
        tree_nodes[i] = Tree((UM_I(0), partition_names[i]))
    end
    # If the partition_names are MPACT spatial hierarchy names, we can do this in
    # a single pass. Otherwise, we need to do a nested loop.
    if all(x -> is_MPACT_partition(x), partition_names)
        # Partition names sorted alphabetically, it's Cell, Lattice, Module.
        # Find the indices of the first and last lattice
        lat_start = findfirst(x -> is_MPACT_lattice(x), partition_names)
        lat_stop = findlast(x -> is_MPACT_lattice(x), partition_names)
        # We now know that the first lat_start - 1 elements are cells, and the last
        # length(partition_names) - lat_stop elements are modules.
        cell_start = 1
        cell_stop = lat_start - 1
        mod_start = lat_stop + 1
        mod_stop = length(partition_names)
        # Lattices contain modules, modules contain cells.
        # Each lattice is mutually exclusive with the other lattices.
        # Likewse for modules and cells.
        # Each should only be added to the tree once, hence we will keep a 
        # Bool vector to track which ones have been added.
        added = falses(length(partition_names))
        # Loop over lattices, adding them as children of the root
        for ilat in lat_start:lat_stop
            push!(root, tree_nodes[ilat])
            added[ilat] = true
            # Loop over modules, adding them as children of the lattice
            for imod in mod_start:mod_stop
                if added[imod]
                    continue
                end
                if elsets[partition_names[imod]] ⊆ elsets[partition_names[ilat]]
                    push!(tree_nodes[ilat], tree_nodes[imod])
                    added[imod] = true
                    # Loop over cells, adding them as children of the module
                    for icell in cell_start:cell_stop
                        if added[icell]
                            continue
                        end
                        if elsets[partition_names[icell]] ⊆ elsets[partition_names[imod]]
                            push!(tree_nodes[imod], tree_nodes[icell])
                            added[icell] = true
                        end
                    end
                end
            end
        end
    else
        for i in eachindex(partition_names)
            for j in eachindex(partition_names)
                if i != j
                    if elsets[partition_names[i]] ⊆ elsets[partition_names[j]]
                        push!(tree_nodes[j], tree_nodes[i])
                    end
                end
            end
        end
    end
    # Add the parentless nodes to the root
    for node in tree_nodes
        if isroot(node)
            push!(root, node)
        end
    end
    # Assign IDs to the leaf nodes
    # Note: partition_names is sorted, so if by == "MPACT", then the order of the
    # leaf nodes will be Cells sorted by ID.
    for (i, node) in enumerate(leaves(root))
        node.data = (UM_I(i), node.data[2])
    end
    return root
end

function _create_leaf_meshes(mesh::AbstractMesh,
                             elsets::Dict{String, Set{UM_I}},
                             root::Tree{Tuple{UM_I, String}})
    # Make a copy of the elsets so we can pop from them
    # as we create the leaf meshes. This reduces the number of set intersections,
    # which can be expensive.
    elsets_copy = Dict{String, Set{UM_I}}()   
    for (k, v) in elsets
        elsets_copy[k] = copy(v)
    end
    leaf_nodes = leaves(root)
    leaf_meshes = Vector{typeof(mesh)}(undef, length(leaf_nodes))
    leaf_elsets = Vector{Dict{String, Set{UM_I}}}(undef, length(leaf_nodes))
    # If all of the leaf nodes have "Cell_" in their name, then we assume this is
    # an MPACT spatial hierarchy. Knowing this, we can immediately discard any
    # sets starting with "Lattice_" or "Module_", since this information is
    # already encoded in the tree.
    if all(x -> is_MPACT_cell(getindex(data(x), 2)), leaf_nodes)
        for key in keys(elsets_copy)
            if is_MPACT_lattice(key) || is_MPACT_module(key)
                delete!(elsets_copy, key)
            end
        end
    end
    # Create the leaf meshes
    for (i, node) in enumerate(leaf_nodes)
        name = getindex(data(node), 2)
        # Create the leaf mesh
        leaf_elset, leaf_mesh = submesh(mesh, elsets_copy, name)
        leaf_elsets[i] = leaf_elset
        leaf_meshes[i] = leaf_mesh
        delete!(elsets_copy, name)
    end
    return leaf_elsets, leaf_meshes
end

# Partitions a mesh based upon the names of its groups 
# If by = "MPACT", partitions based upon:
#   Cell ⊆ Module ⊆ Lattice
function partition_mesh(mesh::AbstractMesh, 
                        elsets::Dict{String, Set{UM_I}};
                        by::String = "MPACT")
    @info "Partitioning mesh: " * name(mesh)
    # Extract the names of all face sets that contain 'by' (the variable)
    partition_names = _get_partition_names(elsets, by)
    # Create a tree to store the partition hierarchy.
    root = _create_tree(mesh, elsets, partition_names)
    # Construct the leaf meshes
    leaf_elsets, leaf_meshes = _create_leaf_meshes(mesh, elsets, root)
    return leaf_elsets, HierarchicalMesh(root, leaf_meshes)
end












function Base.show(io::IO, hm::HierarchicalMesh{M}) where {M}
    println("HierarchicalMesh{", M, "}")
    println(tree(hm))
    return nothing
end
