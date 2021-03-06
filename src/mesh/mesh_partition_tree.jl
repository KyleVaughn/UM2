export MeshPartitionTree
export tree, leaf_meshes, getleaf, nleaves, partition_array_size, 
       lattice_array_size, module_array_size, coarse_cell_array_size,
       lattice_array, module_array, coarse_cell_array,
       lattice_module_ids, module_cc_ids, partition_array_subset,
       lattice_bb, module_bb

# A data structure to hold a hierarchical partition of a mesh.
# Since the mesh is partitioned, we only need to store the leaves of the partition
# tree to reconstruct the mesh.
#
# Each node has the name of the partition and an ID.
#### If the node is an internal node, it has id = 0.
#### If the node is a leaf node, it has a non-zero id, corresponding to the index by 
# which the mesh may be found in the MeshPartitionTree type's leaf_meshes.
struct MeshPartitionTree{M <: AbstractMesh}
    partition_tree::Tree{Tuple{Int64, String}}
    leaf_meshes::Vector{M}
end

tree(mpt::MeshPartitionTree) = mpt.partition_tree
leaf_meshes(mpt::MeshPartitionTree) = mpt.leaf_meshes
getleaf(mpt::MeshPartitionTree, id::Integer) = mpt.leaf_meshes[id]
nleaves(mpt::MeshPartitionTree) = length(mpt.leaf_meshes)

lattice_array_size(mpt::MeshPartitionTree) = partition_array_size(mpt, "Lattice_(")
module_array_size(mpt::MeshPartitionTree) = partition_array_size(mpt, "Module_(")
coarse_cell_array_size(mpt::MeshPartitionTree) = partition_array_size(mpt, "Coarse_Cell_(")

lattice_array(mpt::MeshPartitionTree) = partition_array(mpt, "Lattice_(")
module_array(mpt::MeshPartitionTree) = partition_array(mpt, "Module_(")
coarse_cell_array(mpt::MeshPartitionTree) = partition_array(mpt, "Coarse_Cell_(")

function lattice_module_ids(id::Integer, mpt::MeshPartitionTree)
    return partition_array_subset(mpt, "Lattice_(", id)
end

function module_cc_ids(id::Integer, mpt::MeshPartitionTree)
    return partition_array_subset(mpt, "Module_(", id)
end

function module_bb(id::Integer, mpt::MeshPartitionTree)
    return mapreduce(i->boundingbox(mpt.leaf_meshes[i]), ∪, module_cc_ids(id, mpt))
end
 
function lattice_bb(id::Integer, mpt::MeshPartitionTree)
    return mapreduce(i->module_bb(i, mpt), ∪, lattice_module_ids(id, mpt))
end

# For MPTs representing an MPACT grid hierarchy
function partition_array_size(mpt::MeshPartitionTree, str::String)
    # str == "Lattice_(", "Module_(", or "Coarse_Cell_("
    #
    # Julia is bad at recusive types, so we do this manually since
    # it's only 3 levels deep.
    nx = 0
    ny = 0
    head = length(str)
    node = mpt.partition_tree
    if str == "Lattice_("
        lattices = children(node)
        for lat in lattices
            name = lat.data[2]  
            xcomma_y = name[head+1:end-1]
            x_y = split(xcomma_y, ",")
            x = parse(Int64, x_y[1])
            y = parse(Int64, x_y[2][begin+1:end])
            nx = max(nx, x)
            ny = max(ny, y)
        end
    elseif str == "Module_("
        lattices = children(node)
        for lattice in lattices
            modules = children(lattice)
            for mod in modules
                name = mod.data[2]  
                xcomma_y = name[head+1:end-1]
                x_y = split(xcomma_y, ",")
                x = parse(Int64, x_y[1])
                y = parse(Int64, x_y[2][begin+1:end])
                nx = max(nx, x)
                ny = max(ny, y)
            end
        end
    elseif str == "Coarse_Cell_("
        lattices = children(node)
        for lattice in lattices
            modules = children(lattice)
            for mod in modules
                coarse_cells = children(mod)
                for cc in coarse_cells
                    name = cc.data[2]  
                    xcomma_y = name[head+1:end-1]
                    x_y = split(xcomma_y, ",")
                    x = parse(Int64, x_y[1])
                    y = parse(Int64, x_y[2][begin+1:end])
                    nx = max(nx, x)
                    ny = max(ny, y)
                end
            end
        end
    else
        error("Unknown partition type")
    end
    return nx, ny
end

# For MPTs representing an MPACT grid hierarchy
function partition_array(mpt::MeshPartitionTree, str::String)
    # str == "Lattice_(", "Module_(", or "Coarse_Cell_("
    #
    # Julia is bad at recusive types, so we do this manually since
    # it's only 3 levels deep.
    nx, ny = partition_array_size(mpt, str)
    head = length(str)
    arr = zeros(Int64, ny, nx)
    node = mpt.partition_tree
    if str == "Lattice_("
        lattices = children(node)
        for lat in lattices
            name = lat.data[2]  
            id = lat.data[1]
            xcomma_y = name[head+1:end-1]
            x_y = split(xcomma_y, ",")
            x = parse(Int64, x_y[1])
            y = parse(Int64, x_y[2][begin+1:end])
            arr[y, x] = id
        end
    elseif str == "Module_("
        lattices = children(node)
        for lattice in lattices
            modules = children(lattice)
            for mod in modules
                name = mod.data[2]  
                id = mod.data[1]
                xcomma_y = name[head+1:end-1]
                x_y = split(xcomma_y, ",")
                x = parse(Int64, x_y[1])
                y = parse(Int64, x_y[2][begin+1:end])
                nx = max(nx, x)
                ny = max(ny, y)
                arr[y, x] = id
            end
        end
    elseif str == "Coarse_Cell_("
        lattices = children(node)
        for lattice in lattices
            modules = children(lattice)
            for mod in modules
                coarse_cells = children(mod)
                for cc in coarse_cells
                    name = cc.data[2]  
                    id = cc.data[1]
                    xcomma_y = name[head+1:end-1]
                    x_y = split(xcomma_y, ",")
                    x = parse(Int64, x_y[1])
                    y = parse(Int64, x_y[2][begin+1:end])
                    nx = max(nx, x)
                    ny = max(ny, y)
                    arr[y, x] = id
                end
            end
        end
    else
        error("Unknown partition type")
    end
    return arr 
end

# For MPTs representing an MPACT grid hierarchy    
function partition_array_subset(mpt::MeshPartitionTree, str::String, id::Integer)
    # str == "Lattice_(", "Module_(", or "Coarse_Cell_("    
    #    
    # Julia is bad at recusive types, so we do this manually since    
    # it's only 3 levels deep.    
    node = mpt.partition_tree    
    children_vec = NTuple{3, Int64}[]    
    if str == "Lattice_("    
        head = length("Module_(")    
        lattices = children(node)    
        for lat in lattices    
            latid = lat.data[1]    
            if latid == id    
                for child in children(lat)    
                    name = child.data[2]      
                    xcomma_y = name[head+1:end-1]    
                    x_y = split(xcomma_y, ",")    
                    x = parse(Int64, x_y[1])    
                    y = parse(Int64, x_y[2][begin+1:end])    
                    push!(children_vec, (x, y, child.data[1]))    
                end    
                break    
            end    
        end    
    elseif str == "Module_("    
        head = length("Coarse_Cell_(")
        lattices = children(node)    
        for lattice in lattices    
            modules = children(lattice)    
            for mod in modules    
                modid = mod.data[1]    
                if modid == id    
                    for child in children(mod)    
                        name = child.data[2]      
                        xcomma_y = name[head+1:end-1]    
                        x_y = split(xcomma_y, ",")    
                        x = parse(Int64, x_y[1])    
                        y = parse(Int64, x_y[2][begin+1:end])    
                        push!(children_vec, (x, y, child.data[1]))    
                    end    
                    break    
                end    
            end    
        end    
    else    
        error("Unknown partition type")    
    end    
    sort!(children_vec)    
    yval = children_vec[1][1]
    next_yval = children_vec[1][1]
    cols = 0
    nelements = length(children_vec)
    while next_yval == yval && cols < nelements
        cols += 1
        next_yval = children_vec[cols][1]
    end
    rows = 1
    if cols != 1
        cols -= 1
        rows = nelements ÷ cols
    end
    return reshape(getindex.(children_vec, 3), rows, cols)
end 

# Partitions a mesh based upon the names of its groups 
# For a hierarchical partition, all partitions in the hierarchy must contain face sets 
# of the form "<name>_L<N>" where name is a string, and N is an integer.
# N is the level of the node in the tree: 1, 2, 3, etc...
# Example: "Grid_L1_triangle", or "Partition_L3"
function MeshPartitionTree(mesh::AbstractMesh; by::String = "MPACT")
    @info "Partitioning mesh: " * name(mesh)
    # Extract the names of all face sets that contain 'by' (the variable)
    partition_names = _get_partition_names(mesh, by)
    # Create a tree to store the partition hierarchy.
    root = _create_mesh_partition_tree(mesh, partition_names)
    # Construct the leaf meshes
    leaf_meshes = _create_leaf_meshes(mesh, root)
    return MeshPartitionTree(root, leaf_meshes)
end

function _create_leaf_meshes(mesh::AbstractMesh, root::Tree)
    leaf_nodes = sort!(leaves(root), by = x -> x.data)
    leaf_meshes = Vector{typeof(mesh)}(undef, length(leaf_nodes))
    leaf_ctr = 1
    node_children = children(root)
    if !isnothing(node_children)
        for child in sort(node_children, by = x -> x.data)
            child_mesh = submesh(mesh, data(child)[2])
            leaf_ctr = _create_leaf_meshes!(child_mesh, child, leaf_meshes, leaf_ctr)
        end
    else
        # Remove any Lattice, Module, or Coarse_Cell groups, since this info should now
        # be encoded in the tree
        mpact_groups = filter(x -> isa_MPACT_partition_name(x), keys(groups(mesh)))
        for grp in mpact_groups
            pop!(groups(mesh), grp)
        end
        name = data(root)[2]
        root.data = (leaf_ctr, name)
        leaf_meshes[leaf_ctr] = mesh
    end
    return leaf_meshes
end

function _create_leaf_meshes!(mesh::AbstractMesh,
                              node::Tree,
                              leaf_meshes,
                              leaf_ctr::Int64 = 1)
    node_children = children(node)
    if !isnothing(node_children)
        for child in sort(node_children, by = x -> x.data)
            child_mesh = submesh(mesh, data(child)[2])
            leaf_ctr = _create_leaf_meshes!(child_mesh, child, leaf_meshes, leaf_ctr)
        end
    else
        # Remove any Lattice, Module, or Coarse_Cell groups, since this info should now
        # be encoded in the tree
        mpact_groups = filter(x -> isa_MPACT_partition_name(x), keys(groups(mesh)))
        for grp in mpact_groups
            pop!(mesh.groups, grp)
        end
        leaf_meshes[leaf_ctr] = mesh
        name = data(node)[2]
        node.data = (leaf_ctr, name)
        leaf_ctr += 1
    end
    return leaf_ctr
end

# Create a tree to store grid relationships.
function _create_mesh_partition_tree(mesh::AbstractMesh, partition_names::Vector{String})
    root = Tree((1, name(mesh)))
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

function isa_MPACT_partition_name(x::String)
    return startswith(x, "Coarse_Cell_(") ||
           startswith(x, "Module_(") ||
           startswith(x, "Lattice_(")
end
# Extract partition names
function _get_partition_names(mesh::AbstractMesh, by::String)
    # If by === MPACT, then we will partition by Lattice, Module, and Coarse Cell
    # Otherwise, partition by the string contained in by
    partition_names = collect(keys(groups(mesh)))
    if length(partition_names) === 0
        error("The mesh does not have any groups.")
    end
    if by === "MPACT"
        filter!(x -> isa_MPACT_partition_name(x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any MPACT grid hierarchy groups.")
        end
    else
        filter!(x -> occursin(by, x), partition_names)
        if length(partition_names) === 0
            error("The mesh does not have any groups containing '", by, "'.")
        end
    end
    return sort!(partition_names)
end

function Base.show(io::IO, mpt::MeshPartitionTree{M}) where {M}
    println("MeshPartitionTree{", M, "}")
    println(tree(mpt))
    return nothing
end
