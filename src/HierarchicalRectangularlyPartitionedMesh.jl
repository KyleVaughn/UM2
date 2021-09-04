Base.@kwdef mutable struct HierarchicalRectangularlyPartitionedMesh
    name::String
    rect::Union{Nothing, Quadrilateral_2D} = nothing
    mesh::Union{Nothing, UnstructuredMesh_2D} = nothing
    parent::Union{
                  Nothing,
                  Ref{HierarchicalRectangularlyPartitionedMesh}
                 } = nothing
    children::Vector{Ref{HierarchicalRectangularlyPartitionedMesh}
                    } = Ref{HierarchicalRectangularlyPartitionedMesh}[]
    function HierarchicalRectangularlyPartitionedMesh(
            name::String,
            rect::Union{Nothing, Quadrilateral_2D}, 
            mesh::Union{Nothing, UnstructuredMesh_2D},
            parent::Union{Nothing, Ref{HierarchicalRectangularlyPartitionedMesh}},
            children::Vector{Ref{HierarchicalRectangularlyPartitionedMesh}})
        this = new(name, rect, mesh, parent, children)
        if parent !== nothing
            push!(parent[].children, Ref(this))
        end
        return this
    end 
end

function partition_rectangularly(mesh::UnstructuredMesh_2D)
    @info "Converting UnstructuredMesh_2D into HierarchicalRectangularlyPartitionedMesh"
    # Extract set names, grid names, and max level
    set_names, grid_names, max_level = _process_partition_rectangularly_input(mesh)

    # Create a tree to store grid relationships.
    root = _create_HRPM_tree(mesh, grid_names, max_level)

    # Construct the leaf meshes
    leaf_meshes = _create_HRPM_leaf_meshes(mesh, grid_names, max_level)

    # Construct the mesh hierarchy
    HRPM = _create_HRPM(root, leaf_meshes)
    return HRPM
end

# Extract set names, grid names, and max level
function _process_partition_rectangularly_input(mesh::UnstructuredMesh_2D)
    set_names = [ key for key in keys(mesh.face_sets) ]
    grid_names = copy(set_names)
    for set_name in set_names
        if !occursin("GRID_", uppercase(set_name))
            filter!(x->x ≠ set_name, grid_names)
        end
    end

    if length(grid_names) === 0
        error("No grid face sets in mesh.")
    end

    # Get the number of grid levels
    max_level = 0
    for grid_name in grid_names
        level = parse(Int64, grid_name[7])
        if max_level < level
            max_level = level
        end
    end

    return set_names, grid_names, max_level
end

# Create a tree to store grid relationships.
function _create_HRPM_tree(mesh::UnstructuredMesh_2D, grid_names::Vector{String}, max_level::Int64)
    root = Tree( data = mesh.name )
    current_nodes = Tree[]
    next_nodes = Tree[]
    old_grid_names = copy(grid_names)
    new_grid_names = copy(grid_names)
    # Do first level
    for grid_name in old_grid_names
        grid_level = parse(Int64, grid_name[7])
        if grid_level === 1
            # Add to appropriate node (root)
            push!(next_nodes, Tree(data = grid_name; parent=Ref(root)))
            filter!(x->x ≠ grid_name, new_grid_names)
        end
    end
    # Do all other levels:
    for level in 2:max_level
        old_grid_names = copy(new_grid_names)
        current_nodes = next_nodes
        next_nodes = []
        for grid_name in old_grid_names
            grid_level = parse(Int64, grid_name[7])
            if grid_level == level
                # find the parent for this grid
                grid_faces = Set(mesh.face_sets[grid_name])
                for node in current_nodes
                    node_faces = Set(mesh.face_sets[node.data])
                    if grid_faces ⊆ node_faces
                        push!(next_nodes, Tree(data = grid_name, parent=Ref(node)))
                        filter!(x->x ≠ grid_name, new_grid_names)
                        break
                    end
                end
            end
        end
    end
    return root
end

# Construct the leaf meshes
function _create_HRPM_leaf_meshes(mesh::UnstructuredMesh_2D, 
                                  grid_names::Vector{String},
                                  max_level::Int64) 
    # Generate the leaf meshes (The smallest spatially)
    leaf_meshes = UnstructuredMesh_2D[]
    leaf_names = String[]
    for name in grid_names
        level = parse(Int64, name[7])
        if level == max_level
            push!(leaf_names, name)
        end
    end
    for name in leaf_names
        push!(leaf_meshes, submesh(mesh, name))
    end
    # remove grid levels lower than max_level
    for leaf_mesh in leaf_meshes
        for name in keys(leaf_mesh.face_sets)
            if occursin("GRID_", uppercase(name))
                level = parse(Int64, name[7])
                if level != max_level
                    delete!(leaf_mesh.face_sets, name)
                end
            end
        end
    end
    return leaf_meshes
end

function get_level(HRPM::HierarchicalRectangularlyPartitionedMesh,; current_level=1)
    if HRPM.parent !== nothing
        return get_level(HRPM.parent[]; current_level = current_level + 1)
    else
        return current_level
    end
end
# Is this the last child in the parent's list of children?
# offset determines if the nth-parent is the last child
function _is_last_child(HRPM::HierarchicalRectangularlyPartitionedMesh; relative_offset=0)
    if HRPM.parent == nothing
        return true
    end
    if relative_offset > 0
        return _is_last_child(HRPM.parent[]; relative_offset=relative_offset-1)
    else
        nsiblings = length(HRPM.parent[].children) - 1
        return (HRPM.parent[].children[nsiblings + 1][] == HRPM)
    end
end
function Base.show(io::IO, HRPM::HierarchicalRectangularlyPartitionedMesh; relative_offset=0)
    nsiblings = 0
    for i = relative_offset:-1:1
        if i === 1 && _is_last_child(HRPM, relative_offset=i-1)
            print("└─ ")
        elseif i === 1
            print("├─ ")
        elseif _is_last_child(HRPM, relative_offset=i-1)
            print("   ")
        else
            print("│  ")
        end
    end
    println(HRPM.name)
    for child in HRPM.children
        show(io, child[]; relative_offset = relative_offset + 1)
    end
end

# Construct the HRPM
function _create_HRPM(tree::Tree, leaf_meshes::Vector{UnstructuredMesh_2D})
    # Construct the HRPM from the top down       
    root = HierarchicalRectangularlyPartitionedMesh( name = tree.data )
    _attach_HRPM_children(root, tree, leaf_meshes)
    # Add the rectangles
    AABB(root)
    return root
end

function _attach_HRPM_children(HRPM::HierarchicalRectangularlyPartitionedMesh, 
                               tree::Tree,
                               leaf_meshes::Vector{UnstructuredMesh_2D})
    for child in tree.children 
        name = child[].data
        child_mesh = HierarchicalRectangularlyPartitionedMesh(name = name, 
                                                              parent = Ref(HRPM) )
        for leaf_mesh in leaf_meshes
            if name == leaf_mesh.name
                child_mesh.mesh = leaf_mesh
            end
        end
        _attach_HRPM_children(child_mesh, child[], leaf_meshes)
    end
end

function AABB(HRPM::HierarchicalRectangularlyPartitionedMesh)
    if HRPM.rect !== nothing
        return HRPM.rect
    elseif HRPM.mesh !== nothing
        bb = AABB(HRPM.mesh)
        HRPM.rect = bb
        return bb
    elseif length(HRPM.children) > 0
        children_AABBs = Quadrilateral_2D[]
        for child in HRPM.children
            push!(children_AABBs, AABB(child[]))
        end
        point_tuples = [r.points for r in children_AABBs]
        points = Vector{typeof(point_tuples[1][1])}()
        for tuple in point_tuples
            for p in tuple
                push!(points, p)
            end
        end
        x = map(p->p[1], points)
        y = map(p->p[2], points)
        xmin = minimum(x)
        xmax = maximum(x)
        ymin = minimum(y)
        ymax = maximum(y)
        bb = Quadrilateral_2D(Point_2D(xmin, ymin), 
                              Point_2D(xmax, ymin),
                              Point_2D(xmax, ymax),
                              Point_2D(xmin, ymax))
        HRPM.rect = bb
        return bb
    end
end
