mutable struct HierarchicalRectangularlyPartitionedMesh{T<:AbstractFloat}
    name::String
    rect::Quadrilateral_2D{T}
    mesh::Ref{UnstructuredMesh_2D{T}}
    parent::Ref{HierarchicalRectangularlyPartitionedMesh{T}}
    children::Vector{Ref{HierarchicalRectangularlyPartitionedMesh{T}}}
end

function HierarchicalRectangularlyPartitionedMesh{T}(;
        name::String = "DefaultHRPMName",
        rect::Quadrilateral_2D{T} = Quadrilateral_2D(Point_2D(T, 0), 
                                                     Point_2D(T, 0), 
                                                     Point_2D(T, 0), 
                                                     Point_2D(T, 0)), 
        mesh::Ref{UnstructuredMesh_2D{T}} = Ref{UnstructuredMesh_2D{T}}(),
        parent::Ref{HierarchicalRectangularlyPartitionedMesh{T}}
            = Ref{HierarchicalRectangularlyPartitionedMesh{T}}(),
        children::Vector{Ref{HierarchicalRectangularlyPartitionedMesh{T}}}
            = Ref{HierarchicalRectangularlyPartitionedMesh{T}}[]
        ) where {T<:AbstractFloat}
    this = HierarchicalRectangularlyPartitionedMesh(name, rect, mesh, parent, children)
    if isassigned(parent)
        push!(parent[].children, Ref(this))
    end
    return this
end 

function partition_rectangularly(mesh::UnstructuredMesh_2D{T}) where {T<:AbstractFloat}
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
function _process_partition_rectangularly_input(mesh::UnstructuredMesh_2D{T}) where {T<:AbstractFloat}
    set_names = collect(keys(mesh.face_sets))
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
function _create_HRPM_tree(mesh::UnstructuredMesh_2D{T}, grid_names::Vector{String}, max_level::Int64) where {T<:AbstractFloat}
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
function _create_HRPM_leaf_meshes(mesh::UnstructuredMesh_2D{T}, 
                                  grid_names::Vector{String},
                                  max_level::Int64) where {T<:AbstractFloat}
    # Generate the leaf meshes (The smallest spatially)
    leaf_meshes = UnstructuredMesh_2D{T}[]
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
    # remove grid levels
    for leaf_mesh in leaf_meshes
        for name in keys(leaf_mesh.face_sets)
            if occursin("GRID_", uppercase(name))
                delete!(leaf_mesh.face_sets, name)
            end
        end
    end
    return leaf_meshes
end

function get_level(HRPM::HierarchicalRectangularlyPartitionedMesh,; current_level=1)
    if isassigned(HRPM.parent)
        return get_level(HRPM.parent[]; current_level = current_level + 1)
    else
        return current_level
    end
end
# Is this the last child in the parent's list of children?
# offset determines if the nth-parent is the last child
function _is_last_child(HRPM::HierarchicalRectangularlyPartitionedMesh; relative_offset=0)
    if !isassigned(HRPM.parent)
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
            print(io, "└─ ")
        elseif i === 1
            print(io, "├─ ")
        elseif _is_last_child(HRPM, relative_offset=i-1)
            print(io, "   ")
        else
            print(io, "│  ")
        end
    end
    println(io, HRPM.name)
    for child in HRPM.children
        show(io, child[]; relative_offset = relative_offset + 1)
    end
end

# Construct the HRPM
function _create_HRPM(tree::Tree, leaf_meshes::Vector{UnstructuredMesh_2D{T}}) where {T<:AbstractFloat}
    # Construct the HRPM from the top down       
    root = HierarchicalRectangularlyPartitionedMesh{T}( name = tree.data )
    _attach_HRPM_children(root, tree, leaf_meshes)
    # Add the rectangles
    AABB(root)
    return root
end

function _attach_HRPM_children(HRPM::HierarchicalRectangularlyPartitionedMesh{T}, 
                               tree::Tree,
                               leaf_meshes::Vector{UnstructuredMesh_2D{T}}) where {T<:AbstractFloat}
    for child in tree.children 
        name = child[].data
        child_mesh = HierarchicalRectangularlyPartitionedMesh{T}(name = name, 
                                                              parent = Ref(HRPM) )
        for leaf_mesh in leaf_meshes
            if name == leaf_mesh.name
                child_mesh.mesh[] = leaf_mesh
            end
        end
        _attach_HRPM_children(child_mesh, child[], leaf_meshes)
    end
end

function AABB(HRPM::HierarchicalRectangularlyPartitionedMesh{T}) where {T <: AbstractFloat}
    if HRPM.rect !== Quadrilateral_2D(Point_2D(T, 0), Point_2D(T, 0), Point_2D(T, 0), Point_2D(T, 0))
        return HRPM.rect
    elseif isassigned(HRPM.mesh)
        bb = AABB(HRPM.mesh[], rectangular_boundary=true)
        HRPM.rect = bb
        return bb
    elseif 0 < length(HRPM.children)
        children_AABBs = Quadrilateral_2D[]
        for child in HRPM.children
            push!(children_AABBs, AABB(child[]))
        end
        point_tuples = [r.points for r in children_AABBs]
        points = Vector{typeof(point_tuples[1][1])}()
        for tuple in point_tuples
            append!(points, collect(tuple))
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

function intersect(l::LineSegment_2D{T}, 
        HRPM::HierarchicalRectangularlyPartitionedMesh{T}) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    if (l ∩ HRPM.rect)[1] == 2
        if isassigned(HRPM.mesh)
            append!(intersection_points, l ∩ HRPM.mesh[])
        elseif 0 < length(HRPM.children)
            for child::Ref{HierarchicalRectangularlyPartitionedMesh{T}} in HRPM.children
                append!(intersection_points, l ∩ child[]::HierarchicalRectangularlyPartitionedMesh{T})
            end
            # Sort the points based upon their distance to the first point
            distances = distance.(l.points[1], intersection_points)
            sorted_pairs = sort(collect(zip(distances, intersection_points)); by=first);
            intersection_points::Vector{Point_2D{T}} = getindex.(sorted_pairs, 2)
            if 0 < length(intersection_points)
                # Remove duplicate points
                intersection_points_reduced = Point_2D{T}[]
                push!(intersection_points_reduced, intersection_points[1]) 
                for i = 2:length(intersection_points)
                    if last(intersection_points_reduced) ≉ intersection_points[i]
                        push!(intersection_points_reduced, intersection_points[i])
                    end
                end
                intersection_points = intersection_points_reduced
            end
        end
    end
    return intersection_points::Vector
end

function materialize(HRPM::HierarchicalRectangularlyPartitionedMesh)
    if isassigned(HRPM.mesh)
        HRPM.mesh[] = materialize(HRPM.mesh[])
    elseif 0 < length(HRPM.children)
        for child in HRPM.children
            materialize(child[])
        end
    end
end
