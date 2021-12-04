struct UnstructuredMesh_2D{F <: AbstractFloat, U <: Unsigned}
    name::String
    points::Vector{Point_2D{F}}
    edges::Vector{<:SVector{L, U} where {L}}
    materialized_edges::Vector{<:Edge_2D{F}}
    faces::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}}
    materialized_faces::Vector{<:Face_2D{F}}
    edge_face_connectivity::Vector{SVector{2, U}}
    face_edge_connectivity::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}}
    boundary_edges::Vector{Vector{U}}
    face_sets::Dict{String, Set{U}}
end

function UnstructuredMesh_2D{F, U}(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D{F}} = Point_2D{F}[],
        edges::Vector{<:SVector{L, U} where {L}} = SVector{2, U}[],
        materialized_edges::Vector{<:Edge_2D{F}} = LineSegment_2D{F}[],
        faces::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}} = SVector{4, U}[],
        materialized_faces::Vector{<:Face_2D{F}} = Triangle_2D{F}[],
        edge_face_connectivity::Vector{SVector{2, U}} = SVector{2, U}[],
        face_edge_connectivity ::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}} = SVector{3, U}[],
        boundary_edges::Vector{Vector{U}} = Vector{U}[],
        face_sets::Dict{String, Set{U}} = Dict{String, Set{U}}()
    ) where {F <: AbstractFloat, U <: Unsigned}
        return UnstructuredMesh_2D{F, U}(name,
                                         points,
                                         edges,
                                         materialized_edges,
                                         faces,
                                         materialized_faces,
                                         edge_face_connectivity,
                                         face_edge_connectivity,
                                         boundary_edges,
                                         face_sets,
                                        )
end

Base.broadcastable(mesh::UnstructuredMesh_2D) = Ref(mesh)

# Return a mesh with boundary edges and all necessary prerequisites to find the boundary edges
# Not type-stable
function add_boundary_edges(mesh::UnstructuredMesh_2D{F, U}, 
                            bounding_shape::String) where {F <: AbstractFloat, U <: Unsigned}
    if 0 == length(mesh.edge_face_connectivity)
        mesh = add_connectivity(mesh)
    end
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     materialized_edges = mesh.materialized_edges,
                                     faces = mesh.faces,
                                     materialized_faces = mesh.materialized_faces,
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = boundary_edges(mesh, bounding_shape),
                                     face_sets = mesh.face_sets
                                    )
end

# Return a mesh with face/edge connectivity, edge/face connectivity,
# and all necessary prerequisites to find the boundary edges
# Not type-stable
function add_connectivity(mesh::UnstructuredMesh_2D)
    return add_edge_face_connectivity(add_face_edge_connectivity(mesh))
end

# Return a mesh with edges
# Not type-stable
function add_edges(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat, U <: Unsigned}
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                     points = mesh.points,
                                     edges = edges(mesh.faces),
                                     materialized_edges = mesh.materialized_edges,
                                     faces = mesh.faces,
                                     materialized_faces = mesh.materialized_faces,
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

# Return a mesh with edge/face connectivity
# and all necessary prerequisites to find the boundary edges
# Not type-stable
function add_edge_face_connectivity(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat, U <: Unsigned}
    if 0 == length(mesh.face_edge_connectivity)
        mesh = add_face_edge_connectivity(mesh)
    end
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                    points = mesh.points,
                                    edges = mesh.edges,
                                    materialized_edges = mesh.materialized_edges,
                                    faces = mesh.faces,
                                    materialized_faces = mesh.materialized_faces,
                                    edge_face_connectivity = edge_face_connectivity(mesh.edges, 
                                                                                    mesh.faces, 
                                                                                    mesh.face_edge_connectivity),
                                    face_edge_connectivity = mesh.face_edge_connectivity,
                                    boundary_edges = mesh.boundary_edges,
                                    face_sets = mesh.face_sets
                                   )
end

# Return a mesh with every field created
# Not type-stable
function add_everything(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat, U <: Unsigned}
    return add_boundary_edges(add_materialized_faces(add_materialized_edges(mesh)), "Rectangle")
end

# Return a mesh with face/edge connectivity
# and all necessary prerequisites to find the boundary edges
# Not type-stable
function add_face_edge_connectivity(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat,
                                                                            U <: Unsigned}
    if 0 == length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                    points = mesh.points,
                                    edges = mesh.edges,
                                    materialized_edges = mesh.materialized_edges,
                                    faces = mesh.faces,
                                    materialized_faces = mesh.materialized_faces,
                                    edge_face_connectivity = mesh.edge_face_connectivity,
                                    face_edge_connectivity = face_edge_connectivity(mesh.faces, mesh.edges),
                                    boundary_edges = mesh.boundary_edges,
                                    face_sets = mesh.face_sets
                                   )
end

# Return a mesh with materialized edges
# Not type-stable
function add_materialized_edges(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat, U <: Unsigned}
    if 0 == length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     materialized_edges = materialize_edges(mesh.edges, mesh.points),
                                     faces = mesh.faces,
                                     materialized_faces = mesh.materialized_faces,
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

# Return a mesh with materialized faces
# Not type-stable
function add_materialized_faces(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat,
                                                                        U <: Unsigned}
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     materialized_edges = mesh.materialized_edges,
                                     faces = mesh.faces,
                                     materialized_faces = materialize_faces(mesh.faces, mesh.points),
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

# Return the area of a face set, input by name
# Not type-stable
function area(mesh::UnstructuredMesh_2D{F, U}, set_name::String) where {F <: AbstractFloat, U <: Unsigned}
    if 0 < length(mesh.materialized_faces)
        return area(mesh.materialized_faces, mesh.face_sets[set_name])::F
    else
        return area(mesh.faces, mesh.points, mesh.face_sets[set_name])::F
    end
end

# Axis-aligned bounding box, in 2d a rectangle.
# Type-stable, other than the error message
function bounding_box(mesh::UnstructuredMesh_2D{F, U};
                      rectangular_boundary::Bool = false) where {F <: AbstractFloat, U <: Unsigned}
    # If the mesh does not have any quadratic faces, the bounding_box may be determined entirely from the
    # points. If the mesh does have quadratic cells/faces, we need to find the bounding box of the edges
    # that border the mesh.
    if (any(x->x[1] ∈  UnstructuredMesh_2D_quadratic_cell_types, mesh.faces) &&
        !rectangular_boundary)
        @error "Cannot find bounding_box for a mesh with quadratic faces that does not have a rectangular boundary"
        return Quadrilateral_2D(Point_2D(F, 0),
                                Point_2D(F, 0),
                                Point_2D(F, 0),
                                Point_2D(F, 0))
    else # Can use points
        return bounding_box(mesh.points) 
    end
end

# Return the face containing point p.
# Not type-stable
function find_face(p::Point_2D{F}, mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat,
                                                                           U <: Unsigned}
    if 0 < length(mesh.materialized_faces)
        return U(find_face_explicit(p, mesh.materialized_faces))
    else
        return U(find_face_implicit(p, mesh.faces, mesh.points))
    end
end

# Return the intersection algorithm that will be used for l ∩ mesh
# Not type-stable
function get_intersection_algorithm(mesh::UnstructuredMesh_2D)
    if length(mesh.materialized_edges) !== 0
        return "Edges - Explicit"
    elseif length(mesh.edges) !== 0
        return "Edges - Implicit"
    elseif length(mesh.materialized_faces) !== 0
        return "Faces - Explicit"
    else
        return "Faces - Implicit"
    end
end

# Intersect a line with the mesh. Returns a vector of intersection points, sorted based
# upon distance from the line's start point
# Not type-stable
function intersect(l::LineSegment_2D{F}, 
                   mesh::UnstructuredMesh_2D{F}
                  ) where {F <: AbstractFloat}
    # Edges are faster, so they are the default
    if length(mesh.edges) !== 0 
        if 0 < length(mesh.materialized_edges)
            return intersect_edges_explicit(l, mesh.materialized_edges)
        else
            return intersect_edges_implicit(l, mesh.edges, mesh.points)
        end
    else
        if 0 < length(mesh.materialized_faces)
            return intersect_faces_explicit(l, mesh.materialized_faces)
        else
            return intersect_faces_implicit(l, mesh.faces, mesh.points)
        end
    end
end

function reorder_points_to_hilbert(mesh::UnstructuredMesh_2D{F, U}
                           ) where {F <: AbstractFloat, U <: Unsigned}
    # Points
    point_map_Int64 = remap_points_to_hilbert(mesh.points) 
    point_map = U.(point_map_Int64)
    new_points = mesh.points[point_map] 
    # Adjust face indices
    nfaces = length(mesh.faces)
    new_faces_vec = [ point_map[face] for face in mesh.faces]  
    for i in 1:nfaces
        new_faces_vec[i][1] = mesh.faces[i][1]
    end
    new_faces = SVector.(new_faces_vec)
    # Adjust edge indices
    if 0 < length(mesh.edges)
        new_edges = [ SVector(point_map[edge]) for edge in mesh.edges ]
    else
        new_edges = mesh.edges
    end
    # Adjust face_sets
    # This is done by reference
    for key in keys(mesh.face_sets)
        println(key)
    end
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                     points = new_points,
                                     edges = new_edges,
                                     faces = new_faces,
                                     face_sets = mesh.face_sets
                                    )
end

# How to display a mesh in REPL
function Base.show(io::IO, mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat, U <: Unsigned}
    println(io, "UnstructuredMesh_2D{$F}{$U}")
    name = mesh.name
    println(io, "  ├─ Name      : $name")
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        size_KB = size_MB*1000
        println(io, "  ├─ Size (KB) : $size_KB")
    else
        println(io, "  ├─ Size (MB) : $size_MB")
    end
    npoints = length(mesh.points)
    println(io, "  ├─ Points    : $npoints")
    if 0 < length(mesh.materialized_edges)
        nedges = length(mesh.materialized_edges)
        nlin   = count(x->x isa LineSegment_2D,  mesh.materialized_edges)
        nquad  = count(x->x isa QuadraticSegment_2D,  mesh.materialized_edges)
    elseif 0 < length(mesh.edges)
        nedges = length(mesh.edges)
        nlin   = count(x->length(x) == 2,  mesh.edges)
        nquad  = count(x->length(x) == 3,  mesh.edges)
    else
        nedges = 0
        nlin = 0
        nquad = 0
    end
    ematerialized = length(mesh.materialized_edges) !== 0
    println(io, "  ├─ Edges     : $nedges")
    println(io, "  │  ├─ Linear         : $nlin")
    println(io, "  │  ├─ Quadratic      : $nquad")
    println(io, "  │  └─ Materialized?  : $ematerialized")
    nfaces = length(mesh.faces)
    println(io, "  ├─ Faces     : $nfaces")
    if 0 < nfaces
        ntri   = count(x->x[1] == 5,  mesh.faces)
        nquad  = count(x->x[1] == 9,  mesh.faces)
        ntri6  = count(x->x[1] == 22, mesh.faces)
        nquad8 = count(x->x[1] == 23, mesh.faces)
    else
        ntri   = 0
        nquad  = 0
        ntri6  = 0
        nquad8 = 0
    end
    fmaterialized = length(mesh.materialized_faces) !== 0
    println(io, "  │  ├─ Triangle       : $ntri")
    println(io, "  │  ├─ Quadrilateral  : $nquad")
    println(io, "  │  ├─ Triangle6      : $ntri6")
    println(io, "  │  ├─ Quadrilateral8 : $nquad8")
    println(io, "  │  └─ Materialized?  : $fmaterialized")
    nface_sets = length(keys(mesh.face_sets))
    ef_con = 0 < length(mesh.edge_face_connectivity)
    fe_con = 0 < length(mesh.face_edge_connectivity)
    println(io, "  ├─ Connectivity")
    println(io, "  │  ├─ Edge/Face : $ef_con")
    println(io, "  │  └─ Face/Edge : $fe_con")
    if 0 < length(mesh.boundary_edges)
        nbsides = length(mesh.boundary_edges)
        nbedges = 0
        for side in mesh.boundary_edges
            nbedges += length(side)
        end
        println(io, "  ├─ Boundary edges")
        println(io, "  │  ├─ Edges : $nbedges")
        println(io, "  │  └─ Sides : $nbsides")
    else
        nbsides = 0
        nbedges = 0
        println(io, "  ├─ Boundary edges")
        println(io, "  │  ├─ Edges : $nbedges")
        println(io, "  │  └─ Sides : $nbsides")
    end
    println(io, "  └─ Face sets : $nface_sets")
end

# Return a mesh composed of the faces in the face set set_name
# Not type-stable
function submesh(mesh::UnstructuredMesh_2D{F, U},
                 set_name::String) where {F <: AbstractFloat, U <: Unsigned}
    @debug "Creating submesh for '$set_name'"
    face_ids = mesh.face_sets[set_name]
    return submesh(set_name, mesh.points, mesh.faces, mesh.face_sets, face_ids)
end
