struct UnstructuredMesh_2D{T <: AbstractFloat, I <: Unsigned}
    name::String
    points::Vector{Point_2D{T}}
    edges::Vector{<:Union{NTuple{2, I}, NTuple{3, I}}}
    materialized_edges::Vector{<:Edge_2D{T}}
    faces::Vector{<:Tuple{Vararg{I, N} where N}}
    materialized_faces::Vector{<:Face_2D{T}}
    edge_face_connectivity::Vector{NTuple{2, I}}
    face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}}
    boundary_edges::Vector{Vector{I}}
    face_sets::Dict{String, Set{I}}
end

function UnstructuredMesh_2D{T, I}(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D{T}} = Point_2D{T}[],
        edges::Vector{<:Union{NTuple{2, I}, NTuple{3, I}}} = NTuple{2, I}[],
        materialized_edges::Vector{<:Edge_2D{T}} = LineSegment_2D{T}[],
        faces::Vector{<:Tuple{Vararg{I, N} where N}} = NTuple{4, I}[],
        materialized_faces::Vector{<:Face_2D{T}} = Triangle_2D{T}[],
        edge_face_connectivity::Vector{NTuple{2, I}} = NTuple{2, I}[],
        face_edge_connectivity ::Vector{<:Tuple{Vararg{I, M} where M}} = NTuple{3, I}[],
        boundary_edges::Vector{Vector{I}} = Vector{I}[],
        face_sets::Dict{String, Set{I}} = Dict{String, Set{I}}()
    ) where {T <: AbstractFloat, I <: Unsigned}
        return UnstructuredMesh_2D{T, I}(name,
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
function add_boundary_edges(mesh::UnstructuredMesh_2D{T, I};
                          bounding_shape::String="Rectangle") where {T<:AbstractFloat, I <: Unsigned}
    if 0 == length(mesh.edge_face_connectivity)
        mesh = add_connectivity(mesh)
    end
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     materialized_edges = mesh.materialized_edges,
                                     faces = mesh.faces,
                                     materialized_faces = mesh.materialized_faces,
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = boundary_edges(mesh),
                                     face_sets = mesh.face_sets
                                    )
end

# Return a mesh with face/edge connectivity, edge/face connectivity,
# and all necessary prerequisites to find the boundary edges
function add_connectivity(mesh::UnstructuredMesh_2D{T}) where {T<:AbstractFloat}
    return add_edge_face_connectivity(add_face_edge_connectivity(mesh))
end

# Return a mesh with edges
function add_edges(mesh::UnstructuredMesh_2D{T, I}) where {T<:AbstractFloat, I <: Unsigned}
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                     points = mesh.points,
                                     edges = edges(mesh),
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
function add_edge_face_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T<:AbstractFloat, I<:Unsigned}
    if 0 == length(mesh.face_edge_connectivity)
        mesh = add_face_edge_connectivity(mesh)
    end
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                    points = mesh.points,
                                    edges = edges(mesh),
                                    materialized_edges = mesh.materialized_edges,
                                    faces = mesh.faces,
                                    materialized_faces = mesh.materialized_faces,
                                    edge_face_connectivity = edge_face_connectivity(mesh),
                                    face_edge_connectivity = mesh.face_edge_connectivity,
                                    boundary_edges = mesh.boundary_edges,
                                    face_sets = mesh.face_sets
                                   )
end

# Return a mesh with every field created
function add_everything(mesh::UnstructuredMesh_2D{T, I}) where {T<:AbstractFloat, I<:Unsigned}
    return add_boundary_edges(
           add_connectivity(
           add_materialized_faces(
           add_materialized_edges(
           add_edges(mesh)))))
end

# Return a mesh with face/edge connectivity
# and all necessary prerequisites to find the boundary edges
function add_face_edge_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                            I <: Unsigned}
    if 0 == length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                    points = mesh.points,
                                    edges = mesh.edges,
                                    materialized_edges = mesh.materialized_edges,
                                    faces = mesh.faces,
                                    materialized_faces = mesh.materialized_faces,
                                    edge_face_connectivity = mesh.edge_face_connectivity,
                                    face_edge_connectivity = face_edge_connectivity(mesh),
                                    boundary_edges = mesh.boundary_edges,
                                    face_sets = mesh.face_sets
                                   )
end

# Return a mesh with materialized edges
function add_materialized_edges(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    if 0 == length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     materialized_edges = materialize_edges(mesh),
                                     faces = mesh.faces,
                                     materialized_faces = mesh.materialized_faces,
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

# Return a mesh with materialized faces
function add_materialized_faces(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                        I <: Unsigned}
    return UnstructuredMesh_2D{T, I}(name = mesh.name,
                                     points = mesh.points,
                                     edges = mesh.edges,
                                     materialized_edges = mesh.materialized_edges,
                                     faces = mesh.faces,
                                     materialized_faces = materialize_faces(mesh),
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
end

# Return a vector if the faces adjacent to the face of ID face
function adjacent_faces(face::I,
                        mesh::UnstructuredMesh_2D{T, I}
                        ) where {I <: Unsigned, T <: AbstractFloat}
    return adjacent_faces(face, mesh.face_edge_connectivity, mesh.edge_face_connectivity)
end

# Return the area of a face set, input by name
function area(mesh::UnstructuredMesh_2D{T, I}, set_name::String) where {T <: AbstractFloat, I <: Unsigned}
    return area(mesh, mesh.face_sets[set_name])
end

# Return the area of a face set
function area(mesh::UnstructuredMesh_2D{T, I}, face_set::Set{I}) where {T <: AbstractFloat, I <: Unsigned}
    unsupported = count(x->x[1] ∉  UnstructuredMesh_2D_cell_types, mesh.faces)
    if 0 < unsupported
        @error "Mesh contains an unsupported face type"
    end
    return mapreduce(x->area(mesh, mesh.faces[x]), +, face_set)::T
end

# Axis-aligned bounding box, in 2d a rectangle.
function bounding_box(mesh::UnstructuredMesh_2D{T, I};
                      rectangular_boundary::Bool = false) where {T <: AbstractFloat, I <: Unsigned}
    # If the mesh does not have any quadratic faces, the bounding_box may be determined entirely from the
    # points. If the mesh does have quadratic cells/faces, we need to find the bounding box of the edges
    # that border the mesh.
    if (any(x->x[1] ∈  UnstructuredMesh_2D_quadratic_cell_types, mesh.faces) &&
        !rectangular_boundary)
        @error "Cannot find bounding_box for a mesh with quadratic faces that does not have a rectangular boundary"
        return Quadrilateral_2D(Point_2D{T}(T[0, 0]),
                                Point_2D{T}(T[0, 0]),
                                Point_2D{T}(T[0, 0]),
                                Point_2D{T}(T[0, 0]))
    else # Can use points
        x = map(p->p[1], mesh.points)
        y = map(p->p[2], mesh.points)
        xmin = minimum(x)
        xmax = maximum(x)
        ymin = minimum(y)
        ymax = maximum(y)
        return Quadrilateral_2D(Point_2D(xmin, ymin),
                                Point_2D(xmax, ymin),
                                Point_2D(xmax, ymax),
                                Point_2D(xmin, ymax))
    end
end

# Create the edges for each face
function edges(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    return edges(mesh.faces)
end

# A vector of 2-tuples, denoting the face ID each edge is connected to. If the edge
# is a boundary edge, face ID 0 is returned
function edge_face_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                        I <: Unsigned}
    # Each edge should only border 2 faces if it is an interior edge, and 1 face if it is
    # a boundary edge.
    # Loop through each face in the face_edge_connectivity vector and mark each edge with
    # the faces that it borders.
    if length(mesh.edges) === 0
        @error "Mesh does not have edges!"
        edge_face = MVector{2, I}[]
    elseif length(mesh.face_edge_connectivity) === 0
        edge_face = [MVector{2, I}(zeros(I, 2)) for i in eachindex(mesh.edges)]
        face_edge_conn = face_edge_connectivity(mesh)
        for (iface, edges) in enumerate(face_edge_conn)
            for iedge in edges
                # Add the face id in the first non-zero position of the edge_face conn. vec.
                if edge_face[iedge][1] == 0
                    edge_face[iedge][1] = iface
                elseif edge_face[iedge][2] == 0
                    edge_face[iedge][2] = iface
                else
                    @error "Edge $iedge seems to have 3 faces associated with it!"
                end
            end
        end
    else # has face_edge connectivity
        edge_face = [MVector{2, I}(zeros(I, 2)) for i in eachindex(mesh.edges)]
        for (iface, edges) in enumerate(mesh.face_edge_connectivity)
            for iedge in edges
                # Add the face id in the first non-zero position of the edge_face conn. vec.
                if edge_face[iedge][1] == 0
                    edge_face[iedge][1] = iface
                elseif edge_face[iedge][2] == 0
                    edge_face[iedge][2] = iface
                else
                    @error "Edge $iedge seems to have 3 faces associated with it!"
                end
            end
        end
    end
    return [Tuple(sort(two_faces)) for two_faces in edge_face]
end

# A vector of Tuples, denoting the edge ID each face is connected to.
function face_edge_connectivity(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    # A vector of MVectors of zeros for each face
    # Each MVector is the length of the number of edges
    face_edge = [MVector{Int64(num_edges(face)), I}(zeros(I, num_edges(face))) for face in mesh.faces]
    if length(mesh.edges) === 0
        @error "Mesh does not have edges!"
    else
        # for each face in the mesh, generate the edges.
        # Search for the index of the edge in the mesh.edges vector
        # Insert the index of the edge into the face_edge connectivity vector
        for i in eachindex(mesh.faces)
            for (j, edge) in enumerate(edges(mesh.faces[i]))
                face_edge[i][j] = searchsortedfirst(mesh.edges, Tuple(edge))
            end
        end
    end
    return [Tuple(sort(conn)) for conn in face_edge]::Vector{<:Tuple{Vararg{I, N} where N}}
end

# Return the faces which share the vertex of ID p.
function faces_sharing_vertex(p::P, mesh::UnstructuredMesh_2D{T, I}) where {P <: Integer,
                                                                            T <: AbstractFloat,
                                                                            I <: Unsigned}
    return faces_sharing_vertex(p, mesh.faces)
end

# Return the face containing point p.
function find_face(p::Point_2D{T}, mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                           I <: Unsigned}
    if 0 < length(mesh.materialized_faces)
        return I(find_face_explicit(p, mesh.materialized_faces))
    else
        return I(find_face_implicit(p, mesh, mesh.faces))
    end
end

# Return the intersection algorithm that will be used for l ∩ mesh
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
function intersect(l::LineSegment_2D{T}, mesh::UnstructuredMesh_2D{T}
                   ) where {T <: AbstractFloat}
    # Edges are faster, so they are the default
    if length(mesh.edges) !== 0 || length(mesh.materialized_edges) !== 0
        return intersect_edges(l, mesh)
    else
        return intersect_faces(l, mesh)
    end
end

# How to display a mesh in REPL
function Base.show(io::IO, mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    println(io, "UnstructuredMesh_2D{$T}{$I}")
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
function submesh(mesh::UnstructuredMesh_2D{T, I},
                 set_name::String) where {T <: AbstractFloat, I <: Unsigned}
    @debug "Creating submesh for '$set_name'"
    face_ids = mesh.face_sets[set_name]
    return submesh(mesh, face_ids, name = set_name)
end
