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
                          bounding_shape="Rectangle") where {T<:AbstractFloat, I <: Unsigned}
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
                                     materialized_faces = materialized_faces(mesh),
                                     edge_face_connectivity = mesh.edge_face_connectivity,
                                     face_edge_connectivity = mesh.face_edge_connectivity,
                                     boundary_edges = mesh.boundary_edges,
                                     face_sets = mesh.face_sets
                                    )
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
function AABB(mesh::UnstructuredMesh_2D{T, I};
              rectangular_boundary=false) where {T <: AbstractFloat, I <: Unsigned}
    # If the mesh does not have any quadratic faces, the AABB may be determined entirely from the
    # points. If the mesh does have quadratic cells/faces, we need to find the bounding box of the edges
    # that border the mesh.
    if (any(x->x[1] ∈  UnstructuredMesh_2D_quadratic_cell_types, mesh.faces) &&
        !rectangular_boundary)
        @error "Cannot find AABB for a mesh with quadratic faces that does not have a rectangular boundary"
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

function get_adjacent_faces(face::I,
                            mesh::UnstructuredMesh_2D{T, I}
                            ) where {I <: Unsigned, T <: AbstractFloat}
    return get_adjacent_faces(face, mesh.face_edge_connectivity, mesh.edge_face_connectivity)
end

function get_adjacent_faces(face::I,
                            face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}},
                            edge_face_connectivity::Vector{NTuple{2, I}}
                            )where {I <: Unsigned}
    edges = face_edge_connectivity[face]
    adjacent_faces = I[]
    for edge in edges
        faces = edge_face_connectivity[edge]
        for face_id in faces
            if face_id != face && face_id != 0
                push!(adjacent_faces, face_id)
            end
        end
    end
    return adjacent_faces
end

function get_edge_points(mesh::UnstructuredMesh_2D{T, I},
                         edge::NTuple{2, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (mesh.points[edge[1]], mesh.points[edge[2]])
end

function get_edge_points(mesh::UnstructuredMesh_2D{T, I},
                         edge::NTuple{3, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (mesh.points[edge[1]],
            mesh.points[edge[2]],
            mesh.points[edge[3]]
           )
end

function get_edge_points(points::Vector{Point_2D{T}},
                         edge::NTuple{2, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (points[edge[1]], points[edge[2]])
end

function get_edge_points(points::Vector{Point_2D{T}},
                         edge::NTuple{3, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (points[edge[1]],
            points[edge[2]],
            points[edge[3]])
end

# This gets called enough that ugly code for optimization makes sense
function get_face_points(mesh::UnstructuredMesh_2D{T, I},
                         face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:4])])::NTuple{3, Point_2D{T}}
end

function get_face_points(mesh::UnstructuredMesh_2D{T, I},
                         face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:5])])::NTuple{4, Point_2D{T}}
end

function get_face_points(mesh::UnstructuredMesh_2D{T, I},
                         face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:7])])::NTuple{6, Point_2D{T}}
end

function get_face_points(mesh::UnstructuredMesh_2D{T, I},
                         face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:9])])::NTuple{8, Point_2D{T}}
end

function get_intersection_algorithm(mesh::UnstructuredMesh_2D)
    if length(mesh.materialized_edges) !== 0
        return "Edges - Materialized"
    elseif length(mesh.edges) !== 0
        return "Edges - Implicit"
    elseif length(mesh.materialized_faces) !== 0
        return "Faces - Materialized"
    else
        return "Faces - Implicit"
    end
end

function get_shared_edge(face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}},
                         face1::I, face2::I) where {T <: AbstractFloat, I <: Unsigned}
    edges1 = face_edge_connectivity[face1]
    edges2 = face_edge_connectivity[face1]
    for edge1 in edges1
        for edge2 in edges2
            if edge1 == edge2
                return edge1
            end
        end
    end
    return I(0)
end

function in(p::Point_2D{T}, mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    for point in mesh.points
        if p ≈ point
            return true
        end
    end
    return false
end

function insert_boundary_edge!(edge_index::I, edge_indices::Vector{I}, p_ref::Point_2D{T},
        mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    # Compute the minimum distance from the edge to be inserted to the reference point
    edge_points = get_edge_points(mesh, mesh.edges[edge_index])
    insertion_distance = minimum([ distance(p_ref, p_edge) for p_edge in edge_points ])
    # Loop through the edge indices until an edge with greater distance from the reference point
    # is found, then insert
    nindices = length(edge_indices)
    for i = 1:nindices
        iedge = edge_indices[i]
        iedge_points = get_edge_points(mesh, mesh.edges[iedge])
        iedge_distance = minimum([ distance(p_ref, p_edge) for p_edge in iedge_points ])
        if insertion_distance < iedge_distance
            insert!(edge_indices, i, edge_index)
            return
        end
    end
    insert!(edge_indices, nindices+1, edge_index)
    return nothing
end

function intersect(l::LineSegment_2D{T}, mesh::UnstructuredMesh_2D{T}
                   ) where {T <: AbstractFloat}
    if length(mesh.edges) !== 0 || length(mesh.materialized_edges) !== 0
        return intersect_edges(l, mesh)
    else
        return intersect_faces(l, mesh)
    end
end

function intersect_edges(l::LineSegment_2D{T},
                         mesh::UnstructuredMesh_2D{T, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    if length(mesh.materialized_edges) !== 0
        intersection_points = intersect_edges_explicit(l, mesh.materialized_edges)
        return sort_intersection_points(l, intersection_points)
    else
        intersection_points = intersect_edges_implicit(l, mesh, mesh.edges)
        return sort_intersection_points(l, intersection_points)
    end
end

function intersect_edges_explicit(l::LineSegment_2D{T},
                                  edges::Vector{LineSegment_2D{T}}) where {T <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for edge in edges
        npoints, point = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return intersection_points
end

function intersect_edges_explicit(l::LineSegment_2D{T},
                                  edges::Vector{QuadraticSegment_2D{T}}) where {T <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for edge in edges
        npoints, points = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_edges_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  edges::Vector{NTuple{2, I}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, point = l ∩ LineSegment_2D(get_edge_points(mesh, edge))
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return intersection_points
end

function intersect_edges_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  edges::Vector{NTuple{3, I}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, points = l ∩ QuadraticSegment_2D(get_edge_points(mesh, edge))
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_edge_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 edge::NTuple{2, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ LineSegment_2D(get_edge_points(mesh, edge))
end

function intersect_edge_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 edge::NTuple{3, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ QuadraticSegment_2D(get_edge_points(mesh, edge))
end

function intersect_faces(l::LineSegment_2D{T},
                         mesh::UnstructuredMesh_2D{T, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    if length(mesh.materialized_faces) !== 0
        intersection_points = intersect_faces_explicit(l, mesh.materialized_faces)
        return sort_intersection_points(l, intersection_points)
    else
        # Check if any of the face types are unsupported
        unsupported = sum(x->x[1] ∉  UnstructuredMesh_2D_cell_types, mesh.faces)
        if 0 < unsupported
            @error "Mesh contains an unsupported face type"
        end
        intersection_points = intersect_faces_implicit(l, mesh, mesh.faces)
        return sort_intersection_points(l, intersection_points)
    end
end

function intersect_faces_explicit(l::LineSegment_2D{T},
                                  faces::Vector{<:Union{Triangle_2D{T}, Quadrilateral_2D{T}}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_faces_explicit(l::LineSegment_2D{T},
                                  faces::Vector{<:Union{Triangle6_2D{T}, Quadrilateral8_2D{T}}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_faces_explicit(l::LineSegment_2D{T},
                                  faces::Vector{<:Face_2D{T}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

function intersect_faces_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  faces::Vector{<:Union{NTuple{4, I}, NTuple{5, I}}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for face in faces
        type_id = face[1]
        if type_id == 5 # Triangle
            npoints, points = l ∩ Triangle_2D(get_face_points(mesh,
                                                              face::NTuple{4, I})::NTuple{3, Point_2D{T}})
            # If the intersections yields 1 or more points, push those points to the array of points
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 9 # Quadrilateral
            npoints, points = l ∩ Quadrilateral_2D(get_face_points(mesh,
                                                                   face::NTuple{5, I})::NTuple{4, Point_2D{T}})
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        end
    end
    return intersection_points
end

function intersect_faces_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  faces::Vector{<:Union{NTuple{7, I}, NTuple{9, I}}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for face in faces
        type_id = face[1]
        if type_id == 22 # Triangle6
            npoints, points = l ∩ Triangle6_2D(get_face_points(mesh,
                                                               face::NTuple{7, I})::NTuple{6, Point_2D{T}})
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 23 # Quadrilateral8
            npoints, points = l ∩ Quadrilateral8_2D(get_face_points(mesh,
                                                                    face::NTuple{9, I})::NTuple{8, Point_2D{T}})
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        end
    end
    return intersection_points
end

function intersect_faces_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  faces::Vector{<:Tuple{Vararg{I, N} where N}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for face in faces
        type_id = face[1]
        if type_id == 5 # Triangle
            npoints, points = l ∩ Triangle_2D(get_face_points(mesh,
                                                              face::NTuple{4, I})::NTuple{3, Point_2D{T}})
            # If the intersections yields 1 or more points, push those points to the array of points
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 9 # Quadrilateral
            npoints, points = l ∩ Quadrilateral_2D(get_face_points(mesh,
                                                                   face::NTuple{5, I})::NTuple{4, Point_2D{T}})
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 22 # Triangle6
            npoints, points = l ∩ Triangle6_2D(get_face_points(mesh,
                                                               face::NTuple{7, I})::NTuple{6, Point_2D{T}})
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        elseif type_id == 23 # Quadrilateral8
            npoints, points = l ∩ Quadrilateral8_2D(get_face_points(mesh,
                                                                    face::NTuple{9, I})::NTuple{8, Point_2D{T}})
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
            end
        end
    end
    return intersection_points
end

function materialize_faces(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                    I <: Unsigned}
    return materialize_face.(mesh, mesh.faces)::Vector{<:Face_2D{T}}
end

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
        nlin   = sum(x->x isa LineSegment_2D,  mesh.materialized_edges)
        nquad  = sum(x->x isa QuadraticSegment_2D,  mesh.materialized_edges)
    elseif 0 < length(mesh.edges)
        nedges = length(mesh.edges)
        nlin   = sum(x->length(x) == 2,  mesh.edges)
        nquad  = sum(x->length(x) == 3,  mesh.edges)
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
        ntri   = sum(x->x[1] == 5,  mesh.faces)
        nquad  = sum(x->x[1] == 9,  mesh.faces)
        ntri6  = sum(x->x[1] == 22, mesh.faces)
        nquad8 = sum(x->x[1] == 23, mesh.faces)
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

function sort_intersection_points(l::LineSegment_2D{T},
                                  points::Vector{Point_2D{T}}) where {T <: AbstractFloat}
    if 0 < length(points)
        # Sort the points based upon their distance to the first point in the line
        distances = distance.(l.points[1], points)
        sorted_pairs = sort(collect(zip(distances, points)); by=first)
        # Remove duplicate points
        points_reduced::Vector{Point_2D{T}} = [sorted_pairs[1][2]]
        npoints::Int64 = length(sorted_pairs)
        for i = 2:npoints
            if minimum_segment_length < distance(last(points_reduced), sorted_pairs[i][2])
                push!(points_reduced, sorted_pairs[i][2])
            end
        end
        return points_reduced::Vector{Point_2D{T}}
    else
        return points
    end
end

function submesh(mesh::UnstructuredMesh_2D{T, I},
                 face_ids::Set{I};
                 name::String = "DefaultMeshName") where {T<:AbstractFloat, I <: Unsigned}
    # Setup faces and get all vertex ids
    faces = Vector{Vector{I}}(undef, length(face_ids))
    vertex_ids = Set{I}()
    for (i, face_id) in enumerate(face_ids)
        face = collect(mesh.faces[face_id])
        faces[i] = face
        union!(vertex_ids, Set{I}(face[2:length(face)]))
    end
    # Need to remap vertex ids in faces to new ids
    vertex_ids_sorted = sort(collect(vertex_ids))
    vertex_map = Dict{I, I}()
    for (i,v) in enumerate(vertex_ids_sorted)
        vertex_map[v] = i
    end
    points = Vector{Point_2D{T}}(undef, length(vertex_ids_sorted))
    for (i, v) in enumerate(vertex_ids_sorted)
        points[i] = mesh.points[v]
    end
    # remap vertex ids in faces
    for face in faces
        for (i, v) in enumerate(face[2:length(face)])
            face[i + 1] = vertex_map[v]
        end
    end
    # At this point we have points, faces, & name.
    # Just need to get the face sets
    face_sets = Dict{String, Set{I}}()
    for face_set_name in keys(mesh.face_sets)
        set_intersection = intersect(mesh.face_sets[face_set_name], face_ids)
        if length(set_intersection) !== 0
            face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{I, I}()
    for (i,f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(face_sets)
        new_set = Set{I}()
        for fid in face_sets[face_set_name]
            union!(new_set, face_map[fid])
        end
        face_sets[face_set_name] = new_set
    end
    return UnstructuredMesh_2D{T, I}(name = name,
                                     points = points,
                                     faces = [Tuple(f) for f in faces],
                                     face_sets = face_sets
                                    )
end

function submesh(mesh::UnstructuredMesh_2D{T, I},
                 set_name::String) where {T <: AbstractFloat, I <: Unsigned}
    @debug "Creating submesh for '$set_name'"
    face_ids = mesh.face_sets[set_name]
    return submesh(mesh, face_ids, name = set_name)
end
