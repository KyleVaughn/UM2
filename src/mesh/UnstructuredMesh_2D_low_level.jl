# Return a vector of the faces adjacent to the face of ID face
# @code_warntype checked 2021/11/22
function adjacent_faces(face::U,
                        mesh::UnstructuredMesh_2D{F, U}
                        ) where {U <: Unsigned, F <: AbstractFloat}
    return adjacent_faces(face, mesh.face_edge_connectivity, mesh.edge_face_connectivity)
end

# Return a vector of the faces adjacent to the face of ID face
# @code_warntype checked 2021/11/22
function adjacent_faces(face::U,
                        face_edge_connectivity::Vector{<:SVector{L, U} where {L}},
                        edge_face_connectivity::Vector{SVector{2, U}}
                       ) where {U <: Unsigned}
    edges = face_edge_connectivity[face]
    the_adjacent_faces = U[]
    for edge in edges
        faces = edge_face_connectivity[edge]
        for face_id in faces
            if face_id != face && face_id != 0
                push!(the_adjacent_faces, face_id)
            end
        end
    end
    return the_adjacent_faces
end

# Area of face
# @code_warntype checked 2021/11/22
function area(face::SVector{N, U}, points::Vector{Point_2D{F}}) where {N, F <: AbstractFloat, U <: Unsigned}
    return area(materialize_face(face, points))
end

# Return the area of a face set
# @code_warntype checked 2021/11/22
function area(materialized_faces::Vector{<:Face_2D{F}},
        face_set::Set{U}) where {F <: AbstractFloat, U <: Unsigned}
    return mapreduce(x->area(materialized_faces[x]), +, face_set)
end

# Return the area of a face set
# @code_warntype checked 2021/11/22
function area(faces::Vector{<:SVector{L, U} where {L}},
              points::Vector{Point_2D{F}},
              face_set::Set{U}) where {F <: AbstractFloat, U <: Unsigned}
    unsupported = count(x->x[1] ∉  UnstructuredMesh_2D_cell_types, faces)
    if 0 < unsupported
        @error "Mesh contains an unsupported face type"
    end
    return mapreduce(x->area(faces[x], points), +, face_set)
end

# Return a vector containing vectors of the edges in each side of the mesh's bounding shape, e.g.
# For a rectangular bounding shape the sides are North, East, South, West. Then the output would
# be [ [e1, e2, e3, ...], [e17, e18, e18, ...], ..., [e100, e101, ...]]
# @code_warntype checked 2021/11/23
function boundary_edges(mesh::UnstructuredMesh_2D{F, U};
                       bounding_shape::String = "Rectangle") where {F<:AbstractFloat, U <: Unsigned}
    # edges which have face 0 in their edge_face connectivity are boundary edges
    the_boundary_edges = findall(x->x[1] == 0, mesh.edge_face_connectivity)
    if bounding_shape == "Rectangle"
        # Sort edges into NESW
        bb = bounding_box(mesh, rectangular_boundary=true)
        y_north = bb.points[3].x[2]
        x_east  = bb.points[3].x[1]
        y_south = bb.points[1].x[2]
        x_west  = bb.points[1].x[1]
        p_NW = bb.points[4]
        p_NE = bb.points[3]
        p_SE = bb.points[2]
        p_SW = bb.points[1]
        edges_north = U[]
        edges_east = U[]
        edges_south = U[]
        edges_west = U[]
        # Unsert edges so that indices move from NW -> NE -> SE -> SW -> NW
        for i = 1:length(the_boundary_edges)
            edge = U(the_boundary_edges[i])
            epoints = edge_points(mesh.edges[edge], mesh.points)
            if all(x->abs(x[2] - y_north) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_north, mesh.edges, p_NW, mesh.points)
            elseif all(x->abs(x[1] - x_east) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_east, mesh.edges, p_NE, mesh.points)
            elseif all(x->abs(x[2] - y_south) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_south, mesh.edges, p_SE, mesh.points)
            elseif all(x->abs(x[1] - x_west) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_west, mesh.edges, p_SW, mesh.points)
            else
                @error "Edge $iedge could not be classified as NSEW"
            end
        end
        return [ edges_north, edges_east, edges_south, edges_west ]
    else
        return [ convert(Vector{U}, the_boundary_edges) ]
    end
end

# SVector of MVectors of point IDs representing the 3 edges of a triangle
# @code_warntype checked 2021/11/22
function edges(face::SVector{4, U}) where {U <: Unsigned}
    edges = SVector( MVector{2, U}(face[2], face[3]),
                     MVector{2, U}(face[3], face[4]),
                     MVector{2, U}(face[4], face[2]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadrilateral
# @code_warntype checked 2021/11/22
function edges(face::SVector{5, U}) where {U <: Unsigned}
    edges = SVector( MVector{2, U}(face[2], face[3]),
                     MVector{2, U}(face[3], face[4]),
                     MVector{2, U}(face[4], face[5]),
                     MVector{2, U}(face[5], face[2]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 3 edges of a quadratic triangle
# @code_warntype checked 2021/11/22
function edges(face::SVector{7, U}) where {U <: Unsigned}
    edges = SVector( MVector{3, U}(face[2], face[3], face[5]),
                     MVector{3, U}(face[3], face[4], face[6]),
                     MVector{3, U}(face[4], face[2], face[7]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadratic quadrilateral
# @code_warntype checked 2021/11/22
function edges(face::SVector{9, U}) where {U <: Unsigned}
    edges = SVector( MVector{3, U}(face[2], face[3], face[6]),
                     MVector{3, U}(face[3], face[4], face[7]),
                     MVector{3, U}(face[4], face[5], face[8]),
                     MVector{3, U}(face[5], face[2], face[9]) )
    # Order th linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# The unique edges from a vector of triangles or quadrilaterals represented by point IDs
# @code_warntype checked 2021/11/22
function edges(faces::Vector{<:Union{SVector{4, U}, SVector{5, U}}}) where {U <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(unique(edges_unfiltered))
    return [ SVector(e.data) for e in edges_filtered ]
end

# The unique edges from a vector of quadratic triangles or quadratic quadrilaterals
# represented by point IDs
# @code_warntype checked 2021/11/22
function edges(faces::Vector{<:Union{SVector{7, U}, SVector{9, U}}}) where {U <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered::Vector{MVector{2, U}} = sort(unique(edges_unfiltered))
    return [ SVector(e.data) for e in edges_filtered ]
end

# The unique edges from a vector of faces represented by point IDs
# @code_warntype checked 2021/11/22: Not type stable, but can't be if this is called.
function edges(faces::Vector{<:SVector{N, U} where N}) where {U <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(unique(edges_unfiltered))
    return [ SVector(e.data) for e in edges_filtered ]
end


# A vector of length 2 SVectors, denoting the face ID each edge is connected to. If the edge
# is a boundary edge, face ID 0 is returned
# @code_warntype checked 2021/11/23
function edge_face_connectivity(the_edges::Vector{<:SVector{L, U} where {L}},
                                the_faces::Vector{<:SVector{L, U} where {L}},
                                the_face_edge_connectivity::Vector{<:SVector{L, U} where {L}}
                               ) where {U <: Unsigned}
    # Each edge should only border 2 faces if it is an interior edge, and 1 face if it is
    # a boundary edge.
    # Loop through each face in the face_edge_connectivity vector and mark each edge with
    # the faces that it borders.
    if length(the_edges) === 0
        @error "Does not have edges!"
    end
    if length(the_face_edge_connectivity) === 0
        @error "Does not have face/edge connectivity!"
    end
    edge_face = [MVector{2, U}(zeros(U, 2)) for i in eachindex(the_edges)]
    for (iface, edges) in enumerate(the_face_edge_connectivity)
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
    return [SVector(sort(two_faces).data) for two_faces in edge_face]
end

# Return an SVector of the points in the edge
# @code_warntype checked 2021/11/22
function edge_points(edge::SVector{2, U},
                     points::Vector{Point_2D{F}}
                    ) where {F <: AbstractFloat, U <: Unsigned}
    return SVector(points[edge[1]], points[edge[2]])
end

# Return an SVector of the points in the edge
# @code_warntype checked 2021/11/22
function edge_points(edge::SVector{3, U},
                     points::Vector{Point_2D{F}}
                    ) where {F <: AbstractFloat, U <: Unsigned}
    return SVector(points[edge[1]],
                   points[edge[2]],
                   points[edge[3]]
                  )
end

# A vector of SVectors, denoting the edge ID each face is connected to.
# @code_warntype checked 2021/11/23
function face_edge_connectivity(the_faces::Vector{<:SVector{L, U} where {L}},
                                the_edges::Vector{<:SVector{L, U} where {L}}
                               ) where {U <: Unsigned}
    # A vector of MVectors of zeros for each face
    # Each MVector is the length of the number of edges
    face_edge = [MVector{Int64(num_edges(face)), U}(zeros(U, num_edges(face)))
                    for face in the_faces]::Vector{<:MVector{L, U} where {L}}
    if length(the_edges) === 0
        @error "Does not have edges!"
    end
    # for each face in the mesh, generate the edges.
    # Search for the index of the edge in the mesh.edges vector
    # Insert the index of the edge into the face_edge connectivity vector
    for i in eachindex(the_faces)
        for (j, edge) in enumerate(edges(the_faces[i]))
            face_edge[i][j] = searchsortedfirst(the_edges, SVector(edge.data))
        end
    end
    return [SVector(sort(conn).data) for conn in face_edge]::Vector{<:SVector{L, U} where {L}}
end

# Return an SVector of the points in the face
# @code_warntype checked 2021/11/22
function face_points(face::SVector{N, U}, points::Vector{Point_2D{F}}
    ) where {N, F <: AbstractFloat, U <: Unsigned}
    return SVector{N-1, Point_2D{F}}(points[face[2:N]])
end

# Find the faces which share the vertex of ID v.
# @code_warntype checked 2021/11/23
function faces_sharing_vertex(v::I, faces::Vector{<:SVector{L, U} where L}) where {I <: Integer,
                                                                                   U <: Unsigned}
    shared_faces = U[]
    for i = 1:length(faces)
        N = length(faces[i])
        if v ∈  faces[i][2:N]
            push!(shared_faces, U(i))
        end
    end
    return shared_faces
end

# Return the faces which share the vertex of ID v.
# @code_warntype checked 2021/11/23
function faces_sharing_vertex(v::I, mesh::UnstructuredMesh_2D{F, U}) where {I <: Integer,
                                                                            F <: AbstractFloat,
                                                                            U <: Unsigned}
    return faces_sharing_vertex(v, mesh.faces)
end

# Find the face containing the point p, with explicitly represented faces
# @code_warntype checked 2021/11/23
function find_face_explicit(p::Point_2D{F},
                            faces::Vector{<:Face_2D{F}}
                           ) where {F <: AbstractFloat}
    for i = 1:length(faces)
        if p ∈  faces[i]
            return i
        end
    end
    @error "Could not find face for point $p"
    return 0
end

# Return the face containing the point p, with implicitly represented faces
# @code_warntype checked 2021/11/23
function find_face_implicit(p::Point_2D{F},
                            faces::Vector{<:SVector{N, U} where N},
                            points::Vector{Point_2D{F}}
                            ) where {F <: AbstractFloat, U <: Unsigned}
    for i = 1:length(faces)
        bool = p ∈  materialize_face(faces[i], points)
        if bool
            return i
        end
    end
    @error "Could not find face for point $p"
    return 0
end

# Insert the boundary edge into the correct place in the vector of edge indices, based on
# the distance from some reference point
# @code_warntype checked 2021/11/23
function insert_boundary_edge!(edge_index::U, edge_indices::Vector{U},
                               edges::Vector{<:SVector{L, U} where {L}},
                               p_ref::Point_2D{F}, points::Vector{Point_2D{F}}
                              ) where {F <: AbstractFloat, U <: Unsigned}
    # Compute the minimum distance from the edge to be inserted to the reference point
    epoints = edge_points(edges[edge_index], points)
    insertion_distance = minimum([ distance(p_ref, p_edge) for p_edge in epoints ])
    # Loop through the edge indices until an edge with greater distance from the reference point
    # is found, then insert
    nindices = length(edge_indices)
    for i = 1:nindices
        edge = edge_indices[i]
        epoints = edge_points(edges[edge], points)
        edge_distance = minimum([ distance(p_ref, p_edge) for p_edge in epoints ])
        if insertion_distance < edge_distance
            insert!(edge_indices, i, edge_index)
            return nothing
        end
    end
    insert!(edge_indices, nindices+1, edge_index)
    return nothing
end

# Intersect a line with the edges of a mesh
# @code_warntype checked 2021/11/23
function intersect_edges(l::LineSegment_2D{F},
                         mesh::UnstructuredMesh_2D{F, U}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # Implicit intersection has been faster in every test, so this is the default
    if length(mesh.materialized_edges) !== 0
        intersection_points = intersect_edges_explicit(l, mesh.materialized_edges)
        return sort_intersection_points(l, intersection_points)
    else
        intersection_points = intersect_edges_implicit(l, mesh.edges, mesh.points)
        return sort_intersection_points(l, intersection_points)
    end
end

# Intersect a line with materialized edges
# @code_warntype checked 2021/11/23
function intersect_edges_explicit(l::LineSegment_2D{F},
                                  edges::Vector{LineSegment_2D{F}}) where {F <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for edge in edges
        npoints, point = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return intersection_points
end

# Intersect a line with implicitly defined edges
# @code_warntype checked 2021/11/23
function intersect_edges_explicit(l::LineSegment_2D{F},
                                  edges::Vector{QuadraticSegment_2D{F}}) where {F <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for edge in edges
        npoints, points = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    return intersection_points
end

# Intersect a line with an implicitly defined linear edge
# @code_warntype checked 2021/11/23
function intersect_edge_implicit(l::LineSegment_2D{F},
                                 edge::SVector{2, U},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_edge(edge, points)
end

# Intersect a line with an implicitly defined quadratic edge
# @code_warntype checked 2021/11/23
function intersect_edge_implicit(l::LineSegment_2D{F},
                                 edge::SVector{3, U},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_edge(edge, points)
end

# Intersect a line with a vector of implicitly defined linear edges
# @code_warntype checked 2021/11/23
function intersect_edges_implicit(l::LineSegment_2D{F},
                                  edges::Vector{<:Union{SVector{2, U}, SVector{3, U}}},
                                  points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, point = intersect_edge_implicit(l, edge, points)
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return intersection_points
end

# Intersect a line with an implicitly defined Triangle_2D
# @code_warntype checked 2021/11/23
function intersect_face_implicit(l::LineSegment_2D{F},
                                 face::SVector{4, U},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_face(face, points)
end

# Intersect a line with an implicitly defined Quadrilateral_2D
# @code_warntype checked 2021/11/23
function intersect_face_implicit(l::LineSegment_2D{F},
                                 face::SVector{5, U},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_face(face, points)
end

# Intersect a line with an implicitly defined Triangle6_2D
# @code_warntype checked 2021/11/23
function intersect_face_implicit(l::LineSegment_2D{F},
                                 face::SVector{7, U},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_face(face, points)
end

# Intersect a line with an implicitly defined Quadrilateral8_2D
# @code_warntype checked 2021/11/23
function intersect_face_implicit(l::LineSegment_2D{F},
                                 face::SVector{9, U},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_face(face, points)
end

# Intersect a line with the faces of a mesh
# @code_warntype checked 2021/11/23
function intersect_faces(l::LineSegment_2D{F},
                         mesh::UnstructuredMesh_2D{F, U}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # Explicit faces have been faster in every test, so this is the default
    if length(mesh.materialized_faces) !== 0
        intersection_points = intersect_faces_explicit(l, mesh.materialized_faces)
        return sort_intersection_points(l, intersection_points)
    else
        # Check if any of the face types are unsupported
        unsupported = count(x->x[1] ∉  UnstructuredMesh_2D_cell_types, mesh.faces)
        if 0 < unsupported
            @error "Mesh contains an unsupported face type"
        end
        intersection_points = intersect_faces_implicit(l, mesh.faces, mesh.points)
        return sort_intersection_points(l, intersection_points)
    end
end

# Intersect a line with explicitly defined linear faces
# @code_warntype checked 2021/11/23
function intersect_faces_explicit(l::LineSegment_2D{F},
                                  faces::Vector{<:Union{Triangle_2D{F}, Quadrilateral_2D{F}}}
                        ) where {F <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    return intersection_points
end

# Intersect a line with explicitly defined quadratic faces
# @code_warntype checked 2021/11/23
function intersect_faces_explicit(l::LineSegment_2D{F},
                                  faces::Vector{<:Union{Triangle6_2D{F}, Quadrilateral8_2D{F}}}
                        ) where {F <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    return intersection_points
end

# Intersect a line with explicitly defined faces
# @code_warntype checked 2021/11/23: going to be unstable
function intersect_faces_explicit(l::LineSegment_2D{F},
                                  faces::Vector{<:Face_2D{F}}
                        ) where {F <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    return intersection_points
end

# Intersect a line with implicitly defined faces
# @code_warntype checked 2021/11/23
function intersect_faces_implicit(l::LineSegment_2D{F},
                                  faces::Vector{<:SVector{N, U} where N},
                                  points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    # Intersect the line with each of the faces
    for face in faces
        npoints, ipoints = intersect_face_implicit(l, face, points)
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, ipoints[1:npoints])
        end
    end
    return intersection_points
end

# If a point is a vertex
# @code_warntype checked 2021/11/23
function is_vertex(p::Point_2D{F}, points::Vector{Point_2D{F}}) where {F <: AbstractFloat}
    for point in points
        if p ≈ point
            return true
        end
    end
    return false
end

# Return a LineSegment_2D from the point IDs in an edge
# @code_warntype checked 2021/11/22
function materialize_edge(edge::SVector{2, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return LineSegment_2D(edge_points(edge, points))
end

# Return a QuadraticSegment_2D from the point IDs in an edge
# @code_warntype checked 2021/11/22
function materialize_edge(edge::SVector{3, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return QuadraticSegment_2D(edge_points(edge, points))
end

# Return a materialized edge for each edge in the mesh
# @code_warntype checked 2021/11/22
function materialize_edges(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat, U <: Unsigned}
    return materialize_edge.(mesh.edges, (mesh.points,))::Vector{<:Edge_2D{F}}
end

# Return a Triangle_2D from the point IDs in a face
# @code_warntype checked 2021/11/22
function materialize_face(face::SVector{4, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Triangle_2D(face_points(face, points))
end

# Return a Quadrilateral_2D from the point IDs in a face
# @code_warntype checked 2021/11/22
function materialize_face(face::SVector{5, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Quadrilateral_2D(face_points(face, points))
end

# Return a Triangle6_2D from the point IDs in a face
# @code_warntype checked 2021/11/22
function materialize_face(face::SVector{7, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Triangle6_2D(face_points(face, points))
end

# Return a Quadrilateral8_2D from the point IDs in a face
# @code_warntype checked 2021/11/22
function materialize_face(face::SVector{9, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Quadrilateral8_2D(face_points(face, points))
end

# Return a materialized face for each face in the mesh
# @code_warntype checked 2021/11/22
function materialize_faces(mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat, U <: Unsigned}
    return materialize_face.(mesh.faces, (mesh.points,))::Vector{<:Face_2D{F}}
end

# Return the number of edges in a face type
# @code_warntype checked 2021/11/22
function num_edges(face::SVector{L, U}) where {L, U <: Unsigned}
    cell_type = face[1]
    if cell_type == 5 || cell_type == 22
        return U(3)
    elseif cell_type == 9 || cell_type == 23
        return U(4)
    else
        @error "Unsupported face type"
        return U(0)
    end
end

# Return the ID of the edge shared by two adjacent faces
# @code_warntype checked 2021/11/23
function shared_edge(face1::U, face2::U,
                     face_edge_connectivity::Vector{<:SVector{N, U} where N},
                    ) where {F <: AbstractFloat, U <: Unsigned}
    edges1 = face_edge_connectivity[face1]
    edges2 = face_edge_connectivity[face1]
    for edge1 in edges1
        for edge2 in edges2
            if edge1 == edge2
                return edge1
            end
        end
    end
    return U(0)
end

# Sort points based on their distance from the start of the line
# @code_warntype checked 2021/11/23
function sort_intersection_points(l::LineSegment_2D{F},
                                  points::Vector{Point_2D{F}}) where {F <: AbstractFloat}
    if 0 < length(points)
        # Sort the points based upon their distance to the first point in the line
        distances = distance.(l.points[1], points)
        sorted_pairs = sort(collect(zip(distances, points)); by=first)
        # Remove duplicate points
        points_reduced = [sorted_pairs[1][2]]
        npoints = length(sorted_pairs)
        for i = 2:npoints
            if minimum_segment_length < distance(last(points_reduced), sorted_pairs[i][2])
                push!(points_reduced, sorted_pairs[i][2])
            end
        end
        return points_reduced
    else
        return points
    end
end

# Return a mesh with name name, composed of the faces in the set face_ids
# @code_warntype checked 2021/11/23
function submesh(mesh::UnstructuredMesh_2D{F, U},
                 face_ids::Set{U};
                 name::String = "DefaultMeshName") where {F<:AbstractFloat, U <: Unsigned}
    # Setup faces and get all vertex ids
    faces = Vector{Vector{U}}(undef, length(face_ids))
    vertex_ids = Set{U}()
    for (i, face_id) in enumerate(face_ids)
        face = collect(mesh.faces[face_id])
        faces[i] = face
        union!(vertex_ids, Set(face[2:length(face)]))
    end
    # Need to remap vertex ids in faces to new ids
    vertex_ids_sorted = sort(collect(vertex_ids))
    vertex_map = Dict{U, U}()
    for (i,v) in enumerate(vertex_ids_sorted)
        vertex_map[v] = i
    end
    points = Vector{Point_2D{F}}(undef, length(vertex_ids_sorted))
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
    face_sets = Dict{String, Set{U}}()
    for face_set_name in keys(mesh.face_sets)
        set_intersection = intersect(mesh.face_sets[face_set_name], face_ids)
        if length(set_intersection) !== 0
            face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{U, U}()
    for (i,f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(face_sets)
        new_set = Set{U}()
        for fid in face_sets[face_set_name]
            union!(new_set, face_map[fid])
        end
        face_sets[face_set_name] = new_set
    end
    return UnstructuredMesh_2D{F, U}(name = name,
                                     points = points,
                                     faces = [SVector{length(f), U}(f) for f in faces],
                                     face_sets = face_sets
                                    )
end
