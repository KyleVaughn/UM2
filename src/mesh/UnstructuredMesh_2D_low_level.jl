# Return a vector of the faces adjacent to the face of ID face
# Type-stable if all of the faces/edges are the same
function adjacent_faces(face::U,
                        face_edge_connectivity::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}},
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


# Find the faces which share the vertex of ID v.
# Type-stable
function faces_sharing_vertex(v::I, faces::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}}
    ) where {I <: Integer, U <: Unsigned}
    shared_faces = U[]
    for i = 1:length(faces)
        N = length(faces[i])
        if v ∈  faces[i][2:N]
            push!(shared_faces, U(i))
        end
    end
    return shared_faces
end

# Find the face containing the point p, with explicitly represented faces
# Type-stable
function find_face_explicit(p::Point_2D{F},
                            faces::Vector{<:Face_2D{F}}
                           ) where {F <: AbstractFloat}
    for i = 1:length(faces)
        if p ∈  faces[i]
            return i
        end
    end
    return 0
end

# Return the face containing the point p, with implicitly represented faces
# Type-stable
function find_face_implicit(p::Point_2D{F},
                            faces::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}},
                            points::Vector{Point_2D{F}}
                            ) where {F <: AbstractFloat, U <: Unsigned}
    for i = 1:length(faces)
        bool = p ∈  materialize_face(faces[i], points)
        if bool
            return i
        end
    end
    return 0
end

# Intersect a line with materialized edges
# Type-stable
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
    return sort_points(l.points[1], intersection_points)
end

# Intersect a line with implicitly defined edges
# Type-stable
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
    return sort_points(l.points[1], intersection_points)
end

# Intersect a line with an implicitly defined edge
# Type-stable
function intersect_edge_implicit(l::LineSegment_2D{F},
                                 edge::SVector{L, U} where {L},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_edge(edge, points)
end

# Intersect a line with a vector of implicitly defined linear edges
# Type-stable
function intersect_edges_implicit(l::LineSegment_2D{F},
                                  edges::Vector{SVector{2, U}},
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
    return sort_points(l.points[1], intersection_points) 
end

# Intersect a line with a vector of implicitly defined quadratic edges
# Type-stable
function intersect_edges_implicit(l::LineSegment_2D{F},
                                  edges::Vector{SVector{3, U}},
                                  points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, ipoints = intersect_edge_implicit(l, edge, points)
        if 0 < npoints
            append!(intersection_points, ipoints)
        end
    end
    return sort_points(l.points[1], intersection_points) 
end

# Intersect a line with an implicitly defined face 
# Type-stable
function intersect_face_implicit(l::LineSegment_2D{F},
                                 face::SVector{L, U} where {L},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_face(face, points)
end

# Intersect a line with explicitly defined linear faces
# Type-stable if all faces are one of: linear or quadratic
function intersect_faces_explicit(l::LineSegment_2D{F},
                                  faces::Vector{<:Face_2D{F}} ) where {F <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    return sort_points(l.points[1], intersection_points)
end

# Intersect a line with implicitly defined faces
# Type-stable if all faces are one of: linear or quadratic
function intersect_faces_implicit(l::LineSegment_2D{F},
                                  faces::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}},
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
    return sort_points(l.points[1], intersection_points)
end

# If a point is a vertex
# Type-stable
function is_vertex(p::Point_2D{F}, points::Vector{Point_2D{F}}) where {F <: AbstractFloat}
    for point in points
        if p ≈ point
            return true
        end
    end
    return false
end

function remap_points_to_hilbert(points::Vector{Point_2D{F}}) where {F <: AbstractFloat}
    bb = bounding_box(points) 
    npoints = length(points)
    # Generate a Hilbert curve 
    hilbert_points = hilbert_curve(bb, npoints)
    nhilbert_points = length(hilbert_points)
    # For each point, get the index of the hilbert points that is closest
    point_indices = Vector{Int64}(undef, npoints)
    for i = 1:npoints
        min_distance = F(1.0e10)
        for j = 1:nhilbert_points
            pdistance = distance(points[i], hilbert_points[j])
            if pdistance < min_distance
                min_distance = pdistance
                point_indices[i] = j
            end
        end
    end
    return sortperm(point_indices) 
end

# Return the ID of the edge shared by two adjacent faces
# Type-stable if all faces are the same type
function shared_edge(face1::U, face2::U,
                     face_edge_connectivity::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}},
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

# Return a mesh with name name, composed of the faces in the set face_ids
# Not type-stable
function submesh(name::String,
                 points::Vector{Point_2D{F}},
                 faces::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}},
                 face_sets::Dict{String, Set{U}},
                 face_ids::Set{U}) where {F <: AbstractFloat, U <: Unsigned}
    # Setup faces and get all vertex ids
    submesh_faces = Vector{Vector{U}}(undef, length(face_ids))
    vertex_ids = Set{U}()
    for (i, face_id) in enumerate(face_ids)
        face_vec = collect(faces[face_id].data)
        submesh_faces[i] = face_vec
        union!(vertex_ids, Set(face_vec[2:length(face_vec)]))
    end
    # Need to remap vertex ids in faces to new ids
    vertex_ids_sorted = sort(collect(vertex_ids))
    vertex_map = Dict{U, U}()
    for (i,v) in enumerate(vertex_ids_sorted)
        vertex_map[v] = i
    end
    submesh_points = Vector{Point_2D{F}}(undef, length(vertex_ids_sorted))
    for (i, v) in enumerate(vertex_ids_sorted)
        submesh_points[i] = points[v]
    end
    # remap vertex ids in faces
    for face in submesh_faces
        for (i, v) in enumerate(face[2:length(face)])
            face[i + 1] = vertex_map[v]
        end
    end
    # At this point we have points, faces, & name.
    # Just need to get the face sets
    submesh_face_sets = Dict{String, Set{U}}()
    for face_set_name in keys(face_sets)
        set_intersection = face_sets[face_set_name] ∩ face_ids
        if length(set_intersection) !== 0
            submesh_face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{U, U}()
    for (i,f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(submesh_face_sets)
        new_set = Set{U}()
        for fid in submesh_face_sets[face_set_name]
            union!(new_set, face_map[fid])
        end
        submesh_face_sets[face_set_name] = new_set
    end
    return UnstructuredMesh_2D{F, U}(name = name,
                                     points = submesh_points,
                                     faces = [SVector{length(f), U}(f) for f in submesh_faces],
                                     face_sets = submesh_face_sets
                                    )
end
