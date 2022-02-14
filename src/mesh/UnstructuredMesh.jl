abstract type UnstructuredMesh{Dim, Ord, T, U} end
const UnstructuredMesh2D = UnstructuredMesh{2}
const UnstructuredMesh3D = UnstructuredMesh{3}
const LinearUnstructuredMesh = UnstructuredMesh{Dim, 1} where {Dim}
const LinearUnstructuredMesh2D = UnstructuredMesh{2, 1}
const LinearUnstructuredMesh3D = UnstructuredMesh{3, 1}
const QuadraticUnstructuredMesh = UnstructuredMesh{Dim, 2} where {Dim}
const QuadraticUnstructuredMesh2D = UnstructuredMesh{2, 2}
const QuadraticUnstructuredMesh3D = UnstructuredMesh{3, 2}
Base.broadcastable(mesh::UnstructuredMesh) = Ref(mesh)


# Return the area of face id
function area(id, mesh::UnstructuredMesh)
    return area(materialize_face(id, mesh))
end

# Return the area of the entire mesh
function area(mesh::UnstructuredMesh)
    return mapreduce(x->area(x, mesh), +, 1:length(mesh.faces))
end

# Return the area of a face set
function area(face_set::BitSet, mesh::UnstructuredMesh)
    return mapreduce(x->area(x, mesh), +, face_set)
end

# Return the area of a face set by name
function area(set_name::String, mesh::UnstructuredMesh)
    return area(mesh.face_sets[set_name], mesh)
end

# Axis-aligned bounding box
function boundingbox(mesh::LinearUnstructuredMesh)
    # The bounding box may be determined entirely from the points.
    return boundingbox(mesh.points)
end

# Axis-aligned bounding box
function boundingbox(mesh::QuadraticUnstructuredMesh)
    return union(boundingbox.(materialize_edges(mesh)))
end

@generated function edgepoints(edge::SVector{N}, points::Vector{<:Point}) where {N}
    points_string = "SVector("
    for i ∈ 1:N
        points_string *= "points[edge[$i]], "
    end
    points_string *= ")"
    return Meta.parse(points_string)
end

# Return an SVector of the points in the edge
function edgepoints(edge_id, mesh::UnstructuredMesh)
    return edgepoints(mesh.edges[edge_id], mesh.points)
end

# # A vector of length 2 SVectors, denoting the face ID each edge is connected to. If the edge
# # is a boundary edge, face ID 0 is returned
# function edge_face_connectivity(mesh::UnstructuredMesh{Dim,Ord,T,U}) where {Dim,Ord,T,U}
#     # Each edge should only border 2 faces if it is an interior edge, and 1 face if it is
#     # a boundary edge.
#     # Loop through each face in the face_edge_connectivity vector and mark each edge with
#     # the faces that it borders.
#     if length(mesh.edges) === 0
#         @error "Does not have edges!"
#     end
#     if length(mesh.face_edge_connectivity) === 0
#         @error "Does not have face/edge connectivity!"
#     end
#     edge_face = [MVector{2, U}(0, 0) for _ in eachindex(mesh.edges)]
#     for (iface, edges) in enumerate(mesh.face_edge_connectivity)
#         for iedge in edges
#             # Add the face id in the first non-zero position of the edge_face conn. vec.
#             if edge_face[iedge][1] === U(0)
#                 edge_face[iedge][1] = iface
#             elseif edge_face[iedge][2] === U(0)
#                 edge_face[iedge][2] = iface
#             else
#                 @error "Edge $iedge seems to have 3 faces associated with it!"
#             end
#         end
#     end
#     return [SVector(sort!(two_faces).data) for two_faces in edge_face]
# end

@generated function facepoints(face::SVector{N}, points::Vector{<:Point}) where {N}
    points_string = "SVector("
    for i ∈ 1:N
        points_string *= "points[face[$i]], "
    end
    points_string *= ")"
    return Meta.parse(points_string)
end

# Return an SVector of the points in the face
function facepoints(face_id, mesh::UnstructuredMesh)
    return facepoints(mesh.faces[face_id], mesh.points)
end
# 
# # Return the face containing point p.
# function findface(p::Point2D, mesh::UnstructuredMesh2D{Dim,T,U}) where {Dim,T,U}
#     if 0 < length(mesh.materialized_faces)
#         return U(findface_explicit(p, mesh.materialized_faces))
#     else
#         return U(findface_implicit(p, mesh.faces, mesh.points))
#     end
# end
# 
# # Find the face containing the point p, with explicitly represented faces
# function findface_explicit(p::Point2D, faces::Vector{<:Face2D})
#     for i ∈ 1:length(faces)
#         if @inbounds p ∈ faces[i]
#             return i
#         end
#     end
#     return 0
# end
# 
# # Return the face containing the point p, with implicitly represented faces
# function findface_implicit(p::Point2D, faces::Vector{<:SArray}, points::Vector{<:Point2D})
#     for i ∈ 1:length(faces)
#         bool = @inbounds p ∈ materialize_face(faces[i], points)
#         if bool
#             return i
#         end
#     end
#     return 0
# end
# 
# # Return the intersection algorithm that will be used for l ∩ mesh
# function get_intersection_algorithm(mesh::UnstructuredMesh2D)
#     if length(mesh.materialized_edges) !== 0
#         return "Edges - Explicit"
#     elseif length(mesh.edges) !== 0
#         return "Edges - Implicit"
#     elseif length(mesh.materialized_faces) !== 0
#         return "Faces - Explicit"
#     else
#         return "Faces - Implicit"
#     end
# end
# 
# # Intersect a line with the mesh. Returns a vector of intersection points, sorted based
# # upon distance from the line's start point
# function intersect(l::LineSegment2D, mesh::UnstructuredMesh2D)
#     # Edges are faster, so they are the default
#     if length(mesh.edges) !== 0
#         if 0 < length(mesh.materialized_edges)
#             return intersect_edges_explicit(l, mesh.materialized_edges)
#         else
#             return intersect_edges_implicit(l, mesh.edges, mesh.points)
#         end
#     else
#         if 0 < length(mesh.materialized_faces)
#             return intersect_faces_explicit(l, mesh.materialized_faces)
#         else
#             return intersect_faces_implicit(l, mesh.faces, mesh.points)
#         end
#     end
# end
# 
# # Intersect a line with an implicitly defined edge
# function intersect_edge_implicit(l::LineSegment2D, edge::SVector, points::Vector{<:Point2D})
#     return l ∩ materialize_edge(edge, points)
# end
# 
# # Intersect a line with linear edges 
# function intersect_edges_explicit(l::LineSegment2D{T}, edges::Vector{LineSegment2D{T}}) where {T}
#     intersection_points = Point2D{T}[]
#     for edge in edges
#         hit, point = l ∩ edge
#         if hit
#             push!(intersection_points, point)
#         end
#     end
#     sort_intersection_points!(l.𝘅₁, intersection_points)
#     return intersection_points
# end
# 
# Intersect a line with a vector of implicitly defined linear edges
function intersect_edges(l::LineSegment{Dim, T}, mesh::LinearUnstructuredMesh) where {Dim, T} 
    intersection_points = Point{Dim, T}[]
    for i ∈ 1:length(mesh.edges)
        hit, point = l ∩ materialize_edge(i, mesh)
        if hit
            push!(intersection_points, point)
        end
    end
    sort_intersection_points!(l, intersection_points)
    return intersection_points
end

# Intersect a line with a vector of implicitly defined quadratic edges
function intersect_edges(l::LineSegment{Dim, T}, mesh::QuadraticUnstructuredMesh) where {Dim, T} 
    intersection_points = Point{Dim, T}[]
    for i ∈ 1:length(mesh.edges)
        hits, points = l ∩ materialize_edge(i, mesh)
        if 0 < hits
            append!(intersection_points, view(points, 1:hits))
        end
    end
    sort_intersection_points!(l, intersection_points)
    return intersection_points
end

# Intersect a vector of lines with a vector of quadratic edges
function intersect_edges(lines::Vector{LineSegment{Dim, T}}, 
                         mesh::QuadraticUnstructuredMesh) where {Dim, T} 
    nlines = length(lines)
    intersection_points = [Point{Dim, T}[] for _ = 1:nlines]
    Threads.@threads for j ∈ 1:length(mesh.edges)
        @inbounds for i = 1:nlines
            npoints, points = lines[i] ∩ materialize_edge(j, mesh)
            if 0 < npoints
                append!(intersection_points[i], view(points, 1:npoints))
            end
        end
    end
    Threads.@threads for i = 1:nlines
        sort_intersection_points!(lines[i], intersection_points[i])
    end
    return intersection_points
end

# # Intersect a line with an implicitly defined face
# function intersect_face_implicit(l::LineSegment2D, face::SVector, points::Vector{<:Point2D})
#     return l ∩ materialize_face(face, points)
# end
# 
# # Intersect a line with explicitly defined linear faces
# function intersect_faces_explicit(l::LineSegment2D{T}, faces::Vector{<:Face2D} ) where {T}
#     # An array to hold all of the intersection points
#     intersection_points = Point2D{T}[]
#     for face in faces
#         npoints, points = l ∩ face
#         # If the intersections yields 1 or more points, push those points to the array of points
#         if 0 < npoints
#             append!(intersection_points, @inbounds points[1:npoints])
#         end
#     end
#     sort_intersection_points!(l.𝘅₁, intersection_points)
#     return intersection_points
# end
# 
# # Intersect a line with implicitly defined faces
# function intersect_faces_implicit(l::LineSegment2D{T}, faces::Vector{<:SArray}, 
#                                   points::Vector{<:Point2D}) where {T}
#     # An array to hold all of the intersection points
#     intersection_points = Point2D{T}[]
#     # Intersect the line with each of the faces
#     for face in faces
#         npoints, ipoints = intersect_face_implicit(l, face, points)
#         # If the intersections yields 1 or more points, push those points to the array of points
#         if 0 < npoints
#             append!(intersection_points, @inbounds ipoints[1:npoints])
#         end
#     end
#     sort_intersection_points!(l.𝘅₁, intersection_points)
#     return intersection_points
# end
# 
# Return a LineSegment from the point IDs in an edge
function materialize_edge(edge::SVector{2}, points::Vector{<:Point})
    return LineSegment(edgepoints(edge, points))
end

# Return a QuadraticSegment from the point IDs in an edge
function materialize_edge(edge::SVector{3}, points::Vector{<:Point})
    return QuadraticSegment(edgepoints(edge, points))
end

# Return a materialized edge
function materialize_edge(edge_id, mesh::UnstructuredMesh)
    return materialize_edge(mesh.edges[edge_id], mesh.points)
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::UnstructuredMesh)
    return materialize_edge.(1:length(mesh.edges), mesh)
end

# Return a materialized facee for each facee in the mesh
function materialize_faces(mesh::UnstructuredMesh)
    return materialize_face.(1:length(mesh.faces), mesh)
end

function Base.show(io::IO, mesh::UnstructuredMesh)
    mesh_type = typeof(mesh)
    println(io, mesh_type)
    println(io, "  ├─ Name      : $(mesh.name)")
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        size_KB = size_MB*1000
        println(io, "  ├─ Size (KB) : $size_KB")
    else
        println(io, "  ├─ Size (MB) : $size_MB")
    end
    println(io, "  ├─ Points    : $(length(mesh.points))")
    nedges = length(mesh.edges)
    println(io, "  ├─ Edges     : $nedges")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

function submesh(name::String, mesh::UnstructuredMesh2D{Ord, T, U}) where {Ord, T, U}
    # Setup faces and get all vertex ids
    face_ids = mesh.face_sets[name]
    submesh_faces = [MVector(mesh.faces[id].data) for id ∈ face_ids]
    vertex_ids = U[]
    # This can be sped up substantially by keeping vertex_ids sorted and 
    # checking membership using binary search
    for face in submesh_faces
        for id in face
            if id ∉ vertex_ids
                push!(vertex_ids, id)
            end
        end
    end
    # Need to remap vertex ids in faces to new ids
    sort!(vertex_ids)
    vertex_map = Dict{U, U}()
    for (i, v) in enumerate(vertex_ids)
        vertex_map[v] = i
    end
    points = Vector{Point2D{T}}(undef, length(vertex_ids))
    for (i, v) in enumerate(vertex_ids)
        points[i] = mesh.points[v]
    end
    # remap vertex ids in faces
    for face in submesh_faces
        for (i, v) in enumerate(face)
            face[i] = vertex_map[v]
        end
    end
    # At this point we have points, faces, & name.
    # Just need to get the face sets
    face_sets = Dict{String, BitSet}()
    for face_set_name in keys(mesh.face_sets)
        set_intersection = mesh.face_sets[face_set_name] ∩ face_ids
        if length(set_intersection) !== 0
            face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{Int64, Int64}()
    for (i, f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(face_sets)
        face_sets[face_set_name] = BitSet(map(x->face_map[x], 
                                              collect(face_sets[face_set_name])))
    end
    faces = [SVector{length(f), U}(f) for f in submesh_faces]
    if mesh isa LinearUnstructuredMesh 
        edges = _create_linear_edges_from_faces(faces)
    else # QuadraticUnstructuredMesh
        edges = _create_quadratic_edges_from_faces(faces)
    end
    return typeof(mesh)(name = name,
             points = points,
             edges= edges,
             faces = faces, 
             face_sets = face_sets
            )
end

#########################################################################################
function _convert_faces_edges(::Type{U}, 
                              faces::Vector{<:SArray{S, UInt64, 1} where {S<:Tuple}}, 
                              edges::Vector{<:SVector{N, UInt64}}, 
                             ) where {U, N}
    faces_U = [ convert(SVector{length(face), U}, face) for face in faces ]
    edges_U = [ convert(SVector{length(edge), U}, edge) for edge in edges ] 
    return faces_U, edges_U
end

function _create_linear_edges_from_faces(faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
                                        ) where {U<:Unsigned}
    edge_vecs = linear_edges.(faces)
    num_edges = mapreduce(x->length(x), +, edge_vecs)
    edges_unfiltered = Vector{SVector{2, U}}(undef, num_edges)
    iedge = 1
    for edge in edge_vecs
        for i in eachindex(edge)
            if edge[i][1] < edge[i][2]
                edges_unfiltered[iedge] = edge[i]
            else
                edges_unfiltered[iedge] = SVector(edge[i][2], edge[i][1])
            end
            iedge += 1
        end
    end
    return sort!(unique!(edges_unfiltered))
end

function _create_2d_mesh_from_vector_faces(name::String, points::Vector{Point2D{T}},
                                           faces_vecs::Vector{Vector{UInt64}},
                                           face_sets::Dict{String, BitSet}
                                          ) where {T}
    # Determine face types
    face_lengths = Int64[]
    for face in faces_vecs
        l = length(face)
        if l ∉ face_lengths
            push!(face_lengths, l)
        end
    end
    sort!(face_lengths)
    if all(x->x < 6, face_lengths) # Linear mesh
        if face_lengths == [3]
            faces = [ SVector{3, UInt64}(f) for f in faces_vecs]
            edges = _create_linear_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U = _convert_faces_edges(U, faces, edges)
            return TriangleMesh{2, T, U}(name = name,
                                         points = points,
                                         edges = edges_U,
                                         faces = faces_U,
                                         face_sets = face_sets)
        elseif face_lengths == [4]
            faces = [ SVector{4, UInt64}(f) for f in faces_vecs]
            edges = _create_linear_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U = _convert_faces_edges(U, faces, edges)
            return QuadrilateralMesh{2, T, U}(name = name,
                                              points = points,
                                              edges = edges_U,
                                              faces = faces_U,
                                              face_sets = face_sets)
        elseif face_lengths == [3, 4]
            faces = [ SVector{length(f), UInt64}(f) for f in faces_vecs]
            edges = _create_linear_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U = _convert_faces_edges(U, faces, edges)
            return PolygonMesh{2, T, U}(name = name,
                                        points = points,
                                        edges = edges_U,
                                        faces = faces_U,
                                        face_sets = face_sets)
        end
    else # Quadratic Mesh
        if face_lengths == [6]
            faces = [ SVector{6, UInt64}(f) for f in faces_vecs]
            edges = _create_quadratic_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U = _convert_faces_edges(U, faces, edges)
            return QuadraticTriangleMesh{2, T, U}(name = name,
                                                  points = points,
                                                  edges = edges_U,
                                                  faces = faces_U,
                                                  face_sets = face_sets)
        elseif face_lengths == [8]
            faces = [ SVector{8, UInt64}(f) for f in faces_vecs]
            edges = _create_quadratic_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U = _convert_faces_edges(U, faces, edges)
            return QuadraticQuadrilateralMesh{2, T, U}(name = name,
                                                       points = points,
                                                       edges = edges_U,
                                                       faces = faces_U,
                                                       face_sets = face_sets)
        elseif face_lengths == [6, 8]
            faces = [ SVector{length(f), UInt64}(f) for f in faces_vecs]
            edges = _create_quadratic_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U = _convert_faces_edges(U, faces, edges)
            return QuadraticPolygonMesh{2, T, U}(name = name,
                                                 points = points,
                                                 edges = edges_U,
                                                 faces = faces_U,
                                                 face_sets = face_sets)
        end
    end
    @error "Could not determine mesh type"
end

function _create_quadratic_edges_from_faces(faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
                                        ) where {U<:Unsigned}
    edge_vecs = quadratic_edges.(faces)
    num_edges = mapreduce(x->length(x), +, edge_vecs)
    edges_unfiltered = Vector{SVector{3, U}}(undef, num_edges)
    iedge = 1
    for edge in edge_vecs
        for i in eachindex(edge)
            if edge[i][1] < edge[i][2]
                edges_unfiltered[iedge] = edge[i]
            else
                edges_unfiltered[iedge] = SVector(edge[i][2], edge[i][1], edge[i][3])
            end
            iedge += 1
        end
    end
    return sort!(unique!(edges_unfiltered))
end

function _select_mesh_UInt_type(N::Int64)
    if N ≤ typemax(UInt16) 
        U = UInt16
    elseif N ≤ typemax(UInt32) 
        U = UInt32
    elseif N ≤ typemax(UInt64) 
        U = UInt64
    else 
        @error "How cow, that's a big mesh! Number of edges exceeds typemax(UInt64)"
        U = UInt64
    end
    return U
end
