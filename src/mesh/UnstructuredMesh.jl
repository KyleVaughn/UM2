abstract type UnstructuredMesh{Dim, Ord, T, U} end
const UnstructuredMesh2D = UnstructuredMesh{2}
const LinearUnstructuredMesh = UnstructuredMesh{Dim, 1} where {Dim}
const LinearUnstructuredMesh2D = UnstructuredMesh{2, 1}
const QuadraticUnstructuredMesh = UnstructuredMesh{Dim, 2} where {Dim}
const QuadraticUnstructuredMesh2D = UnstructuredMesh{2, 2}
Base.broadcastable(mesh::UnstructuredMesh) = Ref(mesh)

# # Area of face
# function area(face::SVector, points::Vector{<:Point})
#     return area(materialize_face(face, points))
# end
# 
# # Return the area of a face set
# function area(mesh::UnstructuredMesh, face_set::Set)
#     if 0 < length(mesh.materialized_faces)
#         return mapreduce(x->area(mesh.materialized_faces[x]), +, face_set)
#     else
#         return mapreduce(x->area(mesh.faces[x], mesh.points), +, face_set)
#     end
# end
# 
# # Return the area of a face set by name
# function area(mesh::UnstructuredMesh, set_name::String)
#     return area(mesh, mesh.face_sets[set_name])
# end
# 
# # Axis-aligned bounding box
# function boundingbox(mesh::LinearUnstructuredMesh2D)
#     # If the mesh does not have any quadratic faces, the bounding_box may be determined
#     # entirely from the points.
#     nsides = length(mesh.boundary_edges)
#     if nsides !== 0
#         boundary_edge_IDs = reduce(vcat, mesh.boundary_edges)
#         point_IDs = reduce(vcat, mesh.edges[boundary_edge_IDs])
#         return boundingbox(mesh.points[point_IDs])
#     else
#         return boundingbox(mesh.points)
#     end
# end

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
# # Intersect a line with a vector of implicitly defined linear edges
# function intersect_edges_implicit(l::LineSegment2D{T}, edges::Vector{<:SVector{2}},
#                                   points::Vector{Point2D{T}}) where {T}
#     intersection_points = Point2D{T}[]
#     for edge in edges
#         npoints, point = intersect_edge_implicit(l, edge, points)
#         if 0 < npoints
#             push!(intersection_points, point)
#         end
#     end
#     sort_intersection_points!(l.𝘅₁, intersection_points)
#     return intersection_points
# end
# 
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
# # Return a LineSegment from the point IDs in an edge
# function materialize_edge(edge::SVector{2}, points::Vector{<:Point})
#     return LineSegment(edgepoints(edge, points))
# end
# 
# # Return a QuadraticSegment from the point IDs in an edge
# function materialize_edge(edge::SVector{3}, points::Vector{<:Point})
#     return QuadraticSegment(edgepoints(edge, points))
# end
# 
# # Return a LineSegment or QuadraticSegment
# function materialize_edge(edge_id, mesh::UnstructuredMesh)
#     return materialize_edge(mesh.edges[edge_id], mesh.points)
# end
# 
# # Return a materialized edge for each edge in the mesh
# function materialize_edges(mesh::UnstructuredMesh)
#     return materialize_edge.(mesh.edges, Ref(mesh.points))
# end
# 
# # Return a materialized face from the point IDs in a face
# function materialize_face(face::SVector{N}, points::Vector{<:Point}) where {N}
#     return Polygon{N}(facepoints(face, points))
# end
# 
# # Return an SVector of the points in the edge
# function materialize_face(face_id, mesh::UnstructuredMesh)
#     return materialize_face(mesh.faces[face_id], mesh.points)
# end
# 
# # Return a materialized face for each face in the mesh
# function materialize_faces(mesh::UnstructuredMesh)
#     return materialize_face.(mesh.faces, Ref(mesh.points))
# end
# 
# # Return the number of edges in a face
# function num_edges(face::SVector{L,U}) where {L,U}
#     if L === 3 || L === 6
#         return U(3)
#     elseif L === 4 || L === 8
#         return U(4)
#     else
#         return U(0)
#     end
# end
# 
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
