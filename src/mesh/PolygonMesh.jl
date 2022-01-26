struct PolygonMesh{Dim,T,U} <:LinearUnstructuredMesh{Dim,T,U}
    name::String
    points::Vector{Point{Dim,T}}
    edges::Vector{SVector{2,U}}
    materialized_edges::Vector{LineSegment{Dim,T}}
    faces::Vector{<:SArray{S,U,1} where {S<:Tuple}}
    materialized_faces::Vector{<:Polygon{N,Dim,T} where {N}}
    edge_face_connectivity::Vector{SVector{2,U}}
    face_edge_connectivity::Vector{<:SArray{S,U,1} where {S<:Tuple}}
    boundary_edges::Vector{Vector{U}}
    face_sets::Dict{String, Set{U}}
end

function PolygonMesh{Dim,T,U}(;
    name::String = "default_name",
    points::Vector{Point{Dim,T}} = Point{Dim,T}[],
    edges::Vector{SVector{2,U}} = SVector{2,U}[],
    materialized_edges::Vector{LineSegment{Dim,T}} = LineSegment{Dim,T}[],
    faces::Vector{<:SArray{S,U,1} where {S<:Tuple}} = SVector{3,U}[],
    materialized_faces::Vector{<:Polygon{N,Dim,T} where {N}} = Polygon{3,Dim,T}[],
    edge_face_connectivity::Vector{SVector{2,U}} = SVector{2,U}[],
    face_edge_connectivity::Vector{<:SArray{S,U,1} where {S<:Tuple}} = SVector{3,U}[],
    boundary_edges::Vector{Vector{U}} = Vector{U}[],
    face_sets::Dict{String, Set{U}} = Dict{String, Set{U}}()
    ) where {Dim,T,U}
    return PolygonMesh(name, points, edges, materialized_edges, faces, materialized_faces, 
                       edge_face_connectivity, face_edge_connectivity, boundary_edges, face_sets)
end

struct TriangleMesh{Dim,T,U} <:LinearUnstructuredMesh{Dim,T,U}
    name::String
    points::Vector{Point{Dim,T}}
    edges::Vector{SVector{2,U}}
    materialized_edges::Vector{LineSegment{Dim,T}}
    faces::Vector{SVector{3,U}}
    materialized_faces::Vector{Polygon{3,Dim,T}}
    edge_face_connectivity::Vector{SVector{2,U}}
    face_edge_connectivity::Vector{SVector{3,U}}
    boundary_edges::Vector{Vector{U}}
    face_sets::Dict{String, Set{U}}
end

function TriangleMesh{Dim,T,U}(;
    name::String = "default_name",
    points::Vector{Point{Dim,T}} = Point{Dim,T}[],
    edges::Vector{SVector{2,U}} = SVector{2,U}[],
    materialized_edges::Vector{LineSegment{Dim,T}} = LineSegment{Dim,T}[],
    faces::Vector{SVector{3,U}} = SVector{3,U}[],
    materialized_faces::Vector{Polygon{3,Dim,T}} = Polygon{3,Dim,T}[],
    edge_face_connectivity::Vector{SVector{2,U}} = SVector{2,U}[],
    face_edge_connectivity::Vector{SVector{3,U}} = SVector{3,U}[],
    boundary_edges::Vector{Vector{U}} = Vector{U}[],
    face_sets::Dict{String, Set{U}} = Dict{String, Set{U}}()
    ) where {Dim,T,U}
    return TriangleMesh(name, points, edges, materialized_edges, faces, materialized_faces, 
                        edge_face_connectivity, face_edge_connectivity, boundary_edges, face_sets)
end

struct QuadrilateralMesh{Dim,T,U} <:LinearUnstructuredMesh{Dim,T,U}
    name::String
    points::Vector{Point{Dim,T}}
    edges::Vector{SVector{2,U}}
    materialized_edges::Vector{LineSegment{Dim,T}}
    faces::Vector{SVector{4,U}}
    materialized_faces::Vector{Polygon{4,Dim,T}}
    edge_face_connectivity::Vector{SVector{2,U}}
    face_edge_connectivity::Vector{SVector{4,U}}
    boundary_edges::Vector{Vector{U}}
    face_sets::Dict{String, Set{U}}
end

function QuadrilateralMesh{Dim,T,U}(;
    name::String = "default_name",
    points::Vector{Point{Dim,T}} = Point{Dim,T}[],
    edges::Vector{SVector{2,U}} = SVector{2,U}[],
    materialized_edges::Vector{LineSegment{Dim,T}} = LineSegment{Dim,T}[],
    faces::Vector{SVector{4,U}} = SVector{4,U}[],
    materialized_faces::Vector{Polygon{4,Dim,T}} = Polygon{4,Dim,T}[],
    edge_face_connectivity::Vector{SVector{2,U}} = SVector{2,U}[],
    face_edge_connectivity::Vector{SVector{4,U}} = SVector{4,U}[],
    boundary_edges::Vector{Vector{U}} = Vector{U}[],
    face_sets::Dict{String, Set{U}} = Dict{String, Set{U}}()
    ) where {Dim,T,U}
    return QuadrilateralMesh(name, points, edges, materialized_edges, faces, materialized_faces, 
                             edge_face_connectivity, face_edge_connectivity, boundary_edges, 
                             face_sets)
end


# 
# # Return a mesh with boundary edges
# function add_boundary_edges(mesh::M; boundary_shape="Unknown"
#     ) where {M <: UnstructuredMesh_2D}
#     if 0 === length(mesh.edge_face_connectivity)
#         mesh = add_connectivity(mesh)
#     end
#     return M(name = mesh.name,
#              points = mesh.points,
#              edges = mesh.edges,
#              materialized_edges = mesh.materialized_edges,
#              faces = mesh.faces,
#              materialized_faces = mesh.materialized_faces,
#              edge_face_connectivity = mesh.edge_face_connectivity,
#              face_edge_connectivity = mesh.face_edge_connectivity,
#              boundary_edges = boundary_edges(mesh, boundary_shape),
#              face_sets = mesh.face_sets
#             )
# end
# 
# # Return a mesh with face/edge connectivity and edge/face connectivity
# function add_connectivity(mesh::UnstructuredMesh_2D)
#     return add_edge_face_connectivity(mesh)
# end
# 














# 
# # Return a mesh with edge/face connectivity
# function add_edge_face_connectivity(mesh::M) where {M <: UnstructuredMesh_2D}
#     if 0 === length(mesh.face_edge_connectivity)
#         mesh = add_face_edge_connectivity(mesh)
#     end
#     return M(name = mesh.name,
#              points = mesh.points,
#              edges = mesh.edges,
#              materialized_edges = mesh.materialized_edges,
#              faces = mesh.faces,
#              materialized_faces = mesh.materialized_faces,
#              edge_face_connectivity = edge_face_connectivity(mesh),
#              face_edge_connectivity = mesh.face_edge_connectivity,
#              boundary_edges = mesh.boundary_edges,
#              face_sets = mesh.face_sets
#             )
# end
# 
# # Return a mesh with every field created
# function add_everything(mesh::UnstructuredMesh_2D)
#     return add_materialized_faces(
#              add_materialized_edges(
#                add_boundary_edges(mesh, boundary_shape = "Rectangle")))
# end
# 
# # Return a mesh with face/edge connectivity
# function add_face_edge_connectivity(mesh::M) where {M <: UnstructuredMesh_2D}
#     if 0 === length(mesh.edges)
#         mesh = add_edges(mesh)
#     end
#     return M(name = mesh.name,
#              points = mesh.points,
#              edges = mesh.edges,
#              materialized_edges = mesh.materialized_edges,
#              faces = mesh.faces,
#              materialized_faces = mesh.materialized_faces,
#              edge_face_connectivity = mesh.edge_face_connectivity,
#              face_edge_connectivity = face_edge_connectivity(mesh),
#              boundary_edges = mesh.boundary_edges,
#              face_sets = mesh.face_sets
#             )
# end

# # Return a vector of the faces adjacent to the face of ID face
# function adjacent_faces(face::UInt32, mesh::UnstructuredMesh_2D)
#     edges = mesh.face_edge_connectivity[face]
#     the_adjacent_faces = UInt32[]
#     for edge in edges
#         faces = mesh.edge_face_connectivity[edge]
#         for face_id in faces
#             if face_id != face && face_id != 0
#                 push!(the_adjacent_faces, face_id)
#             end
#         end
#     end
#     return the_adjacent_faces
# end
# 
# 
# # Bounding box of a vector of points
# function boundingbox(points::Vector{Point_2D})
#     x = getindex.(points, 1)
#     y = getindex.(points, 2)
#     return Rectangle_2D(minimum(x), minimum(y), maximum(x), maximum(y))
# end
# 
# # Bounding box of a vector of points
# function boundingbox(points::SVector{L, Point_2D}) where {L}
#     x = getindex.(points, 1)
#     y = getindex.(points, 2)
#     return Rectangle_2D(minimum(x), minimum(y), maximum(x), maximum(y))
# end
# 
# # Return a vector containing vectors of the edges in each side of the mesh's bounding shape, e.g.
# # For a rectangular bounding shape the sides are North, East, South, West. Then the output would
# # be [ [e1, e2, e3, ...], [e17, e18, e18, ...], ..., [e100, e101, ...]]
# function boundary_edges(mesh::UnstructuredMesh_2D, boundary_shape::String)
#     # edges which have face 0 in their edge_face connectivity are boundary edges
#     the_boundary_edges = UInt32.(findall(x->x[1] === 0x00000000, mesh.edge_face_connectivity))
#     if boundary_shape == "Rectangle"
#         # Sort edges into NESW
#         bb = boundingbox(mesh.points)
#         y_north = bb.ymax
#         x_east  = bb.xmax
#         y_south = bb.ymin
#         x_west  = bb.xmin
#         p_NW = Point_2D(x_west, y_north)
#         p_NE = bb.tr
#         p_SE = Point_2D(x_east, y_south)
#         p_SW = bb.bl
#         edges_north = UInt32[] 
#         edges_east = UInt32[] 
#         edges_south = UInt32[] 
#         edges_west = UInt32[] 
#         # Insert edges so that indices move from NW -> NE -> SE -> SW -> NW
#         for edge ∈  the_boundary_edges
#             epoints = edgepoints(mesh.edges[edge], mesh.points)
#             if all(x->abs(x[2] - y_north) < 1e-4, epoints)
#                 insert_boundary_edge!(edge, p_NW, edges_north, mesh)
#             elseif all(x->abs(x[1] - x_east) < 1e-4, epoints)
#                 insert_boundary_edge!(edge, p_NE, edges_east, mesh)
#             elseif all(x->abs(x[2] - y_south) < 1e-4, epoints)
#                 insert_boundary_edge!(edge, p_SE, edges_south, mesh)
#             elseif all(x->abs(x[1] - x_west) < 1e-4, epoints)
#                 insert_boundary_edge!(edge, p_SW, edges_west, mesh)
#             else
#                 @error "Edge $edge could not be classified as NSEW"
#             end
#         end
#         return [ edges_north, edges_east, edges_south, edges_west ]
#     else
#         return [ convert(Vector{UInt32}, the_boundary_edges) ]
#     end 
# end 
# 





# 
# # A vector of length 2 SVectors, denoting the face ID each edge is connected to. If the edge
# # is a boundary edge, face ID 0 is returned
# function edge_face_connectivity(mesh::UnstructuredMesh_2D)
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
#     edge_face = [MVector{2, UInt32}(0, 0) for _ in eachindex(mesh.edges)]
#     for (iface, edges) in enumerate(mesh.face_edge_connectivity)
#         for iedge in edges
#             # Add the face id in the first non-zero position of the edge_face conn. vec.
#             if edge_face[iedge][1] == 0
#                 edge_face[iedge][1] = iface
#             elseif edge_face[iedge][2] == 0
#                 edge_face[iedge][2] = iface
#             else
#                 @error "Edge $iedge seems to have 3 faces associated with it!"
#             end
#         end
#     end
#     return [SVector(sort(two_faces).data) for two_faces in edge_face]
# end

# # Find the faces which share the vertex of ID v.
# function faces_sharing_vertex(v::Integer, mesh::UnstructuredMesh_2D)
#     shared_faces = UInt32[]
#     for i ∈ 1:length(mesh.faces)
#         if v ∈  mesh.faces[i]
#             push!(shared_faces, UInt32(i))
#         end
#     end
#     return shared_faces
# end
# 
# # Return the face containing point p.
# function findface(p::Point_2D, mesh::UnstructuredMesh_2D)
#     if 0 < length(mesh.materialized_faces)
#         return UInt32(findface_explicit(p, mesh.materialized_faces))
#     else
#         return UInt32(findface_implicit(p, mesh.faces, mesh.points))
#     end
# end
# 
# # Find the face containing the point p, with explicitly represented faces
# function findface_explicit(p::Point_2D, faces::Vector{<:Face_2D})
#     for i ∈ 1:length(faces)
#         if p ∈  faces[i]
#             return i
#         end
#     end
#     return 0
# end
# 
# # Return the face containing the point p, with implicitly represented faces
# function findface_implicit(p::Point_2D,
#                             faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}},
#                             points::Vector{Point_2D})
#     for i ∈ 1:length(faces)
#         bool = p ∈  materialize_face(faces[i], points)
#         if bool
#             return i
#         end
#     end
#     return 0
# end
# 
# # Return the intersection algorithm that will be used for l ∩ mesh
# function get_intersection_algorithm(mesh::UnstructuredMesh_2D)
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
# # Insert the boundary edge into the correct place in the vector of edge indices, based on
# # the distance from some reference point
# function insert_boundary_edge!(edge_index::UInt32, p_ref::Point_2D, edge_indices::Vector{UInt32},
#                                mesh::UnstructuredMesh_2D)
#     # Compute the minimum distance from the edge to be inserted to the reference point
#     insertion_distance = minimum(distance.(Ref(p_ref), 
#                                            edgepoints(mesh.edges[edge_index], mesh.points)))
#     # Loop through the edge indices until an edge with greater distance from the reference point
#     # is found, then insert
#     nindices = length(edge_indices)
#     for i ∈ 1:nindices
#         epoints = edgepoints(mesh.edges[edge_indices[i]], mesh.points)
#         edge_distance = minimum(distance.(Ref(p_ref), epoints))
#         if insertion_distance < edge_distance
#             insert!(edge_indices, i, edge_index)
#             return nothing
#         end
#     end
#     insert!(edge_indices, nindices+1, edge_index)
#     return nothing
# end
# 
# # Intersect a line with the mesh. Returns a vector of intersection points, sorted based
# # upon distance from the line's start point
# function intersect(l::LineSegment_2D, mesh::UnstructuredMesh_2D)
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
# function intersect_edge_implicit(l::LineSegment_2D,
#                                  edge::SVector{L, UInt32} where {L},
#                                  points::Vector{Point_2D})
#     return l ∩ materialize_edge(edge, points)
# end
# 
# # Intersect a line with materialized edges
# function intersect_edges_explicit(l::LineSegment_2D, edges::Vector{LineSegment_2D})
#     # A vector to hold all of the intersection points
#     intersection_points = Point_2D[]
#     for edge in edges
#         npoints, point = l ∩ edge
#         # If the intersections yields 1 or more points, push those points to the array of points
#         if 0 < npoints
#             push!(intersection_points, point)
#         end
#     end
#     sort_intersection_points!(l[1], intersection_points)
#     return intersection_points
# end
# 
# # Intersect a line with implicitly defined edges
# function intersect_edges_explicit(l::LineSegment_2D, edges::Vector{QuadraticSegment_2D})
#     # A vector to hold all of the intersection points
#     intersection_points = Point_2D[]
#     for edge in edges
#         npoints, points = l ∩ edge
#         # If the intersections yields 1 or more points, push those points to the array of points
#         if 0 < npoints
#             append!(intersection_points, points[1:npoints])
#         end
#     end
#     sort_intersection_points!(l[1], intersection_points)
#     return intersection_points
# end
# 
# 
# # Intersect a line with a vector of implicitly defined linear edges
# function intersect_edges_implicit(l::LineSegment_2D,
#                                   edges::Vector{SVector{2, UInt32}},
#                                   points::Vector{Point_2D})
#     # A vector to hold all of the intersection points
#     intersection_points = Point_2D[]
#     # Intersect the line with each of the faces
#     for edge in edges
#         npoints, point = intersect_edge_implicit(l, edge, points)
#         if 0 < npoints
#             push!(intersection_points, point)
#         end
#     end
#     sort_intersection_points!(l[1], intersection_points)
#     return intersection_points
# end
# 
# # Intersect a line with a vector of implicitly defined quadratic edges
# function intersect_edges_implicit(l::LineSegment_2D,
#                                   edges::Vector{SVector{3, UInt32}},
#                                   points::Vector{Point_2D})
#     # An array to hold all of the intersection points
#     intersection_points = Point_2D[]
#     # Intersect the line with each of the faces
#     for edge in edges
#         npoints, ipoints = intersect_edge_implicit(l, edge, points)
#         if 0 < npoints
#             append!(intersection_points, ipoints[1:npoints])
#         end
#     end
#     sort_intersection_points!(l[1], intersection_points)
#     return intersection_points
# end
# 
# # Intersect a line with an implicitly defined face
# function intersect_face_implicit(l::LineSegment_2D,
#                                  face::SVector{L, UInt32} where {L},
#                                  points::Vector{Point_2D})
#     return l ∩ materialize_face(face, points)
# end
# 
# # Intersect a line with explicitly defined linear faces
# function intersect_faces_explicit(l::LineSegment_2D, faces::Vector{<:Face_2D} )
#     # An array to hold all of the intersection points
#     intersection_points = Point_2D[]
#     for face in faces
#         npoints, points = l ∩ face
#         # If the intersections yields 1 or more points, push those points to the array of points
#         if 0 < npoints
#             append!(intersection_points, points[1:npoints])
#         end
#     end
#     sort_intersection_points!(l[1], intersection_points)
#     return intersection_points
# end
# 
# # Intersect a line with implicitly defined faces
# function intersect_faces_implicit(l::LineSegment_2D,
#                                   faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}},
#                                   points::Vector{Point_2D})
#     # An array to hold all of the intersection points
#     intersection_points = Point_2D[]
#     # Intersect the line with each of the faces
#     for face in faces
#         npoints, ipoints = intersect_face_implicit(l, face, points)
#         # If the intersections yields 1 or more points, push those points to the array of points
#         if 0 < npoints
#             append!(intersection_points, ipoints[1:npoints])
#         end
#     end
#     sort_intersection_points!(l[1], intersection_points)
#     return intersection_points
# end
# 
# # If a point is a vertex
# function isvertex(p::Point_2D, mesh::UnstructuredMesh_2D)
#     for point in mesh.points
#         if p ≈ point
#             return true
#         end
#     end
#     return false
# end
# 

# 
# function remap_points_to_hilbert(points::Vector{Point_2D})
#     bb = boundingbox(points)
#     npoints = length(points)
#     # Generate a Hilbert curve
#     hilbert_points = hilbert_curve(bb, npoints)
#     nhilbert_points = length(hilbert_points)
#     # For each point, get the index of the hilbert points that is closest
#     point_indices = Vector{Int64}(undef, npoints)
#     for i ∈ 1:npoints
#         min_distance = 1.0e10
#         for j ∈ 1:nhilbert_points
#             pdistance = distance²(points[i], hilbert_points[j])
#             if pdistance < min_distance
#                 min_distance = pdistance
#                 point_indices[i] = j
#             end
#         end
#     end
#     return sortperm(point_indices)
# end
# 
# function reorder_points_to_hilbert!(mesh::UnstructuredMesh_2D)
#     # Points
#     # point_map     maps  new_points[i] == mesh.points[point_map[i]]
#     # point_map_inv maps mesh.points[i] == new_points[point_map_inv[i]]
#     point_map  = remap_points_to_hilbert(mesh.points)
#     point_map_inv = UInt32.(sortperm(point_map))
#     # reordered to resemble a hilbert curve
#     permute!(mesh.points, point_map)
# 
#     # Adjust face indices
#     # Point IDs have changed, so we need to change the point IDs referenced by the faces
#     new_faces_vec = [ point_map_inv[face] for face in mesh.faces]
#     for i in 1:length(mesh.faces)
#         mesh.faces[i] = SVector(point_map_inv[mesh.faces[i]])
#     end
#     return nothing 
# end
# 
# function reorder_faces_to_hilbert!(mesh::UnstructuredMesh_2D)
#     face_map  = UInt32.(remap_points_to_hilbert(centroid.(materialize_faces(mesh))))
#     permute!(mesh.faces, face_map)
#     return nothing 
# end
# 
# function reorder_to_hilbert!(mesh::UnstructuredMesh_2D)
#     reorder_points_to_hilbert!(mesh)
#     reorder_faces_to_hilbert!(mesh)
#     return nothing
# end
# 
# # Return the ID of the edge shared by two adjacent faces
# function shared_edge(face1::UInt32, face2::UInt32, mesh::UnstructuredMesh_2D)
#     for edge1 in mesh.face_edge_connectivity[face1]
#         for edge2 in mesh.face_edge_connectivity[face2]
#             if edge1 == edge2
#                 return edge1
#             end
#         end
#     end
#     return 0x00000000 
# end
# 
function Base.show(io::IO, mesh::TriangleMesh)
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
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_edges) !== 0)")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_faces) !== 0)")
    println(io, "  ├─ Connectivity")
    println(io, "  │  ├─ Edge/Face : $(0 < length(mesh.edge_face_connectivity))")
    println(io, "  │  └─ Face/Edge : $(0 < length(mesh.face_edge_connectivity))")
    println(io, "  ├─ Boundary edges")
    nsides = length(mesh.boundary_edges)
    if nsides !== 0
        println(io, "  │  ├─ Edges : $(mapreduce(x->length(x), +, mesh.boundary_edges))")
    else
        println(io, "  │  ├─ Edges : 0") 
    end
    println(io, "  │  └─ Sides : $nsides")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

function Base.show(io::IO, mesh::QuadrilateralMesh)
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
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_edges) !== 0)")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_faces) !== 0)")
    println(io, "  ├─ Connectivity")
    println(io, "  │  ├─ Edge/Face : $(0 < length(mesh.edge_face_connectivity))")
    println(io, "  │  └─ Face/Edge : $(0 < length(mesh.face_edge_connectivity))")
    println(io, "  ├─ Boundary edges")
    nsides = length(mesh.boundary_edges)
    if nsides !== 0
        println(io, "  │  ├─ Edges : $(mapreduce(x->length(x), +, mesh.boundary_edges))")
    else
        println(io, "  │  ├─ Edges : 0") 
    end
    println(io, "  │  └─ Sides : $nsides")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end


# # Sort intersection points, deleting points that are less than minimum_segment_length apart
# function sort_intersection_points!(p::Point_2D, points::Vector{Point_2D})
#     if 2 <= length(points)
#         sortpoints!(p, points)
#         # Eliminate any points for which the distance between consecutive points 
#         # is less than the minimum segment length
#         delete_ids = Int64[]
#         id_start = 1
#         for id_stop ∈ 2:length(points)
#             if distance²(points[id_start], points[id_stop]) < minimum_segment_length^2
#                 push!(delete_ids, id_stop)
#             else
#                 id_start = id_stop
#             end
#         end
#         deleteat!(points, delete_ids)
#     else
#         return points
#     end
# end
# 
# # Return a mesh with name name, composed of the faces in the set face_ids
# function submesh(name::String, mesh::M) where {M <: UnstructuredMesh_2D}
#     # Setup faces and get all vertex ids
#     face_ids = mesh.face_sets[name]
#     submesh_faces = Vector{Vector{UInt32}}(undef, length(face_ids))
#     vertex_ids = Set{UInt32}()
#     for (i, face_id) in enumerate(face_ids)
#         face_vec = collect(mesh.faces[face_id].data)
#         submesh_faces[i] = face_vec
#         union!(vertex_ids, Set(face_vec))
#     end
#     # Need to remap vertex ids in faces to new ids
#     vertex_ids_sorted = sort(collect(vertex_ids))
#     vertex_map = Dict{UInt32, UInt32}()
#     for (i,v) in enumerate(vertex_ids_sorted)
#         vertex_map[v] = i
#     end
#     submesh_points = Vector{Point_2D}(undef, length(vertex_ids_sorted))
#     for (i, v) in enumerate(vertex_ids_sorted)
#         submesh_points[i] = mesh.points[v]
#     end
#     # remap vertex ids in faces
#     for face in submesh_faces
#         for (i, v) in enumerate(face)
#             face[i] = vertex_map[v]
#         end
#     end
#     # At this point we have points, faces, & name.
#     # Just need to get the face sets
#     submesh_face_sets = Dict{String, Set{UInt32}}()
#     for face_set_name in keys(mesh.face_sets)
#         set_intersection = mesh.face_sets[face_set_name] ∩ face_ids
#         if length(set_intersection) !== 0
#             submesh_face_sets[face_set_name] = set_intersection
#         end
#     end
#     # Need to remap face ids in face sets
#     face_map = Dict{UInt32, UInt32}()
#     for (i,f) in enumerate(face_ids)
#         face_map[f] = i
#     end
#     for face_set_name in keys(submesh_face_sets)
#         new_set = Set{UInt32}()
#         for fid in submesh_face_sets[face_set_name]
#             union!(new_set, face_map[fid])
#         end
#         submesh_face_sets[face_set_name] = new_set
#     end
#     return M(name = name,
#              points = submesh_points,
#              faces = [SVector{length(f), UInt32}(f) for f in submesh_faces],
#              face_sets = submesh_face_sets
#             )
# end
# 
# # Plot
# # -------------------------------------------------------------------------------------------------
# if enable_visualization
#     function convert_arguments(LS::Type{<:LineSegments}, mesh::UnstructuredMesh_2D)
#         if 0 < length(mesh.materialized_edges)
#             return convert_arguments(LS, mesh.materialized_edges)
#         elseif 0 < length(mesh.edges)
#             return convert_arguments(LS, materialize_edges(mesh))
#         else
#             return convert_arguments(LS, materialize_faces(mesh))
#         end
#     end
# 
#     function convert_arguments(P::Type{<:Mesh}, mesh::UnstructuredMesh_2D)
#         if 0 < length(mesh.materialized_faces)
#             return convert_arguments(P, mesh.materialized_faces)
#         else
#             return convert_arguments(P, materialize_faces(mesh))
#         end
#     end
# end
