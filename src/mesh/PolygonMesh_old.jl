#ALL polygons assumed to be complex
struct PolygonMesh{T, U} <:LinearUnstructuredMesh{2, T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
    face_sets::Dict{String, BitSet}
end

function PolygonMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}} = SVector{3, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return PolygonMesh(name, points, faces, face_sets)
end

struct TriangleMesh{T, U} <:LinearUnstructuredMesh{2, T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{3, U}}
    face_sets::Dict{String, BitSet}
end

function TriangleMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{3, U}} = SVector{3, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return TriangleMesh(name, points, faces, face_sets)
end

struct QuadrilateralMesh{T, U} <:LinearUnstructuredMesh{2, T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{4, U}}
    face_sets::Dict{String, BitSet}
end

function QuadrilateralMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{4, U}} = SVector{4, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadrilateralMesh(name, points, faces, face_sets)
end

# Return the edges of the mesh
function edges(mesh::LinearUnstructuredMesh{2, T, U}) where {T, U}
    edge_vecs = linear_edges.(mesh.faces)
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

# Point IDs representing the edges of a polygon.
@generated function linear_edges(face::SVector{N, U}) where {N, U <:Unsigned}
    edges_string = "SVector{N, SVector{2, U}}("
    for i ∈ 1:N
        id₁ = (i - 1) % N + 1
        id₂ = i % N + 1
        edges_string *= "SVector{2, U}(face[$id₁], face[$id₂]), "
    end
    edges_string *= ")"
    return Meta.parse(edges_string)
end

# Return a materialized triangle 
function materialize_face(face_id, mesh::TriangleMesh)
    return materialize_polygon(mesh.faces[face_id], mesh.points)
end

# Return a materialized quadrilateral
function materialize_face(face_id, mesh::QuadrilateralMesh)
    return materialize_polygon(mesh.faces[face_id], mesh.points)
end

# Return a materialized polygon 
function materialize_face(face_id, mesh::PolygonMesh)
    return materialize_polygon(mesh.faces[face_id], mesh.points)
end

# Return a materialized face from the point IDs in a face
function materialize_polygon(face::SVector{N}, points::Vector{<:Point}) where {N}
    return Polygon{N}(facepoints(face, points))
end

# # A vector of SVectors, denoting the edge ID each face is connected to.
# function face_edge_connectivity(mesh::QuadrilateralMesh{Dim,T,U}) where {Dim,T,U}
#     if length(mesh.edges) === 0
#         @error "Mesh does not have edges!"
#     end
#     # A vector of MVectors of zeros for each face
#     # Each MVector is the length of the number of edges
#     face_edge = [MVector{4, U}(0, 0, 0, 0) for _ in eachindex(mesh.faces)]
#     # For each face in the mesh, generate the edges.
#     # Search for the index of the edge in the mesh.edges vector
#     # Insert the index of the edge into the face_edge connectivity vector
#     for i in eachindex(mesh.faces)
#         for (j, edge) in enumerate(edges(mesh.faces[i]))
#             face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
#         end
#         if any(x->x === U(0), face_edge[i])
#             @error "Could not determine the face/edge connectivity of face $i"
#         end
#     end
#     return [SVector(sort!(conn).data) for conn in face_edge]
# end
# 
# # A vector of SVectors, denoting the edge ID each face is connected to.
# function face_edge_connectivity(mesh::TriangleMesh{Dim,T,U}) where {Dim,T,U}
#     if length(mesh.edges) === 0
#         @error "Mesh does not have edges!"
#     end
#     # A vector of MVectors of zeros for each face
#     # Each MVector is the length of the number of edges
#     face_edge = [MVector{3, U}(0, 0, 0) for _ in eachindex(mesh.faces)]
#     # For each face in the mesh, generate the edges.
#     # Search for the index of the edge in the mesh.edges vector
#     # Insert the index of the edge into the face_edge connectivity vector
#     for i in eachindex(mesh.faces)
#         for (j, edge) in enumerate(edges(mesh.faces[i]))
#             face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
#         end
#         if any(x->x === U(0), face_edge[i])
#             @error "Could not determine the face/edge connectivity of face $i"
#         end
#     end
#     return [SVector(sort!(conn).data) for conn in face_edge]
# end
# 
# # 
# # # Return a mesh with boundary edges
# # function add_boundary_edges(mesh::M; boundary_shape="Unknown"
# #     ) where {M <: UnstructuredMesh_2D}
# #     if 0 === length(mesh.edge_face_connectivity)
# #         mesh = add_connectivity(mesh)
# #     end
# #     return M(name = mesh.name,
# #              points = mesh.points,
# #              edges = mesh.edges,
# #              materialized_edges = mesh.materialized_edges,
# #              faces = mesh.faces,
# #              materialized_faces = mesh.materialized_faces,
# #              edge_face_connectivity = mesh.edge_face_connectivity,
# #              face_edge_connectivity = mesh.face_edge_connectivity,
# #              boundary_edges = boundary_edges(mesh, boundary_shape),
# #              face_sets = mesh.face_sets
# #             )
# # end
# # 
# 
# # # Return a vector of the faces adjacent to the face of ID face
# # function adjacent_faces(face::UInt32, mesh::UnstructuredMesh_2D)
# #     edges = mesh.face_edge_connectivity[face]
# #     the_adjacent_faces = UInt32[]
# #     for edge in edges
# #         faces = mesh.edge_face_connectivity[edge]
# #         for face_id in faces
# #             if face_id != face && face_id != 0
# #                 push!(the_adjacent_faces, face_id)
# #             end
# #         end
# #     end
# #     return the_adjacent_faces
# # end
# # 
# # # Return a vector containing vectors of the edges in each side of the mesh's bounding shape, e.g.
# # # For a rectangular bounding shape the sides are North, East, South, West. Then the output would
# # # be [ [e1, e2, e3, ...], [e17, e18, e18, ...], ..., [e100, e101, ...]]
# # function boundary_edges(mesh::UnstructuredMesh_2D, boundary_shape::String)
# #     # edges which have face 0 in their edge_face connectivity are boundary edges
# #     the_boundary_edges = UInt32.(findall(x->x[1] === 0x00000000, mesh.edge_face_connectivity))
# #     if boundary_shape == "Rectangle"
# #         # Sort edges into NESW
# #         bb = boundingbox(mesh.points)
# #         y_north = bb.ymax
# #         x_east  = bb.xmax
# #         y_south = bb.ymin
# #         x_west  = bb.xmin
# #         p_NW = Point_2D(x_west, y_north)
# #         p_NE = bb.tr
# #         p_SE = Point_2D(x_east, y_south)
# #         p_SW = bb.bl
# #         edges_north = UInt32[] 
# #         edges_east = UInt32[] 
# #         edges_south = UInt32[] 
# #         edges_west = UInt32[] 
# #         # Insert edges so that indices move from NW -> NE -> SE -> SW -> NW
# #         for edge ∈  the_boundary_edges
# #             epoints = edgepoints(mesh.edges[edge], mesh.points)
# #             if all(x->abs(x[2] - y_north) < 1e-4, epoints)
# #                 insert_boundary_edge!(edge, p_NW, edges_north, mesh)
# #             elseif all(x->abs(x[1] - x_east) < 1e-4, epoints)
# #                 insert_boundary_edge!(edge, p_NE, edges_east, mesh)
# #             elseif all(x->abs(x[2] - y_south) < 1e-4, epoints)
# #                 insert_boundary_edge!(edge, p_SE, edges_south, mesh)
# #             elseif all(x->abs(x[1] - x_west) < 1e-4, epoints)
# #                 insert_boundary_edge!(edge, p_SW, edges_west, mesh)
# #             else
# #                 @error "Edge $edge could not be classified as NSEW"
# #             end
# #         end
# #         return [ edges_north, edges_east, edges_south, edges_west ]
# #     else
# #         return [ convert(Vector{UInt32}, the_boundary_edges) ]
# #     end 
# # end 
# # 
# 
# 
# # # Find the faces which share the vertex of ID v.
# # function faces_sharing_vertex(v::Integer, mesh::UnstructuredMesh_2D)
# #     shared_faces = UInt32[]
# #     for i ∈ 1:length(mesh.faces)
# #         if v ∈  mesh.faces[i]
# #             push!(shared_faces, UInt32(i))
# #         end
# #     end
# #     return shared_faces
# # end
# # 
# # 
# # # Insert the boundary edge into the correct place in the vector of edge indices, based on
# # # the distance from some reference point
# # function insert_boundary_edge!(edge_index::UInt32, p_ref::Point_2D, edge_indices::Vector{UInt32},
# #                                mesh::UnstructuredMesh_2D)
# #     # Compute the minimum distance from the edge to be inserted to the reference point
# #     insertion_distance = minimum(distance.(Ref(p_ref), 
# #                                            edgepoints(mesh.edges[edge_index], mesh.points)))
# #     # Loop through the edge indices until an edge with greater distance from the reference point
# #     # is found, then insert
# #     nindices = length(edge_indices)
# #     for i ∈ 1:nindices
# #         epoints = edgepoints(mesh.edges[edge_indices[i]], mesh.points)
# #         edge_distance = minimum(distance.(Ref(p_ref), epoints))
# #         if insertion_distance < edge_distance
# #             insert!(edge_indices, i, edge_index)
# #             return nothing
# #         end
# #     end
# #     insert!(edge_indices, nindices+1, edge_index)
# #     return nothing
# # end
# # 
# # 
# 
# # # If a point is a vertex
# # function isvertex(p::Point_2D, mesh::UnstructuredMesh_2D)
# #     for point in mesh.points
# #         if p ≈ point
# #             return true
# #         end
# #     end
# #     return false
# # end
# # 
# 
# # 
# # function remap_points_to_hilbert(points::Vector{Point_2D})
# #     bb = boundingbox(points)
# #     npoints = length(points)
# #     # Generate a Hilbert curve
# #     hilbert_points = hilbert_curve(bb, npoints)
# #     nhilbert_points = length(hilbert_points)
# #     # For each point, get the index of the hilbert points that is closest
# #     point_indices = Vector{Int64}(undef, npoints)
# #     for i ∈ 1:npoints
# #         min_distance = 1.0e10
# #         for j ∈ 1:nhilbert_points
# #             pdistance = distance²(points[i], hilbert_points[j])
# #             if pdistance < min_distance
# #                 min_distance = pdistance
# #                 point_indices[i] = j
# #             end
# #         end
# #     end
# #     return sortperm(point_indices)
# # end
# # 
# # function reorder_points_to_hilbert!(mesh::UnstructuredMesh_2D)
# #     # Points
# #     # point_map     maps  new_points[i] == mesh.points[point_map[i]]
# #     # point_map_inv maps mesh.points[i] == new_points[point_map_inv[i]]
# #     point_map  = remap_points_to_hilbert(mesh.points)
# #     point_map_inv = UInt32.(sortperm(point_map))
# #     # reordered to resemble a hilbert curve
# #     permute!(mesh.points, point_map)
# # 
# #     # Adjust face indices
# #     # Point IDs have changed, so we need to change the point IDs referenced by the faces
# #     new_faces_vec = [ point_map_inv[face] for face in mesh.faces]
# #     for i in 1:length(mesh.faces)
# #         mesh.faces[i] = SVector(point_map_inv[mesh.faces[i]])
# #     end
# #     return nothing 
# # end
# # 
# # function reorder_faces_to_hilbert!(mesh::UnstructuredMesh_2D)
# #     face_map  = UInt32.(remap_points_to_hilbert(centroid.(materialize_faces(mesh))))
# #     permute!(mesh.faces, face_map)
# #     return nothing 
# # end
# # 
# # function reorder_to_hilbert!(mesh::UnstructuredMesh_2D)
# #     reorder_points_to_hilbert!(mesh)
# #     reorder_faces_to_hilbert!(mesh)
# #     return nothing
# # end
# # 
# # # Return the ID of the edge shared by two adjacent faces
# # function shared_edge(face1::UInt32, face2::UInt32, mesh::UnstructuredMesh_2D)
# #     for edge1 in mesh.face_edge_connectivity[face1]
# #         for edge2 in mesh.face_edge_connectivity[face2]
# #             if edge1 == edge2
# #                 return edge1
# #             end
# #         end
# #     end
# #     return 0x00000000 
# # end
# 
# How to display a mesh in REPL
function Base.show(io::IO, mesh::PolygonMesh)
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
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{3},  mesh.faces))")
    println(io, "  │  └─ Quadrilateral  : $(count(x->x isa SVector{4},  mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

# # # Plot
# # # -------------------------------------------------------------------------------------------------
# # if enable_visualization
# #     function convert_arguments(LS::Type{<:LineSegments}, mesh::UnstructuredMesh_2D)
# #         if 0 < length(mesh.materialized_edges)
# #             return convert_arguments(LS, mesh.materialized_edges)
# #         elseif 0 < length(mesh.edges)
# #             return convert_arguments(LS, materialize_edges(mesh))
# #         else
# #             return convert_arguments(LS, materialize_faces(mesh))
# #         end
# #     end
# # 
# #     function convert_arguments(P::Type{<:Mesh}, mesh::UnstructuredMesh_2D)
# #         if 0 < length(mesh.materialized_faces)
# #             return convert_arguments(P, mesh.materialized_faces)
# #         else
# #             return convert_arguments(P, materialize_faces(mesh))
# #         end
# #     end
# # end
