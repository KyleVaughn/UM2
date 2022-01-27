# struct PolygonMesh{Dim,T,U} <:LinearUnstructuredMesh{Dim,T,U}
#     name::String
#     points::Vector{Point{Dim,T}}
#     edges::Vector{SVector{2,U}}
#     materialized_edges::Vector{LineSegment{Dim,T}}
#     faces::Vector{<:SArray{S,U,1} where {S<:Tuple}}
#     materialized_faces::Vector{<:Polygon{N,Dim,T} where {N}}
#     edge_face_connectivity::Vector{SVector{2,U}}
#     face_edge_connectivity::Vector{<:SArray{S,U,1} where {S<:Tuple}}
#     boundary_edges::Vector{Vector{U}}
#     face_sets::Dict{String, Set{U}}
# end
# 
# function PolygonMesh{Dim,T,U}(;
#     name::String = "default_name",
#     points::Vector{Point{Dim,T}} = Point{Dim,T}[],
#     edges::Vector{SVector{2,U}} = SVector{2,U}[],
#     materialized_edges::Vector{LineSegment{Dim,T}} = LineSegment{Dim,T}[],
#     faces::Vector{<:SArray{S,U,1} where {S<:Tuple}} = SVector{3,U}[],
#     materialized_faces::Vector{<:Polygon{N,Dim,T} where {N}} = Polygon{3,Dim,T}[],
#     edge_face_connectivity::Vector{SVector{2,U}} = SVector{2,U}[],
#     face_edge_connectivity::Vector{<:SArray{S,U,1} where {S<:Tuple}} = SVector{3,U}[],
#     boundary_edges::Vector{Vector{U}} = Vector{U}[],
#     face_sets::Dict{String, Set{U}} = Dict{String, Set{U}}()
#     ) where {Dim,T,U}
#     return PolygonMesh(name, points, edges, materialized_edges, faces, materialized_faces,
#                        edge_face_connectivity, face_edge_connectivity, boundary_edges, face_sets)
# end
# 
# struct TriangleMesh{Dim,T,U} <:LinearUnstructuredMesh{Dim,T,U}
#     name::String
#     points::Vector{Point{Dim,T}}
#     edges::Vector{SVector{2,U}}
#     materialized_edges::Vector{LineSegment{Dim,T}}
#     faces::Vector{SVector{3,U}}
#     materialized_faces::Vector{Polygon{3,Dim,T}}
#     edge_face_connectivity::Vector{SVector{2,U}}
#     face_edge_connectivity::Vector{SVector{3,U}}
#     boundary_edges::Vector{Vector{U}}
#     face_sets::Dict{String, Set{U}}
# end
# 
# function TriangleMesh{Dim,T,U}(;
#     name::String = "default_name",
#     points::Vector{Point{Dim,T}} = Point{Dim,T}[],
#     edges::Vector{SVector{2,U}} = SVector{2,U}[],
#     materialized_edges::Vector{LineSegment{Dim,T}} = LineSegment{Dim,T}[],
#     faces::Vector{SVector{3,U}} = SVector{3,U}[],
#     materialized_faces::Vector{Polygon{3,Dim,T}} = Polygon{3,Dim,T}[],
#     edge_face_connectivity::Vector{SVector{2,U}} = SVector{2,U}[],
#     face_edge_connectivity::Vector{SVector{3,U}} = SVector{3,U}[],
#     boundary_edges::Vector{Vector{U}} = Vector{U}[],
#     face_sets::Dict{String, Set{U}} = Dict{String, Set{U}}()
#     ) where {Dim,T,U}
#     return TriangleMesh(name, points, edges, materialized_edges, faces, materialized_faces,
#                         edge_face_connectivity, face_edge_connectivity, boundary_edges, face_sets)
# end
# 
# struct QuadrilateralMesh{Dim,T,U} <:LinearUnstructuredMesh{Dim,T,U}
#     name::String
#     points::Vector{Point{Dim,T}}
#     edges::Vector{SVector{2,U}}
#     materialized_edges::Vector{LineSegment{Dim,T}}
#     faces::Vector{SVector{4,U}}
#     materialized_faces::Vector{Polygon{4,Dim,T}}
#     edge_face_connectivity::Vector{SVector{2,U}}
#     face_edge_connectivity::Vector{SVector{4,U}}
#     boundary_edges::Vector{Vector{U}}
#     face_sets::Dict{String, Set{U}}
# end
# 
# function QuadrilateralMesh{Dim,T,U}(;
#     name::String = "default_name",
#     points::Vector{Point{Dim,T}} = Point{Dim,T}[],
#     edges::Vector{SVector{2,U}} = SVector{2,U}[],
#     materialized_edges::Vector{LineSegment{Dim,T}} = LineSegment{Dim,T}[],
#     faces::Vector{SVector{4,U}} = SVector{4,U}[],
#     materialized_faces::Vector{Polygon{4,Dim,T}} = Polygon{4,Dim,T}[],
#     edge_face_connectivity::Vector{SVector{2,U}} = SVector{2,U}[],
#     face_edge_connectivity::Vector{SVector{4,U}} = SVector{4,U}[],
#     boundary_edges::Vector{Vector{U}} = Vector{U}[],
#     face_sets::Dict{String, Set{U}} = Dict{String, Set{U}}()
#     ) where {Dim,T,U}
#     return QuadrilateralMesh(name, points, edges, materialized_edges, faces, materialized_faces,
#                              edge_face_connectivity, face_edge_connectivity, boundary_edges,
#                              face_sets)
# end



# Axis-aligned bounding box, in 2d a rectangle.
function boundingbox(mesh::QuadraticUnstructuredMesh_2D; boundary_shape::String="Unknown")
    if boundary_shape == "Rectangle"
        return boundingbox(mesh.points)
    else
        # Currently only polygons, so can use the points
        nsides = length(mesh.boundary_edges)
        if nsides !== 0
            boundary_edge_IDs = reduce(vcat, mesh.boundary_edges)
            point_IDs = reduce(vcat, mesh.edges[boundary_edge_IDs])
            return boundingbox(mesh.points[point_IDs]) 
        else
            return reduce(union, boundingbox.(materialize_edge.(edges(mesh), Ref(mesh.points))))
        end
    end
end
 
# A vector of SVectors, denoting the edge ID each face is connected to.
function face_edge_connectivity(mesh::Quadrilateral8Mesh_2D)
    if length(mesh.edges) === 0
        @error "Mesh does not have edges!"
    end
    # A vector of MVectors of zeros for each face
    # Each MVector is the length of the number of edges
    face_edge = [MVector{4, UInt32}(0, 0, 0, 0) for _ in eachindex(mesh.faces)]
    # For each face in the mesh, generate the edges.
    # Search for the index of the edge in the mesh.edges vector
    # Insert the index of the edge into the face_edge connectivity vector
    for i in eachindex(mesh.faces)
        for (j, edge) in enumerate(edges(mesh.faces[i]))
            face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
        end
        if any(x->x === 0x00000000, face_edge[i])
            @error "Could not determine the face/edge connectivity of face $i"
        end
    end
    return [SVector(sort(conn).data) for conn in face_edge]
end

# A vector of SVectors, denoting the edge ID each face is connected to.
function face_edge_connectivity(mesh::Triangle6Mesh_2D)
    if length(mesh.edges) === 0
        @error "Mesh does not have edges!"
    end
    # A vector of MVectors of zeros for each face
    # Each MVector is the length of the number of edges
    face_edge = [MVector{3, UInt32}(0, 0, 0) for _ in eachindex(mesh.faces)]
    # For each face in the mesh, generate the edges.
    # Search for the index of the edge in the mesh.edges vector
    # Insert the index of the edge into the face_edge connectivity vector
    for i in eachindex(mesh.faces)
        for (j, edge) in enumerate(edges(mesh.faces[i]))
            face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
        end
        if any(x->x === 0x00000000, face_edge[i])
            @error "Could not determine the face/edge connectivity of face $i"
        end
    end
    return [SVector(sort(conn).data) for conn in face_edge]
end
