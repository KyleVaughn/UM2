struct GeneralQuadraticUnstructuredMesh_2D <: QuadraticUnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{SVector{3, UInt32}}
    materialized_edges::Vector{QuadraticSegment_2D}
    faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    materialized_faces::Vector{<:Face_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}}
    face_edge_connectivity::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function GeneralQuadraticUnstructuredMesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{SVector{3, UInt32}} = SVector{3, UInt32}[],
        materialized_edges::Vector{QuadraticSegment_2D} = QuadraticSegment_2D[],
        faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}} = SVector{6, UInt32}[],
        materialized_faces::Vector{<:Face_2D} = Triangle6_2D[],
        edge_face_connectivity::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        face_edge_connectivity ::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}} = SVector{3, UInt32}[],
        boundary_edges::Vector{Vector{UInt32}} = Vector{UInt32}[],
        face_sets::Dict{String, Set{UInt32}} = Dict{String, Set{UInt32}}()
    )
        return GeneralUnstructuredMesh_2D(name,
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

struct Triangle6Mesh_2D <: QuadraticUnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{SVector{3, UInt32}}
    materialized_edges::Vector{QuadraticSegment_2D}
    faces::Vector{SVector{6, UInt32}}
    materialized_faces::Vector{Triangle6_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}} 
    face_edge_connectivity::Vector{SVector{3, UInt32}} 
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function Triangle6Mesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{SVector{3, UInt32}} = SVector{3, UInt32}[],
        materialized_edges::Vector{QuadraticSegment_2D} = QuadraticSegment_2D[],
        faces::Vector{SVector{6, UInt32}} = SVector{6, U}[],
        materialized_faces::Vector{Triangle6_2D} = Triangle6_2D[],
        edge_face_connectivity::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        face_edge_connectivity ::Vector{SVector{3, UInt32}} = SVector{3, UInt32}[],
        boundary_edges::Vector{Vector{UInt32}} = Vector{UInt32}[],
        face_sets::Dict{String, Set{UInt32}} = Dict{String, Set{UInt32}}()
    )
        return Triangle6Mesh_2D(name,
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

struct Quadrilateral8Mesh_2D <: QuadraticUnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{SVector{3, UInt32}}
    materialized_edges::Vector{QuadraticSegment_2D}
    faces::Vector{SVector{8, UInt32}}
    materialized_faces::Vector{Quadrilateral8_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}} 
    face_edge_connectivity::Vector{SVector{4, UInt32}} 
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function Quadrilateral8Mesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{SVector{3, UInt32}} = SVector{3, UInt32}[],
        materialized_edges::Vector{QuadraticSegment_2D} = QuadraticSegment_2D[],
        faces::Vector{SVector{8, UInt32}} = SVector{8, U}[],
        materialized_faces::Vector{Quadrilateral8_2D} = Quadrilateral8_D[],
        edge_face_connectivity::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        face_edge_connectivity ::Vector{SVector{4, UInt32}} = SVector{4, UInt32}[],
        boundary_edges::Vector{Vector{UInt32}} = Vector{UInt32}[],
        face_sets::Dict{String, Set{UInt32}} = Dict{String, Set{UInt32}}()
    )
        return Quadrilateral8Mesh_2D(name,
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


# Axis-aligned bounding box, in 2d a rectangle.
# function boundingbox(mesh::M) where {M <: QuadraticUnstructuredMesh_2D}
#     nsides = length(mesh.boundary_edges)
#     if nsides !== 0
#         bb = Rectangle_2D()
#         for iside ∈ 1:nsides
#             bb ∪ boundingbox(materialize_edge(mesh.edges[ 
#         end
#     else
#         return reduce(union, boundingbox.(materialize_edge.(edges(mesh), Ref(mesh.points))))
#     end
# end
# 
