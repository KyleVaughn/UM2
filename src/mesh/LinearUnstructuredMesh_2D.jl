struct LinearTriangleMesh_2D <: UnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{SVector{2, UInt32}}
    materialized_edges::Vector{LineSegment_2D}
    faces::Vector{SVector{3, UInt32}}
    materialized_faces::Vector{Triangle_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}} 
    face_edge_connectivity::Vector{SVector{3, UInt32}} 
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function LinearTriangleMesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        materialized_edges::Vector{LineSegment_2D} = LineSegment_2D[],
        faces::Vector{SVector{3, UInt32}} = SVector{3, U}[],
        materialized_faces::Vector{Triangle_2D} = Triangle_2D[],
        edge_face_connectivity::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        face_edge_connectivity ::Vector{SVector{3, UInt32}} = SVector{3, UInt32}[],
        boundary_edges::Vector{Vector{UInt32}} = Vector{UInt32}[],
        face_sets::Dict{String, Set{UInt32}} = Dict{String, Set{UInt32}}()
    )
        return LinearTriangleMesh_2D(name,
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
