struct GeneralLinearUnstructuredMesh_2D <: LinearUnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{SVector{2, UInt32}}
    materialized_edges::Vector{LineSegment_2D}
    faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    materialized_faces::Vector{<:Face_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}}
    face_edge_connectivity::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function GeneralLinearUnstructuredMesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        materialized_edges::Vector{LineSegment_2D} = LineSegment_2D[],
        faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}} = SVector{3, UInt32}[],
        materialized_faces::Vector{<:Face_2D} = Triangle_2D[],
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

struct TriangleMesh_2D <: LinearUnstructuredMesh_2D
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

function TriangleMesh_2D(;
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
        return TriangleMesh_2D(name,
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

struct QuadrilateralMesh_2D <: LinearUnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{SVector{2, UInt32}}
    materialized_edges::Vector{LineSegment_2D}
    faces::Vector{SVector{4, UInt32}}
    materialized_faces::Vector{Quadrilateral_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}} 
    face_edge_connectivity::Vector{SVector{4, UInt32}} 
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function QuadrilateralMesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        materialized_edges::Vector{LineSegment_2D} = LineSegment_2D[],
        faces::Vector{SVector{4, UInt32}} = SVector{4, U}[],
        materialized_faces::Vector{Quadrilateral_2D} = Quadrilateral_2D[],
        edge_face_connectivity::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        face_edge_connectivity ::Vector{SVector{4, UInt32}} = SVector{4, UInt32}[],
        boundary_edges::Vector{Vector{UInt32}} = Vector{UInt32}[],
        face_sets::Dict{String, Set{UInt32}} = Dict{String, Set{UInt32}}()
    )
        return QuadrilateralMesh_2D(name,
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
function boundingbox(mesh::M) where {M <: LinearUnstructuredMesh_2D}
    # If the mesh does not have any quadratic faces, the bounding_box may be determined 
    # entirely from the points. 
    return boundingbox(mesh.points)
end

# A vector of SVectors, denoting the edge ID each face is connected to.
function face_edge_connectivity(mesh::TriangleMesh_2D)
    if length(the_edges) === 0
        @error "Does not have edges!"
    end
    # A vector of MVectors of zeros for each face
    # Each MVector is the length of the number of edges
    face_edge = [MVector{3, U}(zeros(U, num_edges(face)))
                    for face in the_faces]::Vector{<:MArray{S, U, 1, L} where {S<:Tuple, L}}
    # for each face in the mesh, generate the edges.
    # Search for the index of the edge in the mesh.edges vector
    # Insert the index of the edge into the face_edge connectivity vector
    for i in eachindex(the_faces)
        for (j, edge) in enumerate(edges(the_faces[i]))
            face_edge[i][j] = searchsortedfirst(the_edges, SVector(edge.data))
        end
    end
    return [SVector(sort(conn).data) for conn in face_edge]::Vector{<:SArray{S, U, 1, L} where {S<:Tuple, L}}
end
