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
function boundingbox(mesh::M; bounding_shape::String="None"
    ) where {M <: QuadraticUnstructuredMesh_2D}
    if bounding_shape == "Rectangle"
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
