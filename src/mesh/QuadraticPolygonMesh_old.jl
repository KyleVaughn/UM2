
# # A vector of SVectors, denoting the edge ID each face is connected to.
# function face_edge_connectivity(mesh::Quadrilateral8Mesh_2D)
#     if length(mesh.edges) === 0
#         @error "Mesh does not have edges!"
#     end
#     # A vector of MVectors of zeros for each face
#     # Each MVector is the length of the number of edges
#     face_edge = [MVector{4, UInt32}(0, 0, 0, 0) for _ in eachindex(mesh.faces)]
#     # For each face in the mesh, generate the edges.
#     # Search for the index of the edge in the mesh.edges vector
#     # Insert the index of the edge into the face_edge connectivity vector
#     for i in eachindex(mesh.faces)
#         for (j, edge) in enumerate(edges(mesh.faces[i]))
#             face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
#         end
#         if any(x->x === 0x00000000, face_edge[i])
#             @error "Could not determine the face/edge connectivity of face $i"
#         end
#     end
#     return [SVector(sort(conn).data) for conn in face_edge]
# end
# 
# # A vector of SVectors, denoting the edge ID each face is connected to.
# function face_edge_connectivity(mesh::Triangle6Mesh_2D)
#     if length(mesh.edges) === 0
#         @error "Mesh does not have edges!"
#     end
#     # A vector of MVectors of zeros for each face
#     # Each MVector is the length of the number of edges
#     face_edge = [MVector{3, UInt32}(0, 0, 0) for _ in eachindex(mesh.faces)]
#     # For each face in the mesh, generate the edges.
#     # Search for the index of the edge in the mesh.edges vector
#     # Insert the index of the edge into the face_edge connectivity vector
#     for i in eachindex(mesh.faces)
#         for (j, edge) in enumerate(edges(mesh.faces[i]))
#             face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
#         end
#         if any(x->x === 0x00000000, face_edge[i])
#             @error "Could not determine the face/edge connectivity of face $i"
#         end
#     end
#     return [SVector(sort(conn).data) for conn in face_edge]
# end

# How to display a mesh in REPL
function Base.show(io::IO, mesh::QuadraticPolygonMesh)
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
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{6},  mesh.faces))")
    println(io, "  │  └─ Quadrilateral  : $(count(x->x isa SVector{8},  mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end
