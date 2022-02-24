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
