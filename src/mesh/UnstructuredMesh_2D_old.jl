# Return the face containing point p.
# Not type-stable
function find_face(p::Point_2D{F}, mesh::UnstructuredMesh_2D{F, U}) where {F <: AbstractFloat,
                                                                           U <: Unsigned}
    if 0 < length(mesh.materialized_faces)
        return U(find_face_explicit(p, mesh.materialized_faces))
    else
        return U(find_face_implicit(p, mesh.faces, mesh.points))
    end
end

# Return the intersection algorithm that will be used for l ∩ mesh
# Not type-stable
function get_intersection_algorithm(mesh::UnstructuredMesh_2D)
    if length(mesh.materialized_edges) !== 0
        return "Edges - Explicit"
    elseif length(mesh.edges) !== 0
        return "Edges - Implicit"
    elseif length(mesh.materialized_faces) !== 0
        return "Faces - Explicit"
    else
        return "Faces - Implicit"
    end
end

function reorder_points_to_hilbert(mesh::UnstructuredMesh_2D{F, U}
                           ) where {F <: AbstractFloat, U <: Unsigned}
    # Points
    # point_map     maps  new_points[i] == mesh.points[point_map[i]]
    # point_map_inv maps mesh.points[i] == new_points[point_map_inv[i]]
    point_map  = U.(remap_points_to_hilbert(mesh.points))
    point_map_inv = U.(sortperm(point_map))
    # new_points is the reordered point vector, reordered to resemble a hilbert curve
    new_points = mesh.points[point_map] 

    # Adjust face indices
    # Point IDs have changed, so we need to change the point IDs referenced by the faces
    new_faces_vec = [ point_map_inv[face] for face in mesh.faces]  
    for i in 1:length(mesh.faces)
        new_faces_vec[i][1] = mesh.faces[i][1]
    end
    new_faces = SVector.(new_faces_vec)
    return UnstructuredMesh_2D{F, U}(name = mesh.name,
                                     points = new_points,
                                     faces = new_faces,
                                     face_sets = mesh.face_sets
                                    )
end


# Return a mesh composed of the faces in the face set set_name
# Not type-stable
function submesh(mesh::UnstructuredMesh_2D{F, U},
                 set_name::String) where {F <: AbstractFloat, U <: Unsigned}
    @debug "Creating submesh for '$set_name'"
    face_ids = mesh.face_sets[set_name]
    return submesh(set_name, mesh.points, mesh.faces, mesh.face_sets, face_ids)
end
