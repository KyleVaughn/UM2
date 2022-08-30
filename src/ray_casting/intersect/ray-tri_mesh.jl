export naive_face_intersection

function naive_face_intersection(R::Ray2{T}, mesh::TriMesh{T, I}) where {T, I}
    r_miss = T(INF_POINT)
    rvec = T[]
    for fv_conn in face_conn_iterator(mesh)
        for ev_conn in edge_conn_iterator(fv_conn)
            v1 = mesh.vertices[ev_conn[1]]
            v2 = mesh.vertices[ev_conn[2]]
            r = ray_line_segment_intersection(R, v1, v2)
            if r != r_miss
                push!(rvec, r)
            end
        end
    end
    return rvec
end

function naive_face_intersection(rvec::Vector{T},
                                 R::Ray2{T}, 
                                 mesh::TriMesh{T, I}) where {T, I}
    r_miss = T(INF_POINT)
    N = 0
    rvec_length = length(rvec)
    for fv_conn in face_conn_iterator(mesh)
        for ev_conn in edge_conn_iterator(fv_conn)
            v1 = mesh.vertices[ev_conn[1]]
            v2 = mesh.vertices[ev_conn[2]]
            r = ray_line_segment_intersection(R, v1, v2)
            if r != r_miss
                if N + 1 <= rvec_length
                    rvec[N + 1] = r
                    N += 1
                else
                    push!(rvec, r)
                    N += 1
                    rvec_length += 1
                end
            end
        end
    end
    return rvec
end
