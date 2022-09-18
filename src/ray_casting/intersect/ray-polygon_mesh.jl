export intersect_faces_all,
       intersect_faces_all!,
       intersect_faces_all_fixed_size!

function intersect_faces_all(R::Ray2{T}, mesh::PolygonMesh{N, T}) where {N, T}
    r_miss = T(INF_POINT)
    rvec = T[]
    for fv_conn in fv_conn_iterator(mesh)
        for ev_conn in polygon_ev_conn_iterator(fv_conn)
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

function intersect_faces_all!(rvec::Vector{T},
                              R::Ray2{T}, 
                              mesh::PolygonMesh{N, T}) where {N, T}
    r_miss = T(INF_POINT)
    nintersect = 0
    rvec_length = length(rvec)
    for fv_conn in fv_conn_iterator(mesh)
        for ev_conn in polygon_ev_conn_iterator(fv_conn)
            v1 = mesh.vertices[ev_conn[1]]
            v2 = mesh.vertices[ev_conn[2]]
            r = ray_line_segment_intersection(R, v1, v2)
            if r != r_miss
                if nintersect + 1 <= rvec_length
                    rvec[nintersect + 1] = r
                    nintersect += 1
                else
                    push!(rvec, r)
                    nintersect += 1
                    rvec_length += 1
                end
            end
        end
    end
    return nintersect
end

function intersect_faces_all_fixed_size!(rvec::Vector{T},
                                         R::Ray2{T}, 
                                         mesh::PolygonMesh{N, T}) where {N, T}
    r_miss = T(INF_POINT)
    nintersect = 1
    for fv_conn in fv_conn_iterator(mesh)
        for ev_conn in polygon_ev_conn_iterator(fv_conn)
            v1 = mesh.vertices[ev_conn[1]]
            v2 = mesh.vertices[ev_conn[2]]
            r = ray_line_segment_intersection(R, v1, v2)
            if r != r_miss
                rvec[nintersect] = r
                nintersect += 1
            end
        end
    end
    return nintersect
end
