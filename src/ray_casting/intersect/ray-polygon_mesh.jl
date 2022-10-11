export intersect_faces_all,
       intersect_faces_all!,
       intersect_faces_all_fixed_size!

function intersect_faces_all(R::Ray2{UM_F}, mesh::PolygonMesh{N}) where {N}
    r_miss = UM_F(INF_POINT)
    rvec = UM_F[]
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

function intersect_faces_all!(rvec::Vector{UM_F},
                              R::Ray2{UM_F}, 
                              mesh::PolygonMesh{N}) where {N}
    r_miss = UM_F(INF_POINT)
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

function intersect_faces_all_fixed_size!(rvec::Vector{UM_F},
                                         R::Ray2{UM_F}, 
                                         mesh::PolygonMesh{N}) where {N}
    r_miss = UM_F(INF_POINT)
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
