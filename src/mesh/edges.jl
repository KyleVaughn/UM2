export edge_connectivity

function edge_connectivity(mesh::VolumeMesh{2})
    if islinear(mesh)
        return _volume_mesh_linear_edge_connectivity(mesh)
    else
        return _volume_mesh_quadratic_edge_connectivity(mesh)
    end
end

function _volume_mesh_linear_edge_connectivity(mesh::VolumeMesh{2, T, U}) where {T, U}
    unique_edges = Vec{2, U}[]
    nedges = 0
    for face in face_connectivity(mesh)
        edge_vecs = edges(face)
        for edge::LineSegment{U} in edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge.vertices
            else
                sorted_edge = Vec(edge[2], edge[1])
            end
            index = getsortedfirst(unique_edges, sorted_edge)
            if nedges < index || unique_edges[index] !== sorted_edge
                insert!(unique_edges, index, sorted_edge)
                nedges += 1
            end
        end
    end
    lines = Vector{LineSegment{U}}(undef, nedges)
    lines[1:nedges] = reinterpret(LineSegment{U}, unique_edges)
    return lines
end 

function _volume_mesh_quadratic_edge_connectivity(mesh::VolumeMesh{2, T, U}) where {T, U}
    unique_edges = Vec{3, U}[]
    nedges = 0
    for face in face_connectivity(mesh)
        edge_vecs = edges(face)
        for edge::QuadraticSegment{U} in edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge.vertices
            else
                sorted_edge = Vec(edge[2], edge[1], edge[3])
            end
            index = getsortedfirst(unique_edges, sorted_edge)
            if nedges < index || unique_edges[index] !== sorted_edge
                insert!(unique_edges, index, sorted_edge)
                nedges += 1
            end
        end
    end
    segs = Vector{QuadraticSegment{U}}(undef, nedges)
    segs[1:nedges] = reinterpret(QuadraticSegment{U}, unique_edges)
    return segs
end 

function edge_connectivity(mesh::PolytopeVertexMesh{2})
    if islinear(mesh)
        return _polytope_mesh_linear_edge_connectivity(mesh)
    else
        return _polytope_mesh_quadratic_edge_connectivity(mesh)
    end
end

function _polytope_mesh_linear_edge_connectivity(mesh::PolytopeVertexMesh{2})
    U = vertextype(mesh.polytopes[1])
    unique_edges = Vec{2, U}[]
    nedges = 0
    for face in mesh.polytopes
        edge_vecs = edges(face)
        for edge in edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge.vertices
            else
                sorted_edge = Vec(edge[2], edge[1])
            end
            index = getsortedfirst(unique_edges, sorted_edge)
            if nedges < index || unique_edges[index] !== sorted_edge
                insert!(unique_edges, index, sorted_edge)
                nedges += 1
            end
        end
    end
    lines = Vector{LineSegment{U}}(undef, nedges)
    lines[1:nedges] = reinterpret(LineSegment{U}, unique_edges)
    return lines
end 

function _polytope_mesh_quadratic_edge_connectivity(mesh::PolytopeVertexMesh{2})
    U = vertextype(mesh.polytopes[1])
    unique_edges = Vec{3, U}[]
    nedges = 0
    for face in mesh.polytopes 
        edge_vecs = edges(face)
        for edge in edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge.vertices
            else
                sorted_edge = Vec(edge[2], edge[1], edge[3])
            end
            index = getsortedfirst(unique_edges, sorted_edge)
            if nedges < index || unique_edges[index] !== sorted_edge
                insert!(unique_edges, index, sorted_edge)
                nedges += 1
            end
        end
    end
    segs = Vector{QuadraticSegment{U}}(undef, nedges)
    segs[1:nedges] = reinterpret(QuadraticSegment{U}, unique_edges)
    return segs
end 
