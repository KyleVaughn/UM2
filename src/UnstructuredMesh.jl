struct VertexMesh{V}
    vertices::NTuple{V, Point}
    edges::Vector{Vector{Int}}
    faces::Vector{Vector{Int}}
    cells::Vector{Vector{Int}}
    name::String
end
VertexMesh(vertices; 
                 edges::Vector{Vector{Int}} = [[0]], 
                 faces::Vector{Vector{Int}} = [[0]], 
                 cells::Vector{Vector{Int}} = [[0]],
                 name::String = "DefaultMeshName"
                ) = VertexMesh(vertices, edges, faces, cells, name)
# hasEdges
# hasFaces
# hasCells
# setupEdges
# setupFaces
# setupCells
# pointdata dict -> name of data -> data (array/array) cells over which it is defined and values 
# edgedata
# facedata
# celldata
# visualize
# write
# read
