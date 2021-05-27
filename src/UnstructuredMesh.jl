Base.@kwdef struct UnstructuredMesh
    points::Vector{Point} = Point[] 
    edges::Vector{Vector{Int64}} = Vector{Int64}[]
    faces::Vector{Vector{Int64}} = Vector{Int64}[]
    cells::Vector{Vector{Int64}} = Vector{Int64}[]
    name::String = "DefaultMeshName"
end

function construct_edges_from_cells(points::Vector{Point}, cells::Vector{Vector{Int64}})
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
