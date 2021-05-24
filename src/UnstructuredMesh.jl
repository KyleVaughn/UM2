Base.@kwdef struct UnstructuredMesh
    vertices::Vector{Point} = Point[] 
    edges::Vector{Vector{Int64}} = Vector{Int64}[]
    faces::Vector{Vector{Int64}} = Vector{Int64}[]
    cells::Vector{Vector{Int64}} = Vector{Int64}[]
    name::String = "DefaultMeshName"
end
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
