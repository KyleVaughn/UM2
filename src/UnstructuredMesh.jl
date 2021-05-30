Base.@kwdef struct UnstructuredMesh
    points::Vector{Point} = Point[] 
    edges::Vector{Vector{Int64}} = Vector{Int64}[]
    faces::Vector{Vector{Int64}} = Vector{Int64}[]
    cells::Vector{Vector{Int64}} = Vector{Int64}[]
    dim::Int64 = 3
    name::String = "DefaultMeshName"
end

function construct_edges(cell::Vector{Int64})
    cell_type = cell[1]
    if cell_type == 5 # Triangle
        return [
                [cell[2], cell[3]],  
                [cell[3], cell[4]],  
                [cell[4], cell[2]]
               ]
#    elseif cell_type = 9 # Quadrilateral
#
#    elseif cell_type = 22 # Quadratic Triangle
#
#    elseif cell_type = 23 # Quadratic Quadrilaterial
    else
        error("Unsupported cell type.")
        return [[0]]
    end
end

function construct_edges(cells::Vector{Vector{Int64}})
    edges_unfiltered = Vector{Int64}[]
    for cell in cells
        cell_edges = construct_edges(cell)
        for edge in cell_edges 
            if edge[2] < edge[1]
                e1 = edge[1]
                edge[1] = edge[2]
                edge[2] = e1
            end
            push!(edges_unfiltered, edge)
        end
    end
    return sort(collect(Set(edges_unfiltered)))
end
# hasEdges
# hasFaces
# hasCells
# setupFaces
# setupCells
# pointdata dict -> name of data -> data (array/array) cells over which it is defined and values 
# edgedata
# facedata
# celldata
# visualize
# write
# read
