Base.@kwdef struct UnstructuredMesh
    points::Vector{Point} = Point[] 
    edges::Vector{Vector{Int64}} = Vector{Int64}[]
    faces::Vector{Vector{Int64}} = Vector{Int64}[]
    cells::Vector{Vector{Int64}} = Vector{Int64}[]
    dim::Int64 = 3
    name::String = "DefaultMeshName"
end
# Cell types are the same as VTK
const UnstructuredMesh_cell_types = [5,     # Triangle
                 9,     # Quadrilateral
                 22,    # Triangle6
                 23     # Quadrilateral8
                ]
const UnstructuredMesh_linear_cell_types = [5, 9]
const UnstructuredMesh_quadratic_cell_types = [22, 23]
const UnstructuredMesh_2D_cell_types = [5, 9, 22, 23]
const UnstructuredMesh_3D_cell_types = [10]

function edges(cell::Vector{Int64})
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

function edges(cells::Vector{Vector{Int64}})
    edges_unfiltered = Vector{Int64}[]
    for cell in cells
        cell_edges = edges(cell)
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

## Axis-aligned bounding box, a rectangle.
#function AABB(mesh::UnstructuredMesh; tight::Bool=false)
#    # If the mesh does not have any quadratic cells/faces, the AABB may be determined entirely from the 
#    # points. If the mesh does have quadratic cells/faces, we need to find the bounding box of the edges
#    # that border the mesh. This algorithm naively performs the bounding box for each edge.
#    if mesh.dim == 2
#        if any(x->x ∈ UnstructuredMesh_quadratic_cell_types, getindex.(mesh.faces, 1))
#            for edge in mesh.edges
#                # construct the edge
#                # Check warntype
##                AABB(
#            end
#        else # Can use points
#            x = map(p->p[1], points)
#            y = map(p->p[2], points)
#            xmin = minimum(x)
#            xmax = maximum(x)
#            ymin = minimum(y)
#            ymax = maximum(y)
#            return (xmin, ymin, xmax, ymax)
#        end
#    else
#
#    end
#end

# function AABV (volume, cuboid)
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
