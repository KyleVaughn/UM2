struct UnstructuredMesh_2D{P, F, T}
    points::NTuple{P, Point_2D{T}}
    faces::NTuple{F, Tuple{Vararg{Int64}}}
    name::String
end

#struct UnstructuredMesh_2D{P, E, F, T}
#    points::NTuple{P, Point_2D{T}}
#    edges::NTuple{E, Tuple{Vararg{Int64}}}
#    faces::NTuple{F, Tuple{Vararg{Int64}}}
#    name::String
#end

#Base.@kwdef struct UnstructuredMesh_2D{P, T}

# Cell types are the same as VTK
const UnstructuredMesh_2D_linear_cell_types = [5, # Triangle 
                                               9  # Quadrilateral
                                              ]
const UnstructuredMesh_2D_quadratic_cell_types = [22, # Triangle6
                                                  23  # Quadrilateral8
                                                 ]
const UnstructuredMesh_2D_cell_types = vcat(UnstructuredMesh_2D_linear_cell_types,
                                            UnstructuredMesh_2D_quadratic_cell_types)

# Return each edge for a face
# Note, this returns a vector of vectors because we want to mutate the elements of the edge vectors
function edges(face::Tuple{Vararg{Int64}})
    cell_type = face[1]
    n_vertices = length(face) - 1
    if face[1] ∈  UnstructuredMesh_2D_linear_cell_types 
        edges = [ [face[i], face[i+1]] for i = 2:n_vertices]
        # Add the final edge that connects first and last vertices
        push!(edges, [face[n_vertices + 1], face[2]])
    elseif face[1] ∈  UnstructuredMesh_2D_quadratic_cell_types
        # There are N linear vertices and N quadratic vertices
        N = n_vertices ÷ 2
        edges = [ [face[i], face[i+1], face[N + i]] for i = 2:N]
        # Add the final edge that connects first and last vertices
        push!(edges, [face[N+1], face[2], face[2N+1]])
    else
        error("Unsupported cell type.")
        edges = [[-1, -1]]
    end
    return edges
end

# Create the edges for each face
function edges(faces::NTuple{F, Tuple{Vararg{Int64}}}) where F
    edges_unfiltered = Vector{Int64}[]
    for face in faces
        # Get the edges for each face
        face_edges = edges(face)
        # Order the linear edge vertices by ID
        for edge in face_edges 
            if edge[2] < edge[1]
                e1 = edge[1]
                edge[1] = edge[2]
                edge[2] = e1
            end
            # Add the edge to the list of edges
            push!(edges_unfiltered, edge)
        end
    end
    # Filter the duplicate edges
    edges_filtered = sort(collect(Set(edges_unfiltered)))
    return Tuple([ Tuple(v) for v in edges_filtered ])
end

#
### Axis-aligned bounding box, a rectangle.
##function AABB(mesh::UnstructuredMesh; tight::Bool=false)
##    # If the mesh does not have any quadratic cells/faces, the AABB may be determined entirely from the 
##    # points. If the mesh does have quadratic cells/faces, we need to find the bounding box of the edges
##    # that border the mesh. This algorithm naively performs the bounding box for each edge.
##    if mesh.dim == 2
##        if any(x->x ∈ UnstructuredMesh_quadratic_cell_types, getindex.(mesh.faces, 1))
##            for edge in mesh.edges
##                # construct the edge
##                # Check warntype
###                AABB(
##            end
##        else # Can use points
##            x = map(p->p[1], points)
##            y = map(p->p[2], points)
##            xmin = minimum(x)
##            xmax = maximum(x)
##            ymin = minimum(y)
##            ymax = maximum(y)
##            return (xmin, ymin, xmax, ymax)
##        end
##    else
##
##    end
##end
#
## function AABV (volume, cuboid)
## hasEdges
## hasFaces
## hasCells
## setupFaces
## setupCells
## pointdata dict -> name of data -> data (array/array) cells over which it is defined and values 
## edgedata
## facedata
## celldata
## visualize
## write
## read
