export show_statistics

function show_statistics(mesh::PolytopeVertexMesh{Dim,T,P}; plot::Bool=false
                        ) where {Dim,T,P}
    # Edges
    edges = materialize_edges(mesh)
    nedges = length(edges)
    nstraight = count(isstraight, edges)
    percent_straight = string(@sprintf("%.2f", 100*nstraight/nedges))
    println("Edges:")
    println(" Of the ", nedges, " edges, ", nstraight, " are straight. (", 
            percent_straight, " %)")
    println("Edge lengths (cm)")
    edge_lengths = measure.(edges)
    print_histogram(edge_lengths)
    edges = nothing
    edge_lengths = nothing

    # Faces
    println("\nFace areas (mm²)")
    faces = materialize_faces(mesh)
    face_areas = measure.(faces)
    print_histogram(100*face_areas)
    faces = nothing
    face_ares = nothing

    # Cells
    if polytope_k(P) === 3
        # Cell volumes
    end
    return nothing
end
#    fig = Figure()
#    aspect = 1
#    ax1 = Axis(fig[1, 1])
#    ax2 = Axis(fig[1, 2])
#    ax3 = Axis(fig[2, 1:2])
# Convexity
# Length of edges compared to size field, need have the pos file somewhere.
# Knudson number
# Jacobian determinant?
# Volume
##
## PLOT
##   Edge stats and mesh field stats
##   Knudson number?
