function edge_statistics(mesh::PolytopeVertexMesh; plot::Bool = false)
    # Edges
    edges = materialize_edges(mesh)
    edge_lengths = measure.(edges)
    emed = median(edge_lengths)
    eavg = mean(edge_lengths)
    estd = std(edge_lengths)
    emin = minimum(edge_lengths)
    emax = maximum(edge_lengths)

    emedstr = string(@sprintf("%.3f", emed))
    eavgstr = string(@sprintf("%.3f", eavg))
    estdstr = string(@sprintf("%.3f", estd))
    eminstr = string(@sprintf("%.3f", emin))
    emaxstr = string(@sprintf("%.3f", emax))

    lmaxewidth = maximum(length.((emedstr, eavgstr, eminstr)))
    rmaxewidth = maximum(length.((estdstr, emaxstr)))

    print(" Range ")
    printstyled("("; color=:light_black)
    printstyled("min"; color=:cyan, bold=true)
    print(" … ")
    printstyled("max"; color=:magenta)
    printstyled("):  "; color=:light_black)
    printstyled(lpad(eminstr, lmaxewidth); color=:cyan, bold=true)
    print(" … ")
    printstyled(lpad(emaxstr, rmaxewidth); color=:magenta)
    print("  ")

    

#    fig = Figure()
#    aspect = 1
#    ax1 = Axis(fig[1, 1])
#    ax2 = Axis(fig[1, 2])
#    ax3 = Axis(fig[2, 1:2])
end
# Straightness
# Mesh area
# Length of edges compared to size field, need have the pos file somewhere.
# Knudson number
#
# PLOT
#   Edge stats and mesh field stats
#   Knudson number?
