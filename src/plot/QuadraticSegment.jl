function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment)
    rr = LinRange(0, 1, plot_nonlinear_subdivisions) 
    points = q.(rr)
    coords = reduce(vcat, [[points[i], points[i+1]] for i = 1:length(points)-1])
    return convert_arguments(LS, coords)
end

function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:QuadraticSegment})
    point_sets = [convert_arguments(LS, q) for q in Q]
    return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
end
