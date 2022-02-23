# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
        return convert_arguments(LS, [l.𝘅₁, l.𝘅₂])
    end

    function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment})
        return convert_arguments(LS, reduce(vcat, [[l.𝘅₁, l.𝘅₂] for l in L]))
    end
end
