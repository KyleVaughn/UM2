# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, quad::Quadrilateral)
        l₁ = LineSegment(quad[1], quad[2])
        l₂ = LineSegment(quad[2], quad[3])
        l₃ = LineSegment(quad[3], quad[4])
        l₄ = LineSegment(quad[4], quad[1])
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:Quadrilateral})
        point_sets = [convert_arguments(LS, quad) for quad ∈  Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈  point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, quad::Quadrilateral)
        points = [quad[i].coord for i = 1:4]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, Q::Vector{<:Quadrilateral})
        points = reduce(vcat, [[quad[i].coord for i = 1:4] for quad ∈  Q])
        faces = zeros(Int64, 2*length(Q), 3)
        j = 0
        for i in 1:2:2*length(Q)
            faces[i    , :] = [1 2 3] .+ j
            faces[i + 1, :] = [3 4 1] .+ j
            j += 4
        end
        return convert_arguments(M, points, faces)
    end
end
