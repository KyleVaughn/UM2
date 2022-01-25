function Base.intersect(l::LineSegment_2D{T}, tri::Triangle_2D{T}) where {T}
    # Create the 3 line segments that make up the triangle and intersect each one
    p₁ = Point_2D{T}(0,0)
    p₂ = Point_2D{T}(0,0)
    p₃ = Point_2D{T}(0,0)
    npoints = 0x0000
    for i ∈ 1:3
        hit, point = l ∩ LineSegment_2D(tri[(i - 1) % 3 + 1], 
                                        tri[      i % 3 + 1])
        if hit
            npoints += 0x0001
            if npoints === 0x0001
                p₁ = point
            elseif npoints === 0x0002
                p₂ = point
            else
                p₃ = point
            end
        end
    end
    return npoints, SVector(p₁, p₂, p₃) 
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, tri::Triangle)
        l₁ = LineSegment(tri[1], tri[2])
        l₂ = LineSegment(tri[2], tri[3])
        l₃ = LineSegment(tri[3], tri[1])
        lines = [l₁, l₂, l₃]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, T::Vector{<:Triangle})
        point_sets = [convert_arguments(LS, tri) for tri ∈  T]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, tri::Triangle)
        points = [tri[i].coord for i = 1:3]
        face = [1 2 3]
        return convert_arguments(M, points, face)
    end

    function convert_arguments(M::Type{<:Mesh}, T::Vector{<:Triangle})
        points = reduce(vcat, [[tri[i].coord for i = 1:3] for tri ∈  T])
        faces = zeros(Int64, length(T), 3)
        k = 1
        for i in 1:length(T), j = 1:3
            faces[i, j] = k
            k += 1
        end
        return convert_arguments(M, points, faces)
    end
end
