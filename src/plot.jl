# Point
# ---------------------------------------------------------------------------------------------
function convert_arguments(T::Type{<:Scatter}, p::Point)
    return convert_arguments(T, p.coord)
end 

function convert_arguments(T::Type{<:Scatter}, P::Vector{<:Point})
    return convert_arguments(T, [p.coord for p in P]) 
end 

# LineSegment
# ---------------------------------------------------------------------------------------------
function convert_arguments(T::Type{<:LineSegments}, p::Point)
    return convert_arguments(T, p.coord)
end 

function convert_arguments(T::Type{<:LineSegments}, P::Vector{<:Point})
    return convert_arguments(T, [p.coord for p in P]) 

function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
    return convert_arguments(LS, [l.𝘅₁, l.𝘅₂])
end

function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment})
    return convert_arguments(LS, reduce(vcat, [[l.𝘅₁, l.𝘅₂] for l in L]))
end





# Z-coordinate is in the wrong direction!!!!

if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, aab::AABox2D)
        p₂ = Point2D(aab.xmax, aab.ymin)
        p₄ = Point2D(aab.xmin, aab.ymax)
        l₁ = LineSegment2D(aab.minima, p₂)
        l₂ = LineSegment2D(p₂, aab.maxima)
        l₃ = LineSegment2D(aab.maxima, p₄)
        l₄ = LineSegment2D(p₄, aab.minima)
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, aab::AABox3D{T}) where {T}
        #   7----6
        #  /    /|
        # 4----3 |
        # |    | 5
        # |    |/
        # 1----2
        Δx = (aab.xmax - aab.xmin)
        Δy = (aab.ymax - aab.ymin)
        Δz = (aab.zmax - aab.zmin)
        p₁ = aab.minima
        p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0))
        p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0))
        p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
        p₅ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
        p₆ = aab.maxima
        p₇ = Point3D(p₆ - Point3D{T}(Δx,  0,  0))
        p₈ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))
        #       10
        #     +----+
        #   8/   7/|9
        #   +----+ |
        #  4| 3  | +
        #   |   2|/ 5
        #   +----+
        #     1
        l₁  = LineSegment(p₁, p₂)
        l₂  = LineSegment(p₂, p₃)
        l₃  = LineSegment(p₃, p₄)
        l₄  = LineSegment(p₄, p₁)
        l₅  = LineSegment(p₂, p₅)
        l₆  = LineSegment(p₁, p₈)
        l₇  = LineSegment(p₃, p₆)
        l₈  = LineSegment(p₄, p₇)
        l₉  = LineSegment(p₅, p₆)
        l₁₀ = LineSegment(p₆, p₇)
        l₁₁ = LineSegment(p₇, p₈)
        l₁₂ = LineSegment(p₅, p₈)
        lines = [l₁, l₂, l₃, l₄, l₅, l₆, l₇, l₈, l₉, l₁₀, l₁₁, l₁₂]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABox})
        point_sets = [convert_arguments(LS, aab) for aab in R]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, aab::AABox2D)
        p₂ = Point2D(aab.xmax, aab.ymin)
        p₄ = Point2D(aab.xmin, aab.ymax)
        points = [aab.minima.coord, p₂.coord, aab.maxima.coord, p₄.coord]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, aab::AABox3D{T}) where {T}
        Δx = (aab.xmax - aab.xmin)
        Δy = (aab.ymax - aab.ymin)
        Δz = (aab.zmax - aab.zmin)
        p₁ = aab.minima
        p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0))
        p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0))
        p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
        p₅ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
        p₆ = aab.maxima
        p₇ = Point3D(p₆ - Point3D{T}(Δx,  0,  0))
        p₈ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))

        f₁ = Quadrilateral(p₁, p₂, p₃, p₄)
        f₂ = Quadrilateral(p₅, p₆, p₇, p₈)
        f₃ = Quadrilateral(p₂, p₅, p₆, p₃)
        f₄ = Quadrilateral(p₁, p₈, p₇, p₄)
        f₅ = Quadrilateral(p₄, p₃, p₆, p₇)
        f₆ = Quadrilateral(p₁, p₂, p₅, p₈)
        return convert_arguments(M, [f₁, f₂, f₃, f₄, f₅, f₆])
    end

    function convert_arguments(M::Type{<:Mesh}, R::Vector{<:AABox2D})
        points = reduce(vcat, [[aab.minima.coord,
                                Point2D(aab.xmax, aab.ymin).coord,
                                aab.maxima.coord,
                                Point2D(aab.xmin, aab.ymax).coord] for aab ∈ R])
        faces = zeros(Int64, 2*length(R), 3)
        j = 0
        for i in 1:2:2*length(R)
            faces[i    , :] = [1 2 3] .+ j
            faces[i + 1, :] = [3 4 1] .+ j
            j += 4
        end
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, R::Vector{AABox3D{T}}) where {T}
        faces = Quadrilateral3D{T}[]
        for aab ∈ R
            Δx = (aab.xmax - aab.xmin)
            Δy = (aab.ymax - aab.ymin)
            Δz = (aab.zmax - aab.zmin)
            p₁ = aab.minima
            p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0))
            p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0))
            p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
            p₅ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
            p₆ = aab.maxima
            p₇ = Point3D(p₆ - Point3D{T}(Δx,  0,  0))
            p₈ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))

            f₁ = Quadrilateral(p₁, p₂, p₃, p₄)
            f₂ = Quadrilateral(p₅, p₆, p₇, p₈)
            f₃ = Quadrilateral(p₂, p₅, p₆, p₃)
            f₄ = Quadrilateral(p₁, p₈, p₇, p₄)
            f₅ = Quadrilateral(p₄, p₃, p₆, p₇)
            f₆ = Quadrilateral(p₁, p₂, p₅, p₈)
            append!(faces, [f₁, f₂, f₃, f₄, f₅, f₆])
        end
        return convert_arguments(M, faces)
    end
end

