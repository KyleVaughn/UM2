function convert_arguments(LS::Type{<:LineSegments}, aab::AABox{2})
    return convert_arguments(LS, facets(aab))
end

function convert_arguments(LS::Type{<:LineSegments}, aab::AABox{3})
    return convert_arguments(LS, ridges(aab))
end

function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABox})
    point_sets = [convert_arguments(LS, aab) for aab in R]
    return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
end

function convert_arguments(LS::Type{<:LineSegments}, R::SVector{N, <:AABox}) where {N}
    return convert_arguments(LS, collect(R))
end

#function convert_arguments(M::Type{<:Mesh}, aab::AABox{2})
#    vertices = [v.coords for v in ridges(aab)]
#    faces = [1 2 3;
#             3 4 1]
#    return convert_arguments(M, vertices, faces)
#end

#function convert_arguments(M::Type{<:Mesh}, aab::AABox3D{T}) where {T}
#    #   4----3
#    #  /    /|
#    # 8----7 |
#    # |    | 2
#    # |    |/
#    # 5----6
#    Δx = (aab.xmax - aab.xmin)
#    Δy = (aab.ymax - aab.ymin)
#    Δz = (aab.zmax - aab.zmin)
#    p₁ = aab.minima
#    p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0))
#    p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0))
#    p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
#    p₅ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))
#    p₆ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
#    p₇ = aab.maxima
#    p₈ = Point3D(p₄ + Point3D{T}( 0,  0, Δz))
#
#    f₁ = Quadrilateral(p₁, p₂, p₃, p₄)
#    f₂ = Quadrilateral(p₅, p₆, p₇, p₈)
#    f₃ = Quadrilateral(p₆, p₂, p₃, p₇)
#    f₄ = Quadrilateral(p₇, p₃, p₄, p₈)
#    f₅ = Quadrilateral(p₈, p₄, p₁, p₅)
#    f₆ = Quadrilateral(p₅, p₆, p₂, p₁)
#    return convert_arguments(M, [f₁, f₂, f₃, f₄, f₅, f₆])
#end
#
#function convert_arguments(M::Type{<:Mesh}, R::Vector{<:AABox2D})
#    points = reduce(vcat, [[aab.minima.coord,
#                            Point2D(aab.xmax, aab.ymin).coord,
#                            aab.maxima.coord,
#                            Point2D(aab.xmin, aab.ymax).coord] for aab ∈ R])
#    faces = zeros(Int64, 2*length(R), 3)
#    j = 0
#    for i in 1:2:2*length(R)
#        faces[i    , :] = [1 2 3] .+ j
#        faces[i + 1, :] = [3 4 1] .+ j
#        j += 4
#    end
#    return convert_arguments(M, points, faces)
#end
#
#function convert_arguments(M::Type{<:Mesh}, R::Vector{AABox3D{T}}) where {T}
#    faces = Quadrilateral3D{T}[]
#    for aab ∈ R
#        Δx = (aab.xmax - aab.xmin)
#        Δy = (aab.ymax - aab.ymin)
#        Δz = (aab.zmax - aab.zmin)
#        p₁ = aab.minima
#        p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0))
#        p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0))
#        p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
#        p₅ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
#        p₆ = aab.maxima
#        p₇ = Point3D(p₆ - Point3D{T}(Δx,  0,  0))
#        p₈ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))
#
#        f₁ = Quadrilateral(p₁, p₂, p₃, p₄)
#        f₂ = Quadrilateral(p₅, p₆, p₇, p₈)
#        f₃ = Quadrilateral(p₂, p₅, p₆, p₃)
#        f₄ = Quadrilateral(p₁, p₈, p₇, p₄)
#        f₅ = Quadrilateral(p₄, p₃, p₆, p₇)
#        f₆ = Quadrilateral(p₁, p₂, p₅, p₈)
#        append!(faces, [f₁, f₂, f₃, f₄, f₅, f₆])
#    end
#    return convert_arguments(M, faces)
#end
