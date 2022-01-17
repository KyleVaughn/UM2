# Triangle defined by its 3 vertices.
struct Triangle{N,T} <: Face{N,T}
    points::SVector{3, Point{N,T}}
end

const Triangle_2D = Triangle{2}
const Triangle_3D = Triangle{3}

Base.@propagate_inbounds function Base.getindex(q::Triangle, i::Integer)
    getfield(q, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
function Triangle(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T}) where {N,T}
    return Triangle{N,T}(SVector{3, Point{N,T}}(p₁, p₂, p₃))
end
function Triangle{N}(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T}) where {N,T}
    return Triangle{N,T}(SVector{3, Point{N,T}}(p₁, p₂, p₃))
end

# Methods
# ---------------------------------------------------------------------------------------------
# Interpolation
function (tri::Triangle)(r, s)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return Point((1 - r - s)*tri[1] + r*tri[2] + s*tri[3])
end

function area(tri::Triangle)
    # A = bh/2
    # Let 𝘂 = (v₂ - v₁), 𝘃 = (v₃ - v₁)
    # b = ‖𝘂‖
    # h = ‖sin(θ) 𝘃‖, where θ is the angle between 𝘂 and 𝘃
    # 𝘂 × 𝘃 = ‖𝘂‖‖𝘃‖sin(θ), hence
    # A = ‖𝘂 × 𝘃‖/2 = bh/2
    𝘂 = tri[2] - tri[1]
    𝘃 = tri[3] - tri[1]
    return norm(𝘂 × 𝘃)/2
end

function area(tri::Triangle_2D)
    𝘂 = tri[2] - tri[1]
    𝘃 = tri[3] - tri[1]
    # 2D cross product returns a scalar
    return abs(𝘂 × 𝘃)/2
end

centroid(tri::Triangle) = tri(1//3, 1//3)

function Base.in(p::Point_2D, tri::Triangle_2D)
    # If the point is to the left of every edge
    #  3<-----2
    #  |     ^
    #  | p  /
    #  |   /
    #  |  /
    #  v /
    #  1
    return isleft(p, LineSegment_2D(tri[1], tri[2])) &&
           isleft(p, LineSegment_2D(tri[2], tri[3])) &&
           isleft(p, LineSegment_2D(tri[3], tri[1]))
end

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
