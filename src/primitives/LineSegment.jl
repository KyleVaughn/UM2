# A parametric line segment, defined as the set of all points such that
# 𝘅(r) = 𝘅₁ + r𝘂, where r ∈ [0, 1]. We also define 𝘅₂ = 𝘅₁ + 𝘂 for convenience.
#
# We store 𝘂 instead of 𝘅₂, since 𝘅₂ is needed infrequently, but 𝘂 is needed in
# nearly every method.
struct LineSegment{N,T} <: Edge{N,T}
    𝘅₁::Point{N,T} 
    𝘂::SVector{N,T}
end

const LineSegment_2D = LineSegment{2}
const LineSegment_3D = LineSegment{3}

function Base.getproperty(l::LineSegment, sym::Symbol)
    if sym === :𝘅₂
        return Point(l.𝘅₁ + l.𝘂)
    else # fallback to getfield
        return getfield(l, sym)
    end
end

# Constructors
# ---------------------------------------------------------------------------------------------
LineSegment{N,T}(𝘅₁::Point{N,T}, 𝘅₂::Point{N,T}) where {N,T} = LineSegment{N,T}(𝘅₁, 𝘅₂ - 𝘅₁) 
LineSegment{N}(𝘅₁::Point{N,T}, 𝘅₂::Point{N,T}) where {N,T} = LineSegment{N,T}(𝘅₁, 𝘅₂ - 𝘅₁) 
LineSegment(𝘅₁::Point{N,T}, 𝘅₂::Point{N,T}) where {N,T} = LineSegment{N,T}(𝘅₁, 𝘅₂ - 𝘅₁) 

# Methods
# ---------------------------------------------------------------------------------------------
# Interpolation
@inline (l::LineSegment)(r) = Point(l.𝘅₁.coord + r*l.𝘂)
@inline arclength(l::LineSegment) = distance(l.𝘅₁.coord, l.𝘅₁.coord + l.𝘂)

function Base.intersect(l₁::LineSegment_3D{T}, l₂::LineSegment_3D{T}) where {T}
    # NOTE: Doesn't work for colinear/parallel lines. (𝘂 × 𝘃 = 𝟬).
    # Using the equation of a line in parametric form
    # For l₁ = 𝘅₁ + r𝘂 and l₂ = 𝘅₂ + s𝘃
    # 1) 𝘅₁ + r𝘂 = 𝘅₂ + s𝘃                  subtracting 𝘅₁ from both sides
    # 2) r𝘂 = (𝘅₂-𝘅₁) + s𝘃                  𝘄 = 𝘅₂-𝘅₁
    # 3) r𝘂 = 𝘄 + s𝘃                        cross product with 𝘃 (distributive)
    # 4) r(𝘂 × 𝘃) = 𝘄 × 𝘃 + s(𝘃 × 𝘃)        𝘃 × 𝘃 = 𝟬
    # 5) r(𝘂 × 𝘃) = 𝘄 × 𝘃                   let 𝘄 × 𝘃 = 𝘅 and 𝘂 × 𝘃 = 𝘇
    # 6) r𝘇 = 𝘅                             dot product 𝘇 to each side
    # 7) r𝘇 ⋅ 𝘇 = 𝘅 ⋅ 𝘇                     divide by 𝘇 ⋅ 𝘇
    # 8) r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)                definition of 2-norm
    # 9) r = 𝘅 ⋅ 𝘇/‖𝘇‖
    # We need to ensure r, s ∈ [0, 1], hence we need to solve for s too.
    # 1) 𝘅₂ + s𝘃 = 𝘅₁ + r𝘂                     subtracting 𝘅₂ from both sides
    # 2) s𝘃 = -𝘄 + r𝘂                          cross product with 𝘄
    # 3) s(𝘃 × 𝘄) = -𝘄 × 𝘄 + r(𝘂 × 𝘄)          𝘄 × 𝘄 = 𝟬 
    # 4) s(𝘃 × 𝘄) = r(𝘂 × 𝘄)                   using 𝘂 × 𝘄 = -(𝘄 × 𝘂), likewise for 𝘃 × 𝘄
    # 5) s(𝘄 × 𝘃) = r(𝘄 × 𝘂)                   let 𝘄 × 𝘂 = 𝘆. use 𝘄 × 𝘃 = 𝘅
    # 6) s𝘅 = r𝘆                               dot product 𝘅 to each side
    # 7) s(𝘅 ⋅ 𝘅) = r(𝘆 ⋅ 𝘅)                   definition of 2-norm and divide
    # 9) s = r𝘅 ⋅ 𝘆/‖𝘅‖
    ϵ = T(5e-6)
    𝘄 = l₂.𝘅₁ - l₁.𝘅₁
    𝘇 = l₁.𝘂 × l₂.𝘂
    # Note: 0 ≤ 𝘄 ⋅ 𝘇, and the minimum distance between two lines is d = (𝘄 ⋅𝘇)/‖𝘇‖.
    # Hence 𝘄 ⋅ 𝘇 = 0 for the lines to intersect
    # (https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d)
    𝘄 ⋅ 𝘇 ≤ T(1e-8) || return (false, Point_3D{T}(0,0,0))
    𝘅 = 𝘄 × l₂.𝘂
    𝘆 = 𝘄 × l₁.𝘂
    r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)
    s = r*(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅)
    return (T(1e-8)^2 < 𝘇 ⋅ 𝘇 && -ϵ ≤ r && r ≤ 1 + ϵ && -ϵ ≤ s && s ≤ 1 + ϵ, l₂(s)) # (hit, point)
end

function Base.intersect(l₁::LineSegment_2D{T}, l₂::LineSegment_2D{T}) where {T}
    # NOTE: Doesn't work for colinear/parallel lines. (𝘂 × 𝘃 = 𝟬).
    # The cross product operator for 2D vectors returns a scalar, since the cross product 
    # of two vectors in the plane is a vector of the form (0, 0, z).
    # Using the equation of a line in parametric form
    #
    # From the 3D intersection routine we know:
    # r = 𝘅 ⋅ 𝘇/𝘇 ⋅ 𝘇 
    # s = (𝘅 ⋅ 𝘆)(𝘅 ⋅ 𝘇)/(‖𝘅‖‖𝘇‖) 
    # Since the 2D cross product returns a vector of the form (0, 0, z), the dot products are 
    # essentially scalar multiplication
    # r = 𝘅 ⋅ 𝘇/𝘇 ⋅ 𝘇 = x₃/z₃ 
    # s = (𝘅 ⋅ 𝘆)(𝘅 ⋅ 𝘇)/(‖𝘅‖‖𝘇‖) = y₃/z₃ 
    ϵ = T(5e-6)
    𝘄 = l₂.𝘅₁ - l₁.𝘅₁
    z = l₁.𝘂 × l₂.𝘂
    r = (𝘄 × l₂.𝘂)/z
    s = (𝘄 × l₁.𝘂)/z
    # -ϵ ≤ r ≤ 1 + ϵ introduces a branch, but -ϵ ≤ r && r ≤ 1 + ϵ doesn't for some reason.
    return (T(1e-8) < abs(z) && -ϵ ≤ r && r ≤ 1 + ϵ && -ϵ ≤ s && s ≤ 1 + ϵ, l₂(s)) # (hit, point)
end

# Return if the point is left of the line segment
#   𝗽    ^
#   ^   /
# 𝘃 |  / 𝘂
#   | /
#   o
#   We rely on 𝘂 × 𝘃 = ‖𝘂‖‖𝘃‖sin(θ). We may determine if θ ∈ (0, π] based on the sign of 𝘂 × 𝘃
@inline function isleft(p::Point_2D, l::LineSegment_2D)
    return l.𝘂 × (p - l.𝘅₁) >= 0
end

# A random line within [0, 1] × [0, 1]
function Base.rand(::Type{LineSegment{N,F}}) where {N,F} 
    points = rand(Point{N,F}, 2)
    return LineSegment{N,F}(points[1], points[2])
end

# N random lines within [0, 1] × [0, 1]
function Base.rand(::Type{LineSegment{N,F}}, NL::Int64) where {N,F}
    return [ rand(LineSegment{N,F}) for i ∈ 1:NL ]
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
        return convert_arguments(LS, [l.𝘅₁, l.𝘅₂])
    end

    function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment_2D})
        return convert_arguments(LS, reduce(vcat, [[l.𝘅₁, l.𝘅₂] for l in L]))
    end
end
