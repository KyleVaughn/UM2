# A line segment defined by its two endpoints.
struct LineSegment{N,T} <: Edge{N,T}
    points::SVector{2, Point{N,T}}
end

const LineSegment_2D = LineSegment{2}
const LineSegment_3D = LineSegment{3}

# Constructors & Conversions
# -------------------------------------------------------------------------------------------------
LineSegment(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = LineSegment{N,T}(SVector(p₁, p₂))
LineSegment{N}(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = LineSegment{N,T}(SVector(p₁, p₂))
LineSegment{N,T}(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = LineSegment{N,T}(SVector(p₁, p₂))
function LineSegment{N,T₁}(p₁::Point{N,T₂}, p₂::Point{N,T₂}) where {N,T₁,T₂}
    return LineSegment{N,T₁}(SVector(Point{N, T₁}(p₁), Point{N, T₁}(p₂)))
end

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
# l(0) yields points[1], and l(1) yields points[2]
@inline (l::LineSegment{N,T})(r) where {N,T} = l[1] + (l[2] - l[1])T(r)
@inline arclength(l::LineSegment_2D) = distance(l[1], l[2])

# function intersect(l₁::LineSegment_2D{F}, l₂::LineSegment_2D{F}) where {F <: AbstractFloat}
#     # NOTE: Doesn't work for colinear/parallel lines. (v⃗ × u⃗ = 0). Also, the cross product
#     # operator for 2D points returns a scalar (the 2-norm of the cross product).
#     #
#     # Using the equation of a line in parametric form
#     # For l₁ = x⃗₁ + rv⃗ and l₂ = x⃗₂ + su⃗
#     # x⃗₁ + rv⃗ = x⃗₂ + su⃗                             subtracting x⃗₁ from both sides
#     # rv⃗ = (x⃗₂-x⃗₁) + su⃗                             w⃗ = x⃗₂-x⃗₁
#     # rv⃗ = w⃗ + su⃗                                   cross product with u⃗ (distributive)
#     # r(v⃗ × u⃗) = w⃗ × u⃗ + s(u⃗ × u⃗)                   u⃗ × u⃗ = 0
#     # r(v⃗ × u⃗) = w⃗ × u⃗                              dot product v⃗ × u⃗ to each side
#     # r = (w⃗ × u⃗)/(v⃗ × u⃗)
#     # Note that if the lines are parallel or collinear, v⃗ × u⃗ = 0
#     # We need to ensure r, s ∈ [0, 1].
#     # x⃗₂ + su⃗ = x⃗₁ + rv⃗                             subtracting x⃗₂ from both sides
#     # su⃗ = -w⃗ + rv⃗                                  cross product with w⃗
#     # s(u⃗ × w⃗) = -w⃗ × w⃗ + r(v⃗ × w⃗)                  w⃗ × w⃗ = 0 & substituting for r
#     # s(u⃗ × w⃗) =  (v⃗ × w⃗)(w⃗ × u⃗)/(v⃗ × u⃗)            -(u⃗ × w⃗) = w⃗ × u⃗
#     # s = -(v⃗ × w⃗)/(v⃗ × u⃗)                          -(v⃗ × w⃗) = w⃗ × v⃗
#     # s = (w⃗ × v⃗)/(v⃗ × u⃗)
#     #
#     # Simply evaluating everything removes branches and is faster than failing early with
#     # 1e-8 < abs(wxu) or delaying division by vxu and testing against r and s's numerators.
#     # This has been tested.
#     ϵ = F(5e-6)
#     v⃗ = l₁[2] - l₁[1]
#     u⃗ = l₂[2] - l₂[1]
#     w⃗ = l₂[1] - l₁[1]
#     vxu = v⃗ × u⃗
#     r = w⃗ × u⃗/vxu
#     s = w⃗ × v⃗/vxu
#     # -ϵ ≤ r ≤ 1 + ϵ introduces a branch, but -ϵ ≤ r && r ≤ 1 + ϵ doesn't for some reason.
#     return (1e-8 < abs(vxu) && -ϵ ≤ r && r ≤ 1 + ϵ && -ϵ ≤ s && s ≤ 1 + ϵ  , l₂(s)) # (hit, point)
# end


function Base.intersect(𝐥₁::LineSegment_2D, 𝐥₂::LineSegment_2D)
    # Using the equation of a line in parametric form
    # For 𝐥₁ = 𝐱₁ + r𝐯 and 𝐥₂ = 𝐱₂ + s𝐮
    # 1) 𝐱₁ + r𝐯 = 𝐱₂ + s𝐮                  subtracting 𝐱₁ from both sides
    # 2) r𝐯 = (𝐱₂-𝐱₁) + s𝐮                  𝐰 = 𝐱₂-𝐱₁
    # 3) r𝐯 = 𝐰 + s𝐮                        cross product with 𝐮 (distributive)
    # 4) r(𝐯 × 𝐮) = 𝐰 × 𝐮 + s(𝐮 × 𝐮)        𝐮 × 𝐮 = 0
    # 5) r(𝐯 × 𝐮) = 𝐰 × 𝐮                   let 𝐰 × 𝐮 = 𝐚 and 𝐯 × 𝐮 = 𝐜
    # 6) r𝐜 = 𝐚                             dot product 𝐜 to each side
    # 7) r𝐜 ⋅ 𝐜 = 𝐚 ⋅ 𝐜                     divide by 𝐜 ⋅ 𝐜
    # 8) r = 𝐚 ⋅ 𝐜/𝐜 ⋅ 𝐜                    definition of 2-norm
    # 9) r = 𝐚 ⋅ 𝐜/‖𝐜‖
    # Note that if the lines are parallel or collinear, 𝐜 = 𝐯 × 𝐮 = 0⃗
    # We need to ensure r, s ∈ [0, 1].
    # 𝐱₂ + s𝐮 = 𝐱₁ + r𝐯                     subtracting 𝐱₂ from both sides
    # s𝐮 = -𝐰 + r𝐯                          cross product with 𝐰
    # s(𝐮 × 𝐰) = -𝐰 × 𝐰 + r(𝐯 × 𝐰)          𝐰 × 𝐰 = 0 and substituting for r
    # s(𝐮 × 𝐰) = (𝐯 × 𝐰)[𝐚 ⋅ 𝐜/‖𝐜‖]         using 𝐮 × 𝐰 = -(𝐰 × 𝐮), likewise for 𝐯 × 𝐰
    # s(𝐰 × 𝐮) = (𝐰 × 𝐯)[𝐚 ⋅ 𝐜/‖𝐜‖]         let 𝐰 × 𝐯 = 𝐛. use 𝐰 × 𝐮 = 𝐚
    # s𝐚 = 𝐛[𝐚 ⋅ 𝐜/‖𝐜]                      dot product 𝐚 to each side
    # s(𝐚 ⋅ 𝐚) = (𝐛 ⋅ 𝐚)[𝐚 ⋅ 𝐜/‖𝐜‖]         definition of 2-norm and divide
    # s = (𝐚 ⋅ 𝐛)(𝐚 ⋅ 𝐜)/(‖𝐚‖‖𝐜‖)           substitute for r
    # s = r𝐚 ⋅ 𝐛/‖𝐚‖
    ϵ = 5e-6
    𝐰 = 𝐥₂[1] - 𝐥₁[1]
    𝐯 = 𝐥₁[2] - 𝐥₁[1]
    𝐮 = 𝐥₂[2] - 𝐥₂[1]
    𝐚 = 𝐰 × 𝐮
    𝐛 = 𝐰 × 𝐯
    𝐜 = 𝐯 × 𝐮
    r = (𝐚 ⋅ 𝐜)/(𝐜 ⋅ 𝐜)
    s = r*(𝐚 ⋅ 𝐛)/(𝐚 ⋅ 𝐚)
    return (1e-8 < abs(𝐜 ⋅ 𝐜) && -ϵ ≤ r && r ≤ 1 + ϵ && -ϵ ≤ s && s ≤ 1 + ϵ  , 𝐥₂(s)) # (hit, point)
end

# # Return if the point is left of the line segment
# #   p    ^
# #   ^   /
# # v⃗ |  / u⃗
# #   | /
# #   o
# #   We rely on v⃗ × u⃗ = |v⃗||u⃗|sin(θ). We may determine if θ ∈ (0, π] based on the sign of v⃗ × u⃗
# @inline function isleft(p::Point_2D, l::LineSegment_2D)
#     u⃗ = l[2] - l[1]
#     v⃗ = p - l[1]
#     return u⃗ × v⃗ >= 0
# end
#
# A random line within [0, 1] × [0, 1]
function Base.rand(::Type{LineSegment{N,F}}) where {N,F} 
    return LineSegment{N,F}(rand(Point{N,F}, 2))
end

# N random lines within [0, 1] × [0, 1]
function Base.rand(::Type{LineSegment{N,F}}, NP::Int64) where {N,F}
    return [ rand(LineSegment{N,F}) for i ∈ 1:NP ]
end

# # Plot
# # -------------------------------------------------------------------------------------------------
# if enable_visualization
#     function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment_2D)
#         return convert_arguments(LS, [l[1], l[2]])
#     end
#
#     function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment_2D})
#         return convert_arguments(LS, reduce(vcat, [[l[1], l[2]] for l in L]))
#     end
# end
