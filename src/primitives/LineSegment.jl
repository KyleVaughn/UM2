# A parametric line segment, defined as the set of all points such that
# 𝘅(r) = 𝘅₁ + r𝘂, where r ∈ [0, 1]. We also define 𝘅₂ = 𝘅₁ + 𝘂 for convenience.
#
# We store 𝘂 instead of 𝘅₂, since 𝘅₂ is needed infrequently, but 𝘂 is needed in
# nearly every method.
struct LineSegment{N,T} <: Edge{N,T}
    𝘅₁::Point{N,T} 
    𝘂::Point{N,T}
    
    LineSegment{N,T}(𝘅₁::Point{N,T}, 𝘅₂::Point{N,T}) = new(𝘅₁, 𝘅₂ - 𝘅₁) 
end

const LineSegment_2D = LineSegment{2}
const LineSegment_3D = LineSegment{3}

# Constructors & Conversions
# -------------------------------------------------------------------------------------------------
LineSegment{N,T}(𝘅₁::Point{N,T}, 𝘅₂::Point{N,T}) = new(𝘅₁, 𝘅₂ - 𝘅₁) 
# LineSegment(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = LineSegment{N,T}(SVector(p₁, p₂))
# LineSegment{N}(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = LineSegment{N,T}(SVector(p₁, p₂))
# LineSegment{N,T}(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = LineSegment{N,T}(SVector(p₁, p₂))
# function LineSegment{N,T₁}(p₁::Point{N,T₂}, p₂::Point{N,T₂}) where {N,T₁,T₂}
#     return LineSegment{N,T₁}(SVector(Point{N, T₁}(p₁), Point{N, T₁}(p₂)))
# end
# 
# # Methods
# # -------------------------------------------------------------------------------------------------
# # Interpolation
# # l(0) yields points[1], and l(1) yields points[2]
# @inline (l::LineSegment{N,T})(r) where {N,T} = l[1] + (l[2] - l[1])T(r)
# @inline arclength(l::LineSegment_2D) = distance(l[1], l[2])
# 
# function Base.intersect(𝗹₁::LineSegment_3D{T}, 𝗹₂::LineSegment_3D{T}) where {T}
#     # NOTE: Doesn't work for colinear/parallel lines. (𝘃 × 𝘂 = 𝟬).
#     # Using the equation of a line in parametric form
#     # For 𝗹₁ = 𝗽₁ + r𝘃 and 𝗹₂ = 𝗽₂ + s𝘂
#     # 1) 𝗽₁ + r𝘃 = 𝗽₂ + s𝘂                  subtracting 𝗽₁ from both sides
#     # 2) r𝘃 = (𝗽₂-𝗽₁) + s𝘂                  𝘄 = 𝗽₂-𝗽₁
#     # 3) r𝘃 = 𝘄 + s𝘂                        cross product with 𝘂 (distributive)
#     # 4) r(𝘃 × 𝘂) = 𝘄 × 𝘂 + s(𝘂 × 𝘂)        𝘂 × 𝘂 = 𝟬
#     # 5) r(𝘃 × 𝘂) = 𝘄 × 𝘂                   let 𝘄 × 𝘂 = 𝘅 and 𝘃 × 𝘂 = 𝘇
#     # 6) r𝘇 = 𝘅                             dot product 𝘇 to each side
#     # 7) r𝘇 ⋅ 𝘇 = 𝘅 ⋅ 𝘇                     divide by 𝘇 ⋅ 𝘇
#     # 8) r = 𝘅 ⋅ 𝘇/𝘇 ⋅ 𝘇                    definition of 2-norm
#     # 9) r = 𝘅 ⋅ 𝘇/‖𝘇‖
#     # Note that if the lines are parallel or collinear, 𝘇 = 𝘃 × 𝘂 = 𝟬
#     # We need to ensure r, s ∈ [0, 1].
#     # 𝗽₂ + s𝘂 = 𝗽₁ + r𝘃                     subtracting 𝗽₂ from both sides
#     # s𝘂 = -𝘄 + r𝘃                          cross product with 𝘄
#     # s(𝘂 × 𝘄) = -𝘄 × 𝘄 + r(𝘃 × 𝘄)          𝘄 × 𝘄 = 𝟬 and substituting for r
#     # s(𝘂 × 𝘄) = (𝘅 ⋅ 𝘇/‖𝘇‖)(𝘃 × 𝘄)         using 𝘂 × 𝘄 = -(𝘄 × 𝘂), likewise for 𝘃 × 𝘄
#     # s(𝘄 × 𝘂) = (𝘅 ⋅ 𝘇/‖𝘇‖)(𝘄 × 𝘃)         let 𝘄 × 𝘃 = 𝘆. use 𝘄 × 𝘂 = 𝘅
#     # s𝘅 = (𝘅 ⋅ 𝘇/‖𝘇‖)𝘆                     dot product 𝘅 to each side
#     # s(𝘅 ⋅ 𝘅) = (𝘅 ⋅ 𝘇/‖𝘇‖)(𝘆 ⋅ 𝘅)         definition of 2-norm and divide
#     # s = (𝘅 ⋅ 𝘆)(𝘅 ⋅ 𝘇)/(‖𝘅‖‖𝘇‖)           substitute for r
#     # s = r𝘅 ⋅ 𝘆/‖𝘅‖
#     ϵ = T(5e-6)
#     𝘄 = 𝗹₂[1] - 𝗹₁[1]
#     𝘃 = 𝗹₁[2] - 𝗹₁[1]
#     𝘂 = 𝗹₂[2] - 𝗹₂[1]
#     𝘇 = 𝘃 × 𝘂
#     # Note: 0 ≤ 𝘇 ⋅𝘄, and the minimum distance between two lines is d = (𝘇 ⋅ 𝘄 )/‖𝘇‖.
#     # Hence 𝘇 ⋅𝘄 ≈ 0 for the lines to intersect
#     # (https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d)
#     𝘇 ⋅ 𝘄 ≤ T(1e-8) || return (false, Point_3D{T}(0,0,0))
#     𝘅 = 𝘄 × 𝘂
#     𝘆 = 𝘄 × 𝘃
#     r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)
#     s = r*(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅)
#     return (T(1e-8)^2 < abs(𝘇 ⋅ 𝘇) && -ϵ ≤ r && r ≤ 1 + ϵ && -ϵ ≤ s && s ≤ 1 + ϵ, 𝗹₂(s)) # (hit, point)
# end
# 
# function Base.intersect(𝗹₁::LineSegment_2D{T}, 𝗹₂::LineSegment_2D{T}) where {T}
#     # NOTE: Doesn't work for colinear/parallel lines. (𝘃 × 𝘂 = 𝟬). Also, the cross product
#     # operator for 2D points returns a scalar, since the cross product of two vectors in the
#     # plane is a vector of the form (0, 0, z).
#     # 
#     # From the 3D intersection routine we know:
#     # r = 𝘅 ⋅ 𝘇/𝘇 ⋅ 𝘇 
#     # s = (𝘅 ⋅ 𝘆)(𝘅 ⋅ 𝘇)/(‖𝘅‖‖𝘇‖) 
#     # Since the 2D cross product returns a vector of the form (0, 0, z), the dot products are 
#     # essentially scalar multiplication
#     # r = 𝘅 ⋅ 𝘇/𝘇 ⋅ 𝘇 = x₃/z₃ 
#     # s = (𝘅 ⋅ 𝘆)(𝘅 ⋅ 𝘇)/(‖𝘅‖‖𝘇‖) = y₃/z₃ 
#     #
#     # Simply evaluating everything removes branches and is faster than failing early with
#     # 1e-8 < abs(c) or delaying division by (𝘃 × 𝘂) ⋅𝗸̂ and testing against r and s's numerators.
#     # This has been tested.
#     ϵ = T(5e-6)
#     𝘄 = 𝗹₂[1] - 𝗹₁[1]
#     𝘃 = 𝗹₁[2] - 𝗹₁[1]
#     𝘂 = 𝗹₂[2] - 𝗹₂[1]
#     z = 𝘃 × 𝘂
#     r = 𝘄  × 𝘂/z
#     s = 𝘄  × 𝘃/z
#     # -ϵ ≤ r ≤ 1 + ϵ introduces a branch, but -ϵ ≤ r && r ≤ 1 + ϵ doesn't for some reason.
#     return (T(1e-8) < abs(c) && -ϵ ≤ r && r ≤ 1 + ϵ && -ϵ ≤ s && s ≤ 1 + ϵ, 𝗹₂(s)) # (hit, point)
# end
# 
# # Return if the point is left of the line segment
# #   𝗽    ^
# #   ^   /
# # 𝘃 |  / 𝘂
# #   | /
# #   o
# #   We rely on 𝘃 × 𝘂 = ‖𝘃‖‖𝘂‖sin(θ). We may determine if θ ∈ (0, π] based on the sign of𝘃 × 𝘂
# @inline function isleft(p::Point_2D, l::LineSegment_2D)
#     u = l[2] - l[1]
#     v = p - l[1]
#     return u × v >= 0
# end
# 
# # A random line within [0, 1] × [0, 1]
# function Base.rand(::Type{LineSegment{N,F}}) where {N,F} 
#     return LineSegment{N,F}(rand(Point{N,F}, 2))
# end
# 
# # N random lines within [0, 1] × [0, 1]
# function Base.rand(::Type{LineSegment{N,F}}, NP::Int64) where {N,F}
#     return [ rand(LineSegment{N,F}) for i ∈ 1:NP ]
# end
# 
# # # Plot
# # # -------------------------------------------------------------------------------------------------
# # if enable_visualization
# #     function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment_2D)
# #         return convert_arguments(LS, [l[1], l[2]])
# #     end
# #
# #     function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment_2D})
# #         return convert_arguments(LS, reduce(vcat, [[l[1], l[2]] for l in L]))
# #     end
# # end
