# Intersection of two 2D or 3D line segments
#
# Doesn't work for colinear/parallel lines. (𝘂 × 𝘃 = 𝟬).
# For 𝗹₁(r) = 𝘅₁ + r𝘂 and 𝗹₂(s) = 𝘅₂ + s𝘃
# 1) 𝘅₁ + r𝘂 = 𝘅₂ + s𝘃                  subtracting 𝘅₁ from both sides
# 2) r𝘂 = (𝘅₂-𝘅₁) + s𝘃                  𝘄 = 𝘅₂-𝘅₁
# 3) r𝘂 = 𝘄 + s𝘃                        cross product with 𝘃 (distributive)
# 4) r(𝘂 × 𝘃) = 𝘄 × 𝘃 + s(𝘃 × 𝘃)        𝘃 × 𝘃 = 𝟬
# 5) r(𝘂 × 𝘃) = 𝘄 × 𝘃                   let 𝘄 × 𝘃 = 𝘅 and 𝘂 × 𝘃 = 𝘇
# 6) r𝘇 = 𝘅                             dot product 𝘇 to each side
# 7) r𝘇 ⋅ 𝘇 = 𝘅 ⋅ 𝘇                     divide by 𝘇 ⋅ 𝘇
# 8) r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)
# We need to ensure r, s ∈ [0, 1], hence we need to solve for s too.
# 1) 𝘅₂ + s𝘃 = 𝘅₁ + r𝘂                     subtracting 𝘅₂ from both sides
# 2) s𝘃 = -𝘄 + r𝘂                          cross product with 𝘄
# 3) s(𝘃 × 𝘄) = -𝘄 × 𝘄 + r(𝘂 × 𝘄)          𝘄 × 𝘄 = 𝟬 
# 4) s(𝘃 × 𝘄) = r(𝘂 × 𝘄)                   using 𝘂 × 𝘄 = -(𝘄 × 𝘂), likewise for 𝘃 × 𝘄
# 5) s(𝘄 × 𝘃) = r(𝘄 × 𝘂)                   let 𝘄 × 𝘂 = 𝘆. use 𝘄 × 𝘃 = 𝘅
# 6) s𝘅 = r𝘆                               dot product 𝘅 to each side
# 7) s(𝘅 ⋅ 𝘅) = r(𝘆 ⋅ 𝘅)                   divide by (𝘅 ⋅ 𝘅)
# 9) s = r(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅)
# The cross product of two vectors in the plane is a vector of the form (0, 0, k),
# hence, in 2D:
# r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇) = x₃/z₃ 
# s = r(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅) = y₃/z₃ 

function Base.intersect(l₁::LineSegment{T}, l₂::LineSegment{T}) where {T} 
    p = nan(T)
    𝘄 = l₂[1] - l₁[1]
    𝘂₁= l₁[2] - l₁[1] 
    𝘂₂= l₂[2] - l₂[1] 
    z = 𝘂₁ × 𝘂₂
    r = (𝘄 × 𝘂₂)/z
    s = (𝘄 × 𝘂₁)/z
    if 0 ≤ r ≤ 1 && 0 ≤ s ≤ 1
        p = l₂(s)
    end
    return p 
end

# This version is branchless, but about 8% slower in basic CPU benchmarking
# Should this be using for GPU? How does it perform for multithread?
#function intersect(l₁::LineSegment2D{T}, l₂::LineSegment2D{T}) where {T}
#    ϵ = 5e-6 # Tolerance on r,s ∈ [-ϵ, 1 + ϵ]
#    𝘄 = l₂.𝘅₁ - l₁.𝘅₁
#    z = l₁.𝘂 × l₂.𝘂
#    r = (𝘄 × l₂.𝘂)/z
#    s = (𝘄 × l₁.𝘂)/z
#    return (1e-8 < abs(z) && -ϵ ≤ r && r ≤ 1 + ϵ 
#                          && -ϵ ≤ s && r ≤ 1 + ϵ, l₂(s)) # (hit, point)
#end

# function intersect(l₁::LineSegment3D{T}, l₂::LineSegment3D{T}) where {T}
#     ϵ = 5e-6 # Tolerance on r,s ∈ [-ϵ, 1 + ϵ]
#     𝘂 = l₁.𝘂
#     𝘃 = l₂.𝘂
#     𝘇 = 𝘂 × 𝘃
#     𝘄 = l₂.𝘅₁ - l₁.𝘅₁
#     𝘅 = 𝘄 × 𝘃
#     𝘆 = 𝘄 × 𝘂
#     r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)
#     s = r*(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅)
#     return (1e-16 < norm²(𝘇) && -ϵ ≤ r ≤ 1 + ϵ
#                              && -ϵ ≤ s ≤ 1 + ϵ, l₂(s)) # (hit, point)
# end
