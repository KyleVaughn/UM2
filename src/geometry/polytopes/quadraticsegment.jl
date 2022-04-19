export isstraight
# The segment satisfies the equation Q(r) = P₁ + r𝘂 + r²𝘃, where 
# - 𝘂 = -(3𝗽₁ + 𝗽₂ - 4𝗽₃) and 𝘃 = 2(𝗽₁ + 𝗽₂ - 2𝗽₃),
# - 𝗽ᵢ = Pᵢ - O, for i = 1:3, where O is the origin, 
# - r ∈ [0, 1]
"""
    isstraight(q::QuadraticSegment)

Return if the quadratic segment is effectively straight.
"""
function isstraight(q::QuadraticSegment)
    # Project P₃ onto the line from P₁ to P₂, call it P₄
    𝘃₁₃ = q[3] - q[1] 
    𝘃₁₂ = q[2] - q[1] 
    v₁₂ = norm²(𝘃₁₂)
    𝘃₁₄ = (𝘃₁₃ ⋅ 𝘃₁₂)*inv(v₁₂)*𝘃₁₂
    # Determine the distance from P₃ to P₄ (P₄ - P₃ = P₁ + 𝘃₁₄ - P₃ = 𝘃₁₄ - 𝘃₁₃)
    d = norm(𝘃₁₄ - 𝘃₁₃) 
    return d < ϵ_Point
end
