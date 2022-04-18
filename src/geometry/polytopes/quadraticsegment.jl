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
    return  1e4*norm²(q[1] - q[3] + q[2] - q[3]) < norm²(q[2] - q[1])
end
