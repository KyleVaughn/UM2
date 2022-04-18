export polynomial_coeffs
 
# parametric dimension 1
# P(r) = P₁ + P₂r + P₃r² + ....
polynomial_coeffs(l::LineSegment) = (l[1].coords, l[2] - l[1])
function polynomial_coeffs(q::QuadraticSegment)
    𝗮 = q[1] - q[3]
    𝗯 = q[2] - q[3]
    return(q[1].coords, -3𝗮 - 𝗯, 2(𝗮 + 𝗯)) 
end
