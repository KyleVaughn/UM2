export polynomial_coeffs
 
# parametric dimension 1
# P(r) = P₁ + P₂r + P₃r² + ....
polynomial_coeffs(l::LineSegment) = (l[1], l[2] - l[1])
function polynomial_coeffs(q::QuadraticSegment)
    a = q[1] - q[3]
    b = q[2] - q[3]
    return(q[1], -3a - b, 2(a + b)) 
end
