# Return the Jacobian of q, evalutated at r
# 𝗾′(r) = 2r𝘂 + 𝘃, which is simplified to below.
jacobian(q::QuadraticSegment, r) = (4r - 3)*(q.𝘅₁ - q.𝘅₃) + (4r - 1)*(q.𝘅₂ - q.𝘅₃) 
