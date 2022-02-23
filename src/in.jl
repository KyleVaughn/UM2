# Is left 
# ---------------------------------------------------------------------------------------------
# If the point is left of the line segment in the 2D plane. 
#
# The segment's direction is from 𝘅₁ to 𝘅₂. Let 𝘂 = 𝘅₂ - 𝘅₁ and 𝘃 = 𝗽 - 𝘅₁ 
# We may determine if the angle θ between the point and segment is in [0, π] based on the 
# sign of 𝘂 × 𝘃, since 𝘂 × 𝘃 = ‖𝘂‖‖𝘃‖sin(θ). 
#   𝗽    ^
#   ^   /
# 𝘃 |  / 𝘂
#   | /
#   o
# We allow points on the line (𝘂 × 𝘃 = 0) to be left, since this test is primarily 
# used to determine if a point is inside a polygon. A mesh is supposed to partition
# its domain, so if we do not allow points on the line, there will exist points in the 
# mesh which will not be in any face.
@inline function isleft(p::Point2D, l::LineSegment2D)
    return 0 ≤ l.𝘂 × (p - l.𝘅₁)
end

