# Doesn't work for colinear/parallel lines. (š Ć š = š¬).
# For lā(r) = Pā + rš and lā(s) = Pā + sš
# 1) Pā + rš = Pā + sš                  subtracting Pā from both sides
# 2) rš = (Pā-Pā) + sš                  š = Pā-Pā
# 3) rš = š + sš                        cross product with š (distributive)
# 4) r(š Ć š) = š Ć š + s(š Ć š)        š Ć š = š¬
# 5) r(š Ć š) = š Ć š                   let š Ć š = š and š Ć š = š
# 6) rš = š                             dot product š to each side
# 7) rš ā š = š ā š                     divide by š ā š
# 8) r = (š ā š)/(š ā š)
# We need to ensure r, s ā [0, 1], hence we need to solve for s too.
# 1) Pā + sš = Pā + rš                     subtracting Pā from both sides
# 2) sš = -š + rš                          cross product with š
# 3) s(š Ć š) = -š Ć š + r(š Ć š)          š Ć š = š¬ 
# 4) s(š Ć š) = r(š Ć š)                   using š Ć š = -(š Ć š), likewise for š Ć š
# 5) s(š Ć š) = r(š Ć š)                   let š Ć š = š. use š Ć š = š
# 6) sš = rš                               dot product š to each side
# 7) s(š ā š) = r(š ā š)                   divide by (š ā š)
# 9) s = r(š ā š)/(š ā š)
# The cross product of two vectors in the plane is a vector of the form (0, 0, k),
# hence, in 2D:
# r = (š ā š)/(š ā š) = xā/zā 
# s = r(š ā š)/(š ā š) = yā/zā 
function Base.intersect(lā::LineSegment{Point{2, T}},
                        lā::LineSegment{Point{2, T}}) where {T}
    š = lā[1] - lā[1]
    šā = lā[2] - lā[1]
    šā = lā[2] - lā[1]
    z = šā Ć šā
    r = (š Ć šā) / z
    s = (š Ć šā) / z
    valid = 0 ā¤ r && r ā¤ 1 && 0 ā¤ s && s ā¤ 1
    return valid ? lā(s) : Point{2, T}(INF_POINT, INF_POINT)
end
