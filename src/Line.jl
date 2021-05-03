# Constructors
# -------------------------------------------------------------------------------------------------
struct Line
  p1::Point
  p2::Point
end

# Base methods
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::Line) = Ref(l)

# Methods
# -------------------------------------------------------------------------------------------------
distance(l::Line) = distance(l.p1, l.p2)
(l::Line)(t) = l.p1 + t * (l.p2 - l.p1)
midpoint(l::Line) = l(0.5)
# intersect line
# intersect quad
# point in
# point isleft
