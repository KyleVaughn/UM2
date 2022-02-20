# Specialized methods for a Quadrilateral, aka Polygon{4}
(quad::Quadrilateral)(r, s) = Point(((1 - r)*(1 - s))quad[1] + (r*(1 - s))quad[2] + 
                                                (r*s)quad[3] + ((1 - r)*s)quad[4])
