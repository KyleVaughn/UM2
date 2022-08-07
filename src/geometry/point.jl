export Point,
       Point2,
       Point2f,
       Point2d,
       Point2b,
       Point3f,
       Point3d,
       Point3b,
       EPS_POINT,
       EPS_POINT2,
       INF_POINT

# Points separated by 1e-5 cm = 0.1 micron are treated the same.
const EPS_POINT = 1e-5
const EPS_POINT2 = EPS_POINT * EPS_POINT

# Default coordinate for a point that is essentially infinitely far away.
# Used for when IEEE 754 may not be enforced, such as with fast math. 
const INF_POINT = 1e10

# -- Type aliases --

const Point = Vec
const Point2 = Vec{2}
const Point2f = Vec2f
const Point2d = Vec2d
const Point2b = Vec2b
