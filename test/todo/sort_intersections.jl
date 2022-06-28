# isleft
l = LineSegment2D(Point2D{T}(0, 0), Point2D{T}(1, 0))
@test isleft(Point2D{T}(0, 1), l)
@test !isleft(Point2D{T}(0, -1), l)
@test !isleft(Point2D{T}(0, -1e-6), l)
@test isleft(Point2D{T}(0, 1e-6), l)
@test isleft(Point2D{T}(0.5, 0), l)

# sort_intersection_points
l = LineSegment3D(Point3D{T}(0, 0, 0), Point3D{T}(10, 0, 0))
p₁ = Point3D{T}(1, 0, 0)
p₂ = Point3D{T}(2, 0, 0)
p₃ = Point3D{T}(3, 0, 0)
points = [p₃, p₁, p₂, Point3D{T}(1 + 1 // 1000000, 0, 0)]
sort_intersection_points!(l, points)
@test points[1] == p₁
@test points[2] == p₂
@test points[3] == p₃
