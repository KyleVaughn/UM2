# union
r1 = AABox2D(Point2D{T}(0, 0), Point2D{T}(2, 2))
r2 = AABox2D(Point2D{T}(1, 1), Point2D{T}(4, 4))
r3 = r1 ∪ r2
@test r3.xmin ≈ 0
@test r3.ymin ≈ 0
@test r3.xmax ≈ 4
@test r3.ymax ≈ 4
r1 = AABox2D(Point2D{T}(0, 0), Point2D{T}(2, 2))
r2 = AABox2D(Point2D{T}(1, 1), Point2D{T}(3, 3))
r3 = AABox2D(Point2D{T}(2, 2), Point2D{T}(4, 4))
r4 = union([r1, r2, r3])
@test r4.xmin ≈ 0
@test r4.ymin ≈ 0
@test r4.xmax ≈ 4
@test r4.ymax ≈ 4

#        # union
#        r1 = AABox3D(Point3D{T}(0, 0), Point3D{T}(2, 2)) 
#        r2 = AABox3D(Point3D{T}(1, 1), Point3D{T}(4, 4)) 
#        r3 = r1 ∪ r2
#        @test r3.xmin ≈ 0
#        @test r3.ymin ≈ 0
#        @test r3.xmax ≈ 4
#        @test r3.ymax ≈ 4
