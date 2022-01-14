using MOCNeutronTransport
@testset "QuadraticSegment_2D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # Constructor
            x⃗₁ = Point_2D{F}(0, 0)
            x⃗₂ = Point_2D{F}(2, 0)
            x⃗₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == SVector(x⃗₁, x⃗₂, x⃗₃)
        end

        @testset "Methods" begin
            # interpolation
#             x⃗₁ = Point_2D{F}(0, 0)
#             x⃗₂ = Point_2D{F}(2, 0)
#             x⃗₃ = Point_2D{F}(1, 1)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             for r = LinRange{F}(0, 1, 11)
#                 @test q(r) ≈ Point_2D{F}(2r, -(2r)^2 + 4r)
#             end
# 
#             # gradient
#             for r = LinRange{F}(0, 1, 11)
#                 @test ∇(q, r) ≈ Point_2D{F}(2, -(8r) + 4)
#             end
# 
#             # arclength
#             x⃗₁ = Point_2D{F}(0, 0)
#             x⃗₂ = Point_2D{F}(2, 0)
#             x⃗₃ = Point_2D{F}(1, 0)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             # straight edge
#             @test abs(arclength(q) - 2) < 1.0e-6
#             # curved
#             x⃗₃ = Point_2D{F}(1, 1)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             @test abs(arclength(q) - 2.9578857151786138) < 1.0e-6
# 
#             # boundingbox
#             x⃗₁ = Point_2D{F}(0, 0)
#             x⃗₂ = Point_2D{F}(2, 0)
#             x⃗₃ = Point_2D{F}(1, 1)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             bb = boundingbox(q)
#             @test bb.xmin ≈ 0
#             @test bb.ymin ≈ 0
#             @test bb.xmax ≈ 2
#             @test bb.ymax ≈ 1
#             x⃗₁ = Point_2D{F}(0, 0)
#             x⃗₂ = Point_2D{F}(2, 2)
#             x⃗₃ = Point_2D{F}(1, 1)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             bb = boundingbox(q)
#             @test bb.xmin ≈ 0
#             @test bb.ymin ≈ 0
#             @test bb.xmax ≈ 2
#             @test bb.ymax ≈ 2
#             x⃗₁ = Point_2D{F}(0, 0)
#             x⃗₂ = Point_2D{F}(2, 0)
#             x⃗₃ = Point_2D{F}(2.1, 1)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             bb = boundingbox(q)
#             @test bb.xmin ≈ 0
#             @test bb.ymin ≈ 0
#             @test bb.xmax ≈ 2.3272727272727276
#             @test bb.ymax ≈ 1
# 
#             # isstraight
#             x⃗₁ = Point_2D{F}(0, 0)
#             x⃗₂ = Point_2D{F}(2, 0)
#             x⃗₃ = Point_2D{F}(1, 0)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             @test isstraight(q)
#             x⃗₂ = Point_2D{F}(2, 0.0001)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             @test !isstraight(q)
# 
#             # laplacian
#             x⃗₁ = Point_2D{F}(0, 0)
#             x⃗₂ = Point_2D{F}(2, 0)
#             x⃗₃ = Point_2D{F}(1, 1)
#             q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
#             p = ∇²(q, 0)
#             @test p.x ≈ 0
#             @test p.y ≈ -8
#             p = ∇²(q, 1)
#             @test p.x ≈ 0
#             @test p.y ≈ -8
# 
# 
# #            # closest_point
# #            x⃗₁ = Point_2D{F}(0, 0)           
# #            x⃗₂ = Point_2D{F}(2, 0)
# #            x⃗₃ = Point_2D{F}(1, 1)
# #            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
# #            p = Point_2D{F}(1, 1.1)
# #            r, p_c = closest_point(p, q)
# #            @test x⃗₃ ≈ p_c
# #            p = Point_2D{F}(-0.1, 0)
# #            r, p_c = closest_point(p, q)
# #            @test x⃗₁ ≈ p_c
# #
# #            # intersect
# #            x⃗₁ = Point_2D{F}(0, 0)
# #            x⃗₂ = Point_2D{F}(2, 0)
# #            x⃗₃ = Point_2D{F}(1, 1)
# #            x⃗₄ = Point_2D{F}(1, 0)
# #            x⃗₅ = Point_2D{F}(1, 2)
# #
# #            # 1 intersection
# #            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
# #            l = LineSegment_2D(x⃗₄, x⃗₅)
# #            npoints, (point1, point2) = intersect(l, q)
# #            @test npoints == 1
# #            @test point1 ≈ Point_2D{F}(1, 1)
# #
# #            # 2 intersections
# #            x⃗₄ = Point_2D{F}(0, 3//4)
# #            x⃗₅ = Point_2D{F}(2, 3//4)
# #            l = LineSegment_2D(x⃗₄, x⃗₅)
# #            npoints, (point1, point2) = l ∩ q
# #            @test npoints == 2
# #            @test point1 ≈ Point_2D{F}(1//2, 3//4)
# #            @test point2 ≈ Point_2D{F}(3//2, 3//4)
# #
# #            # 0 intersections
# #            x⃗₄ = Point_2D{F}(0, 3)
# #            x⃗₅ = Point_2D{F}(2, 3)
# #            l = LineSegment_2D(x⃗₄, x⃗₅)
# #            npoints, (point1, point2) = intersect(l, q)
# #            @test npoints == 0
# #            @test point1 ≈ Point_2D{F}(0)
# #            @test point2 ≈ Point_2D{F}(0)
# #
# #            # is_left
# #            x⃗₁ = Point_2D{F}(0, 0)
# #            x⃗₂ = Point_2D{F}(2, 0)
# #            x⃗₃ = Point_2D{F}(1, 1)
# #            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
# #            @test !is_left(Point_2D{F}(1, 0), q)
# #            @test is_left(Point_2D{F}(1, 2), q)
        end
    end
end
