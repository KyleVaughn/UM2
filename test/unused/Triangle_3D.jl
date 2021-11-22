using MOCNeutronTransport
@testset "Triangle_3D" begin
    for T in [Float32, Float64]
        @testset "Constructors" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            tri = Triangle_3D((p₁, p₂, p₃))
            @test tri.points == (p₁, p₂, p₃)

            # single constructor
            tri = Triangle_3D(p₁, p₂, p₃)
            @test tri.points == (p₁, p₂, p₃)
        end

        @testset "Methods" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            tri = Triangle_3D((p₁, p₂, p₃))

            # interpolation
            tri(0, 0) ≈ p₁
            tri(1, 0) ≈ p₂
            tri(0, 1) ≈ p₃
            tri(1//2, 1//2) ≈ Point_3D(T, 1//2, 1//2)

            # area
            a = area(tri)
            @test typeof(a) == typeof(T(1))
            @test a == T(1//2)

            # intersect
            # line is not coplanar with triangle
            p₄ = Point_3D(T, 9//10, 1//10, -5)
            p₅ = Point_3D(T, 9//10, 1//10,  5)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, tri)
            @test bool
            @test point ≈ Point_3D(T, 9//10, 1//10,  0)

            # line is coplanar with triangle
            p₄ = Point_3D(T, 1//2, -1)
            p₅ = Point_3D(T, 1//2,  2)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, tri)
            @test !bool

            # no intersection non-coplanar
            p₄ = Point_3D(T, 2, 1//10, -5)
            p₅ = Point_3D(T, 2, 1//10,  5)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, tri)
            @test !bool

            # no intersection coplanar
            p₄ = Point_3D(T, 2, -1)
            p₅ = Point_3D(T, 2,  2)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, tri)
            @test !bool
        end
    end
end
