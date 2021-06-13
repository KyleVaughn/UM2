using MOCNeutronTransport
@testset "Triangle" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point(T, 0)
            p₂ = Point(T, 1)
            p₃ = Point(T, 1, 1)
            tri = Triangle((p₁, p₂, p₃))
            @test tri.points == (p₁, p₂, p₃)

            # single constructor
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == (p₁, p₂, p₃)
        end

        @testset "Methods" begin
            p₁ = Point(T, 0)
            p₂ = Point(T, 1)
            p₃ = Point(T, 1, 1)
            tri = Triangle((p₁, p₂, p₃))

            # interpolation
            tri(0, 0) ≈ p₁
            tri(1, 0) ≈ p₂
            tri(0, 1) ≈ p₃
            tri(1//2, 1//2) ≈ Point(T, 1//2, 1//2)

            # area
            a = area(tri)
            @test typeof(a) == typeof(T(1))
            @test a == T(1//2)

            # intersect
            # line is not coplanar with triangle
            p₄ = Point(T, 9//10, 1//10, -5)
            p₅ = Point(T, 9//10, 1//10,  5)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, tri)
            @test bool
            @test point ≈ Point(T, 9//10, 1//10,  0)

            # line is coplanar with triangle
            p₄ = Point(T, 1//2, -1)
            p₅ = Point(T, 1//2,  2)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, tri)
            @test !bool

            # no intersection non-coplanar
            p₄ = Point(T, 2, 1//10, -5)
            p₅ = Point(T, 2, 1//10,  5)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, tri)
            @test !bool

            # no intersection coplanar
            p₄ = Point(T, 2.0, -1)
            p₅ = Point(T, 2.0,  2)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, tri)
            @test !bool

            # in
            p = Point(T, 1//2, 1//10)
            @test p ∈  tri
            p = Point(T, 1//2, 0)
            @test p ∈  tri
            p = Point(T, 1//2, 1//10, 1/10)
            @test p ∉ tri
            p = Point(T, 1//2, -1//10, -1//10)
            @test p ∉ tri
            p = Point(T, 1//2, -1//10, -1//10)
            @test p ∉ tri
        end
    end
end
