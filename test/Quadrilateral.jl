using MOCNeutronTransport
@testset "Quadrilateral" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point(T, 0)
            p₂ = Point(T, 1)
            p₃ = Point(T, 1, 1)
            p₄ = Point(T, 0, 1)
            quad = Quadrilateral((p₁, p₂, p₃, p₄))
            @test quad.points == (p₁, p₂, p₃, p₄)

            # single constructor
            quad = Quadrilateral(p₁, p₂, p₃, p₄)
            @test quad.points == (p₁, p₂, p₃, p₄)
        end

        @testset "Methods" begin
            p₁ = Point(T, 0)
            p₂ = Point(T, 1)
            p₃ = Point(T, 1, 1)
            p₄ = Point(T, 0, 1)
            quad = Quadrilateral((p₁, p₂, p₃, p₄))

            # interpolation
            quad(0, 0) ≈ p₁
            quad(1, 0) ≈ p₂
            quad(0, 1) ≈ p₃
            quad(1, 1) ≈ p₄
            quad(1//2, 1//2) ≈ Point(T, 1//2, 1//2)

            # area
            a = area(quad)
            @test typeof(a) == typeof(T(1))
            @test a == T(1)

            # intersect
            # line is not coplanar with Quadrilateral
            p₄ = Point(T, 9//10, 1//10, -5)
            p₅ = Point(T, 9//10, 1//10,  5)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test bool
            @test point ≈ Point(T, 9//10, 1//10, 0)

            # line is coplanar with Quadrilateral
            p₄ = Point(T, 1//2, -1)
            p₅ = Point(T, 1//2,  2)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # no intersection non-coplanar
            p₄ = Point(T, 2, 1//10, -5)
            p₅ = Point(T, 2, 1//10,  5)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # no intersection coplanar
            p₄ = Point(T, 2, -1)
            p₅ = Point(T, 2,  2)
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # in
            p = Point(T, 1//2, 1//10)
            @test p ∈  quad
            p = Point(T, 1//2, 0)
            @test p ∈  quad
            p = Point(T, 1//2, 1//10, 1//10)
            @test p ∉ quad
            p = Point(T, 1//2, -1//10, 1//10)
            @test p ∉ quad
            p = Point(T, 1//2, -1//10, 0)
            @test p ∉ quad
        end
    end
end
