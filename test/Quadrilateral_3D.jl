using MOCNeutronTransport
@testset "Quadrilateral_3D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            p₄ = Point_3D(T, 0, 1)
            quad = Quadrilateral_3D((p₁, p₂, p₃, p₄))
            @test quad.points == (p₁, p₂, p₃, p₄)

            # single constructor
            quad = Quadrilateral_3D(p₁, p₂, p₃, p₄)
            @test quad.points == (p₁, p₂, p₃, p₄)
        end

        @testset "Methods" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            p₄ = Point_3D(T, 0, 1)
            quad = Quadrilateral_3D((p₁, p₂, p₃, p₄))

            # interpolation
            quad(0, 0) ≈ p₁
            quad(1, 0) ≈ p₂
            quad(0, 1) ≈ p₃
            quad(1, 1) ≈ p₄
            quad(1//2, 1//2) ≈ Point_3D(T, 1//2, 1//2)

            # area
            a = area(quad)
            @test typeof(a) == typeof(T(1))
            @test a == T(1)

            # intersect
            # line is not coplanar with Quadrilateral_3D
            p₄ = Point_3D(T, 9//10, 1//10, -5)
            p₅ = Point_3D(T, 9//10, 1//10,  5)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, quad)
            @test bool
            @test point ≈ Point_3D(T, 9//10, 1//10, 0)

            # line is coplanar with Quadrilateral_3D
            p₄ = Point_3D(T, 1//2, -1)
            p₅ = Point_3D(T, 1//2,  2)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # no intersection non-coplanar
            p₄ = Point_3D(T, 2, 1//10, -5)
            p₅ = Point_3D(T, 2, 1//10,  5)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # no intersection coplanar
            p₄ = Point_3D(T, 2, -1)
            p₅ = Point_3D(T, 2,  2)
            l = LineSegment_3D(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # in
            p = Point_3D(T, 1//2, 1//10)
            @test p ∈  quad
            p = Point_3D(T, 1//2, 0)
            @test p ∈  quad
            p = Point_3D(T, 1//2, 1//10, 1//10)
            @test p ∉ quad
            p = Point_3D(T, 1//2, -1//10, 1//10)
            @test p ∉ quad
            p = Point_3D(T, 1//2, -1//10, 0)
            @test p ∉ quad
        end
    end
end
