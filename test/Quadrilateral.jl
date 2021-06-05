using MOCNeutronTransport
@testset "Quadrilateral" begin
    for type in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point( type(0) )
            p₂ = Point( type(1) )
            p₃ = Point( type(1), type(1) )
            p₄ = Point( type(0), type(1) )
            quad = Quadrilateral((p₁, p₂, p₃, p₄))
            @test quad.points == (p₁, p₂, p₃, p₄)

            # single constructor
            quad = Quadrilateral(p₁, p₂, p₃, p₄)
            @test quad.points == (p₁, p₂, p₃, p₄)
        end

        @testset "Methods" begin
            p₁ = Point( type(0) )
            p₂ = Point( type(1) )
            p₃ = Point( type(1), type(1) )
            p₄ = Point( type(0), type(1) )
            quad = Quadrilateral((p₁, p₂, p₃, p₄))

            # evaluation
            quad( type(0), type(0)) ≈ p₁
            quad( type(1), type(0)) ≈ p₂
            quad( type(0), type(1)) ≈ p₃
            quad( type(1), type(1)) ≈ p₄
            quad( type(1//2), type(1//2)) ≈ Point(type(1//2), type(1//2))

            # area
            a = area(quad)
            @test typeof(a) == typeof(type(1))
            @test a == type(1)

            # intersect
            # line is not coplanar with Quadrilateral
            p₄ = Point(type.((0.9, 0.1, -5)))
            p₅ = Point(type.((0.9, 0.1, 5)))
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test bool
            @test point ≈ Point(type.((0.9, 0.1, 0.0)))

            # line is coplanar with Quadrilateral
            p₄ = Point(type.((0.5, -1)))
            p₅ = Point(type.((0.5, 2)))
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # no intersection non-coplanar
            p₄ = Point(type.((2.0, 0.1, -5)))
            p₅ = Point(type.((2.0, 0.1, 5)))
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # no intersection coplanar
            p₄ = Point(type.((2.0, -1)))
            p₅ = Point(type.((2.0, 2)))
            l = LineSegment(p₄, p₅)
            bool, point = intersect(l, quad)
            @test !bool

            # in
            p = Point(type.((0.5, 0.1)))
            @test p ∈  quad
            p = Point(type.((0.5, 0.0)))
            @test p ∈  quad
            p = Point(type.((0.5, 0.1, 0.1)))
            @test p ∉ quad
            p = Point(type.((0.5, -0.1, 0.1)))
            @test p ∉ quad
            p = Point(type.((0.5, -0.1, 0.0)))
            @test p ∉ quad
        end
    end
end
