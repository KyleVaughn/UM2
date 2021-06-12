using MOCNeutronTransport
@testset "Quadrilateral8" begin
    for type in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point( type(0) )
            p₂ = Point( type(1) )
            p₃ = Point( type(1), type(1) )
            p₄ = Point( type(0), type(1) )
            p₅ = Point( type(1//2), type(   0) )
            p₆ = Point( type(   1), type(1//2) )
            p₇ = Point( type(1//2), type(   1) )
            p₈ = Point( type(   0), type(1//2) )
            quad8 = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8.points == (p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)

            # single constructor
            quad8 = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8.points == (p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
        end

        @testset "Methods" begin
            p₁ = Point( type(0) )
            p₂ = Point( type(1) )
            p₃ = Point( type(1), type(1) )
            p₄ = Point( type(0), type(1) )
            p₅ = Point( type(1//2), type(   0) )
            p₆ = Point( type(   1), type(1//2) )
            p₇ = Point( type(1//2), type(   1) )
            p₈ = Point( type(   0), type(1//2) )
            quad8 = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

            # interpolation
            @test quad8(type(0), type(0) ) ≈ p₁
            @test quad8(type(1), type(0) ) ≈ p₂
            @test quad8(type(1), type(1) ) ≈ p₃
            @test quad8(type(0), type(1) ) ≈ p₄
            @test quad8(type(1//2), type(   0) ) ≈ p₅
            @test quad8(type(   1), type(1//2) ) ≈ p₆
            @test quad8(type(1//2), type(   1) ) ≈ p₇
            @test quad8(type(   0), type(1//2) ) ≈ p₈
            @test quad8(type(1//2), type(1//2) ) ≈ Point(type(1//2), type(1//2))          

            # area
            p₁ = Point( type(0) )
            p₂ = Point( type(2) )
            p₃ = Point( type(2), type(3) )
            p₄ = Point( type(0), type(3) )
            p₅ = Point( type(3//2), type(1//2) )
            p₆ = Point( type(5//2), type(3//2) )
            p₇ = Point( type(3//2), type(5//2) )
            p₈ = Point( type(0), type(3//2) )
            quad8 = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

            # 2D default
            @test isapprox(area(quad8; N = 4), 17//3, atol=1.0e-6)
            # 3D default
            p₈ = Point( type(0), type(3//2), type(2) )
            quad8 = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test isapprox(area(quad8; N = 15), 10.09431, atol=1.0e-4)

            # intersect
            p₁ = Point( type(0) )
            p₂ = Point( type(2) )
            p₃ = Point( type(2), type(3) )
            p₄ = Point( type(0), type(3) )
            p₅ = Point( type(3//2), type(1//2) )
            p₆ = Point( type(5//2), type(3//2) )
            p₇ = Point( type(3//2), type(5//2) )
            p₈ = Point( type(0), type(3//2) )
            quad8 = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            l = LineSegment(Point( type(1), type(3//2), type(-2)),
                            Point( type(1), type(3//2), type(2)))
            intersection = l ∩ quad8

            # 1 intersection
            @test intersection[1]
            @test intersection[2] == 1
            @test intersection[3][1] ≈ Point( type(1), type(3//2), type(0))

            # 0 intersection
            l = LineSegment(Point( type(1), type(0), type(-2)),
                            Point( type(1), type(0), type(2)))
            intersection = l ∩ quad8

            # 2 intersection
            @test !intersection[1]

            # 2 intersections
            p₈ = Point( type(0), type(3//2), type(2) )
            quad8 = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            l = LineSegment(Point( type(1), type(0), type(1//2)),
                            Point( type(1), type(3), type(1//2)))
            intersection = l ∩ quad8
        end
    end
end
