using MOCNeutronTransport
@testset "Triangle6" begin
    for type in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point( type(0) )
            p₂ = Point( type(1) )
            p₃ = Point( type(1), type(1) )
            p₄ = Point( type(1)/type(2))
            p₅ = Point( type(1), type(1)/type(2))
            p₆ = Point( type(1)/type(2), type(1)/type(2))
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)

            # single constructor
            tri6 = Triangle6(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)
        end

        @testset "Methods" begin
            p₁ = Point( type(0) )
            p₂ = Point( type(1) )
            p₃ = Point( type(1), type(1) )
            p₄ = Point( type(1)/type(2))
            p₅ = Point( type(1), type(1)/type(2))
            p₆ = Point( type(1)/type(2), type(1)/type(2))
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))

            # evaluation
            tri6( type(0), type(0)) ≈ p₁
            tri6( type(1), type(0)) ≈ p₂
            tri6( type(0), type(1)) ≈ p₃
            tri6( type(1)/type(2), type(0)) ≈ p₄
            tri6( type(1)/type(2), type(1)/type(2)) ≈ p₅
            tri6( type(0), type(1)/type(2)) ≈ p₆
            tri6( type(1//2), type(1//2)) ≈ Point(type(1//2), type(1//2))

            # area
            p₁ = Point( type(0) )
            p₂ = Point( type(2) )
            p₃ = Point( type(2), type(2) )
            p₄ = Point( type(1), type(1)/type(4) )
            p₅ = Point( type(3), type(1) )
            p₆ = Point( type(1), type(1) )
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))
            #Integrates to 3.0
















#            # area
#            a = area(tri)
#            @test typeof(a) == typeof(type(1))
#            @test a == type(1)/type(2)
#
#            # intersect
#            # line is not coplanar with triangle
#            p₄ = Point(type.((0.9, 0.1, -5)))
#            p₅ = Point(type.((0.9, 0.1, 5)))
#            l = LineSegment(p₄, p₅)
#            bool, point = intersect(l, tri)
#            @test bool
#            @test point ≈ Point(type.((0.9, 0.1, 0.0)))
#
#            # line is coplanar with triangle
#            p₄ = Point(type.((0.5, -1)))
#            p₅ = Point(type.((0.5, 2)))
#            l = LineSegment(p₄, p₅)
#            bool, point = intersect(l, tri)
#            @test !bool
#
#            # no intersection non-coplanar
#            p₄ = Point(type.((2.0, 0.1, -5)))
#            p₅ = Point(type.((2.0, 0.1, 5)))
#            l = LineSegment(p₄, p₅)
#            bool, point = intersect(l, tri)
#            @test !bool
#
#            # no intersection coplanar
#            p₄ = Point(type.((2.0, -1)))
#            p₅ = Point(type.((2.0, 2)))
#            l = LineSegment(p₄, p₅)
#            bool, point = intersect(l, tri)
#            @test !bool
#
#            # in
#            p = Point(type.((0.5, 0.1)))
#            @test p ∈  tri
#            p = Point(type.((0.5, 0.0)))
#            @test p ∈  tri
#            p = Point(type.((0.5, 0.1, 0.1)))
#            @test p ∉ tri
#            p = Point(type.((0.5, -0.1, 0.1)))
#            @test p ∉ tri
#            p = Point(type.((0.5, -0.1, 0.0)))
#            @test p ∉ tri
        end
    end
end
