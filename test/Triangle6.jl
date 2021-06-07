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

            # 2D default
            @test isapprox(area(tri6; N = 12), 3, atol=1.0e-6)
            # 3D default
            p₂ = Point( type(2), type(0), type(3) )
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))
            @test isapprox(area(tri6; N = 79), 6.328781460309, atol=1.0e-6)

            # intersect
            # in
        end
    end
end
