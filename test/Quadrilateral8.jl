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


        end
    end
end
