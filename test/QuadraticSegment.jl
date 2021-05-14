using MOCNeutronTransport
@testset "QuadraticSegment" begin
    for type in [Float32, Float64, BigFloat]
        # Constructor
        x⃗₁ = Point( type.((0, 0, 0)) )
        x⃗₂ = Point( type.((2, 0, 0)) )
        x⃗₃ = Point( type.((1, 1, 0)) )

        # Do x₃ barely off of the line.
        seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        #@test seg.ŷ ≈ Point( type.((0, 1, 0)) )
        #@test seg.a ≈ type(-1)
        #@test seg.b ≈ type(2)
        #@test typeof(seg.ŷ) == typeof(Point( type.((0, 1, 0)) ))
        #@test typeof(seg.a) == typeof(type(-1))
        #@test typeof(seg.b) == typeof(type(2))

        #x⃗₃ = Point(type(1), type(1)/sqrt(type(2)), type(1)/sqrt(type(2))) 
        #seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        #@test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        #@test seg.ŷ ≈ Point( type(0), type(1)/sqrt(type(2)), type(1)/sqrt(type(2)) )
        #@test seg.a ≈ type(-1)
        #@test seg.b ≈ type(2)
        #@test typeof(seg.ŷ) == typeof(Point( type(0), type(1)/sqrt(type(2)), type(1)/sqrt(type(2)) ))
        #@test typeof(seg.a) == typeof(type(-1))
        #@test typeof(seg.b) == typeof(type(2))

        #x⃗₃ = Point( type.((1, 0, 0)) )
        #seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        #@test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        #@test seg.ŷ ≈ Point( type.((0, 0, 0)) )
        #@test seg.a ≈ type(0)
        #@test seg.b ≈ type(0)
        #@test typeof(seg.ŷ) == typeof(Point( type.((0, 0, 0)) ))
        #@test typeof(seg.a) == typeof(type(0))
        #@test typeof(seg.b) == typeof(type(0))

        #x⃗₃ = Point( type.((1, eps(type)*100, 0)) )
        #seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        #@test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        #@test seg.ŷ ≈ Point( type.((0, 0, 0)) )
        #@test seg.a ≈ type(0)
        #@test seg.b ≈ type(0)
        #@test typeof(seg.ŷ) == typeof(Point( type.((0, 0, 0)) ))
        #@test typeof(seg.a) == typeof(type(0))
        #@test typeof(seg.b) == typeof(type(0))

        ## Methods
        ## -----------------------------------------------------------------------------------------
        ## evaluation 
        #x⃗₁ = Point( type.((0, 0, 0)) )
        #x⃗₂ = Point( type.((1, 0, 0)) )
        #x⃗₃ = Point( type.((0.5, 0.5, 0)) )
        #
        ## Do x₃ barely off of the line.
        #seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        #@test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        #@test seg.ŷ ≈ Point( type.((0, 1, 0)) )
        #@test seg.a ≈ type(-2)
        #@test seg.b ≈ type(2)
        #for t = type.(LinRange(0,1,11))
        #    @test seg(t) ≈ Point(type(t), type(-2*t^2 + 2t))
        #end
    end
end
