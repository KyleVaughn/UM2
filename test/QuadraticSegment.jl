using MOCNeutronTransport
@testset "QuadraticSegment" begin
    for type in [Float32, Float64, BigFloat]
        # Constructor
        x⃗₁ = Point( type.((0, 0, 0)) )
        x⃗₂ = Point( type.((2, 0, 0)) )
        x⃗₃ = Point( type.((1, 1, 0)) )

        seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        @test seg.r⃗[1] ≈ Point(type.((2, 4, 0)))
        @test seg.r⃗[2] ≈ Point(type.((0, -4, 0)))
        @test typeof(seg.r⃗[1]) == typeof(Point(type.((0, -4, 0))))

        x⃗₃ = Point(type(1), type(1)/sqrt(type(2)), type(1)/sqrt(type(2))) 
        seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        @test seg.r⃗[1] ≈ Point( type(2), type(4)/sqrt(type(2)), type(4)/sqrt(type(2)) )
        @test seg.r⃗[2] ≈ Point( type(0), -type(4)/sqrt(type(2)), -type(4)/sqrt(type(2)) )
        @test typeof(seg.r⃗[1]) == typeof(Point(type.((0, -4, 0))))

        x⃗₃ = Point( type.((1, 0, 0)) )
        seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        @test seg.r⃗[1] ≈ x⃗₂ - x⃗₁
        @test seg.r⃗[2] ≈ zero(x⃗₂)
        @test typeof(seg.r⃗[1]) == typeof(Point(type.((0, -4, 0))))

        x⃗₃ = Point( type.((1, eps(type)*100, 0)) )
        seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        @test seg.r⃗[1] ≈ x⃗₂ - x⃗₁
        @test seg.r⃗[2] ≈ zero(x⃗₂)
        @test typeof(seg.r⃗[1]) == typeof(Point(type.((0, -4, 0))))

        # Methods
        # -----------------------------------------------------------------------------------------
        # evaluation 
        x⃗₁ = Point( type.((0, 0, 0)) )
        x⃗₂ = Point( type.((2, 0, 0)) )
        x⃗₃ = Point( type.((1, 1, 0)) )
        
        seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
        @test seg.r⃗[1] ≈ Point(type.((2, 4, 0)))
        @test seg.r⃗[2] ≈ Point(type.((0, -4, 0)))
        @test typeof(seg.r⃗[1]) == typeof(Point(type.((0, -4, 0))))
        for t = type.(LinRange(0, 1, 11))
            @test seg(t) ≈ Point(type(2t), type(-(2t)^2 + 4t))
        end
    end
end
