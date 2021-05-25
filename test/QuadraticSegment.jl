using MOCNeutronTransport
@testset "QuadraticSegment" begin
    for type in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # Constructor
            x⃗₁ = Point( type.((0, 0, 0)) )
            x⃗₂ = Point( type.((2, 0, 0)) )
            x⃗₃ = Point( type.((1, 1, 0)) )

            seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
            @test seg.ŷ ≈ Point( type.((0, 1, 0)) )
            @test seg.a ≈ type(-1)
            @test seg.b ≈ type(2)
            @test typeof(seg.ŷ) == typeof(Point( type.((0, 1, 0)) ))
            @test typeof(seg.a) == typeof(type(-1))
            @test typeof(seg.b) == typeof(type(2))

            x⃗₃ = Point(type(1), type(1)/sqrt(type(2)), type(1)/sqrt(type(2))) 
            seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
            @test seg.ŷ ≈ Point( type(0), type(1)/sqrt(type(2)), type(1)/sqrt(type(2)) )
            @test seg.a ≈ type(-1)
            @test seg.b ≈ type(2)
            @test typeof(seg.ŷ) == typeof(Point( type(0), type(1)/sqrt(type(2)), type(1)/sqrt(type(2)) ))
            @test typeof(seg.a) == typeof(type(-1))
            @test typeof(seg.b) == typeof(type(2))      

            x⃗₃ = Point( type.((1, 0, 0)) )
            seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
            @test seg.ŷ ≈ Point( type.((0, 0, 0)) )
            @test seg.a ≈ type(0)
            @test seg.b ≈ type(0)
            @test typeof(seg.ŷ) == typeof(Point( type.((0, 0, 0)) ))
            @test typeof(seg.a) == typeof(type(0))
            @test typeof(seg.b) == typeof(type(0))

            x⃗₃ = Point( type.((1, eps(type)*100, 0)) )
            seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
            @test seg.ŷ ≈ Point( type.((0, 0, 0)) )
            @test seg.a ≈ type(0)
            @test seg.b ≈ type(0)
            @test typeof(seg.ŷ) == typeof(Point( type.((0, 0, 0)) ))
            @test typeof(seg.a) == typeof(type(0))
            @test typeof(seg.b) == typeof(type(0))
        end

        @testset "Methods" begin
            # evaluation 
            x⃗₁ = Point( type.((0, 0, 0)) )
            x⃗₂ = Point( type.((2, 0, 0)) )
            x⃗₃ = Point( type.((1, 1, 0)) )
            
            seg = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test seg.x⃗ == (x⃗₁, x⃗₂, x⃗₃)
            for t = type.(LinRange(0, 1, 11))
                @test seg(t) ≈ Point(type(2t), type(-(2t)^2 + 4t))
            end

            # intersect
            x⃗₁ = Point( type.((0, 0, 0)) )
            x⃗₂ = Point( type.((2, 0, 0)) )
            x⃗₃ = Point( type.((1, 1, 0)) )
            x⃗₄ = Point( type.((1, 0, 0)) )
            x⃗₅ = Point( type.((1, 2, 0)) )
            
            # 1 intersection
            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            l = LineSegment(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test bool
            @test npoints == 1
            @test points[1] ≈ Point(type.((1, 1, 0)))
            # 2 intersections
            x⃗₄ = Point( type.((0, 0.75, 0)) )
            x⃗₅ = Point( type.((2, 0.75, 0)) )
            l = LineSegment(x⃗₄, x⃗₅)
            bool, npoints, points = l ∩ q
            @test bool
            @test npoints == 2
            @test points[1] ≈ Point(type.((0.5, 0.75, 0)))
            @test points[2] ≈ Point(type.((1.5, 0.75, 0)))
            # 0 intersections
            x⃗₄ = Point( type.((0, 3, 0)) )       
            x⃗₅ = Point( type.((2, 3, 0)) )
            l = LineSegment(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test !bool
            @test npoints == 0
            @test points[1] ≈ Point(type.((1e9, 1e9, 1e9)))
            @test points[2] ≈ Point(type.((1e9, 1e9, 1e9)))
        end
    end
end
