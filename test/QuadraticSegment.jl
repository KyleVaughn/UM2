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

        # intersects
        x⃗₁ = Point( type.((0, 0, 0)) )
        x⃗₂ = Point( type.((2, 0, 0)) )
        x⃗₃ = Point( type.((1, 1, 0)) )
        x⃗₄ = Point( type.((1, 0, 0)) )
        x⃗₅ = Point( type.((1, 2, 0)) )
        
        # 1 intersection
        q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
        l = LineSegment(x⃗₄, x⃗₅)
        bool, npoints, points = intersects(l, q)
        @test bool
        @test npoints == 1
        @test points[1] ≈ Point(type.((1, 1, 0)))
        # 2 intersections
        x⃗₄ = Point( type.((0, 0.75, 0)) )
        x⃗₅ = Point( type.((2, 0.75, 0)) )
        l = LineSegment(x⃗₄, x⃗₅)
        bool, npoints, points = intersects(l, q)
        @test bool
        @test npoints == 2
        @test points[1] ≈ Point(type.((0.5, 0.75, 0)))
        @test points[2] ≈ Point(type.((1.5, 0.75, 0)))
        # 0 intersections
        x⃗₄ = Point( type.((0, 3, 0)) )       
        x⃗₅ = Point( type.((2, 3, 0)) )
        l = LineSegment(x⃗₄, x⃗₅)
        bool, npoints, points = intersects(l, q)
        @test !bool
        @test npoints == 0
        @test points[1] ≈ Point(type.((1e9, 1e9, 1e9)))
    end
end
