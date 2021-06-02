using MOCNeutronTransport
@testset "QuadraticSegment" begin
    for type in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # Constructor
            x⃗₁ = Point( type.((0, 0, 0)) )
            x⃗₂ = Point( type.((2, 0, 0)) )
            x⃗₃ = Point( type.((1, 1, 0)) )

            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)

            x⃗₃ = Point(type(1), type(1)/sqrt(type(2)), type(1)/sqrt(type(2))) 
            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
        end

        @testset "Methods" begin
            # evaluation 
            x⃗₁ = Point( type.((0, 0, 0)) )
            x⃗₂ = Point( type.((2, 0, 0)) )
            x⃗₃ = Point( type.((1, 1, 0)) )
            
            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
            for t = type.(LinRange(0, 1, 11))
                @test q(t) ≈ Point(type(2t), type(-(2t)^2 + 4t))
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

            # in
            @test x⃗₁ ∈  q
            x⃗₄ = Point( type.((1//2, 3//4, 0)) )       
            @test x⃗₄ ∈  q
            x⃗₄ = Point( type.((1//2, 1, 0)) )       
            @test x⃗₄ ∉ q

#            # in_area
#            @test in_area(x⃗₁, q)             
#            x⃗₄ = Point( type.((1//2, 3//4, 0)) )       
#            @test in_area(x⃗₄, q)             
#            x⃗₄ = Point( type.((1//2, 1, 0)) )       
#            @test !in_area(x⃗₄, q)             
#            x⃗₄ = Point( type.((1//2, 1//2, 0)) )                  
#            @test in_area(x⃗₄, q)             
        end
    end
end
