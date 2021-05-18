using MOCNeutronTransport
@testset "Point" begin
    for type in [Float32, Float64, BigFloat]
        # Constructors
        # ---------------------------------------------------------------------------------------------
        # 3D
        p = Point( type.((1, 2, 3)) )
        @test p.coord == type.((1, 2, 3))
        @test typeof(p.coord) == typeof(type.((1, 2, 3)))

        p = Point(type(1), type(2), type(3))
        @test p.coord == type.((1, 2, 3))
        @test typeof(p.coord) == typeof(type.((1, 2, 3)))

        # 2D
        p = Point(type(1), type(2))
        @test p.coord == type.((1, 2, 0))
        @test typeof(p.coord) == typeof(type.((1, 2, 0)))

        # 1D
        p = Point(type(1))
        @test p.coord == type.((1, 0, 0))
        @test typeof(p.coord) == typeof(type.((1, 0, 0)))

        # Base methods
        # ---------------------------------------------------------------------------------------------
        p = Point( type.((1, 2, 3)) )

        # zero
        p₀ = zero(p)
        @test all(p₀.coord .== type(0.0))
        @test typeof(p.coord[1]) == typeof(type(0.0))

        # getindex
        @test p[1] == type(1.0)
        @test p[2] == type(2.0)
        @test p[3] == type(3.0)

        # broadcastable tested in operators due to reliance on operator correctness
        
        # Operators
        # ---------------------------------------------------------------------------------------------
        p₁ = Point( type.((1, 2, 0)) )
        p₂ = Point( type.((2, 4, 6)) )

        # Point equivalence
        @test p₁ == Point( type.((1, 2, 0)) )

        # Point isapprox
        p = Point( type.((1, 2 - 100*eps(type), 0 - 100*eps(type))) )
        @test type(2) ≈ type(2 - 100*eps(type))
        @test isapprox(type(0), type(0 - 100*eps(type)), atol=sqrt(eps(type)))
        @test p ≈ p₁ 

        # Point addition
        p = p₁ + p₂
        @test p.coord == type.((3,6,6)) 
        @test typeof(p.coord) == typeof(type.((3, 6, 6)))

        # Point subtraction
        p = p₁ - p₂
        @test p.coord == type.((-1, -2, -6)) 
        @test typeof(p.coord) == typeof(type.((-1, -2, -6)))

        # Cross product
        p₁ = Point( type.((2, 3, 4)) )
        p₂ = Point( type.((5, 6, 7)) )
        p = p₁ × p₂
        @test p == Point( type.((-3, 6, -3)) )
        @test typeof(p.coord) == typeof(type.((-3, 6, -3)))

        # Dot product
        @test p₁ ⋅ p₂ == type(10 + 18 + 28)
        
        # Number addition
        p₁ = Point( type.((1, 2, 3)) )
        p₂ = Point( type.((2, 4, 6)) )
        p = p₁ + type(1.0)
        @test p.coord == type.((2, 3, 4)) 
        @test typeof(p.coord) == typeof(type.((2, 3, 4)))

        # Broadcast addition
        parray = [p₁, p₂]
        parray = parray .+ type(1)
        @test parray[1] == p₁ + type(1.0)
        @test parray[2] == p₂ + type(1.0)
        @test typeof(parray[1].coord) == typeof(type.((2, 3, 4)))

        # Subtraction
        p = p₁ - type(1)
        @test p.coord == type.((0, 1, 2)) 
        @test typeof(p.coord) == typeof(type.((0, 1, 2)))

        # Multiplication
        p = type(4)*p₁
        @test p.coord == type.((4, 8, 12))
        @test typeof(p.coord) == typeof(type.((4, 8, 12)))

        # Division
        p = p₁/type(4)
        @test p.coord == (type(1)/type(4), type(2)/type(4), type(3)/type(4))
        @test typeof(p.coord) == typeof(type.((4, 8, 12)))

        # Unary -
        p = -p₁
        @test p.coord == type.((-1, -2, -3))
        @test typeof(p.coord) == typeof(type.((4, 8, 12)))

        # Methods
        # ---------------------------------------------------------------------------------------------
        p₁ = Point( type.((1, 2, 3)) )
        p₂ = Point( type.((2, 4, 6)) )

        # distance
        @test distance(p₁, p₂) == sqrt(type(14))
        @test typeof(distance(p₁, p₂)) == typeof(type(1))

        # norm
        @test norm(p₁) == sqrt(type(14))
        @test norm(p₂) == sqrt(type(4 + 16 + 36))
        @test typeof(norm(p₁)) == typeof(type(1))
    end
end
