using MOCNeutronTransport
@testset "Point" begin
    # Constructors
    # ---------------------------------------------------------------------------------------------
    # 3D
    p = Point((1.0, 2.0, 3.0))
    @test p.coord[1] == 1.0
    @test p.coord[2] == 2.0
    @test p.coord[3] == 3.0

    p = Point(1.0, 2.0, 3.0)
    @test p.coord[1] == 1.0
    @test p.coord[2] == 2.0
    @test p.coord[3] == 3.0

    # 2D
    p = Point((1.0, 2.0))
    @test p.coord[1] == 1.0
    @test p.coord[2] == 2.0
    @test p.coord[3] == 0.0

    p = Point(1.0, 2.0)
    @test p.coord[1] == 1.0
    @test p.coord[2] == 2.0
    @test p.coord[3] == 0.0

    # Base methods
    # ---------------------------------------------------------------------------------------------
    p = Point(1.0, 2.0, 3.0)

    # zero
    p₀ = zero(p)
    @test all(p₀.coord .== 0.0)
    @test p₀.coord[1] isa Float64

    # getindex
    @test p[1] == 1.0
    @test p[2] == 2.0
    @test p[3] == 3.0

    # broadcastable tested in operators due to reliance on operator correctness
    
    # Operators
    # ---------------------------------------------------------------------------------------------
    p₁ = Point(1.0, 2.0, 3.0)
    p₂ = Point(2.0, 4.0, 6.0)

    # Point equivalence
    @test p₁ == Point(1.0, 2.0, 3.0)

    # Point isapprox
    p = Point(1.0, 1.99999999, 3.0)
    @test 2.0 ≈ 1.99999999
    @test p ≈ p₁ 

    # Point addition
    p = p₁ + p₂
    @test p.coord[1] == 3.0 
    @test p.coord[2] == 6.0 
    @test p.coord[3] == 9.0 

    # Point subtraction
    p = p₁ - p₂
    @test p.coord[1] == -1.0 
    @test p.coord[2] == -2.0 
    @test p.coord[3] == -3.0 

    # Cross product
    p₁ = Point(2.0, 3.0, 4.0)
    p₂ = Point(5.0, 6.0, 7.0)
    @test p₁ × p₂ == Point(-3.0, 6.0, -3.0)

    # Dot product
    @test p₁ ⋅ p₂ == 10.0 + 18.0 + 28.0
    
    # Addition
    p₁ = Point(1.0, 2.0, 3.0)
    p₂ = Point(2.0, 4.0, 6.0)
    p = p₁ + 1.0
    @test p.coord[1] == 2.0 
    @test p.coord[2] == 3.0 
    @test p.coord[3] == 4.0 

    # Broadcast addition
    parray = [p₁, p₂]
    parray = parray .+ 1.0
    @test parray[1] == p₁ + 1.0
    @test parray[2] == p₂ + 1.0

    # Subtraction
    p = p₁ - 1.0
    @test p.coord[1] == 0.0 
    @test p.coord[2] == 1.0 
    @test p.coord[3] == 2.0 

    # Multiplication
    p = 4*p₁
    @test p.coord[1] == 4.0
    @test p.coord[2] == 8.0 
    @test p.coord[3] == 12.0 

    # Division
    p = p₁/4
    @test p.coord[1] == 0.25
    @test p.coord[2] == 0.5 
    @test p.coord[3] == 0.75 

    # Methods
    # ---------------------------------------------------------------------------------------------
    p₁ = Point(1.0, 2.0, 3.0)
    p₂ = Point(2.0, 4.0, 6.0)
    
    # distance
    @test distance(p₁, p₂) == sqrt(14.0)
end
