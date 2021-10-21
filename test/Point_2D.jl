using MOCNeutronTransport
using StaticArrays
@testset "Point_2D" begin
    @testset "$T" for T in [Float32, Float64]
        @testset "Constructors" begin
            # 2D single constructor
            p = Point_2D(T(1), T(2))
            @test p.x == SVector(T.((1, 2)))
            @test typeof(p.x) == typeof(SVector(T.((1, 2))))

            # 1D single constructor
            p = Point_2D(T(1))
            @test p.x == SVector(T.((1, 0)))
            @test typeof(p.x) == typeof(SVector(T.((1, 0))))

            # 2D tuple constructor
            p = Point_2D( T.((1, 2)) )
            @test p.x == SVector(T.((1, 2)))
            @test typeof(p.x) == typeof(SVector(T.((1, 2))))

            # 2D single conversion constructor
            p = Point_2D(Float64, 1, 2)
            @test p.x == SVector(Float64.((1, 2)))
            @test typeof(p.x) == typeof(SVector(Float64.((1, 2))))

            # 1D single conversion constructor
            p = Point_2D(Float64, 1)
            @test p.x == SVector(Float64.((1, 0)))
            @test typeof(p.x) == typeof(SVector(Float64.((1, 0))))
        end

        @testset "Base" begin
            p = Point_2D(T, 1, 2)

            # zero
            p₀ = zero(p)
            @test all(p₀.x .== T(0))
            @test typeof(p.x[1]) == typeof(T(0))

            # (::Type)
            q = Float64(p)
            @test q.x == SVector(Float64.((1, 2)))
            @test typeof(q.x) == typeof(SVector(Float64.((1, 2))))
        end

        @testset "Operators" begin
            p₁ = Point_2D(T, 1, 2)
            p₂ = Point_2D(T, 2, 4)

            # Point_2D equivalence
            @test p₁ == Point_2D(T, 1, 2)

            # Point_2D isapprox
            p = Point_2D(T, 1, 2 - 10*eps(T))
            @test T(2) ≈ T(2 - 10*eps(T))
            @test p ≈ p₁

            # Point_2D addition
            p = p₁ + p₂
            @test p.x == SVector(T.((3,6)))
            @test typeof(p.x) == typeof(SVector(T.((3, 6))))

            # Point_2D subtraction
            p = p₁ - p₂
            @test p.x == SVector(T.((-1, -2)))
            @test typeof(p.x) == typeof(SVector(T.((-1, -2))))

            # Cross product
            p₁ = Point_2D(T, 2, 3)
            p₂ = Point_2D(T, 5, 6)
            v = p₁ × p₂
            @test v == -3
            @test typeof(p.x) == typeof(SVector(T.((-3, 6))))

            # Dot product
            @test p₁ ⋅ p₂ == T(10 + 18)

            # Number addition
            p₁ = Point_2D(T, 1, 2)
            p₂ = Point_2D(T, 2, 4)
            p = p₁ + T(1)
            @test p  == Point_2D(T, 2, 3)
            @test typeof(p.x) == typeof(SVector(T.((2, 3))))

            # Broadcast addition
            parray = [p₁, p₂]
            parray = parray .+ T(1)
            @test parray[1] == p₁ + T(1)
            @test parray[2] == p₂ + T(1)
            @test typeof(parray[1].x) == typeof(SVector(T.((2, 3))))

            # Subtraction
            p = p₁ - T(1)
            @test p == Point_2D(T, 0, 1)
            @test typeof(p.x) == typeof(SVector(T.((0, 1))))

            # Multiplication
            p = T(4)*p₁
            @test p == Point_2D(T, 4, 8)
            @test typeof(p.x) == typeof(SVector(T.((4, 8))))

            # Division
            p = p₁/T(4)
            @test p == Point_2D(T, 1//4, 1//2)
            @test typeof(p.x) == typeof(SVector(T.((4, 8))))

            # Unary -
            p = -p₁
            @test p == Point_2D(T, -1, -2)
            @test typeof(p.x) == typeof(SVector(T.((4, 8))))
        end

        @testset "Methods" begin
            p₁ = Point_2D(T, 1, 2)
            p₂ = Point_2D(T, 2, 4)

            # distance
            @test distance(p₁, p₂) == sqrt(T(5))
            @test typeof(distance(p₁, p₂)) == typeof(T(1))

            # norm
            @test norm(p₁) == sqrt(T(5))
            @test norm(p₂) == sqrt(T(4 + 16))
            @test typeof(norm(p₁)) == typeof(T(1))
        end
    end
end
