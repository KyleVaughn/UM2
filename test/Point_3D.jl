using MOCNeutronTransport
using StaticArrays
@testset "Point_3D" begin
    @testset "$T" for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # 3D single constructor
            p = Point_3D(T(1), T(2), T(3))
            @test p.x == SVector(T.((1, 2, 3)))
            @test typeof(p.x) == typeof(SVector(T.((1, 2, 3))))

            # 2D single constructor
            p = Point_3D(T(1), T(2))
            @test p.x == SVector(T.((1, 2, 0)))
            @test typeof(p.x) == typeof(SVector(T.((1, 2, 0))))

            # 1D single constructor
            p = Point_3D(T(1))
            @test p.x == SVector(T.((1, 0, 0)))
            @test typeof(p.x) == typeof(SVector(T.((1, 0, 0))))

            # 3D tuple constructor
            p = Point_3D( T.((1, 2, 3)) )
            @test p.x == SVector(T.((1, 2, 3)))
            @test typeof(p.x) == typeof(SVector(T.((1, 2, 3))))

            # 2D tuple constructor
            p = Point_3D( T.((1, 2)) )
            @test p.x == SVector(T.((1, 2, 0)))
            @test typeof(p.x) == typeof(SVector(T.((1, 2, 0))))

            # 3D single conversion constructor
            p = Point_3D(Float64, 1, 2, 3)
            @test p.x == SVector(Float64.((1, 2, 3)))
            @test typeof(p.x) == typeof(SVector(Float64.((1, 2, 3))))

            # 2D single conversion constructor
            p = Point_3D(Float64, 1, 2)
            @test p.x == SVector(Float64.((1, 2, 0)))
            @test typeof(p.x) == typeof(SVector(Float64.((1, 2, 0))))

            # 1D single conversion constructor
            p = Point_3D(Float64, 1)
            @test p.x == SVector(Float64.((1, 0, 0)))
            @test typeof(p.x) == typeof(SVector(Float64.((1, 0, 0))))
        end

        @testset "Base" begin
            p = Point_3D(T, 1, 2, 3)

            # zero
            p₀ = zero(p)
            @test all(p₀.x .== T(0))
            @test typeof(p.x[1]) == typeof(T(0))

            # (::Type)
            q = Float64(p)
            @test q.x == SVector(Float64.((1, 2, 3)))
            @test typeof(q.x) == typeof(SVector(Float64.((1, 2, 3))))
        end

        @testset "Operators" begin
            p₁ = Point_3D(T, 1, 2, 0)
            p₂ = Point_3D(T, 2, 4, 6)

            # Point_3D equivalence
            @test p₁ == Point_3D(T, 1, 2, 0)

            # Point_3D isapprox
            p = Point_3D(T, 1, 2 - 100*eps(T), 0 - 100*eps(T))
            @test T(2) ≈ T(2 - 100*eps(T))
            @test isapprox(T(0), T(0 - 100*eps(T)), atol=sqrt(eps(T)))
            @test p ≈ p₁

            # Point_3D addition
            p = p₁ + p₂
            @test p.x == SVector(T.((3,6,6)))
            @test typeof(p.x) == typeof(SVector(T.((3, 6, 6))))

            # Point_3D subtraction
            p = p₁ - p₂
            @test p.x == SVector(T.((-1, -2, -6)))
            @test typeof(p.x) == typeof(SVector(T.((-1, -2, -6))))

            # Cross product
            p₁ = Point_3D(T, 2, 3, 4)
            p₂ = Point_3D(T, 5, 6, 7)
            p = p₁ × p₂
            @test p == Point_3D(T, -3, 6, -3)
            @test typeof(p.x) == typeof(SVector(T.((-3, 6, -3))))

            # Dot product
            @test p₁ ⋅ p₂ == T(10 + 18 + 28)

            # Number addition
            p₁ = Point_3D(T, 1, 2, 3)
            p₂ = Point_3D(T, 2, 4, 6)
            p = p₁ + T(1)
            @test p  == Point_3D(T, 2, 3, 4)
            @test typeof(p.x) == typeof(SVector(T.((2, 3, 4))))

            # Broadcast addition
            parray = [p₁, p₂]
            parray = parray .+ T(1)
            @test parray[1] == p₁ + T(1)
            @test parray[2] == p₂ + T(1)
            @test typeof(parray[1].x) == typeof(SVector(T.((2, 3, 4))))

            # Subtraction
            p = p₁ - T(1)
            @test p == Point_3D(T, 0, 1, 2)
            @test typeof(p.x) == typeof(SVector(T.((0, 1, 2))))

            # Multiplication
            p = T(4)*p₁
            @test p == Point_3D(T, 4, 8, 12)
            @test typeof(p.x) == typeof(SVector(T.((4, 8, 12))))

            # Division
            p = p₁/T(4)
            @test p == Point_3D(T, 1//4, 1//2, 3//4)
            @test typeof(p.x) == typeof(SVector(T.((4, 8, 12))))

            # Unary -
            p = -p₁
            @test p == Point_3D(T, -1, -2, -3)
            @test typeof(p.x) == typeof(SVector(T.((4, 8, 12))))
        end

        @testset "Methods" begin
            p₁ = Point_3D(T, 1, 2, 3)
            p₂ = Point_3D(T, 2, 4, 6)

            # distance
            @test distance(p₁, p₂) == sqrt(T(14))
            @test typeof(distance(p₁, p₂)) == typeof(T(1))

            # norm
            @test norm(p₁) == sqrt(T(14))
            @test norm(p₂) == sqrt(T(4 + 16 + 36))
            @test typeof(norm(p₁)) == typeof(T(1))
        end
    end
end
