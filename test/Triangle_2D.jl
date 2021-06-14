using MOCNeutronTransport
@testset "Triangle_2D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            tri = Triangle_2D((p₁, p₂, p₃))
            @test tri.points == (p₁, p₂, p₃)

            # single constructor
            tri = Triangle_2D(p₁, p₂, p₃)
            @test tri.points == (p₁, p₂, p₃)
        end

        @testset "Methods" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            tri = Triangle_2D((p₁, p₂, p₃))

            # interpolation
            tri(0, 0) ≈ p₁
            tri(1, 0) ≈ p₂
            tri(0, 1) ≈ p₃
            tri(1//2, 1//2) ≈ Point_2D(T, 1//2, 1//2)

            # area
            a = area(tri)
            @test typeof(a) == typeof(T(1))
            @test a == T(1//2)

            # in
            p = Point_2D(T, 1//2, 1//10)
            @test p ∈  tri
            p = Point_2D(T, 1//2, 0)
            @test p ∈  tri
            p = Point_2D(T, 1//2, -1//10)
            @test p ∉ tri
        end
    end
end
