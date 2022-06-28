@testset "Gaussian Quadrature" begin
    # NOTE: It is assumed p = 1:4 is sufficient for testing
    @testset "RefLine" begin for T in Floats
        for p in 1:4
            weights, points = gauss_quadrature(LegendreType(), RefLine(), Val(p), T)
            @test abs(sum(weights) - 1) < 10 * eps(T)
            # Test the integration of a polynomial of degree 2p-1, which should
            # be exact for p integration points.
            # 1
            # ∫xᵖdx = 1/(p+1) ≈ ∑ wᵢxᵢᵖ
            # 0
            exact = 1 / big(p + 1)
            approx = zero(T)
            for i in 1:p
                approx += weights[i] * (points[i][1]^p)
            end
            rel_err = T((approx - exact) / exact)
            @test rel_err < 10 * eps(T)
            if T != BigFloat
                @test @allocated(gauss_quadrature(LegendreType(), RefLine(), Val(p), T)) ==
                      0
            end
        end
    end end

    @testset "RefSquare" begin for T in Floats
        for p in 1:3
            weights, points = gauss_quadrature(LegendreType(), RefSquare(), Val(p), T)
            @test abs(sum(weights) - 1) < 10 * eps(T)
            # 
            # q = 2p-1
            #
            # 11                                   5 + (-1)^q + 2q
            # ∫∫(r^q + s^q + (1 - r - s)^q ds dr = --------------
            # 00                                     (q+1)(q+2)
            q = 2p - 1
            exact = big((5 + (-1)^q + 2q)) / big((q + 1) * (q + 2))
            approx = zero(T)
            for i in 1:(p^2)
                r, s = points[i]
                approx += weights[i] * (r^q + s^q + (1 - r - s)^q)
            end
            rel_err = T((approx - exact) / exact)
            @test rel_err < 10 * eps(T)
            # Will allocate past p = 3
            if T != BigFloat
                @test @allocated(gauss_quadrature(LegendreType(), RefSquare(), Val(p), T)) ==
                      0
            end
        end
    end end

    @testset "RefCube" begin for T in Floats
        for p in 1:4
            weights, points = gauss_quadrature(LegendreType(), RefCube(), Val(p), T)
            @test abs(sum(weights) - 1) < 10 * eps(T)
            # 
            # q = 2p-1
            #
            # 111                             
            # ∫∫∫(r^q + s^q + t^q) dt ds dr = 3/(q+1) 
            # 000                            
            q = 2p - 1
            exact = 3 / big(q + 1)
            approx = zero(T)
            for i in 1:(p^3)
                r, s, t = points[i]
                approx += weights[i] * (r^q + s^q + t^q)
            end
            rel_err = T((approx - exact) / exact)
            @test rel_err < 10 * eps(T)
            # Allocates past p = 2
        end
    end end

    @testset "RefTriangle" begin for T in Floats
        for p in 1:4
            weights, points = gauss_quadrature(LegendreType(), RefTriangle(), Val(p), T)
            @test abs(sum(weights) - 1 // 2) < 10 * eps(T)
            # This quadrature is only exact to polynomial degree p, not 2p-1!
            #
            # 1 1-r
            # ∫  ∫ (r^p + s^p + (1 - r - s)^p) ds dr = 3/((p+1)(p+2))
            # 0  0
            #
            exact = 3 / big((p + 1) * (p + 2))
            approx = zero(T)
            for i in 1:length(weights)
                r, s = points[i]
                approx += weights[i] * (r^p + s^p + (1 - r - s)^p)
            end
            rel_err = T((approx - exact) / exact)
            @test rel_err < 10 * eps(T)
            if T != BigFloat
                @test @allocated(gauss_quadrature(LegendreType(), RefTriangle(), Val(p), T)) ==
                      0
            end
        end
    end end
end
