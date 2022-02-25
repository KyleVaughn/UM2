            # jacobian 
            for r = LinRange{T}(0, 1, 11) 
                @test 𝗝(q, r) ≈ SVector{2,T}(2, -(8r) + 4)
            end

                    # jacobian
        J = jacobian(tri6, 0, 0)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(tri6, 1, 0)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(tri6, 0, 1)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(tri6, 1//2, 1//2)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1

                # jacobian
        J = jacobian(quad8, 0, 0)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(quad8, 1, 0)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(quad8, 1, 1)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1

