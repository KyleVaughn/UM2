@testset "Common" begin
    tests = ["sort",
             "select_type",
             "tree"
    ]
    for t in tests
        include("$(t).jl")
    end
end
