using MOCNeutronTransport
@testset "AnyTree" begin
    root = AnyTree("root" )
    @test root.data === "root"
    @test root.parent === nothing
    @test root.children === nothing
    n1 = AnyTree(1, root)
    @test n1.data === 1
    @test n1.parent === root
    @test root.children == [n1] 
    n2 = AnyTree(2, root)
    @test n2.data === 2
    @test n2.parent === root
    @test root.children == [n1, n2] 
    n3 = AnyTree(3, n2)
    
    # isroot
    @test isroot(root)
    @test !isroot(n2)

    # leaves
    @test leaves(root) == [n1, n3]
    @test leaves(n2) == [n3]
    @test length(leaves(n3)) == 0
end
