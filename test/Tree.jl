using MOCNeutronTransport
@testset "Tree" begin
    root = Tree( data = "root" )
    @test root.data === "root"
    @test root.parent === nothing
    @test root.children === nothing
    l1_1 = Tree(data = "L1_1", parent=root)
    @test l1_1.data === "L1_1"
    @test l1_1.parent === root
    @test root.children === [l1_1] 
    l1_2 = Tree(data = "L1_2", parent=root)
    @test l1_2.data === "L1_2"
    @test l1_2.parent === root
    @test root.children === [l1_1, l1_2] 
end
