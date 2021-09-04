using MOCNeutronTransport
@testset "Tree" begin
    root = Tree( data = "root" )
    @test get_level(root) === 1
    l1_1_1 = Tree(data = "L1_1_1"; parent=Ref(root))
    @test get_level(l1_1_1) === 2
    l1_2_1 = Tree(data = "L1_2_1"; parent=Ref(root))
    @test get_level(l1_2_1) === 2
    @test root.children[1][].data == "L1_1_1"
    @test root.children[2][].data == "L1_2_1"
    @test l1_1_1.parent[] === root
    @test l1_2_1.parent[] === root
end
