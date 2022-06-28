@testset "Tree" begin
    root = Tree(0)
    child1 = Tree(1, root)
    child2 = Tree(2, root)
    child3 = Tree(3, child1)
    child4 = Tree(4, child1)
    child5 = Tree(5, child1)

    @test data(root) == 0
    @test @allocated(data(root)) == 0

    @test parent(root) == nothing
    @test @allocated(parent(root)) == 0

    @test children(root) == [child1, child2]
    @test @allocated(children(root)) == 0

    @test isroot(root) == true
    @test @allocated(isroot(root)) == 0

    # child1
    @test data(child1) == 1
    @test @allocated(data(child1)) == 0

    @test parent(child1) == root
    @test @allocated(parent(child1)) == 0

    @test children(child1) == [child3, child4, child5]
    @test @allocated(children(child1)) == 0

    @test isroot(child1) == false
    @test @allocated(isroot(child1)) == 0

    @test leaves(root) == [child3, child4, child5, child2]
    @test leaves(child1) == [child3, child4, child5]
    @test leaves(child2) == [child2]

    @test nleaves(root) == 4
    @test nleaves(child1) == 3
    @test nleaves(child2) == 1
end
