struct MPACTGeometryPartiton2D{T}
    name::String
    bb::AABox2D{T}
    children::Matrix{MPACTGeometryPartiton2D{T}}
end
# RT module minima = 0,0
# union of bbs = parent

mutable struct MPACTCoreGeomTree2D{T}
    name::String
    bb::AABox2D{T}
    pinmesh_id::Int64
    children::Matrix{MPACTCoreGeomTree2D{T}}
end

function MPACTCoreGeomTree2D(name, bb::AABox2D{T}, child_dims::NTuple{2, Int64}=(0,0)) where {T}
    return MPACTCoreGeomTree2D{T}(name, bb, 0, 
                Matrix{MPACTCoreGeomTree2D{T}}(undef, child_dims[1], child_dims[2]))
end

AbstractTrees.children(node::MPACTCoreGeomTree2D) = node.children

AbstractTrees.nodetype(::MPACTCoreGeomTree2D{T}) where {T} = MPACTCoreGeomTree2D{T}

function AbstractTrees.printnode(io::IO, node::MPACTCoreGeomTree2D)
    if node.pinmesh_id === 0
        print(io, node.name)
    else
        print(io, node.name*", id=$(node.pinmesh_id)")
    end
end

Base.eltype(::Type{<:TreeIterator{MPACTCoreGeomTree2D{T}}}
           ) where T = MPACTCoreGeomTree2D{T}

Base.IteratorEltype(::Type{<:TreeIterator{MPACTCoreGeomTree2D{T}}}
                   ) where T = Base.HasEltype()

