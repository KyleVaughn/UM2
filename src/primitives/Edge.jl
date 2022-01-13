abstract type Edge{N,T} end
Base.broadcastable(e::Edge) = Ref(e)
Base.@propagate_inbounds function Base.getindex(e::Edge, i::Int)
    getfield(e, :points)[i]
end
