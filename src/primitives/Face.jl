abstract type Face{N,T} end
Base.broadcastable(f::Face) = Ref(f)
# Base.@propagate_inbounds function Base.getindex(f::Face, i::Int)
#     getfield(f, :points)[i]
# end
