abstract type Edge_2D{F <: AbstractFloat} end
broadcastable(e::Edge_2D) = Ref(e)
@propagate_inbounds function getindex(e::Edge_2D, i::Int)
    getfield(e, :points)[i]
end
