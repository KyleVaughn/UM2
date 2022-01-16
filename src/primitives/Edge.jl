abstract type Edge{N,T} end
Base.broadcastable(e::Edge) = Ref(e)
