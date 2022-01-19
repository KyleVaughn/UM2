abstract type Edge{N,T} end
const Edge_2D = Edge{2}
const Edge_3D = Edge{3}
Base.broadcastable(e::Edge) = Ref(e)
