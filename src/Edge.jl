abstract type Edge{Dim,Ord,T} end
const Edge2D = Edge{2}
const Edge3D = Edge{3}
Base.broadcastable(e::Edge) = Ref(e)
