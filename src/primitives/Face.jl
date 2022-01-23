abstract type Face{Dim,Ord,T} end
const Face2D = Face{2}
const Face3D = Face{3}
Base.broadcastable(f::Face) = Ref(f)
