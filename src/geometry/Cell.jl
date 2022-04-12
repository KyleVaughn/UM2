abstract type Cell{Ord,T} end
Base.broadcastable(c::Cell) = Ref(c)
