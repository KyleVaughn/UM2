struct UnstructuredMesh{V, E, C, T <: AbstractFloat}
    vertices::NTuple{V, Point{T}}
    edges::NTuple{E, Edge}
    cells::NTuple{C, Cell}
end
