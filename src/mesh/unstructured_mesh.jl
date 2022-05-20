export UnstructuredMesh

# See Figure 8-35 of the VTK book.
struct UnstructuredMesh{Dim,T,U}
    points::Vector{Point{Dim,T}}
    cell_array::Vector{U}
    cell_types::Vector{U}
end

const VTK_VERTEX = 1
const VTK_LINE = 3
const VTK_TRIANGLE = 5
const VTK_QUAD = 9
const VTK_TETRA = 10
const VTK_HEXAHEDRON = 12
const VTK_QUADRATIC_EDGE = 21
const VTK_QUADRATIC_TRIANGLE = 22
const VTK_QUADRATIC_QUAD = 23
const VTK_QUADRATIC_TETRA = 24
const VTK_QUADRATIC_HEXAHEDRON = 25
