const VTK_TRIANGLE = 5
const VTK_QUAD = 9
const VTK_QUADRATIC_TRIANGLE = 22
const VTK_QUADRATIC_QUAD = 23
const VTK_TETRA = 10
const VTK_HEXAHEDRON = 12
const VTK_QUADRATIC_TETRA = 24
const VTK_QUADRATIC_HEXAHEDRON = 25

function npoints(vtk_type::I) where {I<:Integer}
    if vtk_type == VTK_TRIANGLE
        return I(3)
    elseif vtk_type == VTK_QUAD
        return I(4)
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        return I(6)
    elseif vtk_type == VTK_QUADRATIC_QUAD
        return I(8)
    elseif vtk_type == VTK_TETRA
        return I(4)
    elseif vtk_type == VTK_HEXAHEDRON
        return I(8)
    elseif vtk_type == VTK_QUADRATIC_TETRA
        return I(10)
    elseif vtk_type == VTK_QUADRATIC_HEXAHEDRON
        return I(12)
    else
        error("Unsupported type.")
        return nothing
    end
end

function vtk_alias_string(vtk_type::I) where {I<:Integer}
    if vtk_type == VTK_TRIANGLE
        return "Triangle"
    elseif vtk_type == VTK_QUAD
        return "Quadrilateral"
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        return "QuadraticTriangle"
    elseif vtk_type == VTK_QUADRATIC_QUAD
        return "QuadraticQuadrilateral"
    elseif vtk_type == VTK_TETRA
        return "Tetrahedron"
    elseif vtk_type == VTK_HEXAHEDRON
        return "Hexahedron"
    elseif vtk_type == VTK_QUADRATIC_TETRA
        return "QuadraticTetrahedron"
    elseif vtk_type == VTK_QUADRATIC_HEXAHEDRON
        return "QuadraticHexahedron"
    else
        error("Unsupported type.")
        return nothing
    end
end
