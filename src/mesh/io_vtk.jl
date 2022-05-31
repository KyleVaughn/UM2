const VTK_TRIANGLE::UInt8 = 5
const VTK_QUAD::UInt8 = 9
const VTK_QUADRATIC_TRIANGLE::UInt8 = 22
const VTK_QUADRATIC_QUAD::UInt8 = 23
const VTK_TETRA::UInt8 = 10
const VTK_HEXAHEDRON::UInt8 = 12
const VTK_QUADRATIC_TETRA::UInt8 = 24
const VTK_QUADRATIC_HEXAHEDRON::UInt8 = 25

function points_in_vtk_type(vtk_type::I) where {I<:Integer}
    if vtk_type == VTK_TRIANGLE
        return 3
    elseif vtk_type == VTK_QUAD
        return 4
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        return 6
    elseif vtk_type == VTK_QUADRATIC_QUAD
        return 8
    elseif vtk_type == VTK_TETRA
        return 4
    elseif vtk_type == VTK_HEXAHEDRON
        return 8
    elseif vtk_type == VTK_QUADRATIC_TETRA
        return 10
    elseif vtk_type == VTK_QUADRATIC_HEXAHEDRON
        return 12
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
