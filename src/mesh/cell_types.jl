const VTK_TRIANGLE           = Int8( 5)
const VTK_QUAD               = Int8( 9)
const VTK_QUADRATIC_TRIANGLE = Int8(22)
const VTK_QUADRATIC_QUAD     = Int8(23)

const ABAQUS_CPS3 = VTK_TRIANGLE
const ABAQUS_CPS4 = VTK_QUAD
const ABAQUS_CPS6 = VTK_QUADRATIC_TRIANGLE
const ABAQUS_CPS8 = VTK_QUADRATIC_QUAD

const XDMF_TRIANGLE             = Int8( 4)
const XDMF_QUAD                 = Int8( 5)
const XDMF_QUADRATIC_TRIANGLE   = Int8(36)
const XDMF_QUADRATIC_QUAD       = Int8(37)

function vtk2xdmf(vtk_type::Int8)
    if vtk_type === VTK_TRIANGLE
        return XDMF_TRIANGLE
    elseif vtk_type === VTK_QUAD
        return XDMF_QUAD
    elseif vtk_type === VTK_QUADRATIC_TRIANGLE
        return XDMF_QUADRATIC_TRIANGLE
    elseif vtk_type === VTK_QUADRATIC_QUAD
        return XDMF_QUADRATIC_QUAD
    else
        error("Invalid VTK type.")
        return nothing
    end
end

function xdmf2vtk(xdmf_type::Int8)
    if xdmf_type === XDMF_TRIANGLE
        return VTK_TRIANGLE
    elseif xdmf_type === XDMF_QUAD
        return VTK_QUAD
    elseif xdmf_type === XDMF_QUADRATIC_TRIANGLE
        return VTK_QUADRATIC_TRIANGLE
    elseif xdmf_type === XDMF_QUADRATIC_QUAD
        return VTK_QUADRATIC_QUAD
    else
        error("Invalid XDMF type.")
        return nothing
    end
end

function points_in_vtk_type(vtk_type::Int8)
    if vtk_type === VTK_TRIANGLE
        return 3
    elseif vtk_type === VTK_QUAD
        return 4
    elseif vtk_type === VTK_QUADRATIC_TRIANGLE
        return 6
    elseif vtk_type === VTK_QUADRATIC_QUAD
        return 8
    else
        error("Unsupported type.")
        return nothing
    end
end
