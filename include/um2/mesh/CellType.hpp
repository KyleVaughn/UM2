#pragma once

#include <um2/config.hpp>

namespace um2
{

enum class VTKCellType : int8_t {

  // Linear cells
  Triangle = 5,
  Quad = 9,

  // Quadratic, isoparametric cells
  QuadraticTriangle = 22,
  QuadraticQuad = 23

};

enum class AbaqusCellType : int8_t {

  // Linear cells
  CPS3 = static_cast<int8_t>(VTKCellType::Triangle),
  CPS4 = static_cast<int8_t>(VTKCellType::Quad),

  // Quadratic, isoparametric cells
  CPS6 = static_cast<int8_t>(VTKCellType::QuadraticTriangle),
  CPS8 = static_cast<int8_t>(VTKCellType::QuadraticQuad)

};

enum class XDMFCellType : int8_t {

  // Linear cells
  Triangle = 4,
  Quad = 5,

  // Quadratic, isoparametric cells
  QuadraticTriangle = 36,
  QuadraticQuad = 37

};

constexpr auto
vtk2xdmf(VTKCellType vtk_type) -> XDMFCellType
{
  switch (vtk_type) {
  case VTKCellType::Triangle:
    return XDMFCellType::Triangle;
  case VTKCellType::Quad:
    return XDMFCellType::Quad;
  case VTKCellType::QuadraticTriangle:
    return XDMFCellType::QuadraticTriangle;
  case VTKCellType::QuadraticQuad:
    return XDMFCellType::QuadraticQuad;
  default:
    assert(false && "Unsupported VTK cell type");
    return XDMFCellType::Triangle;
  }
}

constexpr auto
vtk2xdmf(int8_t vtk_type) -> int8_t
{
  switch (vtk_type) {
  case static_cast<int8_t>(VTKCellType::Triangle):
    return static_cast<int8_t>(XDMFCellType::Triangle);
  case static_cast<int8_t>(VTKCellType::Quad):
    return static_cast<int8_t>(XDMFCellType::Quad);
  case static_cast<int8_t>(VTKCellType::QuadraticTriangle):
    return static_cast<int8_t>(XDMFCellType::QuadraticTriangle);
  case static_cast<int8_t>(VTKCellType::QuadraticQuad):
    return static_cast<int8_t>(XDMFCellType::QuadraticQuad);
  default:
    assert(false && "Unsupported VTK cell type");
    return static_cast<int8_t>(XDMFCellType::Triangle);
  }
}

constexpr auto
abaqus2xdmf(AbaqusCellType abq_type) -> XDMFCellType
{
  switch (abq_type) {
  case AbaqusCellType::CPS3:
    return XDMFCellType::Triangle;
  case AbaqusCellType::CPS4:
    return XDMFCellType::Quad;
  case AbaqusCellType::CPS6:
    return XDMFCellType::QuadraticTriangle;
  case AbaqusCellType::CPS8:
    return XDMFCellType::QuadraticQuad;
  default:
    assert(false && "Unsupported VTK cell type");
    return XDMFCellType::Triangle;
  }
}

constexpr auto
abaqus2xdmf(int8_t abq_type) -> int8_t
{
  switch (abq_type) {
  case static_cast<int8_t>(AbaqusCellType::CPS3):
    return static_cast<int8_t>(XDMFCellType::Triangle);
  case static_cast<int8_t>(AbaqusCellType::CPS4):
    return static_cast<int8_t>(XDMFCellType::Quad);
  case static_cast<int8_t>(AbaqusCellType::CPS6):
    return static_cast<int8_t>(XDMFCellType::QuadraticTriangle);
  case static_cast<int8_t>(AbaqusCellType::CPS8):
    return static_cast<int8_t>(XDMFCellType::QuadraticQuad);
  default:
    assert(false && "Unsupported VTK cell type");
    return static_cast<int8_t>(XDMFCellType::Triangle);
  }
}

constexpr auto
xdmf2vtk(XDMFCellType xdmf_type) -> VTKCellType
{
  switch (xdmf_type) {
  case XDMFCellType::Triangle:
    return VTKCellType::Triangle;
  case XDMFCellType::Quad:
    return VTKCellType::Quad;
  case XDMFCellType::QuadraticTriangle:
    return VTKCellType::QuadraticTriangle;
  case XDMFCellType::QuadraticQuad:
    return VTKCellType::QuadraticQuad;
  default:
    assert(false && "Unsupported XDMF cell type");
    return VTKCellType::Triangle;
  }
}

constexpr auto
xdmf2vtk(int8_t xdmf_type) -> int8_t
{
  switch (xdmf_type) {
  case static_cast<int8_t>(XDMFCellType::Triangle):
    return static_cast<int8_t>(VTKCellType::Triangle);
  case static_cast<int8_t>(XDMFCellType::Quad):
    return static_cast<int8_t>(VTKCellType::Quad);
  case static_cast<int8_t>(XDMFCellType::QuadraticTriangle):
    return static_cast<int8_t>(VTKCellType::QuadraticTriangle);
  case static_cast<int8_t>(XDMFCellType::QuadraticQuad):
    return static_cast<int8_t>(VTKCellType::QuadraticQuad);
  default:
    assert(false && "Unsupported XDMF cell type");
    return static_cast<int8_t>(VTKCellType::Triangle);
  }
}

constexpr auto
isLinear(AbaqusCellType abq_type) -> bool
{
  switch (abq_type) {
  case AbaqusCellType::CPS3:
  case AbaqusCellType::CPS4:
    return true;
  case AbaqusCellType::CPS6:
  case AbaqusCellType::CPS8:
    return false;
  default:
    assert(false && "Unsupported Abaqus cell type");
    return true;
  }
}

constexpr auto
isLinear(VTKCellType vtk_type) -> bool
{
  switch (vtk_type) {
  case VTKCellType::Triangle:
  case VTKCellType::Quad:
    return true;
  case VTKCellType::QuadraticTriangle:
  case VTKCellType::QuadraticQuad:
    return false;
  default:
    assert(false && "Unsupported VTK cell type");
    return true;
  }
}

constexpr auto
pointsInXDMFCell(int8_t xdmf_type) -> Size
{
  switch (xdmf_type) {
  case static_cast<int8_t>(XDMFCellType::Triangle):
    return 3;
  case static_cast<int8_t>(XDMFCellType::Quad):
    return 4;
  case static_cast<int8_t>(XDMFCellType::QuadraticTriangle):
    return 6;
  case static_cast<int8_t>(XDMFCellType::QuadraticQuad):
    return 8;
  default:
    assert(false && "Unsupported XDMF cell type");
    return 3;
  }
}

} // namespace um2
