#pragma once

#include <spdlog/spdlog.h>       // Log
#include <um2/common/config.hpp> // int8_t

namespace um2
{

enum class vtk_cell_type : int8_t {

  // Linear cells
  triangle = 5,
  quad = 9,

  // Quadratic, isoparametric cells
  quadratic_triangle = 22,
  quadratic_quad = 23

};

enum class abaqus_cell_type : int8_t {

  // Linear cells
  cp_s3 = static_cast<int8_t>(vtk_cell_type::triangle),
  cp_s4 = static_cast<int8_t>(vtk_cell_type::quad),

  // Quadratic, isoparametric cells
  cp_s6 = static_cast<int8_t>(vtk_cell_type::quadratic_triangle),
  cp_s8 = static_cast<int8_t>(vtk_cell_type::quadratic_quad)

};

enum class xdmf_cell_type : int8_t {

  // Linear cells
  triangle = 4,
  quad = 5,

  // Quadratic, isoparametric cells
  quadratic_triangle = 36,
  quadratic_quad = 37

};

constexpr auto
vtk2xdmf(vtk_cell_type vtk_type) -> xdmf_cell_type
{
  switch (vtk_type) {
  case vtk_cell_type::triangle:
    return xdmf_cell_type::triangle;
  case vtk_cell_type::quad:
    return xdmf_cell_type::quad;
  case vtk_cell_type::quadratic_triangle:
    return xdmf_cell_type::quadratic_triangle;
  case vtk_cell_type::quadratic_quad:
    return xdmf_cell_type::quadratic_quad;
  default:
    spdlog::error("Unsupported VTK cell type");
    return xdmf_cell_type::triangle;
  }
}

constexpr auto
vtk2xdmf(int8_t vtk_type) -> int8_t
{
  switch (vtk_type) {
  case static_cast<int8_t>(vtk_cell_type::triangle):
    return static_cast<int8_t>(xdmf_cell_type::triangle);
  case static_cast<int8_t>(vtk_cell_type::quad):
    return static_cast<int8_t>(xdmf_cell_type::quad);
  case static_cast<int8_t>(vtk_cell_type::quadratic_triangle):
    return static_cast<int8_t>(xdmf_cell_type::quadratic_triangle);
  case static_cast<int8_t>(vtk_cell_type::quadratic_quad):
    return static_cast<int8_t>(xdmf_cell_type::quadratic_quad);
  default:
    spdlog::error("Unsupported VTK cell type");
    return static_cast<int8_t>(xdmf_cell_type::triangle);
  }
}

constexpr auto
abaqus2xdmf(abaqus_cell_type abq_type) -> xdmf_cell_type
{
  switch (abq_type) {
  case abaqus_cell_type::cp_s3:
    return xdmf_cell_type::triangle;
  case abaqus_cell_type::cp_s4:
    return xdmf_cell_type::quad;
  case abaqus_cell_type::cp_s6:
    return xdmf_cell_type::quadratic_triangle;
  case abaqus_cell_type::cp_s8:
    return xdmf_cell_type::quadratic_quad;
  default:
    spdlog::error("Unsupported VTK cell type");
    return xdmf_cell_type::triangle;
  }
}

constexpr auto
abaqus2xdmf(int8_t abq_type) -> int8_t
{
  switch (abq_type) {
  case static_cast<int8_t>(abaqus_cell_type::cp_s3):
    return static_cast<int8_t>(xdmf_cell_type::triangle);
  case static_cast<int8_t>(abaqus_cell_type::cp_s4):
    return static_cast<int8_t>(xdmf_cell_type::quad);
  case static_cast<int8_t>(abaqus_cell_type::cp_s6):
    return static_cast<int8_t>(xdmf_cell_type::quadratic_triangle);
  case static_cast<int8_t>(abaqus_cell_type::cp_s8):
    return static_cast<int8_t>(xdmf_cell_type::quadratic_quad);
  default:
    spdlog::error("Unsupported VTK cell type");
    return static_cast<int8_t>(xdmf_cell_type::triangle);
  }
}

constexpr auto
xdmf2vtk(xdmf_cell_type xdmf_type) -> vtk_cell_type
{
  switch (xdmf_type) {
  case xdmf_cell_type::triangle:
    return vtk_cell_type::triangle;
  case xdmf_cell_type::quad:
    return vtk_cell_type::quad;
  case xdmf_cell_type::quadratic_triangle:
    return vtk_cell_type::quadratic_triangle;
  case xdmf_cell_type::quadratic_quad:
    return vtk_cell_type::quadratic_quad;
  default:
    spdlog::error("Unsupported XDMF cell type");
    return vtk_cell_type::triangle;
  }
}

constexpr auto
xdmf2vtk(int8_t xdmf_type) -> int8_t
{
  switch (xdmf_type) {
  case static_cast<int8_t>(xdmf_cell_type::triangle):
    return static_cast<int8_t>(vtk_cell_type::triangle);
  case static_cast<int8_t>(xdmf_cell_type::quad):
    return static_cast<int8_t>(vtk_cell_type::quad);
  case static_cast<int8_t>(xdmf_cell_type::quadratic_triangle):
    return static_cast<int8_t>(vtk_cell_type::quadratic_triangle);
  case static_cast<int8_t>(xdmf_cell_type::quadratic_quad):
    return static_cast<int8_t>(vtk_cell_type::quadratic_quad);
  default:
    spdlog::error("Unsupported XDMF cell type");
    return static_cast<int8_t>(vtk_cell_type::triangle);
  }
}

constexpr auto
isLinear(abaqus_cell_type abq_type) -> bool
{
  switch (abq_type) {
  case abaqus_cell_type::cp_s3:
  case abaqus_cell_type::cp_s4:
    return true;
  case abaqus_cell_type::cp_s6:
  case abaqus_cell_type::cp_s8:
    return false;
  default:
    spdlog::error("Unsupported Abaqus cell type");
    return true;
  }
}

constexpr auto
isLinear(vtk_cell_type vtk_type) -> bool
{
  switch (vtk_type) {
  case vtk_cell_type::triangle:
  case vtk_cell_type::quad:
    return true;
  case vtk_cell_type::quadratic_triangle:
  case vtk_cell_type::quadratic_quad:
    return false;
  default:
    spdlog::error("Unsupported VTK cell type");
    return true;
  }
}

constexpr auto
pointsInXdmfCell(int8_t xdmf_type) -> len_t
{
  switch (xdmf_type) {
  case static_cast<int8_t>(xdmf_cell_type::triangle):
    return 3;
  case static_cast<int8_t>(xdmf_cell_type::quad):
    return 4;
  case static_cast<int8_t>(xdmf_cell_type::quadratic_triangle):
    return 6;
  case static_cast<int8_t>(xdmf_cell_type::quadratic_quad):
    return 8;
  default:
    spdlog::error("Unsupported XDMF cell type");
    return 3;
  }
}
} // namespace um2