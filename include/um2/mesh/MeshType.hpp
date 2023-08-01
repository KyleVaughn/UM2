#pragma once

#include <um2/config.hpp>

namespace um2
{

enum class MeshType : int8_t {
  None = 0,
  Tri = 3,
  Quad = 4,
  TriQuad = 7,
  QuadraticTri = 6,
  QuadraticQuad = 8,
  QuadraticTriQuad = 14,
};

// template <std::integral I>
// constexpr auto
// intToMeshtype(I const i) -> mesh_type
//{
//   switch (i) {
//   case static_cast<I>(mesh_type::error):
//     spdlog::error("Invalid mesh type");
//     return mesh_type::error;
//   case static_cast<I>(mesh_type::tri):
//     return mesh_type::tri;
//   case static_cast<I>(mesh_type::quad):
//     return mesh_type::quad;
//   case static_cast<I>(mesh_type::tri_quad):
//     return mesh_type::tri_quad;
//   case static_cast<I>(mesh_type::quadratic_tri):
//     return mesh_type::quadratic_tri;
//   case static_cast<I>(mesh_type::quadratic_quad):
//     return mesh_type::quadratic_quad;
//   case static_cast<I>(mesh_type::quadratic_tri_quad):
//     return mesh_type::quadratic_tri_quad;
//   default:
//     spdlog::error("Invalid mesh type");
//     return mesh_type::error;
//   }
// }
//
// template <std::integral I>
// auto
// meshtypeToString(I const mesh_type) -> std::string
//{
//   switch (mesh_type) {
//   case static_cast<I>(mesh_type::error):
//     return "ERROR";
//   case static_cast<I>(mesh_type::tri):
//     return "TRI";
//   case static_cast<I>(mesh_type::quad):
//     return "QUAD";
//   case static_cast<I>(mesh_type::tri_quad):
//     return "TRI_QUAD";
//   case static_cast<I>(mesh_type::quadratic_tri):
//     return "QUADRATIC_TRI";
//   case static_cast<I>(mesh_type::quadratic_quad):
//     return "QUADRATIC_QUAD";
//   case static_cast<I>(mesh_type::quadratic_tri_quad):
//     return "QUADRATIC_TRI_QUAD";
//   default:
//     spdlog::error("Invalid mesh type");
//     return "ERROR";
//   }
// }

} // namespace um2
