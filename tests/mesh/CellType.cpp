#include <um2/mesh/CellType.hpp>

#include "../test_macros.hpp"

// TEST_CASE(vtk2xdmf)
//{
//   // NOLINTNEXTLINE
//   using namespace um2;
//   VTKCellType tri = VTKCellType::Triangle;
//   VTKCellType quad = VTKCellType::Quad;
//   VTKCellType tri6 = VTKCellType::QuadraticTriangle;
//   VTKCellType quad8 = VTKCellType::QuadraticQuad;
//   ASSERT(vtk2xdmf(tri) == XDMFCellType::Triangle);
//   ASSERT(vtk2xdmf(quad) == XDMFCellType::Quad);
//   ASSERT(vtk2xdmf(tri6) == XDMFCellType::QuadraticTriangle);
//   ASSERT(vtk2xdmf(quad8) == XDMFCellType::QuadraticQuad);
//
//   auto const i8_tri = static_cast<int8_t>(VTKCellType::Triangle);
//   auto const i8_quad = static_cast<int8_t>(VTKCellType::Quad);
//   auto const i8_tri6 = static_cast<int8_t>(VTKCellType::QuadraticTriangle);
//   auto const i8_quad8 = static_cast<int8_t>(VTKCellType::QuadraticQuad);
//   // NOLINTBEGIN(misc-static-assert)
//   ASSERT(vtk2xdmf(i8_tri) == static_cast<int8_t>(XDMFCellType::Triangle));
//   ASSERT(vtk2xdmf(i8_quad) == static_cast<int8_t>(XDMFCellType::Quad));
//   ASSERT(vtk2xdmf(i8_tri6) == static_cast<int8_t>(XDMFCellType::QuadraticTriangle));
//   ASSERT(vtk2xdmf(i8_quad8) == static_cast<int8_t>(XDMFCellType::QuadraticQuad));
//   // NOLINTEND(misc-static-assert)
// }
//
// TEST_CASE(abaqus2xdmf)
//{
//  // NOLINTNEXTLINE
//  using namespace um2;
//  //   AbaqusCellType tri = AbaqusCellType::CPS3;
//  //   AbaqusCellType quad = AbaqusCellType::CPS4;
//  //   AbaqusCellType tri6 = AbaqusCellType::CPS6;
//  //   AbaqusCellType quad8 = AbaqusCellType::CPS8;
//  //   ASSERT(abaqus2xdmf(tri) == XDMFCellType::Triangle);
//  //   ASSERT(abaqus2xdmf(quad) == XDMFCellType::Quad);
//  //   ASSERT(abaqus2xdmf(tri6) == XDMFCellType::QuadraticTriangle);
//  //   ASSERT(abaqus2xdmf(quad8) == XDMFCellType::QuadraticQuad);
//  //
//  auto const i8_tri = static_cast<int8_t>(AbaqusCellType::CPS3);
//  auto const i8_quad = static_cast<int8_t>(AbaqusCellType::CPS4);
//  auto const i8_tri6 = static_cast<int8_t>(AbaqusCellType::CPS6);
//  auto const i8_quad8 = static_cast<int8_t>(AbaqusCellType::CPS8);
//  // NOLINTBEGIN(misc-static-assert)
//  static_assert(abaqus2xdmf(i8_tri) == static_cast<int8_t>(XDMFCellType::Triangle));
//  static_assert(abaqus2xdmf(i8_quad) == static_cast<int8_t>(XDMFCellType::Quad));
//  static_assert(abaqus2xdmf(i8_tri6) ==
//                static_cast<int8_t>(XDMFCellType::QuadraticTriangle));
//  static_assert(abaqus2xdmf(i8_quad8) ==
//                static_cast<int8_t>(XDMFCellType::QuadraticQuad));
//  // NOLINTEND(misc-static-assert)
//}
//
// TEST_CASE(xdmf2vtk)
//{
//   // NOLINTNEXTLINE
//   using namespace um2;
//   XDMFCellType tri = XDMFCellType::Triangle;
//   XDMFCellType quad = XDMFCellType::Quad;
//   XDMFCellType tri6 = XDMFCellType::QuadraticTriangle;
//   XDMFCellType quad8 = XDMFCellType::QuadraticQuad;
//   ASSERT(xdmf2vtk(tri) == VTKCellType::Triangle);
//   ASSERT(xdmf2vtk(quad) == VTKCellType::Quad);
//   ASSERT(xdmf2vtk(tri6) == VTKCellType::QuadraticTriangle);
//   ASSERT(xdmf2vtk(quad8) == VTKCellType::QuadraticQuad);
//
//   auto const i8_tri = static_cast<int8_t>(XDMFCellType::Triangle);
//   auto const i8_quad = static_cast<int8_t>(XDMFCellType::Quad);
//   auto const i8_tri6 = static_cast<int8_t>(XDMFCellType::QuadraticTriangle);
//   auto const i8_quad8 = static_cast<int8_t>(XDMFCellType::QuadraticQuad);
//   // NOLINTBEGIN(misc-static-assert)
//   ASSERT(xdmf2vtk(i8_tri) == static_cast<int8_t>(VTKCellType::Triangle));
//   ASSERT(xdmf2vtk(i8_quad) == static_cast<int8_t>(VTKCellType::Quad));
//   ASSERT(xdmf2vtk(i8_tri6) == static_cast<int8_t>(VTKCellType::QuadraticTriangle));
//   ASSERT(xdmf2vtk(i8_quad8) == static_cast<int8_t>(VTKCellType::QuadraticQuad));
//   // NOLINTEND(misc-static-assert)
// }
//
// TEST_CASE(isLinear)
//{
//   // NOLINTNEXTLINE
//   using namespace um2;
//   // NOLINTBEGIN(misc-static-assert)
//   ASSERT(isLinear(VTKCellType::Triangle));
//   ASSERT(isLinear(VTKCellType::Quad));
//   ASSERT(!isLinear(VTKCellType::QuadraticTriangle));
//   ASSERT(!isLinear(VTKCellType::QuadraticQuad));
//
//   ASSERT(isLinear(AbaqusCellType::CPS3));
//   ASSERT(isLinear(AbaqusCellType::CPS4));
//   ASSERT(!isLinear(AbaqusCellType::CPS6));
//   ASSERT(!isLinear(AbaqusCellType::CPS8));
//   // NOLINTEND(misc-static-assert)
// }
//
// TEST_CASE(pointsInXDMFCell)
//{
//  // NOLINTNEXTLINE
//  using namespace um2;
//  // NOLINTBEGIN(misc-static-assert)
//  static_assert(pointsInXDMFCell(static_cast<int8_t>(XDMFCellType::Triangle)) == 3);
//  static_assert(pointsInXDMFCell(static_cast<int8_t>(XDMFCellType::Quad)) == 4);
//  static_assert(pointsInXDMFCell(static_cast<int8_t>(XDMFCellType::QuadraticTriangle))
//  ==
//                6);
//  static_assert(pointsInXDMFCell(static_cast<int8_t>(XDMFCellType::QuadraticQuad)) ==
//  8);
//  // NOLINTEND(misc-static-assert)
//}

TEST_SUITE(CellType)
{
  //  TEST(vtk2xdmf);
  //  TEST(abaqus2xdmf);
  //  TEST(xdmf2vtk);
  //  TEST(isLinear);
  //  TEST(pointsInXDMFCell);
}

auto
main() -> int
{
  RUN_SUITE(CellType);
  return 0;
}
