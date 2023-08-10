#include <um2/mesh/MeshType.hpp>

#include "../test_macros.hpp"

TEST_CASE(verticesPerCell)
{
  static_assert(um2::verticesPerCell(um2::MeshType::Tri) == 3);
  static_assert(um2::verticesPerCell(um2::MeshType::Quad) == 4);
  static_assert(um2::verticesPerCell(um2::MeshType::QuadraticTri) == 6);
  static_assert(um2::verticesPerCell(um2::MeshType::QuadraticQuad) == 8);
}

TEST_SUITE(CellType) { TEST(verticesPerCell); }

auto
main() -> int
{
  RUN_SUITE(CellType);
  return 0;
}
