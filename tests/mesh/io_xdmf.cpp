#include <um2/mesh/polytope_soup.hpp>

#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

#include <fstream>
#include <iostream>

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri_mesh)
{
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTriPolytopeSoup(mesh_ref);
  mesh_ref.write("./tri.xdmf");

  um2::PolytopeSoup<T, I> mesh;
  mesh.read("./tri.xdmf");

  ASSERT(mesh.compareTo(mesh_ref) == 17); // Don't read elset data

  int stat = std::remove("./tri.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri.h5");
  ASSERT(stat == 0);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(quad_mesh)
{
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceQuadPolytopeSoup(mesh_ref);
  mesh_ref.write("./quad.xdmf");

  um2::PolytopeSoup<T, I> mesh;
  mesh.read("./quad.xdmf");

  ASSERT(mesh.compareTo(mesh_ref) == 17); // Don't read elset data

  int stat = std::remove("./quad.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./quad.h5");
  ASSERT(stat == 0);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri_quad_mesh)
{
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTriQuadPolytopeSoup(mesh_ref);
  mesh_ref.write("./tri_quad.xdmf");

  um2::PolytopeSoup<T, I> mesh;
  mesh.read("./tri_quad.xdmf");

  ASSERT(mesh.compareTo(mesh_ref) == 17); // Don't read elset data

  int stat = std::remove("./tri_quad.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri_quad.h5");
  ASSERT(stat == 0);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri6_mesh)
{
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTri6PolytopeSoup(mesh_ref);
  mesh_ref.write("./tri6.xdmf");

  um2::PolytopeSoup<T, I> mesh;
  mesh.read("./tri6.xdmf");

  ASSERT(mesh.compareTo(mesh_ref) == 17); // Don't read elset data

  int stat = std::remove("./tri6.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri6.h5");
  ASSERT(stat == 0);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(quad8_mesh)
{
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceQuad8PolytopeSoup(mesh_ref);
  mesh_ref.write("./quad8.xdmf");

  um2::PolytopeSoup<T, I> mesh;
  mesh.read("./quad8.xdmf");

  ASSERT(mesh.compareTo(mesh_ref) == 17); // Don't read elset data

  int stat = std::remove("./quad8.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./quad8.h5");
  ASSERT(stat == 0);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri6_quad8_mesh)
{
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTri6Quad8PolytopeSoup(mesh_ref);
  mesh_ref.write("./tri6_quad8.xdmf");

  um2::PolytopeSoup<T, I> mesh;
  mesh.read("./tri6_quad8.xdmf");

  ASSERT(mesh.compareTo(mesh_ref) == 17); // Don't read elset data

  int stat = std::remove("./tri6_quad8.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri6_quad8.h5");
  ASSERT(stat == 0);
}

template <std::floating_point T, std::integral I>
TEST_SUITE(io_xdmf)
{
  TEST((tri_mesh<T, I>));
  TEST((quad_mesh<T, I>));
  TEST((tri_quad_mesh<T, I>));
  TEST((tri6_mesh<T, I>));
  TEST((quad8_mesh<T, I>));
  TEST((tri6_quad8_mesh<T, I>));
}

auto
main() -> int
{
  RUN_SUITE((io_xdmf<float, int16_t>));
  RUN_SUITE((io_xdmf<float, int32_t>));
  RUN_SUITE((io_xdmf<float, int64_t>));
  RUN_SUITE((io_xdmf<double, int16_t>));
  RUN_SUITE((io_xdmf<double, int32_t>));
  RUN_SUITE((io_xdmf<double, int64_t>));
  return 0;
}
