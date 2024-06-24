#include <um2/mesh/element_types.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

TEST_CASE(verticesPerElem)    
{    
  static_assert(um2::verticesPerElem(um2::VTKElemType::Vertex) == 1);    
  static_assert(um2::verticesPerElem(um2::VTKElemType::Line) == 2);    
  static_assert(um2::verticesPerElem(um2::VTKElemType::Triangle) == 3);    
  static_assert(um2::verticesPerElem(um2::VTKElemType::Quad) == 4);    
  static_assert(um2::verticesPerElem(um2::VTKElemType::QuadraticEdge) == 3);    
  static_assert(um2::verticesPerElem(um2::VTKElemType::QuadraticTriangle) == 6);    
  static_assert(um2::verticesPerElem(um2::VTKElemType::QuadraticQuad) == 8);    
} 

TEST_CASE(inferVTKElemType)
{
  static_assert(um2::inferVTKElemType(1) == um2::VTKElemType::Vertex);
  static_assert(um2::inferVTKElemType(2) == um2::VTKElemType::Line);
  static_assert(um2::inferVTKElemType(3) == um2::VTKElemType::Triangle);
  static_assert(um2::inferVTKElemType(4) == um2::VTKElemType::Quad);
  static_assert(um2::inferVTKElemType(6) == um2::VTKElemType::QuadraticTriangle);
  static_assert(um2::inferVTKElemType(8) == um2::VTKElemType::QuadraticQuad);
}

TEST_CASE(getMeshType)
{
  um2::Vector<um2::VTKElemType> types = {um2::VTKElemType::Triangle};
  ASSERT(getMeshType(types) == um2::MeshType::Tri);
  types = {um2::VTKElemType::Quad};
  ASSERT(getMeshType(types) == um2::MeshType::Quad);
  types = {um2::VTKElemType::QuadraticTriangle};
  ASSERT(getMeshType(types) == um2::MeshType::QuadraticTri);
  types = {um2::VTKElemType::QuadraticQuad};
  ASSERT(getMeshType(types) == um2::MeshType::QuadraticQuad);
}

TEST_SUITE(element_types)
{
  TEST_CASE(verticesPerElem);
  TEST_CASE(inferVTKElemType);
  TEST_CASE(getMeshType);
}

auto
main() -> int
{
  RUN_SUITE(element_types);
  return 0;
}
