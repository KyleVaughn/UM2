#pragma once

#include <um2/config.hpp>

namespace um2
{

//==============================================================================
// Element topology identifiers
//==============================================================================

// Element IDs for VTK
enum class VTKElemType : int8_t {
  None = 0,
  Vertex = 1,
  Line = 3,
  Triangle = 5,
  Quad = 9,
  QuadraticEdge = 21,
  QuadraticTriangle = 22,
  QuadraticQuad = 23
};

// Element IDs for XDMF
enum class XDMFElemType : int8_t {
  None = 0,
  Vertex = 1,
  Line = 2,
  Triangle = 4,
  Quad = 5,
  QuadraticEdge = 34,
  QuadraticTriangle = 36,
  QuadraticQuad = 37
};

enum class MeshType : int8_t {
  None = 0,
  Tri = 3,
  Quad = 4,
  TriQuad = 7,
  QuadraticTri = 6,
  QuadraticQuad = 8,
  QuadraticTriQuad = 14
};

CONST constexpr auto
verticesPerElem(VTKElemType const type) -> Int
{
  switch (type) {
  case VTKElemType::Vertex:
    return 1;
  case VTKElemType::Line:
    return 2;
  case VTKElemType::Triangle:
    return 3;
  case VTKElemType::Quad:
    return 4;
  case VTKElemType::QuadraticEdge:
    return 3;
  case VTKElemType::QuadraticTriangle:
    return 6;
  case VTKElemType::QuadraticQuad:
    return 8;
  default:
    ASSERT(false);
    return -1000;
  }
}

// We only support a few element types, so we can uniquely identify a
// VTKElemType by its number of vertices.
CONST constexpr auto
inferVTKElemType(Int const type) -> VTKElemType
{
  switch (type) {
  case 1:
    return VTKElemType::Vertex;
  case 2:
    return VTKElemType::Line;
  case 3:
    return VTKElemType::Triangle;
  case 4:
    return VTKElemType::Quad;
  case 6:
    return VTKElemType::QuadraticTriangle;
  case 8:
    return VTKElemType::QuadraticQuad;
  default:
    ASSERT(false);
    return VTKElemType::None;
  }
}

//
//CONST constexpr auto
//verticesPerElem(MeshType const type) -> Int
//{
//  switch (type) {
//  case MeshType::Tri:
//    return 3;
//  case MeshType::Quad:
//    return 4;
//  case MeshType::QuadraticTri:
//    return 6;
//  case MeshType::QuadraticQuad:
//    return 8;
//  default:
//    ASSERT(false);
//    return -1000;
//  }
//}
//
CONST constexpr auto
xdmfToVTKElemType(int8_t x) -> VTKElemType
{
  switch (x) {
  case static_cast<int8_t>(XDMFElemType::Vertex):
    return VTKElemType::Vertex;
  case static_cast<int8_t>(XDMFElemType::Line):
    return VTKElemType::Line;
  case static_cast<int8_t>(XDMFElemType::Triangle):
    return VTKElemType::Triangle;
  case static_cast<int8_t>(XDMFElemType::Quad):
    return VTKElemType::Quad;
  case static_cast<int8_t>(XDMFElemType::QuadraticEdge):
    return VTKElemType::QuadraticEdge;
  case static_cast<int8_t>(XDMFElemType::QuadraticTriangle):
    return VTKElemType::QuadraticTriangle;
  case static_cast<int8_t>(XDMFElemType::QuadraticQuad):
    return VTKElemType::QuadraticQuad;
  default:
    ASSERT(false);
    return VTKElemType::None;
  }
}

CONST constexpr auto
vtkToXDMFElemType(VTKElemType x) -> XDMFElemType
{
  switch (x) {
  case VTKElemType::Vertex:
    return XDMFElemType::Vertex;
  case VTKElemType::Line:
    return XDMFElemType::Line;
  case VTKElemType::Triangle:
    return XDMFElemType::Triangle;
  case VTKElemType::Quad:
    return XDMFElemType::Quad;
  case VTKElemType::QuadraticEdge:
    return XDMFElemType::QuadraticEdge;
  case VTKElemType::QuadraticTriangle:
    return XDMFElemType::QuadraticTriangle;
  case VTKElemType::QuadraticQuad:
    return XDMFElemType::QuadraticQuad;
  default:
    ASSERT(false);
    return XDMFElemType::None;
  }
}

} // namespace um2
