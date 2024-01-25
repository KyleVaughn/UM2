inline void
makeReferenceTriPolytopeSoup(um2::PolytopeSoup & mesh)
{
  mesh.addVertex(0, 0, 0);
  mesh.addVertex(1, 0, 0);
  mesh.addVertex(1, 1, 0);
  mesh.addVertex(0, 1, 0);
  um2::Vector<I> conn = {0, 1, 2};
  mesh.addElement(um2::VTKElemType::Triangle, conn);
  conn = {2, 3, 0};
  mesh.addElement(um2::VTKElemType::Triangle, conn);
  mesh.addElset("A", {0, 1}, {10, 2});
  mesh.addElset("B", {1});
  mesh.addElset("Material_H2O", {1});
  mesh.addElset("Material_UO2", {0});
}

inline void
makeReferenceQuadPolytopeSoup(um2::PolytopeSoup & mesh)
{
  mesh.addVertex(0, 0, 0);
  mesh.addVertex(1, 0, 0);
  mesh.addVertex(1, 1, 0);
  mesh.addVertex(0, 1, 0);
  mesh.addVertex(2, 0, 0);
  mesh.addVertex(2, 1, 0);
  um2::Vector<I> conn = {0, 1, 2, 3};
  mesh.addElement(um2::VTKElemType::Quad, conn);
  conn = {1, 4, 5, 2};
  mesh.addElement(um2::VTKElemType::Quad, conn);
  mesh.addElset("A", {0, 1}, {10, 2});
  mesh.addElset("B", {1});
  mesh.addElset("Material_H2O", {1});
  mesh.addElset("Material_UO2", {0});
}

inline void
makeReferenceTriQuadPolytopeSoup(um2::PolytopeSoup & mesh)
{
  mesh.addVertex(0, 0, 0);
  mesh.addVertex(1, 0, 0);
  mesh.addVertex(1, 1, 0);
  mesh.addVertex(0, 1, 0);
  mesh.addVertex(2, 0, 0);
  um2::Vector<I> conn = {0, 1, 2, 3};
  mesh.addElement(um2::VTKElemType::Quad, conn);
  conn = {1, 4, 2};
  mesh.addElement(um2::VTKElemType::Triangle, conn);
  mesh.addElset("A", {0, 1}, {10, 2});
  mesh.addElset("B", {1});
  mesh.addElset("Material_H2O", {1});
  mesh.addElset("Material_UO2", {0});
}

inline void
makeReferenceTri6PolytopeSoup(um2::PolytopeSoup & mesh)
{
  mesh.addVertex(0, 0, 0);
  mesh.addVertex(1, 0, 0);
  mesh.addVertex(0, 1, 0);
  mesh.addVertex(condCast<F>(0.5), condCast<F>(0.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.7), condCast<F>(0.5), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.0), condCast<F>(0.5), condCast<F>(0));
  mesh.addVertex(condCast<F>(1.0), condCast<F>(1.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(1.0), condCast<F>(0.5), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.5), condCast<F>(1.0), condCast<F>(0));
  um2::Vector<I> conn = {0, 1, 2, 3, 4, 5};
  mesh.addElement(um2::VTKElemType::QuadraticTriangle, conn);
  conn = {1, 6, 2, 7, 8, 4};
  mesh.addElement(um2::VTKElemType::QuadraticTriangle, conn);
  mesh.addElset("A", {0, 1}, {10, 2});
  mesh.addElset("B", {1});
  mesh.addElset("Material_H2O", {1});
  mesh.addElset("Material_UO2", {0});
}

inline void
makeReferenceQuad8PolytopeSoup(um2::PolytopeSoup & mesh)
{
  mesh.addVertex(0, 0, 0);
  mesh.addVertex(1, 0, 0);
  mesh.addVertex(1, 1, 0);
  mesh.addVertex(0, 1, 0);
  mesh.addVertex(2, 0, 0);
  mesh.addVertex(2, 1, 0);
  mesh.addVertex(condCast<F>(0.5), condCast<F>(0.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(1.1), condCast<F>(0.6), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.5), condCast<F>(1.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.0), condCast<F>(0.5), condCast<F>(0));
  mesh.addVertex(condCast<F>(1.5), condCast<F>(0.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(2.0), condCast<F>(0.5), condCast<F>(0));
  mesh.addVertex(condCast<F>(1.5), condCast<F>(1.0), condCast<F>(0));
  // mesh.nodes_x = {0, 1, 1, 0, 2, 2, 0.5, 1.1, 0.5, 0.0, 1.5, 2.0, 1.5};
  // mesh.nodes_y = {0, 0, 1, 1, 0, 1, 0.0, 0.6, 1.0, 0.5, 0.0, 0.5, 1.0};
  // mesh.nodes_z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  um2::Vector<I> conn = {0, 1, 2, 3, 6, 7, 8, 9};
  mesh.addElement(um2::VTKElemType::QuadraticQuad, conn);
  conn = {1, 4, 5, 2, 10, 11, 12, 7};
  mesh.addElement(um2::VTKElemType::QuadraticQuad, conn);
  mesh.addElset("A", {0, 1}, {10, 2});
  mesh.addElset("B", {1});
  mesh.addElset("Material_H2O", {1});
  mesh.addElset("Material_UO2", {0});
}

inline void
makeReferenceTri6Quad8PolytopeSoup(um2::PolytopeSoup & mesh)
{
  mesh.addVertex(0, 0, 0);
  mesh.addVertex(1, 0, 0);
  mesh.addVertex(1, 1, 0);
  mesh.addVertex(0, 1, 0);
  mesh.addVertex(2, 0, 0);
  mesh.addVertex(condCast<F>(0.5), condCast<F>(0.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.7), condCast<F>(0.6), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.5), condCast<F>(1.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(0.0), condCast<F>(0.5), condCast<F>(0));
  mesh.addVertex(condCast<F>(1.5), condCast<F>(0.0), condCast<F>(0));
  mesh.addVertex(condCast<F>(1.5), condCast<F>(0.5), condCast<F>(0));
  // mesh.nodes_x = {0, 1, 1, 0, 2, 0.5, 0.7, 0.5, 0.0, 1.5, 1.5};
  // mesh.nodes_y = {0, 0, 1, 1, 0, 0.0, 0.6, 1.0, 0.5, 0.0, 1.5};
  // mesh.nodes_z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  um2::Vector<I> conn = {0, 1, 2, 3, 5, 6, 7, 8};
  mesh.addElement(um2::VTKElemType::QuadraticQuad, conn);
  conn = {1, 4, 2, 9, 10, 6};
  mesh.addElement(um2::VTKElemType::QuadraticTriangle, conn);
  mesh.addElset("A", {0, 1}, {10, 2});
  mesh.addElset("B", {1});
  mesh.addElset("Material_H2O", {1});
  mesh.addElset("Material_UO2", {0});
}
