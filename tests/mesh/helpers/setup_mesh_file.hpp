template <std::floating_point T, std::signed_integral I>
static void
makeReferenceTriMeshFile(um2::MeshFile<T, I> & mesh)
{
  mesh.filepath = "./mesh_files/tri.inp";
  mesh.name = "tri";
  mesh.format = um2::MeshFileFormat::Abaqus;
  mesh.type = um2::MeshType::Tri;
  mesh.vertices = {
      {0, 0, 0},
      {1, 0, 0},
      {1, 1, 0},
      {0, 1, 0},
  };
  // mesh.element_types = {5, 5};
  // mesh.element_offsets = {0, 3, 6};
  mesh.element_conn = {0, 1, 2, 2, 3, 0};
  mesh.elset_names = {"A", "B", "Material_H2O", "Material_UO2"};
  mesh.elset_offsets = {0, 2, 3, 4, 5};
  mesh.elset_ids = {0, 1, 1, 1, 0};
}

template <std::floating_point T, std::signed_integral I>
static void
makeReferenceQuadMeshFile(um2::MeshFile<T, I> & mesh)
{
  mesh.filepath = "./mesh_files/quad.inp";
  mesh.name = "quad";
  mesh.format = um2::MeshFileFormat::Abaqus;
  mesh.type = um2::MeshType::Quad;
  mesh.vertices = {
      {0, 0, 0},
      {1, 0, 0},
      {1, 1, 0},
      {0, 1, 0},
      {2, 0, 0},
      {2, 1, 0},
  };
  // mesh.element_types = {9, 9};
  // mesh.element_offsets = {0, 4, 8};
  mesh.element_conn = {0, 1, 2, 3, 1, 4, 5, 2};
  mesh.elset_names = {"A", "B", "Material_H2O", "Material_UO2"};
  mesh.elset_offsets = {0, 2, 3, 4, 5};
  mesh.elset_ids = {0, 1, 1, 1, 0};
}
//
// template <std::floating_point T, std::signed_integral I>
// static void make_tri_quad_reference_mesh_file(um2::MeshFile<T, I> & mesh)
//{
//     mesh.filepath = "./test/mesh/mesh_files/tri_quad.inp";
//     mesh.name = "tri_quad";
//     mesh.format = um2::MeshFileFormat::ABAQUS;
//     mesh.nodes_x = {0, 1, 1, 0, 2};
//     mesh.nodes_y = {0, 0, 1, 1, 0};
//     mesh.nodes_z = {0, 0, 0, 0, 0};
//     mesh.element_types = {9, 5};
//     mesh.element_offsets = {0, 4, 7};
//     mesh.element_conn = {0, 1, 2, 3, 1, 4, 2};
//     mesh.elset_names = {"A", "B", "Material_H2O", "Material_UO2"};
//     mesh.elset_offsets = {0, 2, 3, 4, 5};
//     mesh.elset_ids = {0, 1, 1, 1, 0};
// }
//
template <std::floating_point T, std::signed_integral I>
static void
makeReferenceTri6MeshFile(um2::MeshFile<T, I> & mesh)
{
  mesh.filepath = "./mesh_files/tri6.inp";
  mesh.name = "tri6";
  mesh.format = um2::MeshFileFormat::Abaqus;
  mesh.type = um2::MeshType::QuadraticTri;
  mesh.vertices = {
      {                  0,                   0,                 0},
      {                  1,                   0,                 0},
      {                  0,                   1,                 0},
      {static_cast<T>(0.5), static_cast<T>(0.0), static_cast<T>(0)},
      {static_cast<T>(0.7), static_cast<T>(0.5), static_cast<T>(0)},
      {static_cast<T>(0.0), static_cast<T>(0.5), static_cast<T>(0)},
      {static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(0)},
      {static_cast<T>(1.0), static_cast<T>(0.5), static_cast<T>(0)},
      {static_cast<T>(0.5), static_cast<T>(1.0), static_cast<T>(0)},
  };
  //  mesh.nodes_x = {0.0, 1.0, 0.0, 0.5, 0.7, 0.0, 1.0, 1.0, 0.5};
  //  mesh.nodes_y = {0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0, 0.5, 1.0};
  //  mesh.nodes_z = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  //  mesh.element_types = {22, 22};
  //  mesh.element_offsets = {0, 6, 12};
  mesh.element_conn = {0, 1, 2, 3, 4, 5, 1, 6, 2, 7, 8, 4};
  mesh.elset_names = {"A", "B", "Material_H2O", "Material_UO2"};
  mesh.elset_offsets = {0, 2, 3, 4, 5};
  mesh.elset_ids = {0, 1, 1, 1, 0};
}

template <std::floating_point T, std::signed_integral I>
static void
makeReferenceQuad8MeshFile(um2::MeshFile<T, I> & mesh)
{
  mesh.filepath = "./mesh_files/quad8.inp";
  mesh.name = "quad8";
  mesh.format = um2::MeshFileFormat::Abaqus;
  mesh.type = um2::MeshType::QuadraticQuad;
  mesh.vertices = {
      {                  0,                   0,                 0},
      {                  1,                   0,                 0},
      {                  1,                   1,                 0},
      {                  0,                   1,                 0},
      {                  2,                   0,                 0},
      {                  2,                   1,                 0},
      {static_cast<T>(0.5), static_cast<T>(0.0), static_cast<T>(0)},
      {static_cast<T>(1.1), static_cast<T>(0.6), static_cast<T>(0)},
      {static_cast<T>(0.5), static_cast<T>(1.0), static_cast<T>(0)},
      {static_cast<T>(0.0), static_cast<T>(0.5), static_cast<T>(0)},
      {static_cast<T>(1.5), static_cast<T>(0.0), static_cast<T>(0)},
      {static_cast<T>(2.0), static_cast<T>(0.5), static_cast<T>(0)},
      {static_cast<T>(1.5), static_cast<T>(1.0), static_cast<T>(0)},
  };
  // mesh.nodes_x = {0, 1, 1, 0, 2, 2, 0.5, 1.1, 0.5, 0.0, 1.5, 2.0, 1.5};
  // mesh.nodes_y = {0, 0, 1, 1, 0, 1, 0.0, 0.6, 1.0, 0.5, 0.0, 0.5, 1.0};
  // mesh.nodes_z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  //   mesh.element_types = {23, 23};
  //   mesh.element_offsets = {0, 8, 16};
  mesh.element_conn = {0, 1, 2, 3, 6, 7, 8, 9, 1, 4, 5, 2, 10, 11, 12, 7};
  mesh.elset_names = {"A", "B", "Material_H2O", "Material_UO2"};
  mesh.elset_offsets = {0, 2, 3, 4, 5};
  mesh.elset_ids = {0, 1, 1, 1, 0};
}
//
// template <std::floating_point T, std::signed_integral I>
// static void make_tri6_quad8_reference_mesh_file(um2::MeshFile<T, I> & mesh)
//{
//     mesh.filepath = "./test/mesh/mesh_files/tri6_quad8.inp";
//     mesh.name = "tri6_quad8";
//     mesh.format = um2::MeshFileFormat::ABAQUS;
//     mesh.nodes_x = {0, 1, 1, 0, 2, 0.5, 0.7, 0.5, 0.0, 1.5, 1.5};
//     mesh.nodes_y = {0, 0, 1, 1, 0, 0.0, 0.6, 1.0, 0.5, 0.0, 0.5};
//     mesh.nodes_z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//     mesh.element_types = {23, 22};
//     mesh.element_offsets = {0, 8, 14};
//     mesh.element_conn = {0, 1, 2, 3, 5, 6, 7, 8, 1, 4, 2, 9, 10, 6};
//     mesh.elset_names = {"A", "B", "Material_H2O", "Material_UO2"};
//     mesh.elset_offsets = {0, 2, 3, 4, 5};
//     mesh.elset_ids = {0, 1, 1, 1, 0};
// }
