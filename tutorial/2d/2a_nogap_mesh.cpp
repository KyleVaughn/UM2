// Model reference:
//  VERA Core Physics Benchmark Progression Problem Specifications
//  Revision 4, August 29, 2014
//  CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  // Import the model
  um2::gmsh::open("2a_nogap.brep", /*extra_info=*/true);

  // Create the MPACT spatial partition
  um2::mpact::SpatialPartition model;

  // Nuclear geometry typically consists of repeated shapes in a regular pattern, such
  // as in the pin lattice in this problem. If we don't need to deform our mesh due to
  // other physics (thermal expansion, etc.), it is desirable to only create one fine
  // mesh for each unique coarse cell.
  //
  // Any two coarse cells can be considered to be the same if they contain the exact
  // same fine mesh and materials. Since any two coarse cells with the exact same
  // geometry and materials can have the same mesh, similarly the coarse cells CAN BE
  // BUT ARE NOT NECESSARILY the same. If the same geometry is meshed in two different
  // ways, the coarse cells are not the same.
  //
  // In this case, we have 289 coarse cells, but only 3 of them contain unique geometry.
  // We therefore have the opportunity to dramatically reduce the time and memory
  // necessary to simulate the problem by only creating 3 fine meshes.
  //
  // Fuel rod and guide tube layout (pg. 5)
  std::vector<std::vector<int>> const cc_ids = um2::to_vecvec<int>(R"(
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 1 0 0 2 0 0 1 0 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)");

  // We will create 3 coarse cells of the same size to represent the 3 unique
  // coarse cells. We will also create a ray tracing module for each coarse cell. Ray
  // tracing modules must all be the same size and will typically contain a quarter or
  // full lattice of pin-cells, but in this case we can map each coarse cell to a
  // ray tracing module one-to-one.
  double const pitch = 1.26; // Pitch = 1.26 cm (pg. 4)
  um2::Vec2d const dxdy(pitch, pitch);
  for (int i = 0; i < 3; ++i) {
    model.makeCoarseCell(dxdy); // Create coarse cell i
    model.makeRTM({{i}});       // Create ray tracing module i
  }

  // Lattices are constructed out of ray tracing modules. However, since we mapped each
  // coarse cell to a ray tracing module with the same ID, we can simply use cc_ids
  // to make the lattice.
  model.makeLattice(cc_ids);

  // We have only created one lattice, so we only need to pass in the id of 0
  model.makeAssembly({0});

  // We have only created one assembly, so we only need to pass in the id of 0
  model.makeCore({{0}});

  // We now overlay the various grid hierarchies of the spatial partition onto the
  // CAD geometry. We supply "Water" as an additional argument for the default fill
  // material for empty space.
  um2::gmsh::model::occ::overlaySpatialPartition(model, "Water");

  // If you inspect the model (uncomment the following line), you will see everything but
  // the first occurrence of each unique coarse cell has been deleted. This is because we
  // only need to create a mesh for the 3 unique coarse cells. Keeping the other geometry
  // around would only waste time, and may influence the meshing algorithm.
  //
  // um2::gmsh::fltk::run();

  // Set the characteristic mesh size to 0.1 cm and generate the mesh.
  double const lc = 0.1;
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);
  um2::MeshType const mesh_type = um2::MeshType::QuadraticTri;
  um2::gmsh::model::mesh::generateMesh(mesh_type);

  // Uncomment the following line to see the mesh for each unique pin-cell.
  // um2::gmsh::fltk::run();

  // Write the fine mesh for each coarse cell to file
  um2::gmsh::write("2a_nogap.inp");

  // Import the fine mesh for each coarse cell into the spatial partition
  model.importCoarseCells("2a_nogap.inp");

  // Export the now complete MPACT model to file
  um2::exportMesh("2a_nogap.xdmf", model);

  // May segfault upon clean up of Gmsh due to a bug with deleting entities in the
  // CAD kernel. We have already written the mesh to file, so we can safely ignore this.
  um2::finalize();
  return 0;
}
