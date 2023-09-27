// VERA Core Physics Benchmark Progression Problem Specifications
// Revision 4, August 29, 2014
// CASL-U-2012-0131-004

#include <um2.hpp>

auto
//NOLINTNEXTLINE
main() -> int
{
  um2::initialize();

  // We are importing the .brep file (a cad file format) that we created when running 2a_model.
  // The "extra_info" flag is set to true so that we can import the groups and 
  // colors of the entities in the brep file.
  um2::gmsh::open("2a_nogap.brep", /*extra_info=*/true);

  // We will use this bool later to determine if we should treat each cell as a unique geometry.
  // With repeated cell types in your model, we should try to generate a single mesh for 
  // each unique geometry and reuse that mesh for each cell type. This will allow us to minimize
  // the overhead cost of generating a mesh as much as possible.
  bool const repeated_geom = true;

  // We create a SpatialPartition model in order to map the CAD geometry to
  // MPACT's spacial hierarchy to use in the simulation.
  um2::mpact::SpatialPartition model;

  // Sets the size of an individual coarse cell to a 1.26 cm x 1.26cm size.
  // The coarse cell (dxdy) is implicitly given an id of 0 as in problem 1a, but we will redefine 
  // this course cell id below
  double const pitch = 1.26; // Pitch = 1.26 cm (pg. 4)
  um2::Vec2d const dxdy(pitch, pitch);

  // Fuel rod and guide tube layout (pg. 5)
  std::vector<std::vector<int>> pin_ids(17, std::vector<int>(17, 0));

  if (!repeated_geom) {
    // If we have chosen to not repeat geometry, then we will generate a new mesh for every individual
    // coarse cell. In this case, we do this by giving each of the 17^2 coarse a unique id.
    for (size_t i = 0; i < 17; ++i) {
      for (size_t j = 0; j < 17; ++j) {
        auto const val = static_cast<int>(i * 17 + j);
        pin_ids[i][j] = val;
        model.makeCoarseCell(dxdy);
        model.makeRTM({{val}});
        model.makeLattice({{val}});
      }
    }
  } else {
    // If we have chosen to repeat geometry, then we will generate a new mesh
    // only for meshes with the same ID. In this case, we will only have 3 unique ids,
    // so we will only generate 3 seperate meshes.
    pin_ids = um2::to_vecvec<int>(R"(
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
    
    // 3 calls are made for makeCoarseCell since we have exactly 3 meshes to generate
    model.makeCoarseCell(dxdy);
    model.makeCoarseCell(dxdy);
    model.makeCoarseCell(dxdy);

    // We create 3 ray tracing modules for each of the unique ids we have created. 
    // We should create individual ray tracing modules whenever we have identicle cell sizes, but
    // this will not be possible with varying cell sizes.
    model.makeRTM({{0}});
    model.makeRTM({{1}});
    model.makeRTM({{2}});

    
  }
  // We can use the pin_ids in this case since the Ray Tracing modules have the same IDs 
  // as the coarse cells in this case
  model.makeLattice(pin_ids);

  // We have only created one lattice, so we only need to pass in the id of 0
  model.makeAssembly({0});

  // We have only created one Assembly, so we only need to pass in the id of 0
  model.makeCore({{0}});

  // We now overlay the full spacial hierarchy model onto the geometry for each coarse cell.
  // As in problem 1a, unassigned materials will be named "Material Moderator" and has a color of "royalblue",
  // but can be changed using the optional argument
  um2::gmsh::model::occ::overlaySpatialPartition(model, "Water");

  // Sets the characteristic mesh size for the whole model to 0.1 cm as we did in problem 1a
  // MAKE SURE TO SPECIFY MESH SIZE AFTER THE OVERLAY. When doing the overlay of the grid it
  // deltes certian cad entities, which may cause seg faults if you specify before the overlay.
  double const lc = 0.1;
  um2::gmsh::model::mesh::setGlobalMeshSize(lc);

  // Sets the mesh type to a triangular mesh type, but note that we should
  // opt to use a quadratic mesh for more complex problems as mentioned in 1a
  um2::MeshType const mesh_type = um2::MeshType::Tri;
  
  // We can now generate the mesh. This will create a mesh for the entire model.
  um2::gmsh::model::mesh::generateMesh(mesh_type);

  // Uncomment the following line to see the mesh.
  um2::gmsh::fltk::run();

  //write the generated mesh to file
  um2::gmsh::write("2a_nogap.inp");

  //import generated mesh into current model
  model.importCoarseCells("2a_nogap.inp");

  //export the mesh model to a file 
  um2::exportMesh("2a_nogap.xdmf", model);
  um2::finalize();
  return 0;
}
