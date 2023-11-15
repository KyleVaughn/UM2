// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  // Parameters
  double const radius = 0.54;          // Pin radius = 0.54 cm (pg. 3)
  double const pin_pitch = 1.26;       // Pin pitch = 1.26 cm (pg. 3)
  double const assembly_pitch = 21.42; // Assembly pitch = 21.42 cm (pg. 3)

  // Materials
  um2::Material const uo2("UO2", "forestgreen");
  um2::Material const mox43("MOX_4.3", "orange");
  um2::Material const mox70("MOX_7.0", "yellow");
  um2::Material const mox87("MOX_8.7", "red");
  um2::Material const fiss_chamber("Fission Chamber", "black");
  um2::Material const guide_tube("Guide Tube", "darkgrey");

  // Pin ID  |  Material
  // --------+----------------
  // 0       |  UO2
  // 1       |  MOX 4.3%
  // 2       |  MOX 7.0%
  // 3       |  MOX 8.7%
  // 4       |  Fission Chamber
  // 5       |  Guide Tube
  // 6       |  Moderator

  // Each pin has the same radius and pitch
  std::vector<std::vector<double>> const pin_radii(6, {radius});
  um2::Vec2d const pin_size = {pin_pitch, pin_pitch};
  std::vector<um2::Vec2d> const dxdy(6, pin_size);

  // Each pin contains a single material
  std::vector<std::vector<um2::Material>> const pin_mats = {
      {uo2}, {mox43}, {mox70}, {mox87}, {fiss_chamber}, {guide_tube}};

  // UO2 lattice pins (pg. 7)
  std::vector<std::vector<int>> const uo2_lattice = um2::to_vecvec<int>(R"(
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 4 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    )");

  // MOX lattice pins (pg. 7)
  std::vector<std::vector<int>> const mox_lattice = um2::to_vecvec<int>(R"(
      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
      1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
      1 2 2 2 2 5 2 2 5 2 2 5 2 2 2 2 1
      1 2 2 5 2 3 3 3 3 3 3 3 2 5 2 2 1
      1 2 2 2 3 3 3 3 3 3 3 3 3 2 2 2 1
      1 2 5 3 3 5 3 3 5 3 3 5 3 3 5 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 5 3 3 5 3 3 4 3 3 5 3 3 5 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 2 3 3 3 3 3 3 3 3 3 3 3 2 2 1
      1 2 5 3 3 5 3 3 5 3 3 5 3 3 5 2 1
      1 2 2 2 3 3 3 3 3 3 3 3 3 2 2 2 1
      1 2 2 5 2 3 3 3 3 3 3 3 2 5 2 2 1
      1 2 2 2 2 5 2 2 5 2 2 5 2 2 2 2 1
      1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1
      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    )");

  // Moderator lattice
  std::vector<std::vector<int>> const h2o_lattice = um2::to_vecvec<int>(R"(
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
    )");

  // Make the calls a bit more readable using an alias
  namespace factory = um2::gmsh::model::occ;

  // We want to set up the problem to look like the following:
  //  +---------+---------+---------+
  //  |         |         |         |
  //  |   UO2   |   MOX   |   H2O   |
  //  |         |         |         |
  //  +---------+---------+---------+
  //  |         |         |         |
  //  |   MOX   |   UO2   |   H2O   |
  //  |         |         |         |
  //  +---------+---------+---------+
  //  |         |         |         |
  //  |   H2O   |   H2O   |   H2O   |
  //  |         |         |         |
  //  +---------+---------+---------+
  //
  // Create UO2 lattices
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy, uo2_lattice,
                                      assembly_pitch * um2::Point2d(0, 2));
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy, uo2_lattice,
                                      assembly_pitch * um2::Point2d(1, 1));

  // Create MOX lattices
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy, mox_lattice,
                                      assembly_pitch * um2::Point2d(0, 1));
  factory::addCylindricalPinLattice2D(pin_radii, pin_mats, dxdy, mox_lattice,
                                      assembly_pitch * um2::Point2d(1, 2));

  // Uncomment to view the geometry in Gmsh
  // um2::gmsh::fltk::run();

  // Construct the MPACT spatial partition using pin-modular ray tracing
  // (coarse cells map one-to-one with ray tracing modules)
  um2::mpact::SpatialPartition model;

  for (int i = 0; i < 7; ++i) {
    model.makeCoarseCell(pin_size);
    model.makeRTM({{i}});
  }

  model.makeLattice(uo2_lattice);
  model.makeLattice(mox_lattice);
  model.makeLattice(h2o_lattice);

  // The problem is 2D, so we may map each lattice one-to-one to an assembly
  model.makeAssembly({0}); // UO2
  model.makeAssembly({1}); // MOX
  model.makeAssembly({2}); // H2O

  // Make core
  // Core assembly IDs (pg. 6)
  std::vector<std::vector<int>> const core_assembly_ids = um2::to_vecvec<int>(R"(
      0 1 2
      1 0 2
      2 2 2
    )");
  model.makeCore(core_assembly_ids);

  // Overlay the spatial partition onto the domain
  um2::gmsh::model::occ::overlaySpatialPartition(model);
  um2::gmsh::model::mesh::setGlobalMeshSize(0.25);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);

  um2::gmsh::write("c5g7.inp");
  model.importCoarseCells("c5g7.inp");
  um2::exportMesh("c5g7.xdmf", model);
  um2::finalize();
  return 0;
}
