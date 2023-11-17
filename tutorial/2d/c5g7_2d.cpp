// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

#include <um2.hpp>

#include <iostream>

auto
main() -> int
{
  um2::initialize("trace");

  // Parameters
  double const radius = 0.54;          // Pin radius = 0.54 cm (pg. 3)
  double const pin_pitch = 1.26;       // Pin pitch = 1.26 cm (pg. 3)
  double const assembly_pitch = 21.42; // Assembly pitch = 21.42 cm (pg. 3)

  // Cross sections
  // We only need total cross section to compute the Knudsen number
  um2::Vector<double> const uo2_xs = {2.12450e-01, 3.55470e-01, 4.85540e-01, 5.59400e-01,
                                      3.18030e-01, 4.01460e-01, 5.70610e-01};
  um2::Vector<double> const mox43_xs = {2.11920e-01, 3.55810e-01, 4.88900e-01,
                                        5.71940e-01, 4.32390e-01, 6.84950e-01,
                                        6.88910e-01};
  um2::Vector<double> const mox70_xs = {2.14540e-01, 3.59350e-01, 4.98910e-01,
                                        5.96220e-01, 4.80350e-01, 8.39360e-01,
                                        8.59480e-01};
  um2::Vector<double> const mox87_xs = {2.16280e-01, 3.61700e-01, 5.05630e-01,
                                        6.11170e-01, 5.08900e-01, 9.26670e-01,
                                        9.60990e-01};
  um2::Vector<double> const fiss_chamber_xs = {1.90730e-01, 4.56520e-01, 6.40700e-01,
                                               6.49840e-01, 6.70630e-01, 8.75060e-01,
                                               1.43450e+00};
  um2::Vector<double> const guide_tube_xs = {1.90730e-01, 4.56520e-01, 6.40670e-01,
                                             6.49670e-01, 6.70580e-01, 8.75050e-01,
                                             1.43450e+00};
  um2::Vector<double> const moderator_xs = {2.30070e-01, 7.76460e-01, 1.48420e+00,
                                            1.50520e+00, 1.55920e+00, 2.02540e+00,
                                            3.30570e+00};

  // Materials
  um2::Material<double> uo2("UO2", "forestgreen");
  um2::Material<double> mox43("MOX_4.3", "orange");
  um2::Material<double> mox70("MOX_7.0", "yellow");
  um2::Material<double> mox87("MOX_8.7", "red");
  um2::Material<double> fiss_chamber("Fission_Chamber", "black");
  um2::Material<double> guide_tube("Guide_Tube", "darkgrey");
  um2::Material<double> moderator("Moderator", "royalblue");

  uo2.xs.t = uo2_xs;
  mox43.xs.t = mox43_xs;
  mox70.xs.t = mox70_xs;
  mox87.xs.t = mox87_xs;
  fiss_chamber.xs.t = fiss_chamber_xs;
  guide_tube.xs.t = guide_tube_xs;
  moderator.xs.t = moderator_xs;

  std::vector<um2::Material<double>> const materials = {
      uo2, mox43, mox70, mox87, fiss_chamber, guide_tube, moderator};

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
  std::vector<std::vector<um2::Material<double>>> const pin_mats = {
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
  um2::mpact::SpatialPartition<double, int> model;

  for (int i = 0; i < 7; ++i) {
    model.makeCoarseCell(pin_size);
    model.makeRTM({{i}});
  }

  model.stdMakeLattice(uo2_lattice);
  model.stdMakeLattice(mox_lattice);
  model.stdMakeLattice(h2o_lattice);

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
  model.stdMakeCore(core_assembly_ids);

  // Overlay the spatial partition onto the domain
  um2::gmsh::model::occ::overlaySpatialPartition(model);

  // Create the mesh
  double const target_kn = 10.0;
  um2::XSReductionStrategy const kn_strategy = um2::XSReductionStrategy::Mean;
  um2::gmsh::model::mesh::setMeshFieldFromKnudsenNumber(2, materials, target_kn,
                                                        kn_strategy);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);

  um2::gmsh::write("c5g7.inp");
  model.importCoarseCells("c5g7.inp");
  model.materials[6].xs.t = uo2_xs;
  model.materials[2].xs.t = mox43_xs;
  model.materials[3].xs.t = mox70_xs;
  model.materials[4].xs.t = mox87_xs;
  model.materials[0].xs.t = fiss_chamber_xs;
  model.materials[1].xs.t = guide_tube_xs;
  model.materials[5].xs.t = moderator_xs;
  um2::PolytopeSoup<double, int> soup;
  model.toPolytopeSoup(soup, /*write_kn=*/true);
  soup.write("c5g7.xdmf");
  //  um2::exportMesh("c5g7.xdmf", model);
  um2::finalize();
  return 0;
}
