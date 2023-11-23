// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize("trace");

  // Geometry parameters
  double const radius = 0.54;    // Pin radius = 0.54 cm (pg. 3)
  double const pin_pitch = 1.26; // Pin pitch = 1.26 cm (pg. 3)

  // Mesh parameters
  double const water_radius = 0.62;
  Size const num_azimuthal = 8;
  Size const num_fuel_rings = 3;
  Size const num_water_rings = 2;

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

  // Hack for xsec assignment
  uo2.xs.t = uo2_xs;
  mox43.xs.t = mox43_xs;
  mox70.xs.t = mox70_xs;
  mox87.xs.t = mox87_xs;
  fiss_chamber.xs.t = fiss_chamber_xs;
  guide_tube.xs.t = guide_tube_xs;
  moderator.xs.t = moderator_xs;

  // Pin ID  |  Material
  // --------+----------------
  // 0       |  UO2
  // 1       |  MOX 4.3%
  // 2       |  MOX 7.0%
  // 3       |  MOX 8.7%
  // 4       |  Fission Chamber
  // 5       |  Guide Tube
  // 6       |  Moderator

  um2::Vector<um2::Material<double>> const materials = {
      uo2, mox43, mox70, mox87, fiss_chamber, guide_tube, moderator};

  // Construct the MPACT spatial partition using pin-modular ray tracing
  // (coarse cells map one-to-one with ray tracing modules)
  um2::mpact::SpatialPartition<double, int> model;

  // Set the model materials
  model.materials = materials;

  // Make the single pin mesh
  um2::Vec2d const pin_size = {pin_pitch, pin_pitch};
  model.makeCylindricalPinMesh({radius, water_radius}, pin_pitch,
                               {num_fuel_rings, num_water_rings}, num_azimuthal, 2);
  Size const nfaces = model.quadratic_quad.back().numFaces();

  // Make the coarse cells
  um2::Vector<MaterialID> mat_ids(nfaces, 6);
  for (Size i = 0; i < 6; ++i) {
    um2::fill(mat_ids.begin(),
              mat_ids.begin() + static_cast<ptrdiff_t>(num_fuel_rings * num_azimuthal),
              i);
    model.makeCoarseCell(pin_size, um2::MeshType::QuadraticQuad, 0, mat_ids);
    model.makeRTM({{i}});
  }

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

  //  // Moderator lattice
  //  std::vector<std::vector<int>> const h2o_lattice = um2::to_vecvec<int>(R"(
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //      6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
  //    )");

  model.stdMakeLattice(uo2_lattice);
  model.stdMakeLattice(mox_lattice);
  //  model.stdMakeLattice(h2o_lattice);

  // The problem is 2D, so we may map each lattice one-to-one to an assembly
  model.makeAssembly({0}); // UO2
  model.makeAssembly({1}); // MOX
                           //  model.makeAssembly({2}); // H2O

  // Make core
  // Core assembly IDs (pg. 6)
  //  std::vector<std::vector<int>> const core_assembly_ids = um2::to_vecvec<int>(R"(
  //      0 1 2
  //      1 0 2
  //      2 2 2
  //    )");
  std::vector<std::vector<int>> const core_assembly_ids = um2::to_vecvec<int>(R"(
      0 1
      1 0
    )");
  model.stdMakeCore(core_assembly_ids);

  um2::PolytopeSoup<double, int> soup;
  model.toPolytopeSoup(soup, /*write_kn=*/true);
  soup.write("c5g7.xdmf");
  um2::finalize();
  return 0;
}
