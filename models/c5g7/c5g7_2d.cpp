// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

#include <um2.hpp>

auto
main(int argc, char** argv) -> int
{
  um2::initialize();

  // Check the number of arguments
  if (argc != 2) {
    um2::logger::error("Usage: ./c5g7_2d num_coarse_cells");
    return 1;
  }

  //===========================================================================
  // Parametric study parameters
  //===========================================================================

  char * end = nullptr;
  Int const num_coarse_cells = um2::strto<Int>(argv[1], &end);
  ASSERT(end != nullptr);
  ASSERT(num_coarse_cells > 0);

  //===========================================================================
  // Model parameters
  //===========================================================================

  Float const radius = 0.54;          // Pin radius = 0.54 cm (pg. 3)
  Float const pin_pitch = 1.26;       // Pin pitch = 1.26 cm (pg. 3)
  Float const assembly_pitch = 21.42; // Assembly pitch = 21.42 cm (pg. 3)

  //===========================================================================
  // Materials
  //===========================================================================
  // See tables for cross sections

  um2::Material uo2;
  uo2.setName("UO2");
  uo2.setColor(um2::forestgreen);
  uo2.xsec().t() = {2.12450e-01, 3.55470e-01, 4.85540e-01, 5.59400e-01,
                    3.18030e-01, 4.01460e-01, 5.70610e-01};
  uo2.xsec().isMacro() = true;

  um2::Material mox43;
  mox43.setName("MOX_4.3");
  mox43.setColor(um2::yellow);
  mox43.xsec().t() = {2.11920e-01, 3.55810e-01, 4.88900e-01, 5.71940e-01,
    4.32390e-01, 6.84950e-01, 6.88910e-01};
  mox43.xsec().isMacro() = true;

  um2::Material mox70;
  mox70.setName("MOX_7.0");
  mox70.setColor(um2::orange);
  mox70.xsec().t() = {2.14540e-01, 3.59350e-01, 4.98910e-01,
                      5.96220e-01, 4.80350e-01, 8.39360e-01,
                      8.59480e-01};
  mox70.xsec().isMacro() = true;

  um2::Material mox87;
  mox87.setName("MOX_8.7");
  mox87.setColor(um2::red);
  mox87.xsec().t() = {2.16280e-01, 3.61700e-01, 5.05630e-01,
 6.11170e-01, 5.08900e-01, 9.26670e-01,
 9.60990e-01};
  mox87.xsec().isMacro() = true;

  um2::Material fiss_chamber;
  fiss_chamber.setName("Fission_Chamber");
  fiss_chamber.setColor(um2::black);
  fiss_chamber.xsec().t() = {1.90730e-01, 4.56520e-01, 6.40700e-01,
 6.49840e-01, 6.70630e-01, 8.75060e-01,
 1.43450e+00};
  fiss_chamber.xsec().isMacro() = true;

  um2::Material guide_tube;
  guide_tube.setName("Guide_Tube");
  guide_tube.setColor(um2::darkgrey);
  guide_tube.xsec().t() = {1.90730e-01, 4.56520e-01, 6.40670e-01,
 6.49670e-01, 6.70580e-01, 8.75050e-01,
 1.43450e+00};
  guide_tube.xsec().isMacro() = true;

  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::royalblue);
  moderator.xsec().t() = {2.30070e-01, 7.76460e-01, 1.48420e+00,
 1.50520e+00, 1.55920e+00, 2.02540e+00,
 3.30570e+00};
  moderator.xsec().isMacro() = true;

  // Safety checks
  uo2.validateXSec();
  mox43.validateXSec();
  mox70.validateXSec();
  mox87.validateXSec();
  fiss_chamber.validateXSec();
  guide_tube.validateXSec();
  moderator.validateXSec();

  //===========================================================================
  // Geometry
  //===========================================================================

  // Pin ID  |  Material
  // --------+----------------
  // 0       |  UO2
  // 1       |  MOX 4.3%
  // 2       |  MOX 7.0%
  // 3       |  MOX 8.7%
  // 4       |  Fission Chamber
  // 5       |  Guide Tube

  // Each pin has the same radius and pitch
  um2::Vector<um2::Vector<Float>> const pin_radii(6, {radius});
  um2::Vec2F const pin_size = {pin_pitch, pin_pitch};
  um2::Vector<um2::Vec2F> const xy_extents(6, pin_size);

  // Each pin contains a single material
  um2::Vector<um2::Vector<um2::Material>> const pin_mats = {
      {uo2}, {mox43}, {mox70}, {mox87}, {fiss_chamber}, {guide_tube}};

  // UO2 lattice pins (pg. 7)
  um2::Vector<um2::Vector<Int>> const uo2_lattice = um2::stringToLattice<Int>(R"(
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
  um2::Vector<um2::Vector<Int>> const mox_lattice = um2::stringToLattice<Int>(R"(
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

  // Ensure the lattices are the correct size
  ASSERT(uo2_lattice.size() == 17);
  ASSERT(uo2_lattice[0].size() == 17);
  ASSERT(mox_lattice.size() == 17);
  ASSERT(mox_lattice[0].size() == 17);

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
  factory::addCylindricalPinLattice2D(uo2_lattice, xy_extents,
                                      pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2(0, 2));
  factory::addCylindricalPinLattice2D(uo2_lattice, xy_extents,
                                      pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2(1, 1));

  // Create MOX lattices
  factory::addCylindricalPinLattice2D(mox_lattice, xy_extents,
                                      pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2(0, 1));
  factory::addCylindricalPinLattice2D(mox_lattice, xy_extents,
                                      pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2(1, 2));

  //===========================================================================
  // Overlay CMFD mesh
  //===========================================================================

  // Construct the MPACT model
  um2::mpact::Model model;
  model.addMaterial(uo2);
  model.addMaterial(mox43);
  model.addMaterial(mox70);
  model.addMaterial(mox87);
  model.addMaterial(fiss_chamber);
  model.addMaterial(guide_tube);
  model.addMaterial(moderator);
  std::vector<int> const is_fuel = {1, 1, 1, 1, 0, 0, 0};

  // Add a coarse grid that evenly subdivides the domain
  um2::Vec2F const domain_extents(3 * assembly_pitch, 3 * assembly_pitch);
  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
  model.addCoarseGrid(domain_extents, num_cells);
  um2::gmsh::model::occ::overlayCoarseGrid(model, moderator);

  //===========================================================================
  // Generate the mesh
  //===========================================================================

//  um2::gmsh::model::mesh::setGlobalMeshSize(pin_pitch / 12);
  Float const kn_target = 5.0;
  Float const mfp_threshold = 4.0;
  Float const mfp_scale = 1.2;
  um2::gmsh::model::mesh::setMeshFieldFromKnudsenNumber(
      2, model.materials(), kn_target, mfp_threshold, mfp_scale, is_fuel);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::write("c5g7_2d.inp");

  //===========================================================================
  // Complete the MPACT model and write the mesh
  //===========================================================================

  model.importCoarseCellMeshes("c5g7_2d.inp");
  model.writeOpticalThickness("c5g7_2d_optical_thickness.xdmf");
  model.write("c5g7_2d.xdmf", /*write_knudsen_data=*/true, /*write_xsec_data=*/true);
  um2::finalize();
  return 0;
}
