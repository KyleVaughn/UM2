// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

// NOLINTBEGIN(misc-include-cleaner)

#include <um2.hpp>

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  //===========================================================================
  // Parse command line arguments
  //===========================================================================

  // Check the number of arguments
  if (argc != 4) {
    um2::String const exec_name(argv[0]);
    um2::logger::error("Usage: ", exec_name, " target_kn mfp_threshold mfp_scale");
    return 1;
  }

  char * end = nullptr;
  Float const target_kn = um2::strto<Float>(argv[1], &end);
  ASSERT(end != nullptr);
  ASSERT(target_kn > 0.0);
  end = nullptr;

  Float const mfp_threshold = um2::strto<Float>(argv[2], &end);
  ASSERT(end != nullptr);
  end = nullptr;

  Float const mfp_scale = um2::strto<Float>(argv[3], &end);
  ASSERT(end != nullptr);

  um2::logger::info("Target Knudsen number: ", target_kn);
  um2::logger::info("MFP threshold: ", mfp_threshold);
  um2::logger::info("MFP scale: ", mfp_scale);

  //===========================================================================
  // Model parameters
  //===========================================================================

  Float const radius = 0.54;          // Pin radius = 0.54 cm (pg. 3)
  Float const pin_pitch = 1.26;       // Pin pitch = 1.26 cm (pg. 3)
  Float const assembly_pitch = 21.42; // Assembly pitch = 21.42 cm (pg. 3)

  //===========================================================================
  // Materials
  //===========================================================================

  um2::Vector<um2::Material> const materials = um2::getC5G7Materials();
  auto const & uo2 = materials[0];
  auto const & mox43 = materials[1];
  auto const & mox70 = materials[2];
  auto const & mox87 = materials[3];
  auto const & fiss_chamber = materials[4];
  auto const & guide_tube = materials[5];
  auto const & moderator = materials[6];

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
  um2::Vector<um2::Vec2F> const xy_extents(6, {pin_pitch, pin_pitch});

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
  factory::addCylindricalPinLattice2D(uo2_lattice, xy_extents, pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2F(0, 2));
  factory::addCylindricalPinLattice2D(uo2_lattice, xy_extents, pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2F(1, 1));

  // Create MOX lattices
  factory::addCylindricalPinLattice2D(mox_lattice, xy_extents, pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2F(0, 1));
  factory::addCylindricalPinLattice2D(mox_lattice, xy_extents, pin_radii, pin_mats,
                                      assembly_pitch * um2::Point2F(1, 2));

  //===========================================================================
  // Overlay CMFD grid
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

  // Add a coarse grid that evenly subdivides the domain
  um2::Vec2F const domain_extents(3 * assembly_pitch, 3 * assembly_pitch);
  Int constexpr num_coarse_cells = 51;
  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
  model.addCoarseGrid(domain_extents, num_cells);
  um2::gmsh::model::occ::overlayCoarseGrid(model, moderator);

  //===========================================================================
  // Generate the mesh
  //===========================================================================

  um2::gmsh::model::mesh::setMeshFieldFromKnudsenNumber(2, model.materials(), target_kn,
                                                        mfp_threshold, mfp_scale);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::write("c5g7_2d.inp");

  //===========================================================================
  // Complete the MPACT model
  //===========================================================================

  model.importCoarseCellMeshes("c5g7_2d.inp");
  model.write("c5g7_2d.xdmf", /*write_knudsen_data=*/true);
  um2::finalize();
  return 0;
}
// NOLINTEND(misc-include-cleaner)
