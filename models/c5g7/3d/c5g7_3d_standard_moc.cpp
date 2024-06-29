// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

#include <um2.hpp>
#include <um2/stdlib/algorithm/fill.hpp> // um2::fill

// NOLINTBEGIN(misc-include-cleaner)

auto
main() -> int
{
  um2::initialize();

  um2::mpact::Model model;

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

  model.addMaterial(uo2);
  model.addMaterial(mox43);
  model.addMaterial(mox70);
  model.addMaterial(mox87);
  model.addMaterial(fiss_chamber);
  model.addMaterial(guide_tube);
  model.addMaterial(moderator);

  //===========================================================================
  // Geometry
  //===========================================================================

  // Create pin meshes
  //---------------------------------------------------------------------------
  // For moderator only pin-cells, use a rectangular mesh
  // For all other cells, use a cylindrical mesh

  // Cylindrical pin mesh
  // 8 azimuthal divisions, order 2 mesh
  // The first 8 * 3 = 24 faces are the inner material
  // The next 8 * 2 + 8 = 24 faces are moderator
  Float const pin_pitch = 1.26; // Pin pitch = 1.26 cm (pg. 3)
  Float const radius = 0.54;    // Pin radius = 0.54 cm (pg. 3)
  um2::Vector<Int> const rings = {3, 2};
  um2::Vector<Float> const radii = {radius, 0.62};
  Int const num_azi = 8;
  auto const cyl_pin_mesh_type = um2::MeshType::QuadraticQuad;
  auto const cyl_pin_id =
      model.addCylindricalPinMesh(pin_pitch, radii, rings, num_azi, 2);

  // Rectangular pin mesh
  // 5 by 5 mesh for the reflector
  um2::Vec2F const xy_extents(pin_pitch, pin_pitch);
  auto const rect_pin_mesh_type = um2::MeshType::Quad;
  auto const rect_pin_id = model.addRectangularPinMesh(xy_extents, 5, 5);

  // Create coarse cells
  //---------------------------------------------------------------------------
  // Pin ID  |  Material
  // --------+----------------
  // 0       |  UO2
  // 1       |  MOX 4.3%
  // 2       |  MOX 7.0%
  // 3       |  MOX 8.7%
  // 4       |  Fission Chamber
  // 5       |  Guide Tube
  // 6       |  Moderator

  // Add the 6 cylindrical pins
  um2::Vector<MatID> mat_ids(48, 6);
  for (MatID i = 0; i < 6; ++i) {
    um2::fill(mat_ids.begin(), mat_ids.begin() + 24, i);
    model.addCoarseCell(xy_extents, cyl_pin_mesh_type, cyl_pin_id, mat_ids);
  }

  // Add the 1 rectangular pin
  mat_ids.resize(25);
  um2::fill(mat_ids.begin(), mat_ids.end(), static_cast<MatID>(6));
  model.addCoarseCell(xy_extents, rect_pin_mesh_type, rect_pin_id, mat_ids);

  // RTMs
  //---------------------------------------------------------------------------
  // Use pin-modular ray tracing

  um2::Vector<um2::Vector<Int>> ids = {{0}};
  for (Int i = 0; i < 7; ++i) {
    ids[0][0] = i;
    model.addRTM(ids);
  }

  // Lattices
  //---------------------------------------------------------------------------

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

  // Moderator lattice
  um2::Vector<um2::Vector<Int>> const h2o_lattice = um2::stringToLattice<Int>(R"( 
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

  // Ensure the lattices are the correct size
  ASSERT(uo2_lattice.size() == 17);
  ASSERT(uo2_lattice[0].size() == 17);
  ASSERT(mox_lattice.size() == 17);
  ASSERT(mox_lattice[0].size() == 17);
  ASSERT(h2o_lattice.size() == 17);
  ASSERT(h2o_lattice[0].size() == 17);

  model.addLattice(uo2_lattice);
  model.addLattice(mox_lattice);
  model.addLattice(h2o_lattice);

  // Assemblies
  //---------------------------------------------------------------------------
  // Evenly divide into 10 slices
  // The normal model is 60 slices, but use 10 for the test
  // The model is 9 parts fuel, 1 part moderator
  Float const model_height = 214.2;
  auto const num_slices = 10;
  auto const num_fuel_slices = 9 * num_slices / 10;
  um2::Vector<Int> lattice_ids(num_slices, 2); // Fill with H20
  um2::Vector<Float> z_slices(num_slices + 1);
  for (Int i = 0; i <= num_slices; ++i) {
    z_slices[i] = i * model_height / num_slices;
  }

  // uo2 assembly
  um2::fill(lattice_ids.begin(), lattice_ids.begin() + num_fuel_slices, 0);
  model.addAssembly(lattice_ids, z_slices);

  // mox assembly
  um2::fill(lattice_ids.begin(), lattice_ids.begin() + num_fuel_slices, 1);
  model.addAssembly(lattice_ids, z_slices);

  // moderator assembly
  um2::fill(lattice_ids.begin(), lattice_ids.begin() + num_slices, 2);
  model.addAssembly(lattice_ids, z_slices);

  // Core
  //---------------------------------------------------------------------------
  ids = um2::stringToLattice<Int>(R"(
      0 1 2
      1 0 2
      2 2 2
  )");
  ASSERT(ids.size() == 3);
  ASSERT(ids[0].size() == 3);

  model.addCore(ids);

  model.write("c5g7.xdmf");

  um2::finalize();
  return 0;
}

// NOLINTEND(misc-include-cleaner)
