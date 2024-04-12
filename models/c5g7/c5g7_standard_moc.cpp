// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

#include <um2.hpp>
#include <um2/stdlib/algorithm.hpp>

auto
main() -> int
{
  um2::initialize();

  um2::mpact::Model model;

  //===========================================================================
  // Materials
  //===========================================================================

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
  uo2.validate();
  mox43.validate();
  mox70.validate();
  mox87.validate();
  fiss_chamber.validate();
  guide_tube.validate();
  moderator.validate();

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

  // Pin meshes
  //---------------------------------------------------------------------------
  auto const radius = castIfNot<Float>(0.54);
  auto const pin_pitch = castIfNot<Float>(1.26);

  um2::Vec2F const xy_extents = {pin_pitch, pin_pitch};

  // Use the same mesh for all pins except the reflector
  um2::Vector<Float> const radii = {radius, castIfNot<Float>(0.62)};
  um2::Vector<Int> const rings = {3, 2};

  // 8 azimuthal divisions, order 2 mesh
  // The first 8 * 3 = 24 faces are the inner material
  // The next 8 * 2 + 8 = 24 faces are moderator
  auto const cyl_pin_mesh_type = um2::MeshType::QuadraticQuad; 
  auto const cyl_pin_id = model.addCylindricalPinMesh(pin_pitch, radii, rings, 8, 2);

  // 5 by 5 mesh for the reflector
  auto const rect_pin_mesh_type = um2::MeshType::Quad;
  auto const rect_pin_id = model.addRectangularPinMesh(xy_extents, 5, 5);

  // Coarse cells
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
  auto const model_height = castIfNot<Float>(214.2);
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
