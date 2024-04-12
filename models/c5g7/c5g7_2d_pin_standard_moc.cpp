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

  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::royalblue);
  moderator.xsec().t() = {2.30070e-01, 7.76460e-01, 1.48420e+00,
 1.50520e+00, 1.55920e+00, 2.02540e+00,
 3.30570e+00};
  moderator.xsec().isMacro() = true;

  // Safety checks
  uo2.validate();
  moderator.validate();

  model.addMaterial(uo2);
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

  um2::Vector<MatID> mat_ids(48, 1);
  um2::fill(mat_ids.begin(), mat_ids.begin() + 24, static_cast<MatID>(0));
  model.addCoarseCell(xy_extents, cyl_pin_mesh_type, cyl_pin_id, mat_ids);
  model.addRTM({{0}});
  model.addLattice({{0}});
  model.addAssembly({0});
  model.addCore({{0}});

  model.write("c5g7_pin.xdmf");

  um2::finalize();
  return 0;
}
