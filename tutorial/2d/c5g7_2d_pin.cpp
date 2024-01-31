// Model reference:
//  BENCHMARK SPECIFICATION FOR DETERMINISTIC 2-D/3-D MOX FUEL ASSEMBLY
//  TRANSPORT CALCULATIONS WITHOUT SPATIAL HOMOGENISATION (C5G7 MOX)
//  NEA/NSC/DOC(2001)4

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize("info", /*init_gmsh=*/true, 99);

  // Parameters
  double const radius = 0.54;
  double const pin_pitch = 10 * 1.26; // Larger for demonstration purposes

  // Cross sections
  um2::Vector<double> const uo2_sigma_t = {2.12450e-01, 3.55470e-01, 4.85540e-01,
                                           5.59400e-01, 3.18030e-01, 4.01460e-01,
                                           5.70610e-01};
  um2::Vector<double> const moderator_sigma_t = {2.30070e-01, 7.76460e-01, 1.48420e+00,
                                                 1.50520e+00, 1.55920e+00, 2.02540e+00,
                                                 3.30570e+00};
  um2::CrossSection const uo2_xs(uo2_sigma_t);
  um2::CrossSection const moderator_xs(moderator_sigma_t);

  // Materials
  um2::Material uo2("UO2", um2::forestgreen);
  um2::Material moderator("Moderator", um2::royalblue);

  // Assign xs
  uo2.xs() = uo2_xs;
  moderator.xs() = moderator_xs;

  // Each pin has the same radius and pitch
  um2::Vec2d const pin_size = {pin_pitch, pin_pitch};

  // Make the geometry
  um2::gmsh::model::occ::addCylindricalPin2D({0.7 * pin_pitch, 0.9 * pin_pitch}, {radius},
                                             {uo2});
  um2::gmsh::model::occ::addCylindricalPin2D({0.9 * pin_pitch, 0.7 * pin_pitch}, {radius},
                                             {uo2});

  // Construct the MPACT spatial partition
  um2::mpact::SpatialPartition model;
  model.makeCoarseCell(pin_size);
  model.makeRTM({{0}});
  model.makeLattice({{0}});
  model.makeAssembly({0});
  model.makeCore({{0}});

  // Overlay the spatial partition onto the domain
  um2::gmsh::model::occ::overlaySpatialPartition(model);

  // Create the mesh
  double const target_kn = 8;
  um2::gmsh::model::mesh::setMeshFieldFromKnudsenNumber(
      2, {uo2, moderator}, target_kn,
      /*mfp_threshold=*/3.0, /*mfp_scale=*/2.0, /*is_fuel=*/{1, 0});
  //  um2::gmsh::model::mesh::coarsenModeratorFieldByFuelDistance(2, kn_field, {uo2},
  //  moderator);
  um2::gmsh::model::mesh::generate(2);
  // um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);

  um2::gmsh::fltk::run();

  //  um2::gmsh::write("c5g7.inp");
  //  model.importCoarseCells("c5g7.inp");
  //  for (auto const & cc : model.coarse_cells) {
  //    um2::Log::info("CC has " + um2::toString(cc.numFaces()) + " faces");
  //  }
  //  model.materials[6].xs.t = uo2_xs;
  //  model.materials[2].xs.t = mox43_xs;
  //  model.materials[3].xs.t = mox70_xs;
  //  model.materials[4].xs.t = mox87_xs;
  //  model.materials[0].xs.t = fiss_chamber_xs;
  //  model.materials[1].xs.t = guide_tube_xs;
  //  model.materials[5].xs.t = moderator_xs;
  //  // um2::PolytopeSoup<double, int> soup;
  //  // model.toPolytopeSoup(soup, /*write_kn=*/true);
  //  // soup.write("c5g7.xdmf");
  //  model.write("c5g7.xdmf", /*write_kn=*/true);
  um2::finalize();
  return 0;
}
