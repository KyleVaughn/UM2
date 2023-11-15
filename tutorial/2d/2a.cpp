// Model reference:
//  VERA Core Physics Benchmark Progression Problem Specifications
//  Revision 4, August 29, 2014
//  CASL-U-2012-0131-004

#include <um2.hpp>

auto
main() -> int
{
  um2::initialize();

  // This will follow a similar structure to the 2a model model without
  // inter-assembly gap.

  // Parameters
  double const r_fuel = 0.4096;    // Pellet radius = 0.4096 cm (pg. 4)
  double const r_gap = 0.418;      // Inner clad radius = 0.418 cm (pg. 4)
  double const r_clad = 0.475;     // Outer clad radius = 0.475 cm (pg.4)
  double const r_gt_inner = 0.561; // Inner Guide Tube Radius = 0.561 cm (pg. 5)
  double const r_gt_outer = 0.602; // Outer Guide Tube Radius = 0.602 cm (pg. 5)
  double const r_it_inner = 0.559; // Inner Instrument Tube Radius = 0.559 cm (pg. 5)
  double const r_it_outer = 0.605; // Outer Instrument Tube Radius = 0.605 cm (pg. 5)
  double const pitch = 1.26;       // Pitch = 1.26 cm (pg. 4)
  double const half_gap = 0.04;    // Inter-assembly half gap = 0.04 cm (pg. 5)

  // Materials
  um2::Material const fuel("Fuel", "forestgreen");
  um2::Material const gap("Gap", "red");
  um2::Material const clad("Clad", "slategray");
  um2::Material const water("Water", "royalblue");

  // Group radii and materials into pins
  std::vector<double> const p0_radii = {r_fuel, r_gap, r_clad};  // Fuel pin
  std::vector<double> const p1_radii = {r_gt_inner, r_gt_outer}; // Guide tube
  std::vector<double> const p2_radii = {r_it_inner, r_it_outer}; // Instrument tube
  std::vector<um2::Material> const p0_mats = {fuel, gap, clad};
  std::vector<um2::Material> const p1_mats = {water, clad};
  std::vector<um2::Material> const p2_mats = {water, clad};

  // Fuel rod and guide tube layout (pg. 5)
  std::vector<std::vector<int>> const pin_ids = um2::to_vecvec<int>(R"(
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

  // Each pin will have the same spacing
  um2::Vec2d const pin_pitch = {pitch, pitch};

  // This time we offset the lattice by the inter-assembly half gap.
  um2::gmsh::model::occ::addCylindricalPinLattice2D(
      {p0_radii, p1_radii, p2_radii},    // radii of each pin
      {p0_mats, p1_mats, p2_mats},       // materials of each pin
      {pin_pitch, pin_pitch, pin_pitch}, // pitch of each pin
      pin_ids,                           // pin ids
      {half_gap, half_gap}               // offset (account for half gap)
  );

  // We can write the geometry to a file if we wish, but we will skip this
  // and move on to meshing.
  // um2::gmsh::write("2a.brep", /*extra_info=*/true);

  // Since the geometry has been offset by the half gap, the coarse cells
  // on the perimeter of the lattice will be different sizes. This means
  // we need more than just 3 unique coarse cell sizes.
  um2::Vec2d const interior_dxdy(pitch, pitch);
  um2::Vec2d const tall_dxdy(pitch, pitch + half_gap);
  um2::Vec2d const wide_dxdy(pitch + half_gap, pitch);
  um2::Vec2d const corner_dxdy(pitch + half_gap, pitch + half_gap);

  um2::mpact::SpatialPartition model;

  // Due to the half gap, the cells on the perimeter are extended
  std::vector<std::vector<int>> const cc_ids = um2::to_vecvec<int>(R"(
     7 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 8
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 4
     6 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 1 0 0 1 0 0 2 0 0 1 0 0 1 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
     6 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 4
     6 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 4
     6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
    10 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 9)");

  // Construct coarse cells
  model.makeCoarseCell(interior_dxdy); // 0
  model.makeCoarseCell(interior_dxdy); // 1
  model.makeCoarseCell(interior_dxdy); // 2
  model.makeCoarseCell(tall_dxdy);     // 3
  model.makeCoarseCell(wide_dxdy);     // 4
  model.makeCoarseCell(tall_dxdy);     // 5
  model.makeCoarseCell(wide_dxdy);     // 6
  model.makeCoarseCell(corner_dxdy);   // 7
  model.makeCoarseCell(corner_dxdy);   // 8
  model.makeCoarseCell(corner_dxdy);   // 9
  model.makeCoarseCell(corner_dxdy);   // 10

  // All ray tracing modules must be identical in size. Therefore, we cannot map
  // the coarse cells one-to-one to RTMS, like in the model without the gap. Instead, we
  // create a single RTM corresponding to the entire lattice
  model.makeRTM(cc_ids);
  model.makeLattice({{0}});
  model.makeAssembly({0});
  model.makeCore({{0}});
  um2::gmsh::model::occ::overlaySpatialPartition(model, "Water");
  um2::gmsh::model::mesh::setGlobalMeshSize(0.1);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);

  um2::gmsh::write("2a.inp");
  model.importCoarseCells("2a.inp");
  um2::exportMesh("2a.xdmf", model);
  um2::finalize();
  return 0;
}
