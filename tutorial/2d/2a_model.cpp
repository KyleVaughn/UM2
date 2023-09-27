// Model reference:
//   VERA Core Physics Benchmark Progression Problem Specifications
//   Revision 4, August 29, 2014
//   CASL-U-2012-0131-004

#include <um2.hpp>

auto
//NOLINTNEXTLINE
main() -> int
{
  um2::initialize();

  // Parameters
  double const r_fuel = 0.4096;    // Pellet radius = 0.4096 cm (pg. 4)
  double const r_gap = 0.418;      // Inner clad radius = 0.418 cm (pg. 4)
  double const r_clad = 0.475;     // Outer clad radius = 0.475 cm (pg.4)
  double const r_gt_inner = 0.561; // Inner Guide Tube Radius = 0.561 cm (pg. 5)
  double const r_gt_outer = 0.602; // Outer Guide Tube Radius = 0.602 cm (pg. 5)
  double const r_it_inner = 0.559; // Inner Instrument Tube Radius = 0.559 cm (pg. 5)
  double const r_it_outer = 0.605; // Outer Instrument Tube Radius = 0.605 cm (pg. 5)
  double const pitch = 1.26;       // Pitch = 1.26 cm (pg. 4)
  double const half_gap = 0.4;     // Inter-Assembly Half Gap  = 0.04 cm (pg. 7)

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
  // Note that the IDs on the perimeter will change in the mesh
  // due to the half gap
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

  std::vector<um2::Vec2d> const dxdy(3, {pitch, pitch});
  um2::gmsh::model::occ::addCylindricalPinLattice2D(
      {p0_radii, p1_radii, p2_radii}, // radii
      {p0_mats, p1_mats, p2_mats},    // materials
      dxdy,                           // dx, dy of each pin cell
      pin_ids,                        // pin ids
      {half_gap, half_gap}            // offset (account for half gap)
  );

  //uncomment this line to view the output

  um2::gmsh::fltk::run();
  um2::gmsh::write("2a.brep", /*extra_info=*/true);

  um2::finalize();
  return 0;
}
