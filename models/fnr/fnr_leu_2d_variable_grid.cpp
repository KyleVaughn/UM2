// Model references:
// (1) Kerr, William, et al.
//     The Ford Nuclear Reactor demonstration project for the evaluation and
//     analysis of low enrichment fuel.
//     No. ANL/RERTR/TM--17. Argonne National Lab., 1991.
//
// (2) Kerr, William, et al.
//     The Ford Nuclear Reactor description and operation.
//     Rev 1, 4-62 (April 1965)
//     (Note this is the Red book)

//----------------------------------------------------------------------------
// ASSUMPTIONS
//----------------------------------------------------------------------------
// - The fuel plates are manufactured curved, and therefore the clad thickness
//   is the same on each side of the fuel meat. If the plates are manufactured
//   flat and then curved to fit into the fuel elements, then the clad thickness
//   will need to be adjusted to conserve volume.
//
//   To see this: flat_width * flat_thickness = area = (theta / 2) * (R^2 - r^2)
//   where theta is the angle of the circular arc, R is outer radius, and r is inner
//   radius. Let R = r + t, where t is the clad thickness. Then the area becomes
//   theta * t * (r + t/2). If theta is fixed (boundaries of fuel and clad align),
//   then A_inner = A_outer and r_inner != r_outer implies t_inner != t_outer.
//
// - The bottom fuel plate in an assembly is offset by water_gap / 2 from the bottom
//   of the side plate. (2), pg. III-2 states that the plates ate approximately 3
//   inches long. Assuming they maintain their curvature into the side plates, we
//   see that the plates must be shifted up by some amount.
//
// - The core is configured according to (1) Figure B-6 on page 353, BUT THE FUEL
//   IF FRESH

//----------------------------------------------------------------------------
// ISSUES
//----------------------------------------------------------------------------
// - I have no reference for the exact length of the fuel plates. Therefore the
//   indents in the side plates of empty elements are not modeled, since it is
//   assumed that this is not vital to the neutronics.
//
// - I have no reference for what an empty element looks like. Since in some core
//   configurations, the empty spaces are filled with full fuel elements, I am
//   going to assume that only the side plates are present in the empty elements.
//
// - I have no reference for the exact location of the guide plates. They are simply
//   places at what seems to be a reasonable location.
//
// - We approximate the control rods for no reason other than convenience. All that
//   needs to be done is to completely round the corners of the rod. The current
//   approximation stops just short of fully rounding the corners.
//
// - I have no reference for the tank size. I am using Riley's OpenMC model as a
//   reference for the tank size.
//

#include <um2.hpp>

// NOLINTBEGIN(misc-include-cleaner)

#include <um2/stdlib/utility/pair.hpp>

#include <vector>

//----------------------------------------------------------------------------
// Global geometry parameters
//----------------------------------------------------------------------------
Float constexpr in_to_cm = 2.54;
Float constexpr fuel_curvature_radius = 5.5 * in_to_cm; // (1), pg. 347
Float constexpr elem_x_pitch = 3.031 * in_to_cm;        // (1), pg. 348
Float constexpr elem_y_pitch = 3.189 * in_to_cm;        // (1), pg. 348

//----------------------------------------------------------------------------
// Global variables
//----------------------------------------------------------------------------
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> fuel_tags;
std::vector<int> clad_tags;
std::vector<int> borated_steel_tags;
std::vector<int> steel_tags;
std::vector<int> heavy_water_tags;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

//----------------------------------------------------------------------------
// getCircleCenter
//----------------------------------------------------------------------------
// Given two points on a circle, return the center of the circle
CONST auto
getCircleCenter(um2::Point2d const p0, um2::Point2d const p1,
                Float const r) -> um2::Point2d
{
  //        pc
  //       / |
  //   r  /  |
  //     /   | b
  //    /    |
  //   /     |
  // p0-----p2-----p1
  //     a      a

  Float const a = p0.distanceTo(p1) / 2;
  Float const b = um2::sqrt(r * r - a * a);
  um2::Point2d const p2 = um2::midpoint(p0, p1);
  um2::Vec2d const v = (p1 - p0).normalized();
  um2::Vec2d const u(v[1], -v[0]);
  return p2 + b * u;
}

//----------------------------------------------------------------------------
// makeFuelPlateLEURegular
//----------------------------------------------------------------------------
// Create an LEU Regular fuel plate given the x, y coordinates of the bottom left corner
// of the plate, where the fuel plate meets the side plate.
// Return the surface tags of the fuel and clad.
auto
makeFuelPlateLEURegular(Float const x0, Float const y0) -> um2::Vec2I
{
  // Given parameters
  //------------------
  Float constexpr fuel_meat_width = 2.4 * in_to_cm;       // (1), pg. 347
  Float constexpr fuel_meat_thickness = 0.032 * in_to_cm; // (1), pg. 347
  Float constexpr clad_thickness = 0.015 * in_to_cm;      // (1), pg. 347
  Float constexpr plate_to_plate = 2.564 * in_to_cm;      // (1), pg. 347

  // Computed parameters
  //---------------------
  // It is not clear whether the plates are manufactured flat or curved.
  // Therefore it is not clear if we need to modify the clad thickness/curvature
  // in order to conserve volume. For now, we will assume that the plates are
  // manufactured curved and that the clad thickness is the same on each side.

  // Total thickness of the plate: fuel_meat_thickness + 2 * clad_thickness
  Float constexpr plate_thickness = fuel_meat_thickness + 2 * clad_thickness;

  // Angle in radians of the arc that the fuel meat makes
  // arc_length = r * angle
  Float constexpr fuel_angle = fuel_meat_width / fuel_curvature_radius;

  // First create the the plate boundary curves. Note line0 and line1 are
  // straight up and down, since the side plate truncates the fuel plate
  // arc.
  //
  //                            arc1
  //                        ..........
  //              ..........          .........
  // p3..........                               ...........p2
  // |                                                     |
  // | line1                ..........                     | line0
  // |            ..........   arc0    .........           |
  // p0..........                               ...........p1

  um2::Point2d const p0(x0, y0);
  um2::Point2d const p1(x0 + plate_to_plate, y0);
  um2::Point2d const p2(x0 + plate_to_plate, y0 + plate_thickness);
  um2::Point2d const p3(x0, y0 + plate_thickness);

  auto const p0_tag = um2::gmsh::model::occ::addPoint(p0[0], p0[1], 0);
  auto const p1_tag = um2::gmsh::model::occ::addPoint(p1[0], p1[1], 0);
  auto const p2_tag = um2::gmsh::model::occ::addPoint(p2[0], p2[1], 0);
  auto const p3_tag = um2::gmsh::model::occ::addPoint(p3[0], p3[1], 0);

  // Add the circular arc between p0--p1 (arc0)
  um2::Point2d const c0 = getCircleCenter(p0, p1, fuel_curvature_radius);
  auto const c0_tag = um2::gmsh::model::occ::addPoint(c0[0], c0[1], 0);
  auto const arc0_tag = um2::gmsh::model::occ::addCircleArc(p0_tag, c0_tag, p1_tag);

  // Add the line between p1--p2 (line0)
  auto const line0_tag = um2::gmsh::model::occ::addLine(p1_tag, p2_tag);

  // Add the circular arc between p2--p3 (arc1)
  um2::Point2d const c1 = getCircleCenter(p3, p2, fuel_curvature_radius);
  auto const c1_tag = um2::gmsh::model::occ::addPoint(c1[0], c1[1], 0);
  auto const arc1_tag = um2::gmsh::model::occ::addCircleArc(p3_tag, c1_tag, p2_tag);

  // Add the line between p3--p0 (line1)
  auto const line1_tag = um2::gmsh::model::occ::addLine(p3_tag, p0_tag);

  // Next add the fuel meat curves. Note that line2 and line3 are not straight
  // up and down. They lie on lines which meet at the center of the circle
  // with radius fuel_curvature_radius that passes through the center of the
  // fuel meat
  //                                arc3
  //                            ..........
  //                  ..........          .........
  // p7..............                               ..............p6
  // \                                                           /
  //  \ line3                   ..........                      / line2
  //   \             ..........   arc2    .........            /
  //   p4..........                               ...........p5

  // In can be shown that the center of the circle that passes through the center
  // of the fuel meat is at cc = (c0 + c1) / 2. Additionally, it can be shown that
  // the thickness of the fuel meat is the same as if the plate were flat, since it
  // lies at the center of the arc.
  // Hence:
  //  - p4 = cc + (fuel_curvature_radius - fuel_meat_thickness / 2) * Vec(-theta/2)
  //  - p5 = cc + (fuel_curvature_radius - fuel_meat_thickness / 2) * Vec(theta/2)
  //  - p6 = cc + (fuel_curvature_radius + fuel_meat_thickness / 2) * Vec(theta/2)
  //  - p7 = cc + (fuel_curvature_radius + fuel_meat_thickness / 2) * Vec(-theta/2)
  // where theta = fuel_angle and Vec(theta) is the unit vector in the direction of theta,
  // centered at the midpoint of the arc.
  auto const cc = (c0 + c1) / 2;
  um2::Vec2d v(-um2::sin(fuel_angle / 2), um2::cos(fuel_angle / 2));
  ASSERT(v[0] < 0);
  ASSERT(v[1] > 0);
  um2::Point2d const p4 = cc + (fuel_curvature_radius - fuel_meat_thickness / 2) * v;
  um2::Point2d const p7 = cc + (fuel_curvature_radius + fuel_meat_thickness / 2) * v;
  v[0] = -v[0];
  um2::Point2d const p5 = cc + (fuel_curvature_radius - fuel_meat_thickness / 2) * v;
  um2::Point2d const p6 = cc + (fuel_curvature_radius + fuel_meat_thickness / 2) * v;

  auto const p4_tag = um2::gmsh::model::occ::addPoint(p4[0], p4[1], 0);
  auto const p5_tag = um2::gmsh::model::occ::addPoint(p5[0], p5[1], 0);
  auto const p6_tag = um2::gmsh::model::occ::addPoint(p6[0], p6[1], 0);
  auto const p7_tag = um2::gmsh::model::occ::addPoint(p7[0], p7[1], 0);

  // Add the circular arc between p4--p5 (arc2)
  um2::Point2d const c2 = getCircleCenter(p4, p5, fuel_curvature_radius);
  auto const c2_tag = um2::gmsh::model::occ::addPoint(c2[0], c2[1], 0);
  auto const arc2_tag = um2::gmsh::model::occ::addCircleArc(p4_tag, c2_tag, p5_tag);

  // Add the line between p5--p6 (line2)
  auto const line2_tag = um2::gmsh::model::occ::addLine(p5_tag, p6_tag);

  // Add the circular arc between p6--p7 (arc3)
  um2::Point2d const c3 = getCircleCenter(p7, p6, fuel_curvature_radius);
  auto const c3_tag = um2::gmsh::model::occ::addPoint(c3[0], c3[1], 0);
  auto const arc3_tag = um2::gmsh::model::occ::addCircleArc(p7_tag, c3_tag, p6_tag);

  // Add the line between p7--p4 (line3)
  auto const line3_tag = um2::gmsh::model::occ::addLine(p7_tag, p4_tag);

  // Add the curve loops
  auto const outer_loop_tag =
      um2::gmsh::model::occ::addCurveLoop({arc0_tag, line0_tag, arc1_tag, line1_tag});
  auto const inner_loop_tag =
      um2::gmsh::model::occ::addCurveLoop({arc2_tag, line2_tag, arc3_tag, line3_tag});

  // Add the inner and outer surfaces
  auto const outer_surface_tag =
      um2::gmsh::model::occ::addPlaneSurface({outer_loop_tag, inner_loop_tag});
  auto const inner_surface_tag = um2::gmsh::model::occ::addPlaneSurface({inner_loop_tag});

  return {inner_surface_tag, outer_surface_tag};
}

//----------------------------------------------------------------------------
// makeFuelElementLEURegular
//----------------------------------------------------------------------------
// Create an LEU Regular fuel element given the x, y coordinates of the bottom left corner
// of the element
void
makeFuelElementLEURegular(Float const x0, Float const y0)
{
  // Given parameters
  //------------------
  Float constexpr side_plate_width = 3.15 * in_to_cm;      // (1), pg. 347
  Float constexpr side_plate_thickness = 0.189 * in_to_cm; // (1), pg. 347
  Float constexpr plate_to_plate = 2.564 * in_to_cm;       // (1), pg. 347
  Float constexpr water_gap = 0.115 * in_to_cm;            // (1), pg. 347
  Float constexpr unit_cell_thickness = 0.177 * in_to_cm;  // (1), pg. 347
  Int constexpr num_plates = 18;                           // (1), pg. 347

  // The x and y extents of the fuel element
  Float constexpr x_extent = plate_to_plate + 2 * side_plate_thickness;
  Float constexpr y_extent = side_plate_width;

  // Therefore, to center plates in the element we need to shift the geometry by:
  Float constexpr x_shift = (elem_x_pitch - x_extent) / 2;
  Float constexpr y_shift = (elem_y_pitch - y_extent) / 2;

  for (Int i = 0; i < num_plates; ++i) {
    Float const x = x0 + side_plate_thickness + x_shift;
    Float const y = y0 + i * unit_cell_thickness + water_gap / 2 + y_shift;
    auto const tags = makeFuelPlateLEURegular(x, y);
    fuel_tags.push_back(tags[0]);
    clad_tags.push_back(tags[1]);
  }
  Float x = x0 + x_shift;
  Float y = y0 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, side_plate_thickness,
                                                          side_plate_width));
  x = x0 + plate_to_plate + side_plate_thickness + x_shift;
  y = y0 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, side_plate_thickness,
                                                          side_plate_width));
}

//----------------------------------------------------------------------------
// makeFuelElementEmpty
//----------------------------------------------------------------------------
// Create an empty fuel element given the x, y coordinates of the bottom left corner
// of the element
void
makeFuelElementEmpty(Float const x0, Float const y0)
{
  // Given parameters
  //------------------
  Float constexpr side_plate_width = 3.15 * in_to_cm;      // (1), pg. 347
  Float constexpr side_plate_thickness = 0.189 * in_to_cm; // (1), pg. 347
  Float constexpr plate_to_plate = 2.564 * in_to_cm;       // (1), pg. 347

  // The x and y extents of the fuel element
  Float constexpr x_extent = plate_to_plate + 2 * side_plate_thickness;
  Float constexpr y_extent = side_plate_width;

  // Therefore, to center plates in the element we need to shift the geometry by:
  Float constexpr x_shift = (elem_x_pitch - x_extent) / 2;
  Float constexpr y_shift = (elem_y_pitch - y_extent) / 2;

  Float x = x0 + x_shift;
  Float y = y0 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, side_plate_thickness,
                                                          side_plate_width));
  x = x0 + plate_to_plate + side_plate_thickness + x_shift;
  y = y0 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, side_plate_thickness,
                                                          side_plate_width));
}

//----------------------------------------------------------------------------
// makeControlRod
//----------------------------------------------------------------------------
// Create a control rod given the x, y of the center of the rod and a vector
// of tags which will be used to tag the rod
void
makeControlRod(Float const x0, Float const y0, std::vector<int> & rod_tags)
{
  // Given parameters
  //------------------
  // These values are converted from mm to cm
  Float constexpr rod_thickness = 2.198; // (1), pg. 346
  Float constexpr rod_width = 5.668;     // (1), pg. 346
  Float constexpr rod_radius =
      rod_thickness / 2; // (1), pg. 346 perfectly rounded corners

  Float const x = x0 - rod_width / 2;
  Float const y = y0 - rod_thickness / 2;
  rod_tags.push_back(um2::gmsh::model::occ::addRectangle(
      x, y, 0, rod_width, rod_thickness, -1, rod_radius * 0.999));
}

//----------------------------------------------------------------------------
// makeFuelElementLEUSpecial
//----------------------------------------------------------------------------
// Create an LEU Special fuel element given the x, y coordinates of the bottom left corner
// of the element
void
makeFuelElementLEUSpecial(Float const x0, Float const y0, bool make_ctrl,
                          std::vector<int> & rod_tags)
{
  // Given parameters
  //------------------
  Float constexpr side_plate_width = 3.15 * in_to_cm;      // (1), pg. 347
  Float constexpr side_plate_thickness = 0.189 * in_to_cm; // (1), pg. 347
  Float constexpr plate_to_plate = 2.564 * in_to_cm;       // (1), pg. 347
  Float constexpr water_gap = 0.115 * in_to_cm;            // (1), pg. 347
  Float constexpr unit_cell_thickness = 0.177 * in_to_cm;  // (1), pg. 347
  // This is num plates in the regular assembly. Actual num plates here is 9
  // (from same source and page)
  Int constexpr num_plates = 18;                            // (1), pg. 347
  Float constexpr guide_plate_width = 2.564 * in_to_cm;     // (1), pg. 347
  Float constexpr guide_plate_thickness = 0.125 * in_to_cm; // (1), pg. 347

  // The x and y extents of the fuel element
  Float constexpr x_extent = plate_to_plate + 2 * side_plate_thickness;
  Float constexpr y_extent = side_plate_width;

  // Therefore, to center plates in the element we need to shift the geometry by:
  Float constexpr x_shift = (elem_x_pitch - x_extent) / 2;
  Float constexpr y_shift = (elem_y_pitch - y_extent) / 2;

  for (Int i = 0; i < 5; ++i) {
    Float const x = x0 + side_plate_thickness + x_shift;
    Float const y = y0 + i * unit_cell_thickness + water_gap / 2 + y_shift;
    auto const tags = makeFuelPlateLEURegular(x, y);
    fuel_tags.push_back(tags[0]);
    clad_tags.push_back(tags[1]);
  }
  // The middle 9 plates are skipped
  for (Int i = 14; i < num_plates; ++i) {
    Float const x = x0 + side_plate_thickness + x_shift;
    Float const y = y0 + i * unit_cell_thickness + water_gap / 2 + y_shift;
    auto const tags = makeFuelPlateLEURegular(x, y);
    fuel_tags.push_back(tags[0]);
    clad_tags.push_back(tags[1]);
  }

  // Side plates
  Float x = x0 + x_shift;
  Float y = y0 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, side_plate_thickness,
                                                          side_plate_width));
  x = x0 + plate_to_plate + side_plate_thickness + x_shift;
  y = y0 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, side_plate_thickness,
                                                          side_plate_width));

  // Guide plates
  x = x0 + side_plate_thickness + x_shift;
  y = y0 + 6 * unit_cell_thickness + water_gap / 2 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, guide_plate_width,
                                                          guide_plate_thickness));
  y = y0 + 13 * unit_cell_thickness + water_gap / 2 + y_shift;
  clad_tags.push_back(um2::gmsh::model::occ::addRectangle(x, y, 0, guide_plate_width,
                                                          guide_plate_thickness));

  // Control rod
  if (make_ctrl) {
    x += guide_plate_width / 2;
    // Evenly space between the guide plates
    Float const ybot =
        y0 + 6 * unit_cell_thickness + water_gap / 2 + y_shift + guide_plate_thickness;
    Float const ytop = y0 + 13 * unit_cell_thickness + water_gap / 2 + y_shift;
    y = (ybot + ytop) / 2;
    makeControlRod(x, y, rod_tags);
  }
}

//----------------------------------------------------------------------------
// makeD2OTank
//----------------------------------------------------------------------------
// Given the lower left corener of the tank, create the D2O tank
void
makeD2OTank(Float const x0, Float const y0)
{
  // These values are from Riley's OpenMC model. Need to find the source.
  Float constexpr tank_thickness = 2.54 / 2;
  Float constexpr tank_outer_height = 64.80084;
  Float constexpr tank_outer_width = 30.8356;

  // Create the outer then inner points, lines, loops, and surfaces
  // Outer points
  auto const p0_tag = um2::gmsh::model::occ::addPoint(x0, y0, 0);
  auto const p1_tag = um2::gmsh::model::occ::addPoint(x0 + tank_outer_width, y0, 0);
  auto const p2_tag =
      um2::gmsh::model::occ::addPoint(x0 + tank_outer_width, y0 + tank_outer_height, 0);
  auto const p3_tag = um2::gmsh::model::occ::addPoint(x0, y0 + tank_outer_height, 0);
  // Inner points
  auto const p4_tag =
      um2::gmsh::model::occ::addPoint(x0 + tank_thickness, y0 + tank_thickness, 0);
  auto const p5_tag = um2::gmsh::model::occ::addPoint(
      x0 + tank_outer_width - tank_thickness, y0 + tank_thickness, 0);
  auto const p6_tag = um2::gmsh::model::occ::addPoint(
      x0 + tank_outer_width - tank_thickness, y0 + tank_outer_height - tank_thickness, 0);
  auto const p7_tag = um2::gmsh::model::occ::addPoint(
      x0 + tank_thickness, y0 + tank_outer_height - tank_thickness, 0);

  // Outer lines
  auto const line0_tag = um2::gmsh::model::occ::addLine(p0_tag, p1_tag);
  auto const line1_tag = um2::gmsh::model::occ::addLine(p1_tag, p2_tag);
  auto const line2_tag = um2::gmsh::model::occ::addLine(p2_tag, p3_tag);
  auto const line3_tag = um2::gmsh::model::occ::addLine(p3_tag, p0_tag);
  // Inner lines
  auto const line4_tag = um2::gmsh::model::occ::addLine(p4_tag, p5_tag);
  auto const line5_tag = um2::gmsh::model::occ::addLine(p5_tag, p6_tag);
  auto const line6_tag = um2::gmsh::model::occ::addLine(p6_tag, p7_tag);
  auto const line7_tag = um2::gmsh::model::occ::addLine(p7_tag, p4_tag);

  // Outer loop
  auto const outer_loop_tag =
      um2::gmsh::model::occ::addCurveLoop({line0_tag, line1_tag, line2_tag, line3_tag});
  // Inner loop
  auto const inner_loop_tag =
      um2::gmsh::model::occ::addCurveLoop({line4_tag, line5_tag, line6_tag, line7_tag});

  // Outer surface
  auto const outer_surface_tag =
      um2::gmsh::model::occ::addPlaneSurface({outer_loop_tag, inner_loop_tag});
  auto const inner_surface_tag = um2::gmsh::model::occ::addPlaneSurface({inner_loop_tag});

  clad_tags.push_back(outer_surface_tag);
  heavy_water_tags.push_back(inner_surface_tag);
}

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  // Check the number of arguments
  if (argc != 2) {
    um2::String const exec_name(argv[0]);
    um2::logger::error("Usage: ", exec_name, " num_coarse_cells");
    return 1;
  }

  //===========================================================================
  // Parametric study parameters
  //===========================================================================

  char * end = nullptr;
  Int const num_coarse_cells = um2::strto<Int>(argv[1], &end);
  ASSERT(end != nullptr);
  ASSERT(num_coarse_cells > 0);

  //============================================================================
  // Materials
  //============================================================================
  // From Riley's OpenMC model
  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);

  // Aluminum
  //---------------------------------------------------------------------------
  um2::Material aluminum;
  aluminum.setName("Aluminum");
  aluminum.setDensity(2.7); // g/cm^3
  aluminum.setTemperature(300.0);
  aluminum.setColor(um2::gray);
  aluminum.addNuclideWt("Al27", 0.9725);
  aluminum.addNuclideWt("Mg00", 0.01);
  aluminum.addNuclideWt("Si00", 0.006);
  aluminum.addNuclideWt("Fe00", 0.0035);
  aluminum.addNuclideWt("Cu63", 0.00205437585615717);
  aluminum.addNuclideWt("Cu65", 0.00094562414384283);
  aluminum.addNuclideWt("Cr00", 0.003);
  // No Zinc
  aluminum.addNuclideWt("Ti00", 0.0005);
  aluminum.addNuclideWt("Mn55", 0.0005);
  aluminum.populateXSec(xslib);

  // Water
  //---------------------------------------------------------------------------
  um2::Material h2o;
  h2o.setName("Water");
  h2o.setDensity(0.99821); // g/cm^3
  h2o.setTemperature(300.0);
  h2o.setColor(um2::blue);
  h2o.addNuclidesAtomPercent({"H1", "O16"}, {2.0 / 3.0, 1.0 / 3.0});
  h2o.populateXSec(xslib);

  // Heavy water
  //---------------------------------------------------------------------------
  um2::Material d2o;
  d2o.setName("HeavyWater");
  d2o.setDensity(1.11); // g/cm^3
  d2o.setTemperature(300.0);
  d2o.setColor(um2::darkblue);
  d2o.addNuclidesAtomPercent({"H1", "H2", "O16"}, {0.005 / 3.0, 1.995 / 3.0, 1.0 / 3.0});
  d2o.populateXSec(xslib);

  // LEU fuel
  //---------------------------------------------------------------------------
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setDensity(2.89924);
  fuel.setTemperature(300.0);
  fuel.setColor(um2::red);
  // Weights from (1), pg. 347
  Float const u235_wt = 140.61;
  Float const u234_wt = 1.51;
  Float const u236_wt = 0.75;
  Float const u238_wt = 8.04;
  Float const aluminum_wt = 908;
  Float const iron_wt = 3.7;
  Float const heu_wt = u235_wt + u234_wt + u236_wt + u238_wt + aluminum_wt + iron_wt;
  fuel.addNuclideWt("U235", u235_wt / heu_wt);
  fuel.addNuclideWt("U234", u234_wt / heu_wt);
  fuel.addNuclideWt("U236", u236_wt / heu_wt);
  fuel.addNuclideWt("U238", u238_wt / heu_wt);
  fuel.addNuclideWt("Al27", 0.9725 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Mg00", 0.01 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Si00", 0.006 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Cu63", 0.00205437585615717 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Cu65", 0.00094562414384283 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Cr00", 0.003 * aluminum_wt / heu_wt);
  // No Zinc
  fuel.addNuclideWt("Ti00", 0.0005 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Mn55", 0.0005 * aluminum_wt / heu_wt);
  fuel.addNuclideWt("Fe00", iron_wt / heu_wt + 0.0035 * aluminum_wt / heu_wt);
  fuel.populateXSec(xslib);

  // Borated steel
  //---------------------------------------------------------------------------
  um2::Material borated_steel;
  borated_steel.setName("BoratedSteel");
  borated_steel.setDensity(8.0369);
  borated_steel.setTemperature(300.0);
  borated_steel.setColor(um2::black);
  // From (1), pg. 346
  borated_steel.addNuclide("B10", 0.001108);
  borated_steel.addNuclide("B11", 0.005184);
  borated_steel.addNuclide("Fe00", 0.05644);
  borated_steel.addNuclide("Ni00", 0.0113);
  borated_steel.addNuclide("Cr00", 0.0164);
  borated_steel.populateXSec(xslib);

  // Steel
  //---------------------------------------------------------------------------
  um2::Material steel;
  steel.setName("Steel");
  steel.setDensity(7.85);
  steel.setTemperature(300.0);
  steel.setColor(um2::darkgray);
  // Same as above without boron
  steel.addNuclide("Fe00", 0.05644);
  steel.addNuclide("Ni00", 0.0113);
  steel.addNuclide("Cr00", 0.0164);
  steel.populateXSec(xslib);

  //============================================================================
  // Geometry
  //============================================================================
  // Element ID | Description
  // -----------|------------
  // 0          | Empty
  // 1          | Regular LEU
  // 2          | Special LEU / Shim rod
  // 3          | Special LEU / Control rod

  um2::Vec2F const domain_extents(100, 100); // Bounding box of the domain
  um2::Vec2F const core_bounds(77.02904 - 0.5 * 30.8356,
                               64.80084); // Bounding box of the core - 0.5 * D2O tank
  um2::Vec2F const core_offset = (domain_extents - core_bounds) / 2;
  ASSERT(core_offset[0] > 0);
  ASSERT(core_offset[1] > 0);

  //(1) Figure B-6 on page 353
  um2::Vector<um2::Vector<Int>> const core_layout = um2::stringToLattice<Int>(R"(
      0 0 0 0 0 0 
      0 0 0 1 0 0
      0 1 1 1 1 1
      0 1 2 1 2 1
      0 1 1 1 1 1
      0 1 3 1 2 1
      0 1 1 1 1 1
      0 0 0 0 0 0
    )");

  // Place each fuel element
  for (Int i = core_layout.size() - 1; i >= 0; --i) {
    Float const y = core_offset[1] + (core_layout.size() - i - 1) * elem_y_pitch;
    for (Int j = 0; j < core_layout[i].size(); ++j) {
      Float const x = core_offset[0] + j * elem_x_pitch;
      if (core_layout[i][j] == 0) {
        makeFuelElementEmpty(x, y);
      } else if (core_layout[i][j] == 1) {
        makeFuelElementLEURegular(x, y);
      } else if (core_layout[i][j] == 2) {
        makeFuelElementLEUSpecial(x, y, /*make_ctrl=*/true, borated_steel_tags);
      } else if (core_layout[i][j] == 3) {
        makeFuelElementLEUSpecial(x, y, /*make_ctrl=*/true, steel_tags);
      }
    }
  }
  // Place the D2O tank
  makeD2OTank(core_offset[0] + 6 * elem_x_pitch, core_offset[1]);
  um2::gmsh::model::occ::synchronize();

  // Add the physical groups
  um2::gmsh::model::addPhysicalGroup(2, clad_tags, -1, "Material_Aluminum");
  um2::gmsh::model::addPhysicalGroup(2, heavy_water_tags, -1, "Material_HeavyWater");
  um2::gmsh::model::addPhysicalGroup(2, fuel_tags, -1, "Material_Fuel");
  um2::gmsh::model::addPhysicalGroup(2, borated_steel_tags, -1, "Material_BoratedSteel");
  um2::gmsh::model::addPhysicalGroup(2, steel_tags, -1, "Material_Steel");

  // Color the physical groups
  std::vector<um2::Material> const materials = {aluminum, d2o, fuel, borated_steel,
                                                steel};
  // std::vector<um2::Material> const materials = {aluminum, d2o, fuel};
  um2::gmsh::model::occ::colorMaterialPhysicalGroupEntities(materials);

  //===========================================================================
  // Overlay CMFD mesh
  //===========================================================================

  // Construct the MPACT model
  um2::mpact::Model model;
  model.addMaterial(aluminum);
  model.addMaterial(h2o);
  model.addMaterial(d2o);
  model.addMaterial(fuel);
  model.addMaterial(borated_steel);
  model.addMaterial(steel);

  // Add a coarse grid that evenly subdivides the domain (quarter core)
  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
  model.addCoarseGrid(domain_extents, num_cells);
  um2::gmsh::model::occ::overlayCoarseGrid(model, h2o);

  //===========================================================================
  // Generate the mesh
  //===========================================================================

  um2::gmsh::model::mesh::setGlobalMeshSize(0.30);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::Tri);
  um2::gmsh::write("fnr_2d.inp");

  //===========================================================================
  // Complete the MPACT model and write the mesh
  //===========================================================================

  model.importCoarseCellMeshes("fnr_2d.inp");
  model.write("fnr_2d.xdmf");
  model.writeCMFDInfo("fnr_2d_cmfd_info.xdmf");
  um2::finalize();
  return 0;
}

// NOLINTEND(misc-include-cleaner)
