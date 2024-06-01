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
//

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
// - The core is configures according to (1) Figure B-7 on page 354.
//  

#include <um2.hpp>
#include <iostream>

//----------------------------------------------------------------------------
// Global geometry parameters
//----------------------------------------------------------------------------
Float constexpr in_to_cm = 2.54;
Float constexpr fuel_curvature_radius = 5.5 * in_to_cm; // (1), pg. 347

//----------------------------------------------------------------------------
// getCircleCenter
//----------------------------------------------------------------------------
// Given two points on a circle, return the center of the circle
CONST auto
getCircleCenter(um2::Point2 const p0, um2::Point2 const p1, Float const r) -> um2::Point2
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
  um2::Point2 const p2 = um2::midpoint(p0, p1);
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
  Float constexpr fuel_meat_width = 2.4 * in_to_cm; // (1), pg. 347
  Float constexpr fuel_meat_thickness = 0.032 * in_to_cm; // (1), pg. 347
  Float constexpr clad_thickness = 0.015 * in_to_cm; // (1), pg. 347
  Float constexpr plate_to_plate = 2.564 * in_to_cm; // (1), pg. 347

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

  um2::Point2 const p0(x0, y0);
  um2::Point2 const p1(x0 + plate_to_plate, y0);
  um2::Point2 const p2(x0 + plate_to_plate, y0 + plate_thickness);
  um2::Point2 const p3(x0, y0 + plate_thickness);

  auto const p0_tag = um2::gmsh::model::occ::addPoint(p0[0], p0[1], 0);
  auto const p1_tag = um2::gmsh::model::occ::addPoint(p1[0], p1[1], 0);
  auto const p2_tag = um2::gmsh::model::occ::addPoint(p2[0], p2[1], 0);
  auto const p3_tag = um2::gmsh::model::occ::addPoint(p3[0], p3[1], 0);

  // Add the circular arc between p0--p1 (arc0)
  um2::Point2 const c0 = getCircleCenter(p0, p1, fuel_curvature_radius);
  auto const c0_tag = um2::gmsh::model::occ::addPoint(c0[0], c0[1], 0);
  auto const arc0_tag = um2::gmsh::model::occ::addCircleArc(p0_tag, c0_tag, p1_tag);

  // Add the line between p1--p2 (line0)
  auto const line0_tag = um2::gmsh::model::occ::addLine(p1_tag, p2_tag);

  // Add the circular arc between p2--p3 (arc1)
  um2::Point2 const c1 = getCircleCenter(p3, p2, fuel_curvature_radius);
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
  um2::Point2 const p4 = cc + (fuel_curvature_radius - fuel_meat_thickness / 2) * v;
  um2::Point2 const p7 = cc + (fuel_curvature_radius + fuel_meat_thickness / 2) * v;
  v[0] = -v[0];
  um2::Point2 const p5 = cc + (fuel_curvature_radius - fuel_meat_thickness / 2) * v;
  um2::Point2 const p6 = cc + (fuel_curvature_radius + fuel_meat_thickness / 2) * v;

  auto const p4_tag = um2::gmsh::model::occ::addPoint(p4[0], p4[1], 0);
  auto const p5_tag = um2::gmsh::model::occ::addPoint(p5[0], p5[1], 0);
  auto const p6_tag = um2::gmsh::model::occ::addPoint(p6[0], p6[1], 0);
  auto const p7_tag = um2::gmsh::model::occ::addPoint(p7[0], p7[1], 0);

  // Add the circular arc between p4--p5 (arc2)
  um2::Point2 const c2 = getCircleCenter(p4, p5, fuel_curvature_radius);
  auto const c2_tag = um2::gmsh::model::occ::addPoint(c2[0], c2[1], 0);
  auto const arc2_tag = um2::gmsh::model::occ::addCircleArc(p4_tag, c2_tag, p5_tag);
  
  // Add the line between p5--p6 (line2)
  auto const line2_tag = um2::gmsh::model::occ::addLine(p5_tag, p6_tag);

  // Add the circular arc between p6--p7 (arc3)
  um2::Point2 const c3 = getCircleCenter(p7, p6, fuel_curvature_radius);
  auto const c3_tag = um2::gmsh::model::occ::addPoint(c3[0], c3[1], 0);
  auto const arc3_tag = um2::gmsh::model::occ::addCircleArc(p7_tag, c3_tag, p6_tag);

  // Add the line between p7--p4 (line3)
  auto const line3_tag = um2::gmsh::model::occ::addLine(p7_tag, p4_tag);

  // Add the curve loops
  auto const outer_loop_tag = um2::gmsh::model::occ::addCurveLoop({arc0_tag, line0_tag, arc1_tag, line1_tag});
  auto const inner_loop_tag = um2::gmsh::model::occ::addCurveLoop({arc2_tag, line2_tag, arc3_tag, line3_tag});

  // Add the inner and outer surfaces
  auto const outer_surface_tag = um2::gmsh::model::occ::addPlaneSurface({outer_loop_tag, inner_loop_tag});
  auto const inner_surface_tag = um2::gmsh::model::occ::addPlaneSurface({inner_loop_tag});

  return {inner_surface_tag, outer_surface_tag};
}

//----------------------------------------------------------------------------
// makeFuelElementLEURegular
//----------------------------------------------------------------------------
// Create an LEU Regular fuel element given the x, y coordinates of the bottom left corner
// of the side plate 
void
makeFuelElementLEURegular(Float const x0, Float const y0)
{
  // Given parameters
  //------------------
  Float constexpr side_plate_width = 3.15 * in_to_cm; // (1), pg. 347
  Float constexpr side_plate_thickness = 0.189 * in_to_cm; // (1), pg. 347
  Float constexpr plate_to_plate = 2.564 * in_to_cm; // (1), pg. 347
  Float constexpr water_gap = 0.115 * in_to_cm; // (1), pg. 347
  Float constexpr unit_cell_thickness = 0.177 * in_to_cm; // (1), pg. 347
  Int constexpr num_plates = 18; // (1), pg. 347

  um2::Vector<int> fuel_tags(18);
  um2::Vector<int> clad_tags(20); 
  for (Int i = 0; i < num_plates; ++i) {
    Float const x = x0 + side_plate_thickness;
    Float const y = y0 + i * unit_cell_thickness + water_gap / 2;
    auto const tags = makeFuelPlateLEURegular(x, y);
    fuel_tags[i] = tags[0];
    clad_tags[i] = tags[1];
  }
  clad_tags[18] =
    um2::gmsh::model::occ::addRectangle(x0, y0, 0, side_plate_thickness, side_plate_width); 
  clad_tags[19] =
    um2::gmsh::model::occ::addRectangle(x0 + plate_to_plate + side_plate_thickness, 
        y0, 0, side_plate_thickness, side_plate_width);
}


auto
//main(int argc, char** argv) -> int
main() -> int
{
  um2::initialize();

  // TODO: Shim. Control. Tank. Empty. Special

  //============================================================================
  // Materials
  //============================================================================

  // Moderator
  //---------------------------------------------------------------------------

  //============================================================================
  // Geometry
  //============================================================================

  // LEU Regular elements
  //---------------------------------------------------------------------------
//  Float const asy_width = 3.031 * in_to_cm; // (1), pg. 348
//  Float const asy_height = 3.189 * in_to_cm; // (1), pg. 348
  
  makeFuelElementLEURegular(0, 0);
  um2::gmsh::model::occ::synchronize();
  um2::gmsh::fltk::run();

//  // Fuel
//  //---------------------------------------------------------------------------
//  //um2::Vector<Float> const fuel_radii = {r_fuel, r_gap, r_clad};
//  //um2::Vector<um2::Material> const fuel_2110_materials = {fuel_2110, gap, clad};
//  //um2::Vector<um2::Material> const fuel_2619_materials = {fuel_2619, gap, clad};
//  um2::Vector<Float> const fuel_radii = {r_fuel, r_clad};
//  um2::Vector<um2::Material> const fuel_2110_materials = {fuel_2110, clad};
//  um2::Vector<um2::Material> const fuel_2619_materials = {fuel_2619, clad};
//
//  // Empty guide tube
//  //---------------------------------------------------------------------------
//  um2::Vector<Float> const gt_radii = {r_gt_inner, r_gt_outer};
//  um2::Vector<um2::Material> const gt_materials = {moderator, clad};
//
//  // Empty instrument tube
//  //---------------------------------------------------------------------------
//  um2::Vector<Float> const it_radii = {r_it_inner, r_it_outer};
//  um2::Vector<um2::Material> const it_materials = {moderator, clad};
//
//  // Pyrex
//  //---------------------------------------------------------------------------
//  // um2::Vector<Float> const pyrex_radii = {
//  //   r_pyrex_it_inner, r_pyrex_it_outer, r_pyrex_inner, r_pyrex_outer, r_pyrex_clad_inner,
//  //   r_pyrex_clad_outer, r_it_inner, r_it_outer
//  // };
//  // um2::Vector<um2::Material> const pyrex_materials = {
//  //   gap, ss304, gap, pyrex, gap, ss304, moderator, clad
//  // };
//  // Extend the ss304 to touch the pyrex (eliminate the gap)
//  um2::Vector<Float> const pyrex_radii = {
//    r_pyrex_it_inner, r_pyrex_inner, r_pyrex_outer,
//    r_pyrex_clad_outer, r_it_inner, r_it_outer
//  };
//  um2::Vector<um2::Material> const pyrex_materials = {
//    gap, ss304, pyrex, ss304, moderator, clad
//  };
//
//  // AIC
//  //---------------------------------------------------------------------------
//  //um2::Vector<Float> const aic_radii = {r_aic, r_aic_inner, r_aic_outer};
//  //um2::Vector<um2::Material> const aic_materials = {aic, gap, ss304};
//
//  um2::Vector<Float> const aic_radii = {r_aic, r_aic_outer};
//  um2::Vector<um2::Material> const aic_materials = {aic, ss304};
//
//  // Materials, radii, and xy_extents for each pin
//  //---------------------------------------------------------------------------
//  um2::Vector<um2::Vector<um2::Material>> const materials = {
//    fuel_2110_materials, gt_materials, it_materials, fuel_2619_materials, pyrex_materials,
//    aic_materials
//  };
//  um2::Vector<um2::Vector<Float>> const radii = {
//    fuel_radii, gt_radii, it_radii, fuel_radii, pyrex_radii, aic_radii
//  };
//  um2::Vector<um2::Vec2F> const xy_extents(6, pin_size);
//
//  // Lattice layout (Fig. 3, pg. 5)
//  um2::Vector<um2::Vector<Int>> const fuel_2110_lattice = um2::stringToLattice<Int>(R"(
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0
//      0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 1 0 0 1 0 0 2 0 0 1 0 0 1 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
//      0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//    )");
//
//  um2::Vector<um2::Vector<Int>> const fuel_2619_lattice = um2::stringToLattice<Int>(R"(
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 3 3 3 4 3 3 4 3 3 4 3 3 3 3 3
//      3 3 3 4 3 3 3 3 3 3 3 3 3 4 3 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 4 3 3 4 3 3 4 3 3 4 3 3 4 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 4 3 3 4 3 3 2 3 3 4 3 3 4 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 4 3 3 4 3 3 4 3 3 4 3 3 4 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 3 4 3 3 3 3 3 3 3 3 3 4 3 3 3
//      3 3 3 3 3 4 3 3 4 3 3 4 3 3 3 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
//    )");
//
//  um2::Vector<um2::Vector<Int>> const fuel_2110_aic_lattice = um2::stringToLattice<Int>(R"(
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
//      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 5 0 0 5 0 0 2 0 0 5 0 0 5 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
//      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//    )");
//
//  ASSERT(fuel_2110_lattice.size() == 17);
//  ASSERT(fuel_2110_lattice[0].size() == 17);
//  ASSERT(fuel_2619_lattice.size() == 17);
//  ASSERT(fuel_2619_lattice[0].size() == 17);
//
//  // Make the calls a bit more readable using an alias
//  namespace factory = um2::gmsh::model::occ;
//
//  // Create lattices
//  factory::addCylindricalPinLattice2D(
//      fuel_2110_lattice,
//      xy_extents,
//      radii,
//      materials,
//      /*offset=*/{assembly_pitch / 2 + inter_assembly_gap, inter_assembly_gap});
//
//  factory::addCylindricalPinLattice2D(
//      fuel_2619_lattice,
//      xy_extents,
//      radii,
//      materials,
//      /*offset=*/{-assembly_pitch / 2 + inter_assembly_gap, inter_assembly_gap});
//
//  factory::addCylindricalPinLattice2D(
//      fuel_2619_lattice,
//      xy_extents,
//      radii,
//      materials,
//      /*offset=*/{assembly_pitch / 2 + inter_assembly_gap, assembly_pitch + inter_assembly_gap});
//
//  factory::addCylindricalPinLattice2D(
//      fuel_2110_aic_lattice,
//      xy_extents,
//      radii,
//      materials,
//      /*offset=*/{-assembly_pitch / 2 + inter_assembly_gap, assembly_pitch + inter_assembly_gap});
//
//  //===========================================================================
//  // Overlay CMFD mesh
//  //===========================================================================
//
//  // Construct the MPACT model
//  um2::mpact::Model model;
//  model.addMaterial(fuel_2110);
//  model.addMaterial(fuel_2619);
//  model.addMaterial(gap);
//  model.addMaterial(clad);
//  model.addMaterial(moderator);
//  model.addMaterial(pyrex);
//  model.addMaterial(ss304);
//  model.addMaterial(aic);
//
//   // Add a coarse grid that evenly subdivides the domain (quarter core)
//  um2::Vec2F const domain_extents(1.5 * assembly_pitch, 1.5 * assembly_pitch);
//  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
//  model.addCoarseGrid(domain_extents, num_cells);
//  um2::gmsh::model::occ::overlayCoarseGrid(model, moderator);
//
//  //===========================================================================
//  // Generate the mesh
//  //===========================================================================
//
//  um2::gmsh::model::mesh::setGlobalMeshSize(pitch / 12);
//  um2::gmsh::model::mesh::generateMesh(um2::MeshType::Tri);
//  um2::gmsh::write("4_2d.inp");
//
//  //===========================================================================
//  // Complete the MPACT model and write the mesh
//  //===========================================================================
//
//  model.importCoarseCellMeshes("4_2d.inp");
//  model.writeCMFDInfo("4_2d_cmfd_info.xdmf");
//  model.write("4_2d.xdmf", /*write_knudsen_data=*/true, /*write_xsec_data=*/true);
  um2::finalize();
  return 0;
}
