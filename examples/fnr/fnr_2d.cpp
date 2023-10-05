// Standard FNR model uses LEU fuel
//
// Model reference:
//  THE FORD NUCLEAR REACTOR DEMONSTRATION PROJECT FOR THE EVALUATION AND ANALYSIS OF
//  LOW ENRICHMENT FUEL
//  William Kerr, John S. King, John C. Lee, William R. Martin, and David K. Wehe
//  Report Number: ANL/RERTR/TM-17

#include <um2.hpp>

//==============================================================================
// Parameters (pg 347 Table B-2)
//==============================================================================
double constexpr fuel_meat_width_inch = 2.40; // Flat width of the fuel meat
double constexpr fuel_meat_thickness_inch = 0.032;
double constexpr water_gap_inch = 0.115; // Plate 1 bottom - Plate 0 top
double constexpr clad_thickness_inch = 0.015;
double constexpr unit_cell_pitch_inch = 0.177; // Plate 1 top - Plate 0 top
int constexpr num_fuel_plates = 18;
double constexpr side_plate_width_inch = 3.15;
double constexpr side_plate_thickness_inch = 0.189;
double constexpr side_plate_to_plate_spacing_inch = 2.564;
double constexpr fuel_curvature_radius_inch = 5.5; // The centerline

// Convert everything to cm (1 inch = 2.54 cm)
double constexpr fuel_meat_width = fuel_meat_width_inch * 2.54;
double constexpr fuel_meat_thickness = fuel_meat_thickness_inch * 2.54;
double constexpr water_gap = water_gap_inch * 2.54;
double constexpr clad_thickness = clad_thickness_inch * 2.54;
double constexpr unit_cell_pitch = unit_cell_pitch_inch * 2.54;
double constexpr side_plate_width = side_plate_width_inch * 2.54;
double constexpr side_plate_thickness = side_plate_thickness_inch * 2.54;
double constexpr side_plate_to_plate_spacing = side_plate_to_plate_spacing_inch * 2.54;
double constexpr fuel_curvature_radius = fuel_curvature_radius_inch * 2.54;

// Computed parameters
double constexpr fuel_thickness = fuel_meat_thickness + 2.0 * clad_thickness;

// Bottom left corner of the side plate is at (x, y)
void
addSidePlate(double x0, double y0, std::vector<int> & tags)
{
  auto const tag = um2::gmsh::model::occ::addRectangle(x0, y0, 0.0, side_plate_thickness,
                                                       side_plate_width);
  tags.push_back(tag);
}

auto
getMidpointArcOffset(double r, double a) -> double
{
  // Compute how much to offset the midpoint
  // in order to get the point that is the center of the circle arc
  // Suppose we have a triangle like so:
  //        a
  // p0-----------pm-----------p1
  //  \           |           /
  //   \          |          /
  //    \         |         /
  //     \        |        /
  //      \      b|       /
  //       \      |      /
  //      r \     |     /
  //         \    |    /
  //          \   |   /
  //           \  | t/ <-- theta
  //            \ | /
  //             \|/
  //              pc
  return std::sqrt(r * r - a * a);
}

// Bottom left corner of the fuel plate is at (x, y)
void
addFuelPlate(double x0, double y0, std::vector<int> & clad_tags,
             std::vector<int> & fuel_tags)
{
  // Get the angle from the center of the bottom clad arc to the ends of the plate
  double const r = fuel_curvature_radius;
  double const rb = r - fuel_thickness / 2.0;
  double const rt = r + fuel_thickness / 2.0;
  double const rbf = r - fuel_meat_thickness / 2.0;
  double const rtf = r + fuel_meat_thickness / 2.0;
  double const theta_b = std::asin(side_plate_to_plate_spacing / (2.0 * rb));
  double const theta = fuel_meat_width / (2.0 * r); // L = theta * r

  // Let p0 be the bottom left corner of the fuel plate
  // Let p1 be the bottom right corner of the fuel plate
  // Let p2 be the top right corner of the fuel plate
  // Let p3 be the top left corner of the fuel plate
  // y0 == y1
  // y2 == y3
  // x0 == x3
  // x1 == x2

  double const x1 = x0 + side_plate_to_plate_spacing;
  double const y1 = y0 + fuel_thickness * std::cos(theta_b);
  double const xm = (x0 + x1) / 2.0;

  auto const p0 = um2::gmsh::model::occ::addPoint(x0, y0, 0.0);
  auto const p1 = um2::gmsh::model::occ::addPoint(x1, y0, 0.0);
  auto const p2 = um2::gmsh::model::occ::addPoint(x1, y1, 0.0);
  auto const p3 = um2::gmsh::model::occ::addPoint(x0, y1, 0.0);
  double const ycb = y0 - getMidpointArcOffset(rb, side_plate_to_plate_spacing / 2.0);
  double const yct = y1 - getMidpointArcOffset(rt, side_plate_to_plate_spacing / 2.0);
  auto const pcb = um2::gmsh::model::occ::addPoint(xm, ycb, 0.0);
  auto const pct = um2::gmsh::model::occ::addPoint(xm, yct, 0.0);

  // Get the midpoint of the bottom arc
  double const yb = ycb + rb;
  // Get the center of the fuel midline arc
  double const yf = yb + fuel_thickness / 2.0;
  double const ycf = yf - r;

  // Compute the points for the fuel
  double const x4 = xm - rbf * std::sin(theta);
  double const x5 = xm + rbf * std::sin(theta);
  double const y4 = ycf + rbf * std::cos(theta); // y4 == y5
  double const x7 = xm - rtf * std::sin(theta);
  double const x6 = xm + rtf * std::sin(theta);
  double const y6 = ycf + rtf * std::cos(theta); // y6 == y7

  auto const pcf = um2::gmsh::model::occ::addPoint(xm, ycf, 0.0);
  auto const p4 = um2::gmsh::model::occ::addPoint(x4, y4, 0.0);
  auto const p5 = um2::gmsh::model::occ::addPoint(x5, y4, 0.0);
  auto const p6 = um2::gmsh::model::occ::addPoint(x6, y6, 0.0);
  auto const p7 = um2::gmsh::model::occ::addPoint(x7, y6, 0.0);

  auto const clad_bottom = um2::gmsh::model::occ::addCircleArc(p0, pcb, p1);
  auto const clad_right = um2::gmsh::model::occ::addLine(p1, p2);
  auto const clad_top = um2::gmsh::model::occ::addCircleArc(p2, pct, p3);
  auto const clad_left = um2::gmsh::model::occ::addLine(p3, p0);

  auto const fuel_bottom = um2::gmsh::model::occ::addCircleArc(p4, pcf, p5);
  auto const fuel_right = um2::gmsh::model::occ::addLine(p5, p6);
  auto const fuel_top = um2::gmsh::model::occ::addCircleArc(p6, pcf, p7);
  auto const fuel_left = um2::gmsh::model::occ::addLine(p7, p4);

  // Create a curve loop and surface for the clad
  auto const clad_loop =
      um2::gmsh::model::occ::addCurveLoop({clad_bottom, clad_right, clad_top, clad_left});
  auto const fuel_loop =
      um2::gmsh::model::occ::addCurveLoop({fuel_bottom, fuel_right, fuel_top, fuel_left});
  // auto const clad_surface =
  auto const clad_tag = um2::gmsh::model::occ::addPlaneSurface({clad_loop, fuel_loop});
  auto const fuel_tag = um2::gmsh::model::occ::addPlaneSurface({fuel_loop});
  clad_tags.push_back(clad_tag);
  fuel_tags.push_back(fuel_tag);
}

// Bottom left corner of the fuel assembly is at (x, y)
void
addFuelAssembly(double x, double y, std::vector<int> & clad_tags,
                std::vector<int> & fuel_tags)
{
  // Add left side plate
  addSidePlate(x, y, clad_tags);
  // Add right side plate
  addSidePlate(x + side_plate_thickness + side_plate_to_plate_spacing, y, clad_tags);
  // Add fuel plates
  for (int i = 0; i < num_fuel_plates; ++i) {
    double const x0 = x + side_plate_thickness;
    double const y0 = y + i * unit_cell_pitch + water_gap / 2.0;
    addFuelPlate(x0, y0, clad_tags, fuel_tags);
  }
}

auto
main() -> int
{
  um2::initialize();

  um2::Material const clad("Clad", "slategray");
  um2::Material const fuel("Fuel", "forestgreen");

  std::vector<int> clad_tags;
  std::vector<int> fuel_tags;

  double const center = 5.0;

  double const x0 = center - side_plate_thickness - side_plate_to_plate_spacing / 2.0;
  double const y0 = center - side_plate_width / 2.0; 
  addFuelAssembly(x0, y0, clad_tags, fuel_tags);

  // Set the physical groups
  um2::gmsh::model::occ::synchronize();
  um2::gmsh::model::addPhysicalGroup(2, clad_tags, -1, "Material_Clad");
  um2::gmsh::model::addPhysicalGroup(2, fuel_tags, -1, "Material_Fuel");

  um2::gmsh::vectorpair clad_dimtags(clad_tags.size());
  um2::gmsh::vectorpair fuel_dimtags(fuel_tags.size());
  for (size_t i = 0; i < clad_tags.size(); ++i) {
    clad_dimtags[i].first = 2;
    clad_dimtags[i].second = clad_tags[i];
  }
  for (size_t i = 0; i < fuel_tags.size(); ++i) {
    fuel_dimtags[i].first = 2;
    fuel_dimtags[i].second = fuel_tags[i];
  }
  um2::gmsh::model::setColor(clad_dimtags, clad.color.r(), clad.color.g(),
                             clad.color.b());
  um2::gmsh::model::setColor(fuel_dimtags, fuel.color.r(), fuel.color.g(),
                             fuel.color.b());

  um2::gmsh::vectorpair out_dimtags;
  std::vector<um2::gmsh::vectorpair> out_dimtags_map;
  um2::gmsh::model::occ::groupPreservingFragment(fuel_dimtags, clad_dimtags, out_dimtags,
                                                 out_dimtags_map, {fuel, clad});


  // Overlay a regular grid on the geometry
  um2::mpact::SpatialPartition model;
  size_t const num_cells = 2;

  // Size of the coarse cells
  double const dx = 2.0 * center / static_cast<double>(num_cells);
  um2::Vec2d const dxdy(dx, dx);

  std::vector<std::vector<int>> cc_ids(num_cells, std::vector<int>(num_cells, 0));
  for (size_t i = 0; i < num_cells; ++i) {
    for (size_t j = 0; j < num_cells; ++j) {
      cc_ids[i][j] = static_cast<int>(i * num_cells + j);
      model.makeCoarseCell(dxdy);
      model.makeRTM({{cc_ids[i][j]}});
    }
  }
  model.makeLattice(cc_ids);
  model.makeAssembly({0});
  model.makeCore({{0}});

  // Overlay the spatial partition
  um2::gmsh::model::occ::overlaySpatialPartition(model);

  um2::gmsh::model::mesh::setGlobalMeshSize(0.05);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::fltk::run();

  um2::gmsh::write("fnr.inp");

  um2::MeshFile<double, int> meshfile;
  um2::importMesh("fnr.inp", meshfile);
  um2::QuadraticTriMesh<2, double, int> const mesh(meshfile);
  um2::printStats(mesh);

//  model.importCoarseCells("fnr.inp");
//  um2::exportMesh("fnr.xdmf", model);
  um2::finalize();
  return 0;
}
