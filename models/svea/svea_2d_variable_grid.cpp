// This model has no reference material, since actual SVEA assembly properties
// are proprietary. This model is simply a test model to demonstrate UM2.

// NOLINTBEGIN(misc-include-cleaner)

#include <um2.hpp>

#include <numeric>
#include <vector>

bool constexpr omit_gap = true; // Omit gap material in pins, fill with clad

Float constexpr r_fuel = 0.424;
Float constexpr r_gap = 0.4315;
Float constexpr r_clad = 0.492;
Float constexpr box_radius = 0.5;
Float constexpr box_thickness = 0.14;
Float constexpr water_gap = 0.69;
Float constexpr pin_pitch = 1.3;
Float constexpr sla = 0.95;
Float constexpr fftf = 1.485;             // width of water hole at the center
Float constexpr sth = (1.525 - fftf) / 2; // thickness of water hole cladding
Float constexpr bundle_width = 2 * water_gap + 2 * box_thickness + 10 * pin_pitch + sla;
Float constexpr wct = 0.04;           // clad thickness
Float constexpr wt = 0.1;             // wings water hole thickness
Float constexpr wl = 2.5 * pin_pitch; // wings water hole length

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> clad_tags;
std::vector<int> cool_tags;
std::vector<int> mod_tags;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

void
makeChannelBox()
{
  // Make the channel box geometry and return the surface tag of the box.
  // TODO(kcvaughn): Make an addChannelBox function for UM2's geometry API
  //
  // Diagram for the layout of the channel box vertices
  // Vertices 0-3 are center of the circles that are used to round
  // the corners of the box.
  // If would be nice to do this using a boolean difference of two rectangles with rounded
  // corners, but it is not clear how gmsh's radius parameter works. (using 0.5, produces
  // a corner with a radius that is not 0.5...)
  //
  //            17------------------------16
  //
  //             9-------------------------8
  //
  //   18  10    3                         2    7   15
  //   |    |                                   |    |
  //   |    |                                   |    |
  //   |    |                                   |    |
  //   |    |                                   |    |
  //   |    |                                   |    |
  //   |    |                                   |    |
  //   |    |                                   |    |
  //   19  11    0                         1    6   14
  //
  //             4------------------------ 5
  //
  //            12------------------------13

  // We store the offset from 0 to the box outer surface, box inner surface, and
  // the center of the circle.
  Float constexpr offset[3] = {water_gap, water_gap + box_thickness,
                               water_gap + box_thickness + box_radius};

  Float constexpr bw = bundle_width; // Alias for convenience

  um2::Vector<int> pt_tags;
  um2::Vector<int> curve_tags;

  namespace factory = um2::gmsh::model::occ;

  // Add the centers of the circles
  pt_tags.push_back(factory::addPoint(offset[2], offset[2], 0));
  pt_tags.push_back(factory::addPoint(bw - offset[2], offset[2], 0));
  pt_tags.push_back(factory::addPoint(bw - offset[2], bw - offset[2], 0));
  pt_tags.push_back(factory::addPoint(offset[2], bw - offset[2], 0));

  // Add the inner curve vertices
  pt_tags.push_back(factory::addPoint(offset[2], offset[1], 0));           // 4
  pt_tags.push_back(factory::addPoint(bw - offset[2], offset[1], 0));      // 5
  pt_tags.push_back(factory::addPoint(bw - offset[1], offset[2], 0));      // 6
  pt_tags.push_back(factory::addPoint(bw - offset[1], bw - offset[2], 0)); // 7
  pt_tags.push_back(factory::addPoint(bw - offset[2], bw - offset[1], 0)); // 8
  pt_tags.push_back(factory::addPoint(offset[2], bw - offset[1], 0));      // 9
  pt_tags.push_back(factory::addPoint(offset[1], bw - offset[2], 0));      // 10
  pt_tags.push_back(factory::addPoint(offset[1], offset[2], 0));           // 11

  // Add the outer curve vertices
  pt_tags.push_back(factory::addPoint(offset[2], offset[0], 0));           // 12
  pt_tags.push_back(factory::addPoint(bw - offset[2], offset[0], 0));      // 13
  pt_tags.push_back(factory::addPoint(bw - offset[0], offset[2], 0));      // 14
  pt_tags.push_back(factory::addPoint(bw - offset[0], bw - offset[2], 0)); // 15
  pt_tags.push_back(factory::addPoint(bw - offset[2], bw - offset[0], 0)); // 16
  pt_tags.push_back(factory::addPoint(offset[2], bw - offset[0], 0));      // 17
  pt_tags.push_back(factory::addPoint(offset[0], bw - offset[2], 0));      // 18
  pt_tags.push_back(factory::addPoint(offset[0], offset[2], 0));           // 19

  // Add the inner curves
  curve_tags.push_back(factory::addCircleArc(pt_tags[11], pt_tags[0], pt_tags[4]));
  curve_tags.push_back(factory::addLine(pt_tags[4], pt_tags[5]));
  curve_tags.push_back(factory::addCircleArc(pt_tags[5], pt_tags[1], pt_tags[6]));
  curve_tags.push_back(factory::addLine(pt_tags[6], pt_tags[7]));
  curve_tags.push_back(factory::addCircleArc(pt_tags[7], pt_tags[2], pt_tags[8]));
  curve_tags.push_back(factory::addLine(pt_tags[8], pt_tags[9]));
  curve_tags.push_back(factory::addCircleArc(pt_tags[9], pt_tags[3], pt_tags[10]));
  curve_tags.push_back(factory::addLine(pt_tags[10], pt_tags[11]));

  // Add the outer curves
  curve_tags.push_back(factory::addCircleArc(pt_tags[19], pt_tags[0], pt_tags[12]));
  curve_tags.push_back(factory::addLine(pt_tags[12], pt_tags[13]));
  curve_tags.push_back(factory::addCircleArc(pt_tags[13], pt_tags[1], pt_tags[14]));
  curve_tags.push_back(factory::addLine(pt_tags[14], pt_tags[15]));
  curve_tags.push_back(factory::addCircleArc(pt_tags[15], pt_tags[2], pt_tags[16]));
  curve_tags.push_back(factory::addLine(pt_tags[16], pt_tags[17]));
  curve_tags.push_back(factory::addCircleArc(pt_tags[17], pt_tags[3], pt_tags[18]));
  curve_tags.push_back(factory::addLine(pt_tags[18], pt_tags[19]));

  // Add the curve loops
  auto const cloop0 =
      factory::addCurveLoop({curve_tags[0], curve_tags[1], curve_tags[2], curve_tags[3],
                             curve_tags[4], curve_tags[5], curve_tags[6], curve_tags[7]});
  auto const cloop1 = factory::addCurveLoop(
      {curve_tags[8], curve_tags[9], curve_tags[10], curve_tags[11], curve_tags[12],
       curve_tags[13], curve_tags[14], curve_tags[15]});

  // Create the coolant region and box outer surface
  auto const coolant = factory::addPlaneSurface({cloop0});
  auto const box_outer_surf = factory::addPlaneSurface({cloop1, cloop0});

  // Create the central water cross
  Float const x = bw / 2 - fftf / 2;
  auto const rect0 =
      factory::addRectangle(x - sth, x - sth, 0.0, (fftf + 2 * sth), (fftf + 2 * sth));
  auto const rect1 = factory::addRectangle(x, x, 0.0, fftf, fftf);
  factory::rotate(
      {
          {2, rect0},
          {2, rect1}
  },
      bw / 2, bw / 2, 0, 0, 0, 1, 0.25 * um2::pi<double>);
  // Get the difference of the two rectangles
  std::vector<std::pair<int, int>> out_dim_tags;
  std::vector<std::vector<std::pair<int, int>>> out_dim_tags_map;
  factory::cut(
      {
          {2, rect0}
  },
      {{2, rect1}}, out_dim_tags, out_dim_tags_map, -1,
      /*removeObject=*/false, /*removeTool=*/true);
  auto const central_cross = out_dim_tags[0].second;

  // Produce the wings of the water cross
  auto skinny_wing = factory::addRectangle(water_gap + box_thickness, bw / 2 - wct / 2,
                                           0.0, bw / 2, wct);
  // Clip the skinny portion to the outer rectangle
  factory::cut(
      {
          {2, skinny_wing}
  },
      {{2, rect0}}, out_dim_tags, out_dim_tags_map);
  skinny_wing = out_dim_tags[0].second;
  auto thick_wing =
      factory::addRectangle(water_gap + box_thickness + box_radius + pin_pitch / 4,
                            bw / 2 - wct - wt / 2, 0.0, wl + wct * 2, wt + wct * 2);
  // Fuse the thick wing with the skinny wing
  factory::fuse(
      {
          {2, skinny_wing}
  },
      {{2, thick_wing}}, out_dim_tags, out_dim_tags_map);
  thick_wing = out_dim_tags[0].second;
  // Punch a hole for the water
  auto const wing_hole =
      factory::addRectangle(water_gap + box_thickness + box_radius + pin_pitch / 4 + wct,
                            bw / 2 - wt / 2, 0.0, wl, wt);
  factory::cut(
      {
          {2, thick_wing}
  },
      {{2, wing_hole}}, out_dim_tags, out_dim_tags_map);

  um2::Vector<int> wing_tags(4);
  wing_tags[0] = out_dim_tags[0].second;

  // Produce the other 3 water wings
  std::vector<std::pair<int, int>> tmp_dim_tags;
  for (int i = 0; i < 3; ++i) {
    // NOLINTNEXTLINE(readability*)
    factory::copy(out_dim_tags, tmp_dim_tags);
    wing_tags[i + 1] = tmp_dim_tags[0].second;
    factory::rotate(tmp_dim_tags, bw / 2, bw / 2, 0, 0, 0, 1,
                    0.5 * (i + 1) * um2::pi<double>);
  }

  // Fuse outer surface, central cross, and wings
  tmp_dim_tags.clear();
  for (Int i = 0; i < 4; ++i) {
    tmp_dim_tags.emplace_back(2, wing_tags[i]);
  }
  factory::fuse(
      {
          {2, box_outer_surf},
          {2,  central_cross}
  },
      tmp_dim_tags, out_dim_tags, out_dim_tags_map);

  auto const channel_box = out_dim_tags[0].second;
  clad_tags.push_back(channel_box);

  // Cut the coolant region based upon the box
  factory::cut(
      {
          {2, coolant}
  },
      {{2, channel_box}}, out_dim_tags, out_dim_tags_map, -1,
      /*removeObject=*/true, /*removeTool=*/false);
  // The first 4 entities are the coolant region
  // The last 4 entities are moderator
  for (size_t i = 0; i < 4; ++i) {
    cool_tags.push_back(out_dim_tags[i].second);
  }
  for (size_t i = 4; i < 8; ++i) {
    mod_tags.push_back(out_dim_tags[i].second);
  }
  factory::synchronize();
}

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  //===========================================================================
  // Parse command line arguments
  //===========================================================================

  // Check the number of arguments
  if (argc != 2) {
    um2::String const exec_name(argv[0]);
    um2::logger::error("Usage: ", exec_name, " num_coarse_cells");
    return 1;
  }

  char * end = nullptr;
  Int const num_coarse_cells = um2::strto<Int>(argv[1], &end);
  ASSERT(end != nullptr);
  ASSERT(num_coarse_cells > 0);

  //============================================================================
  // Materials
  //============================================================================
  // NOTE: some materials were just ripped from PWR models, so may need to be
  // thermally expanded or adjusted for pressure.
  // Many quantities are approximated. DO NOT USE FOR REAL ANALYSIS.

  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);

  um2::Vector<um2::Material> materials;

  // UO2 - 3% enrichment
  //---------------------------------------------------------------------------
  um2::Material uo2_300;
  uo2_300.setName("UO2_300");
  uo2_300.setDensity(10.257);
  uo2_300.setTemperature(900.0);
  uo2_300.setColor(um2::pink);
  uo2_300.addNuclidesAtomPercent({"U235", "U238", "O16"}, {0.03 / 3, 0.97 / 3, 2.0 / 3});
  uo2_300.populateXSec(xslib);
  materials.push_back(uo2_300);

  // UO2 - 3.8% enrichment
  //---------------------------------------------------------------------------
  um2::Material uo2_380;
  uo2_380.setName("UO2_380");
  uo2_380.setDensity(10.257);
  uo2_380.setTemperature(900.0);
  uo2_380.setColor(um2::lightblue);
  uo2_380.addNuclidesAtomPercent({"U235", "U238", "O16"},
                                 {0.038 / 3, 0.962 / 3, 2.0 / 3});
  uo2_380.populateXSec(xslib);
  materials.push_back(uo2_380);

  // UO2 - 4.1% enrichment + Gd2O3 (5 wt %)
  //---------------------------------------------------------------------------
  um2::Material uo2_410gd;
  uo2_410gd.setName("UO2_410Gd");
  uo2_410gd.setDensity(10.111);
  uo2_410gd.setTemperature(900.0);
  uo2_410gd.setColor(um2::purple);
  uo2_410gd.addNuclide(92235, 8.78236e-04);
  uo2_410gd.addNuclide(92238, 2.05422e-02);
  uo2_410gd.addNuclide(64152, 3.35960e-06);
  uo2_410gd.addNuclide(64154, 3.66190e-05);
  uo2_410gd.addNuclide(64155, 2.48606e-04);
  uo2_410gd.addNuclide(64156, 3.43849e-04);
  uo2_410gd.addNuclide(64157, 2.62884e-04);
  uo2_410gd.addNuclide(64158, 4.17255e-04);
  uo2_410gd.addNuclide(64160, 3.67198e-04);
  uo2_410gd.addNuclide(8016, 4.53705e-02);
  uo2_410gd.populateXSec(xslib);
  materials.push_back(uo2_410gd);

  // UO2 - 4.6% enrichment
  //-----------------------------------------------------------------------
  um2::Material uo2_460;
  uo2_460.setName("UO2_460");
  uo2_460.setDensity(10.257);
  uo2_460.setTemperature(900.0);
  uo2_460.setColor(um2::brown);
  uo2_460.addNuclidesAtomPercent({"U235", "U238", "O16"},
                                 {0.046 / 3, 0.954 / 3, 2.0 / 3});
  uo2_460.populateXSec(xslib);
  materials.push_back(uo2_460);

  // UO2 - 4.7% enrichment + Gd2O3 (5 wt %)
  //---------------------------------------------------------------------------
  um2::Material uo2_470gd;
  uo2_470gd.setName("UO2_470Gd");
  uo2_470gd.setDensity(10.111);
  uo2_470gd.setTemperature(900.0);
  uo2_470gd.setColor(um2::green);
  uo2_470gd.addNuclide(92235, 1.00676e-04);
  uo2_470gd.addNuclide(92238, 2.04136e-02);
  uo2_470gd.addNuclide(64152, 3.35960e-06);
  uo2_470gd.addNuclide(64154, 3.66190e-05);
  uo2_470gd.addNuclide(64155, 2.48606e-04);
  uo2_470gd.addNuclide(64156, 3.43849e-04);
  uo2_470gd.addNuclide(64157, 2.62884e-04);
  uo2_470gd.addNuclide(64158, 4.17255e-04);
  uo2_470gd.addNuclide(64160, 3.67198e-04);
  uo2_470gd.addNuclide(8016, 4.53705e-02);
  uo2_470gd.populateXSec(xslib);
  materials.push_back(uo2_470gd);

  // UO2 - 4.95% enrichment
  //---------------------------------------------------------------------------
  um2::Material uo2_495;
  uo2_495.setName("UO2_495");
  uo2_495.setDensity(10.257);
  uo2_495.setTemperature(900.0);
  uo2_495.setColor(um2::orange);
  uo2_495.addNuclidesAtomPercent({"U235", "U238", "O16"},
                                 {0.0495 / 3, 0.9505 / 3, 2.0 / 3});
  uo2_495.populateXSec(xslib);
  materials.push_back(uo2_495);

  // Gap
  //---------------------------------------------------------------------------
  um2::Material gap;
  gap.setName("Gap");
  gap.setDensity(0.006);     // g/cm^3
  gap.setTemperature(600.0); // K
  gap.setColor(um2::white);
  gap.addNuclide("He4", 1.38714e-05);
  gap.populateXSec(xslib);
  if (!omit_gap) {
    materials.push_back(gap);
  }

  // Clad
  //---------------------------------------------------------------------------
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(6.56);
  clad.setTemperature(600.0);
  clad.setColor(um2::slategray);
  clad.addNuclide(24050, 3.30121e-06);
  clad.addNuclide(24052, 6.36606e-05);
  clad.addNuclide(24053, 7.21860e-06);
  clad.addNuclide(24054, 1.79686e-06);
  clad.addNuclide(26054, 8.68307e-06);
  clad.addNuclide(26056, 1.36306e-04);
  clad.addNuclide(26057, 3.14789e-06);
  clad.addNuclide(26058, 4.18926e-07);
  clad.addNuclide(40090, 2.18865e-02);
  clad.addNuclide(40091, 4.77292e-03);
  clad.addNuclide(40092, 7.29551e-03);
  clad.addNuclide(40094, 7.39335e-03);
  clad.addNuclide(40096, 1.19110e-03);
  clad.addNuclide(50112, 4.68066e-06);
  clad.addNuclide(50114, 3.18478e-06);
  clad.addNuclide(50115, 1.64064e-06);
  clad.addNuclide(50116, 7.01616e-05);
  clad.addNuclide(50117, 3.70592e-05);
  clad.addNuclide(50118, 1.16872e-04);
  clad.addNuclide(50119, 4.14504e-05);
  clad.addNuclide(50120, 1.57212e-04);
  clad.addNuclide(50122, 2.23417e-05);
  clad.addNuclide(50124, 2.79392e-05);
  clad.addNuclide(72174, 3.54138e-09);
  clad.addNuclide(72176, 1.16423e-07);
  clad.addNuclide(72177, 4.11686e-07);
  clad.addNuclide(72178, 6.03806e-07);
  clad.addNuclide(72179, 3.01460e-07);
  clad.addNuclide(72180, 7.76449e-07);
  clad.populateXSec(xslib);
  materials.push_back(clad);

  // Coolant
  //---------------------------------------------------------------------------
  um2::Material coolant;
  coolant.setName("Coolant");
  coolant.setDensity(0.4);
  coolant.setTemperature(580.0);
  coolant.setColor(um2::yellow);
  coolant.addNuclidesAtomPercent({"H1", "O16"}, {2.0 / 3.0, 1.0 / 3.0});
  coolant.populateXSec(xslib);
  materials.push_back(coolant);

  // Moderator
  //---------------------------------------------------------------------------
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setDensity(0.743);
  moderator.setTemperature(580.0);
  moderator.setColor(um2::blue);
  moderator.addNuclidesAtomPercent({"H1", "O16"}, {2.0 / 3.0, 1.0 / 3.0});
  moderator.populateXSec(xslib);
  materials.push_back(moderator);

  //============================================================================
  // Geometry
  //============================================================================
  // Pin ID  |  Description
  // --------+----------------
  // 0       |  UO2 3.0%
  // 1       |  UO2 3.8%
  // 2       |  UO2 4.1% + Gd2O3 5 wt%
  // 3       |  UO2 4.6%
  // 4       |  UO2 4.7% + Gd2O3 5 wt%
  // 5       |  UO2 4.95%

  // Add pins
  //---------------------------------------------------------------------------

  um2::Vector<Float> pin_radii;
  um2::Vector<um2::Material> pin_mats0;
  um2::Vector<um2::Material> pin_mats1;
  um2::Vector<um2::Material> pin_mats2;
  um2::Vector<um2::Material> pin_mats3;
  um2::Vector<um2::Material> pin_mats4;
  um2::Vector<um2::Material> pin_mats5;
  if (omit_gap) {
    pin_radii = {r_fuel, r_clad};
    pin_mats0 = {uo2_300, clad};
    pin_mats1 = {uo2_380, clad};
    pin_mats2 = {uo2_410gd, clad};
    pin_mats3 = {uo2_460, clad};
    pin_mats4 = {uo2_470gd, clad};
    pin_mats5 = {uo2_495, clad};
  } else {
    pin_radii = {r_fuel, r_gap, r_clad};
    pin_mats0 = {uo2_300, gap, clad};
    pin_mats1 = {uo2_380, gap, clad};
    pin_mats2 = {uo2_410gd, gap, clad};
    pin_mats3 = {uo2_460, gap, clad};
    pin_mats4 = {uo2_470gd, gap, clad};
    pin_mats5 = {uo2_495, gap, clad};
  }
  um2::Vector<um2::Vector<um2::Material>> const pin_mats = {
      pin_mats0, pin_mats1, pin_mats2, pin_mats3, pin_mats4, pin_mats5};

  um2::Vector<um2::Vector<int>> pin_ids = um2::stringToLattice<Int>(R"(
    0 1 2 2 1 2 4 4 2 1
    1 4 4 5 5 5 5 3 5 2
    2 3 4 5 5 5 5 5 3 4
    2 5 5 5 4 4 5 5 5 4
    1 4 5 4 9 9 4 5 5 2
    1 4 5 4 9 9 4 5 5 2
    2 5 5 5 4 4 5 5 5 4
    2 3 4 5 5 5 5 5 3 4
    1 4 4 5 5 5 5 3 5 2
    0 1 2 2 1 2 4 4 2 1
  )");

  namespace factory = um2::gmsh::model::occ;
  Int constexpr nrow = 10;
  Int constexpr ncol = 10;
  Float const wg_bt = water_gap + box_thickness;
  for (Int i = 0; i < nrow; ++i) {
    Float yy = (static_cast<Float>(i) + 0.5) * pin_pitch + wg_bt;
    if (i > 4) {
      yy += sla;
    }
    for (Int j = 0; j < ncol; ++j) {
      Float xx = (static_cast<Float>(j) + 0.5) * pin_pitch + wg_bt;
      if (j > 4) {
        xx += sla;
      }
      auto const & this_pin_id = pin_ids[i][j];
      if (this_pin_id == 9) {
        continue;
      }
      auto const & this_pin_mats = pin_mats[this_pin_id];
      factory::addCylindricalPin2D({xx, yy}, pin_radii, this_pin_mats);
    } // for (Int j = 0; j < ncol; ++j)
  } // for (Int i = 0; i < nrow; ++i)

  // Add the channel box
  //---------------------------------------------------------------------------
  makeChannelBox();

  // Add channel box physical groups
  um2::gmsh::model::addToPhysicalGroup(2, clad_tags, -1, "Material_Clad");
  um2::gmsh::model::addToPhysicalGroup(2, cool_tags, -1, "Material_Coolant");
  um2::gmsh::model::addToPhysicalGroup(2, mod_tags, -1, "Material_Moderator");

  // Color the box physical groups
  {
    std::vector<um2::Material> const box_materials = {clad, coolant, moderator};
    um2::gmsh::model::occ::colorMaterialPhysicalGroupEntities(box_materials);
  }

  //===========================================================================
  // Overlay CMFD mesh
  //===========================================================================

  // Construct the MPACT model
  um2::mpact::Model model;
  for (auto const & material : materials) {
    model.addMaterial(material);
  }

  // Add a coarse grid that evenly subdivides the domain
  um2::Vec2F const domain_extents(bundle_width, bundle_width);
  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
  model.addCoarseGrid(domain_extents, num_cells);
  um2::gmsh::model::occ::overlayCoarseGrid(model, moderator);

  //===========================================================================
  // Generate the mesh
  //===========================================================================

  um2::gmsh::model::mesh::setGlobalMeshSize(pin_pitch / 8);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::write("svea_2d.inp");

  //===========================================================================
  // Complete the MPACT model and write the mesh
  //===========================================================================

  model.importCoarseCellMeshes("svea_2d.inp");
  model.writeCMFDInfo("svea_2d_cmfd_info.xdmf");
  model.write("svea_2d.xdmf");
  um2::finalize();
  return 0;
}
// NOLINTEND(misc-include-cleaner)
