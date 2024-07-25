// Model reference:
//  Baker, Una, et al.
//  "Simulation of the NuScale SMR and Investigation of the Effect of Load-following
//  on Component Lifetimes." Nuclear Technology 210.1 (2024): 1-22.
//  https://doi.org/10.1080/00295450.2023.2216973

#include <um2.hpp>

#include <fstream>

bool constexpr omit_gap = true; // Omit the gap material in the fuel pins

// NOLINTBEGIN(misc-include-cleaner)

Float constexpr r_fuel = 0.405765;
Float constexpr r_gap = 0.41402;
Float constexpr r_clad = 0.47498;
Float constexpr r_gt_inner = 0.57150;
Float constexpr r_gt_outer = 0.61214;
Float constexpr pin_pitch = 1.26;
// Int constexpr npins = 17;
Float constexpr asy_pitch = 21.50364;
// Float constexpr asy_gap = (asy_pitch - npins * pin_pitch) / 2;
Float constexpr uo2_density = 10.3458;
Float constexpr h2o_density = 0.84478;
Float constexpr temp_fuel = 900.0;
Float constexpr temp_water = 531.26;

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  //===========================================================================
  // Parse command line arguments
  //===========================================================================

  // Check the number of arguments
  if (argc != 4) {
    um2::String const exec_name(argv[0]);
    um2::logger::error("Usage: ", exec_name, " target_kn mfp_threshold mfp_scale");
    return 1;
  }

  char * end = nullptr;
  Float const target_kn = um2::strto<Float>(argv[1], &end);
  ASSERT(end != nullptr);
  ASSERT(target_kn > 0);

  Float const mfp_threshold = um2::strto<Float>(argv[2], &end);
  ASSERT(end != nullptr);
  end = nullptr;

  Float const mfp_scale = um2::strto<Float>(argv[3], &end);
  ASSERT(end != nullptr);

  um2::logger::info("Target Knudsen number: ", target_kn);
  um2::logger::info("MFP threshold: ", mfp_threshold);
  um2::logger::info("MFP scale: ", mfp_scale);

  Int constexpr num_coarse_cells = 119;
  um2::String const model_name = "nuscale_2d_" + um2::String(num_coarse_cells) + ".brep";

  //============================================================================
  // Materials
  //============================================================================
  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);

  // fuel U175 10.3458 96.0 / 1.75  ! A01,A02,A03     1.75%wt [r2]
  // fuel U200 10.3458 96.0 / 2.00  ! A04             2.00%wt [r2]
  // fuel U208 10.3458 96.0 / 2.08  ! A05 20% cutback of 2.60%wt [r2]
  // fuel U260 10.3458 96.0 / 2.60  ! A05             2.60%wt [r2]
  // fuel U280 10.3458 96.0 / 2.80  ! A06 20% cutback of 3.50%wt [r2]
  // fuel U292 10.3458 96.0 / 2.92  ! A07 20% cutback of 3.65%wt [r2]
  // fuel U300 10.3458 96.0 / 3.00  ! A08 20% cutback of 3.75%wt [r2]
  // fuel U350 10.3458 96.0 / 3.50  ! A06             3.50%wt [r2]
  // fuel U365 10.3458 96.0 / 3.65  ! A07             3.65%wt [r2]
  // fuel U375 10.3458 96.0 / 3.75  ! A08             3.75%wt [r2]
  // fuel G260 10.3458 96.0 / 2.60 / gad=6.0 ! A05 w/ 6%wt Gd 2.60%wt [r2]
  // fuel G350 10.3458 96.0 / 3.50 / gad=4.0 ! A06 w/ 4%wt Gd 3.50%wt
  um2::Vector<um2::Material> materials(10);
  um2::Vector<Float> enrichments = {1.75, 2.0, 2.08, 2.6,  2.8,
                                    2.92, 3.0, 3.5,  3.65, 3.75};
  um2::Vector<um2::Color> colors = {um2::Color(242, 233, 38), um2::Color(231, 208, 35),
                                    um2::Color(220, 184, 32), um2::Color(209, 159, 29),
                                    um2::Color(198, 134, 26), um2::Color(188, 110, 22),
                                    um2::Color(177, 85, 19),  um2::Color(166, 60, 16),
                                    um2::Color(155, 36, 13),  um2::Color(144, 11, 10)};
  um2::Vector<um2::String> names = {"U175", "U200", "U208", "U260", "U280",
                                    "U292", "U300", "U350", "U365", "U375"};
  for (Int i = 0; i < 10; ++i) {
    um2::Material fuel;
    fuel.setName(names[i]);
    fuel.setColor(colors[i]);
    fuel.setDensity(uo2_density); // This should be smaller, but we match the MPACT model
    fuel.setTemperature(temp_fuel);
    fuel.setUO2(enrichments[i] / 100);
    fuel.populateXSec(xslib);
    materials[i] = fuel;
  }

  // Gad pins
  um2::Material g260;
  g260.setName("G260");
  g260.setColor(um2::lightpink);
  g260.setDensity(uo2_density);
  g260.setTemperature(temp_fuel);
  g260.setUO2(0.026, /*wt_gad=*/0.06); // 6% wt Gd
  g260.populateXSec(xslib);
  materials.push_back(g260);

  um2::Material g350;
  g350.setName("G350");
  g350.setColor(um2::pink);
  g350.setDensity(uo2_density);
  g350.setTemperature(temp_fuel);
  g350.setUO2(0.035, /*wt_gad=*/0.04); // 4% wt Gd
  g350.populateXSec(xslib);
  materials.push_back(g350);

  // Gap
  um2::Material gap;
  gap.setName("Gap");
  gap.setColor(um2::white);
  gap.setTemperature(temp_fuel);
  gap.setDensity(0.0001786);
  gap.addNuclideWt("He4", 1.0);
  gap.populateXSec(xslib);
  if (!omit_gap) {
    materials.push_back(gap);
  }

  // M5
  um2::Material m5;
  m5.setName("M5");
  m5.setColor(um2::slategray);
  m5.setTemperature(700.0); // Just a guess
  m5.setDensity(6.5);
  m5.addNuclideWt(40000, 0.98827);
  m5.addNuclideWt("Nb93", 0.01);
  m5.addNuclideWt("O16", 0.00135);
  m5.addNuclideWt(26000, 0.00038);
  m5.populateXSec(xslib);
  materials.push_back(m5);

  // Zirc4
  um2::Material zirc4;
  zirc4.setName("Zirc4");
  zirc4.setColor(um2::darkgray);
  zirc4.setTemperature(700.0); // Just a guess
  zirc4.setDensity(6.55);
  // Verify this is correct
  zirc4.addNuclide(24050, 3.30121e-06);
  zirc4.addNuclide(24052, 6.36606e-05);
  zirc4.addNuclide(24053, 7.21860e-06);
  zirc4.addNuclide(24054, 1.79686e-06);
  zirc4.addNuclide(26054, 8.68307e-06);
  zirc4.addNuclide(26056, 1.36306e-04);
  zirc4.addNuclide(26057, 3.14789e-06);
  zirc4.addNuclide(26058, 4.18926e-07);
  zirc4.addNuclide(40090, 2.18865e-02);
  zirc4.addNuclide(40091, 4.77292e-03);
  zirc4.addNuclide(40092, 7.29551e-03);
  zirc4.addNuclide(40094, 7.39335e-03);
  zirc4.addNuclide(40096, 1.19110e-03);
  zirc4.addNuclide(50112, 4.68066e-06);
  zirc4.addNuclide(50114, 3.18478e-06);
  zirc4.addNuclide(50115, 1.64064e-06);
  zirc4.addNuclide(50116, 7.01616e-05);
  zirc4.addNuclide(50117, 3.70592e-05);
  zirc4.addNuclide(50118, 1.16872e-04);
  zirc4.addNuclide(50119, 4.14504e-05);
  zirc4.addNuclide(50120, 1.57212e-04);
  zirc4.addNuclide(50122, 2.23417e-05);
  zirc4.addNuclide(50124, 2.79392e-05);
  zirc4.addNuclide(72174, 3.54138e-09);
  zirc4.addNuclide(72176, 1.16423e-07);
  zirc4.addNuclide(72177, 4.11686e-07);
  zirc4.addNuclide(72178, 6.03806e-07);
  zirc4.addNuclide(72179, 3.01460e-07);
  zirc4.addNuclide(72180, 7.76449e-07);
  zirc4.populateXSec(xslib);
  materials.push_back(zirc4);

  // Stainless steel
  um2::Material ss;
  ss.setName("SS304");
  ss.setColor(um2::gray);
  ss.setTemperature(temp_water);
  ss.setDensity(8.0);
  ss.addNuclide(6000, 3.20895e-04);
  // ss.addNuclide(14028, 1.58197e-03);
  // ss.addNuclide(14029, 8.03653e-05);
  // ss.addNuclide(14030, 5.30394e-05);
  ss.addNuclide(14000, 1.7153747e-03);
  ss.addNuclide(15031, 6.99938e-05);
  ss.addNuclide(24050, 7.64915e-04);
  ss.addNuclide(24052, 1.47506e-02);
  ss.addNuclide(24053, 1.67260e-03);
  ss.addNuclide(24054, 4.16346e-04);
  ss.addNuclide(25055, 1.75387e-03);
  ss.addNuclide(26054, 3.44776e-03);
  ss.addNuclide(26056, 5.41225e-02);
  ss.addNuclide(26057, 1.24992e-03);
  ss.addNuclide(26058, 1.66342e-04);
  ss.addNuclide(28058, 5.30854e-03);
  ss.addNuclide(28060, 2.04484e-03);
  ss.addNuclide(28061, 8.88879e-05);
  ss.addNuclide(28062, 2.83413e-04);
  ss.addNuclide(28064, 7.21770e-05);
  ss.populateXSec(xslib);
  materials.push_back(ss);

  // Water
  um2::Material water;
  water.setName("Water");
  water.setColor(um2::blue);
  water.setTemperature(temp_water);
  water.setDensity(h2o_density);
  water.addNuclidesAtomPercent({"H1", "O16"}, {2.0 / 3.0, 1.0 / 3.0});
  water.populateXSec(xslib);
  materials.push_back(water);

  //============================================================================
  // Geometry
  //============================================================================

  namespace factory = um2::gmsh::model::occ;

  bool model_brep_exists = false;
  {
    std::ifstream const file(model_name.data());
    model_brep_exists = file.good();
    LOG_INFO("Model BREP exists: ", model_brep_exists);
  }

  bool vessel_brep_exists = false;
  {
    std::ifstream const file("./vessel.brep");
    vessel_brep_exists = file.good();
    LOG_INFO("Vessel BREP exists: ", vessel_brep_exists);
  }

  if (!model_brep_exists) {
    if (!vessel_brep_exists) {
      // Add the reflector and vessel first, then cut out space for the fuel
      // We add the concentric cylinders as a "pin"
      // vessel mod  0.0        ! For heavy reflector
      //         ss  93.680     ! Heavy reflector outer radius [r1 p.189],(0.3 cm gap
      //         matches figure in r??]
      //        mod  93.980     ! Barrel Inner Radius                     [r1 p.13]*2.54
      //        cm/in
      //         ss  99.060     ! Barrel outer radius (cm)     [r1 p.189],[r1 p.13]*2.54
      //         cm/in
      //        mod 134.620     ! Vessel liner inner radius               [r5 p.76]
      //         ss 135.255     ! Vessel inner radius (cm)     [r5 p.32], [r5 p.76] 309L
      //         and 308L cs 146.685     ! Vessel outer radius          [r5 p.50], [r5
      //         p.76] ss 147.0025    ! Vessel liner outer radius    [r5 p.32], [r5 p.76]
      //         309L
      // Unsure what cs is , so we just use ss

      auto const vessel_tags =
          factory::addCylindricalPin2D({0, 0},                                  // center
                                       {93.68, 93.98, 99.06, 134.62, 147.0025}, // radii
                                       {ss, water, ss, water, ss} // materials
          );
      // It is important to note that the returned vessel tags are from inside out,
      // so the innermost tag is the 0th index.

      // Create an assembly sized cutout for the fuel at each assembly location
      um2::Vector<um2::Vec2d> const asy_offsets = {
          {-17 * pin_pitch / 2, -17 * pin_pitch / 2},
          {      asy_pitch / 2, -17 * pin_pitch / 2},
          {      asy_pitch / 2,       asy_pitch / 2},
          {-17 * pin_pitch / 2,       asy_pitch / 2},
          {    1.5 * asy_pitch, -17 * pin_pitch / 2},
          {-17 * pin_pitch / 2,     1.5 * asy_pitch},
          {    1.5 * asy_pitch,     0.5 * asy_pitch},
          {    0.5 * asy_pitch,     1.5 * asy_pitch},
          {    2.5 * asy_pitch, -17 * pin_pitch / 2},
          {-17 * pin_pitch / 2,     2.5 * asy_pitch},
          {    1.5 * asy_pitch,     1.5 * asy_pitch},
          {    2.5 * asy_pitch,     0.5 * asy_pitch},
          {    0.5 * asy_pitch,     2.5 * asy_pitch},
      };
      std::vector<std::pair<int, int>> cut_dim_tags;
      for (auto const & offset : asy_offsets) {
        auto const rect_tag =
            factory::addRectangle(offset[0], offset[1], 0, asy_pitch, asy_pitch);
        cut_dim_tags.emplace_back(2, rect_tag);
      }

      // Create the holes in the heavy reflector
      um2::Vector<um2::Vec3d> const hole_xyr = {
          { 0.0000, 79.8046, 0.6300},
          { 5.2500, 79.8046, 0.6300},
          {10.5000, 79.8046, 0.6300},
          {15.7500, 79.8046, 0.6300},
          {21.0000, 79.8046, 0.6300},
          {26.2500, 79.8046, 0.6300},
          {31.5000, 79.8046, 0.6300},
          {36.7073, 74.3387, 0.6300},
          {36.7073, 68.9627, 0.6300},
          {36.7073, 63.5868, 0.6300},
          {36.7073, 58.2109, 1.2600},
          {42.0832, 58.2109, 0.6300},
          {47.4591, 58.2109, 0.6300},
          {52.8350, 58.2109, 0.6300},
          {58.2109, 58.2109, 0.6300},
          { 2.6880, 84.4655, 0.6300},
          { 7.9380, 84.4655, 0.6300},
          {13.1880, 84.4655, 0.6300},
          {18.4380, 84.4655, 0.6300},
          {23.6880, 85.8605, 0.6300},
          {28.9380, 84.4655, 0.6300},
          {40.5086, 62.0123, 0.6300},
          {44.7711, 62.8666, 0.6300},
          {41.3630, 66.2748, 0.6300},
          {49.0337, 63.7209, 0.6300},
          {42.2173, 70.5373, 0.6300},
          {61.5504, 66.0047, 0.6300},
          {51.1182, 74.3774, 0.6300},
          {39.8918, 81.1753, 0.6300},
          {14.2355, 89.3646, 0.6300},
          { 5.5650, 88.8124, 0.6300},
          {79.8046,  0.0000, 0.6300},
          {79.8046,  5.2500, 0.6300},
          {79.8046, 10.5000, 0.6300},
          {79.8046, 15.7500, 0.6300},
          {79.8046, 21.0000, 0.6300},
          {79.8046, 26.2500, 0.6300},
          {79.8046, 31.5000, 0.6300},
          {74.3387, 36.7073, 0.6300},
          {68.9627, 36.7073, 0.6300},
          {63.5868, 36.7073, 0.6300},
          {58.2109, 36.7073, 1.2600},
          {58.2109, 42.0832, 0.6300},
          {58.2109, 47.4591, 0.6300},
          {58.2109, 52.8350, 0.6300},
          {84.4655,  2.6880, 0.6300},
          {84.4655,  7.9380, 0.6300},
          {84.4655, 13.1880, 0.6300},
          {84.4655, 18.4380, 0.6300},
          {85.8605, 23.6880, 0.6300},
          {84.4655, 28.9380, 0.6300},
          {62.0123, 40.5086, 0.6300},
          {62.8666, 44.7711, 0.6300},
          {66.2748, 41.3630, 0.6300},
          {63.7209, 49.0337, 0.6300},
          {70.5373, 42.2173, 0.6300},
          {66.0047, 61.5504, 0.6300},
          {74.3774, 51.1182, 0.6300},
          {81.1753, 39.8918, 0.6300},
          {89.3646, 14.2355, 0.6300},
          {88.8124,  5.5650, 0.6300},
      };

      for (auto const & hole : hole_xyr) {
        auto const disk_tag = factory::addDisk(hole[0], hole[1], 0, hole[2], hole[2]);
        cut_dim_tags.emplace_back(2, disk_tag);
      }
      // Cut the rectangles and disks from the innermost vessel tag
      std::vector<std::pair<int, int>> out_dim_tags;
      std::vector<std::vector<std::pair<int, int>>> out_dim_tags_map;
      factory::groupPreservingCut(
          {
              {2, vessel_tags[0]}
      },
          cut_dim_tags, out_dim_tags, out_dim_tags_map);
      factory::synchronize();
      // The cut causes the vessel to lost it's color, so we reapply it
      factory::colorMaterialPhysicalGroupEntities({water, ss});

      um2::gmsh::write("vessel.brep", /*extra_info=*/true);
      // This is a hack to prevent a segfault in Gmsh.
      // We s
      LOG_WARN("Gmsh is likely now going to segfault, but the BREP file should be "
               "written. Rerun the program.");
      um2::gmsh::finalize();
      um2::gmsh::initialize();
    } // if (!vessel_brep_exists)
    um2::gmsh::open("vessel.brep", /*extra_info=*/true);

    // cell  1 0.405765 0.41402 0.47498 / U175 he m5
    // cell  4 0.405765 0.41402 0.47498 / U200 he m5
    // cell 6c 0.405765 0.41402 0.47498 / U280 he m5
    // cell 6g 0.405765 0.41402 0.47498 / G350 he m5
    // cell  7 0.405765 0.41402 0.47498 / U365 he m5
    // cell 7c 0.405765 0.41402 0.47498 / U292 he m5
    // cell  8 0.405765 0.41402 0.47498 / U375 he m5
    // cell 8c 0.405765 0.41402 0.47498 / U300 he m5
    // GT == IT
    // cell GT          0.57150 0.61214 / mod  zirc4

    um2::Vector<Float> const gt_radii = {r_gt_inner, r_gt_outer};
    um2::Vector<um2::Material> const pin_gt_mats = {water, zirc4};

    um2::Vector<Float> fuel_radii = {r_fuel, r_gap, r_clad};
    um2::Vector<um2::Material> pin1_mats = {materials[0], gap, m5};
    um2::Vector<um2::Material> pin4_mats = {materials[1], gap, m5};
    um2::Vector<um2::Material> pin5_mats = {materials[3], gap, m5};
    um2::Vector<um2::Material> pin5c_mats = {materials[2], gap, m5};
    um2::Vector<um2::Material> pin5g_mats = {g260, gap, m5};
    um2::Vector<um2::Material> pin6_mats = {materials[7], gap, m5};
    um2::Vector<um2::Material> pin6c_mats = {materials[4], gap, m5};
    um2::Vector<um2::Material> pin6g_mats = {g350, gap, m5};
    um2::Vector<um2::Material> pin7_mats = {materials[8], gap, m5};
    um2::Vector<um2::Material> pin7c_mats = {materials[5], gap, m5};
    um2::Vector<um2::Material> pin8_mats = {materials[9], gap, m5};
    um2::Vector<um2::Material> pin8c_mats = {materials[6], gap, m5};

    if (omit_gap) {
      fuel_radii = {r_fuel, r_clad};
      pin1_mats = {materials[0], m5};
      pin4_mats = {materials[1], m5};
      pin5_mats = {materials[3], m5};
      pin5c_mats = {materials[2], m5};
      pin5g_mats = {g260, m5};
      pin6_mats = {materials[7], m5};
      pin6c_mats = {materials[4], m5};
      pin6g_mats = {g350, m5};
      pin7_mats = {materials[8], m5};
      pin7c_mats = {materials[5], m5};
      pin8_mats = {materials[9], m5};
      pin8c_mats = {materials[6], m5};
    }

    // lattice LAT_A01 == A02 == A03
    //   GT
    //    1  1
    //    1  1  1
    //   GT  1  1 GT
    //    1  1  1  1  1
    //    1  1  1  1  1 GT
    //   GT  1  1 GT  1  1  1
    //    1  1  1  1  1  1  1  1
    //    1  1  1  1  1  1  1  1  1
    //
    // lattice LAT_A04
    //   GT
    //    4  4
    //    4  4  4
    //   GT  4  4 GT
    //    4  4  4  4  4
    //    4  4  4  4  4 GT
    //   GT  4  4 GT  4  4  4
    //    4  4  4  4  4  4  4  4
    //    4  4  4  4  4  4  4  4  4

    um2::Vector<um2::Vector<Int>> const lat_a01 = um2::stringToLattice<Int>(R"(
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 1
        1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1
        1 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
      )");

    // lattice LAT_A05 ! A05 for cycle 1 [??]
    //   IT
    //    5  5
    //    5  5  5
    //   GT  5  5 GT
    //    5  5  5  5 5g
    //    5  5  5  5  5 GT
    //   GT  5  5 GT  5  5  5
    //    5  5  5  5  5  5  5  5
    //   5c 5c 5c 5c 5c 5c 5c 5c 5c
    //
    // lattice LAT_A06 ! A06 for cycle 1 [??]
    //   IT
    //    6  6
    //    6  6  6
    //   GT  6  6 GT
    //    6  6  6  6  6
    //    6  6  6  6 6g GT
    //   GT  6  6 GT  6  6  6
    //    6  6  6  6  6  6  6  6
    //   6c 6c 6c 6c 6c 6c 6c 6c 6c

    um2::Vector<um2::Vector<Int>> const lat_a05 = um2::stringToLattice<Int>(R"(
        2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 2
        2 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 2
        2 1 1 1 3 1 1 1 1 1 1 1 3 1 1 1 2
        2 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 2
        2 1 1 1 3 1 1 1 1 1 1 1 3 1 1 1 2
        2 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 2
        2 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
      )");

    // lattice LAT_A07 ! A07 for cycle 1 [??]
    //   IT
    //    7  7
    //    7  7  7
    //   GT  7  7 GT
    //    7  7  7  7  7
    //    7  7  7  7  7 GT
    //   GT  7  7 GT  7  7  7
    //    7  7  7  7  7  7  7  7
    //   7c 7c 7c 7c 7c 7c 7c 7c 7c
    //
    // lattice LAT_A08 ! A08 for cycle 1 [??]
    //   IT
    //    8  8
    //    8  8  8
    //   GT  8  8 GT
    //    8  8  8  8  8
    //    8  8  8  8  8 GT
    //   GT  8  8 GT  8  8  8
    //    8  8  8  8  8  8  8  8
    //   8c 8c 8c 8c 8c 8c 8c 8c 8c

    um2::Vector<um2::Vector<Int>> const lat_a07 = um2::stringToLattice<Int>(R"(
        2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 2
        2 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 0 1 1 0 1 1 0 1 1 0 1 1 0 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 2
        2 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 2
        2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
      )");

    // assm_map
    // A01                 A01 A01 A04 A06
    // A02 A03        -->  A01 A01 A05 A08
    // A04 A05 A07         A04 A05 A07
    // A06 A08             A06 A08

    // We will use a layout like so:
    //  ^
    //  |  A06 A08
    //  |  A04 A05 A07
    //  |  A01 A01 A05 A08
    //  |  A01 A01 A04 A06
    //  +----------------->

    um2::Vec2d const dxdy(pin_pitch, pin_pitch);
    // A01
    factory::addCylindricalPinLattice2D(lat_a01,                  // Pin IDs
                                        {dxdy, dxdy},             // Pitch of the pins
                                        {gt_radii, fuel_radii},   // Radii of the pins
                                        {pin_gt_mats, pin1_mats}, // Materials of the pins
                                        {-17 * pin_pitch / 2, -17 * pin_pitch / 2}
                                        // Offset of the lattice
    );
    factory::addCylindricalPinLattice2D(lat_a01,                  // Pin IDs
                                        {dxdy, dxdy},             // Pitch of the pins
                                        {gt_radii, fuel_radii},   // Radii of the pins
                                        {pin_gt_mats, pin1_mats}, // Materials of the pins
                                        {asy_pitch / 2, -17 * pin_pitch / 2}
                                        // Offset of the lattice
    );
    factory::addCylindricalPinLattice2D(lat_a01,                  // Pin IDs
                                        {dxdy, dxdy},             // Pitch of the pins
                                        {gt_radii, fuel_radii},   // Radii of the pins
                                        {pin_gt_mats, pin1_mats}, // Materials of the pins
                                        {asy_pitch / 2, asy_pitch / 2}
                                        // Offset of the lattice
    );
    factory::addCylindricalPinLattice2D(lat_a01,                  // Pin IDs
                                        {dxdy, dxdy},             // Pitch of the pins
                                        {gt_radii, fuel_radii},   // Radii of the pins
                                        {pin_gt_mats, pin1_mats}, // Materials of the pins
                                        {-17 * pin_pitch / 2, asy_pitch / 2}
                                        // Offset of the lattice
    );
    // A04
    // Reuse A01 layout, but swap pin1 for pin4
    factory::addCylindricalPinLattice2D(lat_a01,                  // Pin IDs
                                        {dxdy, dxdy},             // Pitch of the pins
                                        {gt_radii, fuel_radii},   // Radii of the pins
                                        {pin_gt_mats, pin4_mats}, // Materials of the pins
                                        {1.5 * asy_pitch, -17 * pin_pitch / 2}
                                        // Offset of the lattice
    );
    factory::addCylindricalPinLattice2D(lat_a01,                  // Pin IDs
                                        {dxdy, dxdy},             // Pitch of the pins
                                        {gt_radii, fuel_radii},   // Radii of the pins
                                        {pin_gt_mats, pin4_mats}, // Materials of the pins
                                        {-17 * pin_pitch / 2, 1.5 * asy_pitch}
                                        // Offset of the lattice
    );
    // A05
    factory::addCylindricalPinLattice2D(
        lat_a05,                                          // Pin IDs
        {dxdy, dxdy, dxdy, dxdy},                         // Pitch of the pins
        {gt_radii, fuel_radii, fuel_radii, fuel_radii},   // Radii of the pins
        {pin_gt_mats, pin5_mats, pin5c_mats, pin5g_mats}, // Materials of the pins
        {1.5 * asy_pitch, 0.5 * asy_pitch}                // Offset of the lattice
    );
    factory::addCylindricalPinLattice2D(
        lat_a05,                                          // Pin IDs
        {dxdy, dxdy, dxdy, dxdy},                         // Pitch of the pins
        {gt_radii, fuel_radii, fuel_radii, fuel_radii},   // Radii of the pins
        {pin_gt_mats, pin5_mats, pin5c_mats, pin5g_mats}, // Materials of the pins
        {0.5 * asy_pitch, 1.5 * asy_pitch}                // Offset of the lattice
    );
    // A06
    factory::addCylindricalPinLattice2D(
        lat_a05,                                          // Pin IDs
        {dxdy, dxdy, dxdy, dxdy},                         // Pitch of the pins
        {gt_radii, fuel_radii, fuel_radii, fuel_radii},   // Radii of the pins
        {pin_gt_mats, pin6_mats, pin6c_mats, pin6g_mats}, // Materials of the pins
        {2.5 * asy_pitch, -17 * pin_pitch / 2}            // Offset of the lattice
    );
    factory::addCylindricalPinLattice2D(
        lat_a05,                                          // Pin IDs
        {dxdy, dxdy, dxdy, dxdy},                         // Pitch of the pins
        {gt_radii, fuel_radii, fuel_radii, fuel_radii},   // Radii of the pins
        {pin_gt_mats, pin6_mats, pin6c_mats, pin6g_mats}, // Materials of the pins
        {-17 * pin_pitch / 2, 2.5 * asy_pitch}            // Offset of the lattice
    );
    // A07
    factory::addCylindricalPinLattice2D(
        lat_a07,                              // Pin IDs
        {dxdy, dxdy, dxdy},                   // Pitch of the pins
        {gt_radii, fuel_radii, fuel_radii},   // Radii of the pins
        {pin_gt_mats, pin7_mats, pin7c_mats}, // Materials of the pins
        {1.5 * asy_pitch, 1.5 * asy_pitch}    // Offset of the lattice
    );
    // A08
    factory::addCylindricalPinLattice2D(
        lat_a07,                              // Pin IDs
        {dxdy, dxdy, dxdy},                   // Pitch of the pins
        {gt_radii, fuel_radii, fuel_radii},   // Radii of the pins
        {pin_gt_mats, pin8_mats, pin8c_mats}, // Materials of the pins
        {2.5 * asy_pitch, 0.5 * asy_pitch}    // Offset of the lattice
    );
    factory::addCylindricalPinLattice2D(
        lat_a07,                              // Pin IDs
        {dxdy, dxdy, dxdy},                   // Pitch of the pins
        {gt_radii, fuel_radii, fuel_radii},   // Radii of the pins
        {pin_gt_mats, pin8_mats, pin8c_mats}, // Materials of the pins
        {0.5 * asy_pitch, 2.5 * asy_pitch}    // Offset of the lattice
    );

  } // if (!model_brep_exists)

  //===========================================================================
  // Overlay CMFD mesh
  //===========================================================================

  // Construct the MPACT model
  um2::mpact::Model model;
  for (auto const & mat : materials) {
    model.addMaterial(mat);
  }

  // Add a coarse grid that evenly subdivides the domain
  um2::Vec2F const domain_extents(7 * 17 * 1.26, 7 * 17 * 1.26);
  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
  model.addCoarseGrid(domain_extents, num_cells);
  // If a file with the model name exists, we don't need to recreate the model
  if (model_brep_exists) {
    um2::gmsh::open(model_name.data(), /*extra_info=*/true);
  } else {
    factory::overlayCoarseGrid(model, water);
    um2::gmsh::write(model_name.data(), /*extra_info=*/true);
  }

  //===========================================================================
  // Generate the mesh
  //===========================================================================

  um2::gmsh::model::mesh::setMeshFieldFromKnudsenNumber(2, model.materials(), target_kn,
                                                        mfp_threshold, mfp_scale);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::write("nuscale_2d.inp");

  //===========================================================================
  // Complete the MPACT model and write the mesh
  //===========================================================================

  model.importCoarseCellMeshes("nuscale_2d.inp");
  model.write("nuscale_2d.xdmf");
  um2::finalize();
  return 0;
}

// NOLINTEND(misc-include-cleaner)
