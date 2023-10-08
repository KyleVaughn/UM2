#include <um2/mpact/SpatialPartition.hpp>

namespace um2::mpact
{

// template <std::floating_point T, std::signed_integral I>
// int SpatialPartition::make_cylindrical_pin_mesh(
//     std::vector<double> const & radii,
//     double const pitch,
//     std::vector<int> const & num_rings,
//     int const num_azimuthal,
//     int const mesh_order)
//{
//   if ((num_azimuthal & (num_azimuthal - 1)) != 0)
//   {
//     Log::error("The number of azimuthal divisions must be a power of 2");
//   }
//   if (num_azimuthal < 8)
//   {
//     Log::error("The number of azimuthal divisions must be at least 8 until someone
//     fixes the AABB mesh for 4 divisions");
//   }
//   if (radii.size() != num_rings.size())
//   {
//     Log::error("The number of radii must match the size of the subradial rings
//     vector");
//   }
//   if (std::any_of(radii.begin(), radii.end(), [pitch](double r) { return r > pitch / 2;
//   }))
//   {
//     Log::error("The radii must be less than half the pitch");
//   }
//
//   // radial region = region containing different materials (rings + outside of the last
//   radius)
//   //   radial_region_areas = area of each radial region, including outside of the last
//   radius
//   // ring = an equal area division of a radial region containing the same material
//   //   ring_radii = the radius of each ring, NOT including the outside of the last
//   radius
//   //   ring_areas = the area of each ring, including the outside of the last radius
//   Size mesh_id = -1;
//   if (mesh_order == 1) {
//     mesh_id = this->quad.size();
//     Log::info("Making linear quadrilateral cylindrical pin mesh " +
//     std::to_string(mesh_id));
//   } else if (mesh_order == 2) {
//     mesh_id = this->quadratic_quad.size();
//     Log::info("Making quadratic quadrilateral cylindrical pin mesh " +
//     std::to_string(mesh_id));
//   } else {
//     Log::error("Invalid mesh order");
//   }
//
//   // Get the area of each radial region (rings + outside of the last ring)
//   // --------------------------------------
//   size_t const num_radial_regions = radii.size() + 1;
//   std::vector<double> radial_region_areas(num_radial_regions); // Each of the radii +
//   the pin pitch using std::numbers::pi;
//   // A0 = pi * r0^2
//   // Ai = pi * (ri^2 - ri-1^2)
//   radial_region_areas[0] = pi * radii[0] * radii[0];
//   for (size_t i = 1; i < num_radial_regions - 1; ++i)
//   {
//     radial_region_areas[i] = pi * (radii[i] * radii[i] - radii[i - 1] * radii[i - 1]);
//   }
//   radial_region_areas[num_radial_regions - 1] = pitch * pitch -
//   radial_region_areas[num_radial_regions - 2];
//
//   // Get the radii and areas of each ring after splitting the radial regions
//   // This includes outside of the last ring
//   // --------------------------------------
//   size_t const total_rings = std::reduce(num_rings.begin(), num_rings.end(), 0);
//   std::vector<double> ring_radii(total_rings);
//   std::vector<double> ring_areas(total_rings + 1);
//   // Inside the innermost region
//   ring_areas[0] = radial_region_areas[0] / num_rings[0];
//   ring_radii[0] = std::sqrt(ring_areas[0] / pi);
//   for (size_t i = 1; i < num_rings[0]; ++i)
//   {
//     ring_areas[i] = ring_areas[0];
//     ring_radii[i] = std::sqrt(ring_areas[i] / pi + ring_radii[i - 1] * ring_radii[i -
//     1]);
//   }
//   size_t ctr = num_rings[0];
//   for (size_t ireg = 1; ireg < num_radial_regions - 1; ++ireg)
//   {
//     size_t const num_rings_in_region = num_rings[ireg];
//     double const area_per_ring = radial_region_areas[ireg] / num_rings_in_region;
//     for (size_t iring = 0; iring < num_rings_in_region; ++iring, ++ctr)
//     {
//       ring_areas[ctr] = area_per_ring;
//       ring_radii[ctr] = std::sqrt(area_per_ring / pi + ring_radii[ctr - 1] *
//       ring_radii[ctr - 1]);
//     }
//   }
//   // Outside of the last ring
//   ring_areas[ctr] = pitch * pitch - pi * ring_radii.back() * ring_radii.back();
//   // Log the radii and areas in debug mode
//   for (size_t i = 0; i < total_rings; ++i)
//   {
//     Log::debug("Ring " + std::to_string(i) + " radius: " +
//     std::to_string(ring_radii[i])); Log::debug("Ring " + std::to_string(i) + " area: "
//     + std::to_string(ring_areas[i]));
//   }
//   Log::debug("The area outside of the last ring is " +
//   std::to_string(ring_areas[ctr]));
//   // Ensure the sum of the ring areas is equal to pitch^2
//   double const sum_ring_areas = std::reduce(ring_areas.begin(), ring_areas.end(), 0.0);
//   UM2_ASSERT(std::abs(sum_ring_areas - pitch * pitch) < 1e-6)
//
//   if (mesh_order == 1)
//   {
//     // Get the equivalent radius of each ring if it were a quadrilateral
//     double const theta = 2 * pi / num_azimuthal;
//     double const sin_theta = std::sin(theta);
//     std::vector<double> eq_radii(total_rings);
//     // The innermost radius is a special case, and is essentially a triangle.
//     // A_t = l² * sin(θ) / 2
//     // A_ring = num_azi * A_t = l² * sin(θ) * num_azi / 2
//     // l = sqrt(2 * A_ring / (sin(θ) * num_azi))
//     eq_radii[0] = sqrt(2 * ring_areas[0] / (sin_theta * num_azimuthal));
//     // A_q = (l² - l²₀) * sin(θ) / 2
//     // A_ring = num_azi * A_q = (l² - l²₀) * sin(θ) * num_azi / 2
//     // l = sqrt(2 * A_ring / (sin(θ) * num_azi) + l²₀)
//     for (size_t i = 1; i < total_rings; ++i)
//     {
//       eq_radii[i] = sqrt(2 * ring_areas[i] / (sin_theta * num_azimuthal) + eq_radii[i -
//       1] * eq_radii[i - 1]);
//     }
//     // Log the equivalent radii in debug mode
//     for (size_t i = 0; i < total_rings; ++i)
//     {
//       Log::debug("Ring " + std::to_string(i) + " equivalent radius: " +
//       std::to_string(eq_radii[i]));
//     }
//     // If any of the equivalent radii are larger than half the pitch, error
//     if (std::any_of(eq_radii.begin(), eq_radii.end(), [pitch](double r) { return r >
//     pitch / 2; }))
//     {
//       Log::error("The equivalent radius of a ring is larger than half the pitch.");
//       return -1;
//     }
//     // Sanity check: ensure the sum of the quadrilateral areas in a ring is equal to
//     the ring area UM2_ASSERT(std::abs(eq_radii[0] * eq_radii[0] * sin_theta / 2 -
//     ring_areas[0] / num_azimuthal) < 1e-4) for (size_t i = 0; i < total_rings; ++i)
//     {
//       double const area = (eq_radii[i] * eq_radii[i] - eq_radii[i - 1] * eq_radii[i -
//       1]) * sin_theta / 2; UM2_ASSERT(std::abs(area - ring_areas[i] / num_azimuthal) <
//       1e-4)
//     }
//
//     // Get the points that make up the mesh
//     // -------------------------------------
//     // The number of points is:
//     //   Center point
//     //   num_azimuthal / 2, for the points in the innermost ring to make the quads
//     "triangular"
//     //   (num_rings + 1) * num_azimuthal
//     size_t const num_points = 1 + (total_rings + 1) * num_azimuthal + num_azimuthal /
//     2; std::vector<Point2d> vertices(num_points);
//     // Center point
//     vertices[0] = {0, 0};
//     // Triangular points
//     double const rt = eq_radii[0] / 2;
//     for (size_t ia = 0; ia < num_azimuthal / 2; ++ia)
//     {
//       double const sin_ia_theta = std::sin(theta * (2 * ia + 1));
//       double const cos_ia_theta = std::cos(theta * (2 * ia + 1));
//       vertices[1 + ia] = {rt * cos_ia_theta, rt * sin_ia_theta};
//     }
//     // Quadrilateral points
//     // Points on rings, not including the boundary of the pin (pitch / 2 box)
//     for (size_t ir = 0; ir < total_rings; ++ir)
//     {
//       size_t const num_prev_points = 1 + num_azimuthal / 2 + ir * num_azimuthal;
//       for (size_t ia = 0; ia < num_azimuthal; ++ia)
//       {
//         double sin_ia_theta = std::sin(theta * ia);
//         double cos_ia_theta = std::cos(theta * ia);
//         if (std::abs(sin_ia_theta) < 1e-6)
//         {
//           sin_ia_theta = 0;
//         }
//         if (std::abs(cos_ia_theta) < 1e-6)
//         {
//           cos_ia_theta = 0;
//         }
//         vertices[num_prev_points + ia] = {eq_radii[ir] * cos_ia_theta, eq_radii[ir] *
//         sin_ia_theta};
//       }
//     }
//     // Points on the boundary of the pin (pitch / 2)
//     size_t const num_prev_points = 1 + num_azimuthal / 2 + total_rings * num_azimuthal;
//     for (size_t ia = 0; ia < num_azimuthal; ++ia)
//     {
//       double sin_ia_theta = std::sin(theta * ia);
//       double cos_ia_theta = std::cos(theta * ia);
//       if (std::abs(sin_ia_theta) < 1e-6)
//       {
//         sin_ia_theta = 0;
//       }
//       if (std::abs(cos_ia_theta) < 1e-6)
//       {
//         cos_ia_theta = 0;
//       }
//       double const rx = std::abs(pitch / (2 * cos_ia_theta));
//       double const ry = std::abs(pitch / (2 * sin_ia_theta));
//       double const rb = std::min(rx, ry);
//       vertices[num_prev_points + ia] = {rb * cos_ia_theta, rb * sin_ia_theta};
//     }
//     for (size_t i = 0; i < num_points; ++i)
//     {
//       Log::debug("Point " + std::to_string(i) + ": " + std::to_string(vertices[i][0]) +
//       ", " + std::to_string(vertices[i][1]));
//     }
//
//     // Get the faces that make up the mesh
//     // -----------------------------------
//     size_t const num_faces = num_azimuthal * (total_rings + 1);
//     std::vector<Vec<4, size_t>> faces(num_faces);
//     // Establish a few aliases
//     size_t const na = num_azimuthal;
//     size_t const nr = total_rings;
//     size_t const ntric = 1 + na / 2; // Number of triangular points + center point
//     // Triangular quads
//     for (size_t ia = 0; ia < na / 2; ++ia)
//     {
//       size_t const p0 = 0;                  // Center point
//       size_t const p1 = ntric + ia * 2;      // Bottom right point on ring
//       size_t const p2 = ntric + ia * 2 + 1;  // Top right point on ring
//       size_t const p3 = 1 + ia;             // The triangular point
//       size_t       p4 = ntric + ia * 2 + 2;  // Top left point on ring
//       // If we're at the end of the ring, wrap around
//       if (p4 == ntric + na)
//       {
//         p4 = ntric;
//       }
//       faces[2 * ia    ] = {p0, p1, p2, p3};
//       faces[2 * ia + 1] = {p0, p3, p2, p4};
//     }
//     // Non-boundary and boundary quads
//     for (size_t ir = 1; ir < nr + 1; ++ir)
//     {
//       for (size_t ia = 0; ia < na; ++ia)
//       {
//         size_t const p0 = ntric + (ir - 1) * na + ia;     // Bottom left point
//         size_t const p1 = ntric + (ir    ) * na + ia;     // Bottom right point
//         size_t       p2 = ntric + (ir    ) * na + ia + 1; // Top right point
//         size_t       p3 = ntric + (ir - 1) * na + ia + 1; // Top left point
//         // If we're at the end of the ring, wrap around
//         if (ia + 1 == na)
//         {
//           p2 -= na;
//           p3 -= na;
//         }
//         faces[ir * na + ia] = {p0, p1, p2, p3};
//       }
//     }
//     Vector<Point2<Float>> vertices_t(static_cast<Size>(num_points));
//     double const half_pitch = pitch / 2;
//     for (size_t i = 0; i < num_points; ++i)
//     {
//       vertices_t[i] = {static_cast<Float>(vertices[i][0] + half_pitch),
//                        static_cast<Float>(vertices[i][1] + half_pitch)};
//       // Fix close to zero values
//       if (std::abs(vertices_t[i][0]) < 1e-6)
//       {
//         vertices_t[i][0] = 0;
//       }
//       if (std::abs(vertices_t[i][1]) < 1e-6)
//       {
//         vertices_t[i][1] = 0;
//       }
//     }
//     Vector<Int> faces_t(static_cast<Size>(num_faces * 4));
//     for (size_t i = 0; i < num_faces; ++i)
//     {
//       faces_t[4 * i    ] = static_cast<Int>(faces[i][0]);
//       faces_t[4 * i + 1] = static_cast<Int>(faces[i][1]);
//       faces_t[4 * i + 2] = static_cast<Int>(faces[i][2]);
//       faces_t[4 * i + 3] = static_cast<Int>(faces[i][3]);
//     }
//     this->quad.push_back(QuadMesh<Float, Int>(vertices_t, {}, faces_t));
//     return mesh_id;
//   } else if (mesh_order == 2) {
//     // Get the equivalent radius of each ring if it were a quadratic quadrilateral
//     double const theta = 2 * pi / num_azimuthal;
//     double const gamma = theta / 2;
//     double const sin_gamma = std::sin(gamma);
//     double const cos_gamma = std::cos(gamma);
//     double const sincos_gamma = sin_gamma * cos_gamma;
//     std::vector<double> eq_radii(total_rings);
//     // The innermost radius is a special case, and is essentially a triangle.
//     // Each quadratic shape is made up of the linear shape plus quadratic edges
//     // A_t = l² * sin(θ) / 2 = l² * sin(θ/2) * cos(θ/2)
//     // A_q = (l² - l²₀) * sin(θ) / 2 = (l² - l²₀) * sin(θ/2) * cos(θ/2)
//     // A_edge = (4 / 3) the area of the triangle formed by the vertices of the edge.
//     //        = (4 / 3) * 2l sin(θ/2) * (L - l cos(θ/2)) / 2 = (4 / 3) * l sin(θ/2) *
//     (L - l cos(θ/2))
//     //
//     // For N + 1 rings
//     // A_0 = pi R_0² = Na ( A_t + A_e0)
//     // A_i = pi (R_i² - R_{i-1}²) = Na ( A_q + A_ei - A_ei-1)
//     // A_N = P² - pi R_N² = P² - sum_i=0^N A_i
//     // Constraining L_N to be the value which minimizes the 2-norm of the integral of
//     the quadratic segment
//     // minus the circle arc is the correct thing to do, but holy cow the integral is a
//     mess.
//     // Therefore we settle for constraining l_i = r_i
//     double tri_area = ring_radii[0] * ring_radii[0] * sincos_gamma;
//     double ring_area = ring_areas[0] / num_azimuthal;
//     double l = ring_radii[0];
//     eq_radii[0] = 0.75 * (ring_area - tri_area) / (l * sin_gamma) + l * cos_gamma;
//     for (size_t i = 1; i < total_rings; ++i)
//     {
//       double const l_im1 = ring_radii[i - 1];
//       double const L_im1 = eq_radii[i - 1];
//       double const A_edge_im1 = (4.0 / 3.0) * l_im1 * sin_gamma * (L_im1 - l_im1 *
//       cos_gamma); double const l = ring_radii[i]; double const A_quad = (l * l - l_im1
//       * l_im1) * sincos_gamma; double const A_ring = ring_areas[i] / num_azimuthal;
//       eq_radii[i] = 0.75 * (A_ring - A_quad + A_edge_im1) / (l * sin_gamma) + l *
//       cos_gamma;
//     }
//     // Log the equivalent radii in debug mode
//     for (size_t i = 0; i < total_rings; ++i)
//     {
//       Log::debug("Ring " + std::to_string(i) + " equivalent radius: " +
//       std::to_string(eq_radii[i]));
//     }
//     // If any of the equivalent radii are larger than half the pitch, error
//     if (std::any_of(eq_radii.begin(), eq_radii.end(), [pitch](double r) { return r >
//     pitch / 2; }))
//     {
//       Log::error("The equivalent radius of a ring is larger than half the pitch.");
//       return -1;
//     }
//
//     // Get the points that make up the mesh
//     // -------------------------------------
//     // The number of points is:
//     //   Center point
//     //   2 * num_azimuthal for the triangular points inside the first ring
//     //   2 * num_azimuthal for the triangular points on the first ring
//     //   3 * num_azimuthal * total_rings
//     //
//     // Some aliases to make the code more readable
//     size_t const na = num_azimuthal;
//     size_t const nr = total_rings;
//     size_t const num_points = 1 + 4 * na + 3 * na * nr;
//     std::vector<Point2d> vertices(num_points);
//     // Center point
//     vertices[0] = {0, 0};
//     // Triangular points
//     double const rt = ring_radii[0] / 2;
//     for (size_t ia = 0; ia < na; ++ia)
//     {
//       double const sin_ia_theta = std::sin(ia * theta);
//       double const cos_ia_theta = std::cos(ia * theta);
//       // if ia is 0 or even, just do the 1 center point, otherwise we need 3 points at
//       (1/4, 2/4, 3/4) of the radius if (ia % 2 == 0)
//       {
//         vertices[1 + 2 * ia] = {rt * cos_ia_theta, rt * sin_ia_theta};
//       }
//       else
//       {
//         vertices[2 * ia] = {rt * cos_ia_theta / 2, rt * sin_ia_theta / 2};
//         vertices[2 * ia + 1] = {rt * cos_ia_theta, rt * sin_ia_theta};
//         vertices[2 * ia + 2] = {3 * rt * cos_ia_theta / 2, 3 * rt * sin_ia_theta / 2};
//       }
//     }
//     // Points on the first ring
//     size_t num_prev_points = 1 + 2 * na;
//     for (size_t ia = 0; ia < 2 * na; ++ia)
//     {
//       double sin_ia_gamma = std::sin(ia * gamma);
//       double cos_ia_gamma = std::cos(ia * gamma);
//       if (std::abs(sin_ia_gamma) < 1e-6)
//       {
//         sin_ia_gamma = 0;
//       }
//       if (std::abs(cos_ia_gamma) < 1e-6)
//       {
//         cos_ia_gamma = 0;
//       }
//       // if ia is 0 or even, we want the point at ring_radii[ir], otherwise we want the
//       point at eq_radii[ir] if (ia % 2 == 0)
//       {
//         vertices[num_prev_points + ia] = {ring_radii[0] * cos_ia_gamma, ring_radii[0] *
//         sin_ia_gamma};
//       }
//       else
//       {
//         vertices[num_prev_points + ia] = {eq_radii[0] * cos_ia_gamma, eq_radii[0] *
//         sin_ia_gamma};
//       }
//     }
//     // Points on and between the rings
//     for (size_t ir = 1; ir < total_rings; ++ir)
//     {
//       num_prev_points = 1 + 4 * na + 3 * na * (ir - 1);
//       // Between the rings
//       for (size_t ia = 0; ia < num_azimuthal; ++ia)
//       {
//         double sin_ia_theta = std::sin(ia * theta);
//         double cos_ia_theta = std::cos(ia * theta);
//         if (std::abs(sin_ia_theta) < 1e-6)
//         {
//           sin_ia_theta = 0;
//         }
//         if (std::abs(cos_ia_theta) < 1e-6)
//         {
//           cos_ia_theta = 0;
//         }
//         double const r = (ring_radii[ir] + ring_radii[ir - 1]) / 2;
//         vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
//       }
//       num_prev_points += num_azimuthal;
//       for (size_t ia = 0; ia < 2 * num_azimuthal; ++ia)
//       {
//         double sin_ia_gamma = std::sin(ia * gamma);
//         double cos_ia_gamma = std::cos(ia * gamma);
//         if (std::abs(sin_ia_gamma) < 1e-6)
//         {
//           sin_ia_gamma = 0;
//         }
//         if (std::abs(cos_ia_gamma) < 1e-6)
//         {
//           cos_ia_gamma = 0;
//         }
//         // if ia is 0 or even, we want the point at ring_radii[ir], otherwise we want
//         the point at eq_radii[ir] if (ia % 2 == 0)
//         {
//           vertices[num_prev_points + ia] = {ring_radii[ir] * cos_ia_gamma,
//           ring_radii[ir] * sin_ia_gamma};
//         }
//         else
//         {
//           vertices[num_prev_points + ia] = {eq_radii[ir] * cos_ia_gamma, eq_radii[ir] *
//           sin_ia_gamma};
//         }
//       }
//     }
//     // Quadratic points before the boundary
//     num_prev_points = 1 + 4 * na + 3 * na * (total_rings - 1);
//     for (size_t ia = 0; ia < num_azimuthal; ++ia)
//     {
//       double sin_ia_theta = std::sin(ia * theta);
//       double cos_ia_theta = std::cos(ia * theta);
//       if (std::abs(sin_ia_theta) < 1e-6)
//       {
//         sin_ia_theta = 0;
//       }
//       if (std::abs(cos_ia_theta) < 1e-6)
//       {
//         cos_ia_theta = 0;
//       }
//       // pitch and last ring radius
//       double const rx = std::abs(pitch / (2 * cos_ia_theta));
//       double const ry = std::abs(pitch / (2 * sin_ia_theta));
//       double const rb = std::min(rx, ry);
//       double const r = (rb + ring_radii[total_rings - 1]) / 2;
//       vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
//     }
//     // Points on the boundary of the pin (pitch / 2)
//     num_prev_points += num_azimuthal;
//     for (size_t ia = 0; ia < 2 * num_azimuthal; ++ia)
//     {
//       double sin_ia_gamma = std::sin(gamma * ia);
//       double cos_ia_gamma = std::cos(gamma * ia);
//       if (std::abs(sin_ia_gamma) < 1e-6)
//       {
//         sin_ia_gamma = 0;
//       }
//       if (std::abs(cos_ia_gamma) < 1e-6)
//       {
//         cos_ia_gamma = 0;
//       }
//       double const rx = std::abs(pitch / (2 * cos_ia_gamma));
//       double const ry = std::abs(pitch / (2 * sin_ia_gamma));
//       double const rb = std::min(rx, ry);
//       vertices[num_prev_points + ia] = {rb * cos_ia_gamma, rb * sin_ia_gamma};
//     }
//     for (size_t i = 0; i < num_points; ++i)
//     {
//       Log::debug("Point " + std::to_string(i) + ": " + std::to_string(vertices[i][0]) +
//       ", " + std::to_string(vertices[i][1]));
//     }
//
//     // Get the faces that make up the mesh
//     // -----------------------------------
//     size_t const num_faces = na * (nr + 1);
//     std::vector<Vec<8, size_t>> faces(num_faces);
//     // Triangular quads
//     for (size_t ia = 0; ia < na / 2; ++ia)
//     {
//       size_t const p0 = 0;                      // Center point
//       size_t const p1 = 1 + 2 * na + 4 * ia;    // Bottom right point on ring
//       size_t const p2 = p1 + 2;                 // Top right point on ring
//       size_t const p3 = 3 + 4 * ia;             // The triangular point
//       size_t       p4 = p2 + 2;                 // Top left point on ring
//       size_t const p5 = 1 + 4 * ia;             // Bottom quadratic point
//       size_t const p6 = p1 + 1;                 // Right quadratic point
//       size_t const p7 = p3 + 1;                 // Top tri quadratic point
//       size_t const p8 = p3 - 1;                 // Bottom tri quadratic point
//       size_t const p9 = p2 + 1;                 // Top right quadratic point
//       size_t       p10 = p7 + 1;                // Top left quadratic point
//       // If we're at the end of the ring, wrap around
//       if (p10 == 1 + 2 * na)
//       {
//         p4 -= 2 * na;
//         p10 -= 2 * na;
//       }
//       faces[2 * ia    ] = {p0, p1, p2, p3, p5, p6, p7, p8};
//       faces[2 * ia + 1] = {p0, p3, p2, p4, p8, p7, p9, p10};
//     }
//     // All other faces
//     for (size_t ir = 1; ir < nr + 1; ++ir)
//     {
//       size_t const np = 1 + 2 * na + 3 * na * (ir - 1);
//       for (size_t ia = 0; ia < na; ++ia)
//       {
//         size_t const p0 = np + 2 * ia;
//         size_t const p1 = p0 + 3 * na;
//         size_t       p2 = p1 + 2;
//         size_t       p3 = p0 + 2;
//         size_t const p4 = np + 2 * na + ia;
//         size_t const p5 = p1 + 1;
//         size_t       p6 = p4 + 1;
//         size_t const p7 = p0 + 1;
//         // If we're at the end of the ring, wrap around
//         if (ia + 1 == na)
//         {
//           p2 -= 2 * na;
//           p3 -= 2 * na;
//           p6 -= na;
//         }
//         faces[ir * na + ia] = {p0, p1, p2, p3, p4, p5, p6, p7};
//       }
//     }
//     // Print the faces
//     for (size_t i = 0; i < num_faces; ++i)
//     {
//       Log::debug("Face " + std::to_string(i) + ": " + std::to_string(faces[i][0]) + ",
//       " + std::to_string(faces[i][1]) + ", " + std::to_string(faces[i][2]) + ", " +
//       std::to_string(faces[i][3]) + ", " + std::to_string(faces[i][4]) + ", " +
//       std::to_string(faces[i][5]) + ", " + std::to_string(faces[i][6]) + ", " +
//       std::to_string(faces[i][7]));
//     }
//
//     Vector<Point2<Float>> vertices_t(static_cast<Size>(num_points));
//     double const half_pitch = pitch / 2;
//     for (size_t i = 0; i < num_points; ++i)
//     {
//       vertices_t[i] = {static_cast<Float>(vertices[i][0] + half_pitch),
//                        static_cast<Float>(vertices[i][1] + half_pitch)};
//       // Fix close to zero values
//       if (std::abs(vertices_t[i][0]) < 1e-6)
//       {
//         vertices_t[i][0] = 0;
//       }
//       if (std::abs(vertices_t[i][1]) < 1e-6)
//       {
//         vertices_t[i][1] = 0;
//       }
//     }
//     Vector<Int> faces_t(static_cast<Size>(num_faces * 8));
//     for (size_t i = 0; i < num_faces; ++i)
//     {
//       faces_t[8 * i    ] = static_cast<Int>(faces[i][0]);
//       faces_t[8 * i + 1] = static_cast<Int>(faces[i][1]);
//       faces_t[8 * i + 2] = static_cast<Int>(faces[i][2]);
//       faces_t[8 * i + 3] = static_cast<Int>(faces[i][3]);
//       faces_t[8 * i + 4] = static_cast<Int>(faces[i][4]);
//       faces_t[8 * i + 5] = static_cast<Int>(faces[i][5]);
//       faces_t[8 * i + 6] = static_cast<Int>(faces[i][6]);
//       faces_t[8 * i + 7] = static_cast<Int>(faces[i][7]);
//     }
//     this->quadratic_quad.push_back(QuadraticQuadMesh<Float, Int>(vertices_t, {},
//     faces_t)); return mesh_id;
//   } else {
//     Log::error("Only linear meshes are supported for cylindrical pin meshes.");
//     return -1;
//   }
//   return 0;
// }
//
// template <std::floating_point T, std::signed_integral I>
// int SpatialPartition::make_rectangular_pin_mesh(Vec2<Float> const dxdy,
//                                                       int const nx,
//                                                       int const ny)
//{
//   if (dxdy[0] <= 0 || dxdy[1] <= 0)
//   {
//     Log::error("Pin dimensions must be positive");
//   }
//   if ( nx <= 0 || ny <= 0 )
//   {
//     Log::error("Number of divisions in x and y must be positive");
//   }
//
//   Size mesh_id = mesh_id = this->quad.size();
//   Log::info("Making rectangular pin mesh " + std::to_string(mesh_id));
//
//   // Make the vertices
//   Vector<Point2<Float>> vertices(static_cast<Size>((nx + 1) * (ny + 1)));
//   Float const delta_x = dxdy[0] / nx;
//   Float const delta_y = dxdy[1] / ny;
//   for (Size j = 0; j < ny + 1; ++j)
//   {
//     for (Size i = 0; i < nx + 1; ++i)
//     {
//       vertices[j * (nx + 1) + i] = {i * delta_x, j * delta_y};
//     }
//   }
//   // Make the faces
//   Vector<Int> faces(4 * static_cast<Size>(nx * ny));
//   // Left to right, bottom to top
//   for (Size j = 0; j < ny; ++j)
//   {
//     for (Size i = 0; i < nx; ++i)
//     {
//       faces[4 * (j * nx + i)    ] = (j    ) * (nx + 1) + i    ;
//       faces[4 * (j * nx + i) + 1] = (j    ) * (nx + 1) + i + 1;
//       faces[4 * (j * nx + i) + 2] = (j + 1) * (nx + 1) + i + 1;
//       faces[4 * (j * nx + i) + 3] = (j + 1) * (nx + 1) + i    ;
//     }
//   }
//   this->quad.push_back(QuadMesh<Float, Int>(vertices, {}, faces));
//   return mesh_id;
// }
//
// template <std::floating_point T, std::signed_integral I>
// int SpatialPartition::makeCoarseCell(I const mesh_type,
//                                              I const mesh_id,
//                                              std::vector<Material> const &
//                                              cc_materials)
//{
//     Size const cc_id = this->coarse_cells.size();
//     Log::info("Making coarse cell " + std::to_string(cc_id));
//
//     // Ensure valid mesh type
//     if (mesh_type != -1) {
//         if (!(1 <= mesh_type && mesh_type <= 6)) {
//             Log::error("Invalid mesh type: " + std::to_string(mesh_type));
//             return -1;
//         }
//     }
//     // Ensure that the mesh exists and get its AABB
//     AABox2<Float> aabb;
//     if (mesh_id != -1) {
//         switch (mesh_type) {
//             case 1:
//                 if (!(0 <= mesh_id && mesh_id < this->tri.size())) {
//                     Log::error("Tri mesh " + std::to_string(mesh_id) + " does not
//                     exist"); return -1;
//                 }
//                 aabb = bounding_box(this->tri[mesh_id].vertices);
//                 UM2_ASSERT(cc_materials.size() == num_faces(this->tri[mesh_id]));
//                 break;
//             case 2:
//                 if (!(0 <= mesh_id && mesh_id < this->quad.size())) {
//                     Log::error("Quad mesh " + std::to_string(mesh_id) + " does not
//                     exist"); return -1;
//                 }
//                 aabb = bounding_box(this->quad[mesh_id].vertices);
//                 UM2_ASSERT(cc_materials.size() == num_faces(this->quad[mesh_id]));
//                 break;
//             case 3:
//                 if (!(0 <= mesh_id && mesh_id < this->tri_quad.size())) {
//                     Log::error("Tri-quad mesh " + std::to_string(mesh_id) + " does not
//                     exist"); return -1;
//                 }
//                 aabb = bounding_box(this->tri_quad[mesh_id].vertices);
//                 UM2_ASSERT(cc_materials.size() == num_faces(this->tri_quad[mesh_id]));
//                 break;
//             case 4:
//                 if (!(0 <= mesh_id && mesh_id < this->quadratic_tri.size())) {
//                     Log::error("Quadratic tri mesh " + std::to_string(mesh_id) + " does
//                     not exist"); return -1;
//                 }
//                 aabb = bounding_box(this->quadratic_tri[mesh_id].vertices);
//                 UM2_ASSERT(cc_materials.size() ==
//                 num_faces(this->quadratic_tri[mesh_id])); break;
//             case 5:
//                 if (!(0 <= mesh_id && mesh_id < this->quadratic_quad.size())) {
//                     Log::error("Quadratic quad mesh " + std::to_string(mesh_id) + "
//                     does not exist"); return -1;
//                 }
//                 aabb = bounding_box(this->quadratic_quad[mesh_id].vertices);
//                 UM2_ASSERT(cc_materials.size() ==
//                 num_faces(this->quadratic_quad[mesh_id])); break;
//             case 6:
//                 if (!(0 <= mesh_id && mesh_id < this->quadratic_tri_quad.size())) {
//                     Log::error("Quadratic tri-quad mesh " + std::to_string(mesh_id) + "
//                     does not exist"); return -1;
//                 }
//                 aabb = bounding_box(this->quadratic_tri_quad[mesh_id].vertices);
//                 UM2_ASSERT(cc_materials.size() ==
//                 num_faces(this->quadratic_tri_quad[mesh_id])); break;
//             default: Log::error("Invalid mesh type: " + std::to_string(mesh_type));
//                 return -1;
//         }
//     }
//
//     Vector<MaterialID> material_ids_vec(cc_materials.size());
//     for (Size i = 0; i < cc_materials.size(); ++i) {
//         auto it = std::find(this->materials.begin(), this->materials.end(),
//         cc_materials[i]); ptrdiff_t const idx = std::distance(this->materials.begin(),
//         it); if (idx >= this->materials.size()) {
//           this->materials.push_back(cc_materials[i]);
//         }
//         material_ids_vec[i] = idx;
//     }
//     auto const dxdy = aabb.maxima - aabb.minima;
//     // Create the coarse cell
//     this->coarse_cells.push_back({dxdy, mesh_type, mesh_id, material_ids_vec});
//     return cc_id;
// }

//=============================================================================
// makeCoarseCell
//=============================================================================

auto
SpatialPartition::makeCoarseCell(Vec2<Float> const dxdy, MeshType const mesh_type,
                                 Size const mesh_id,
                                 Vector<MaterialID> const & material_ids) -> Size
{
  Size const cc_id = coarse_cells.size();
  Log::info("Making coarse cell " + std::to_string(cc_id));
  // Ensure dx and dy are positive
  if (dxdy[0] <= 0 || dxdy[1] <= 0) {
    Log::error("dx and dy must be positive:; " + std::to_string(dxdy[0]) + ", " +
               std::to_string(dxdy[1]));
    return -1;
  }
  // Ensure that the mesh exists
  if (mesh_id != -1) {
    checkMeshExists(mesh_type, mesh_id);
    // Make sure materials are specified
    if (material_ids.empty()) {
      Log::error("No materials specified");
      return -1;
    }
  }

  // Create the coarse cell
  this->coarse_cells.push_back(CoarseCell{dxdy, mesh_type, mesh_id, material_ids});
  return cc_id;
}

//=============================================================================
// makeRTM
//=============================================================================

auto
SpatialPartition::makeRTM(std::vector<std::vector<Size>> const & cc_ids) -> Size
{
  Size const rtm_id = rtms.size();
  Log::info("Making ray tracing module " + std::to_string(rtm_id));
  std::vector<Size> unique_cc_ids;
  std::vector<Vec2<Float>> dxdy;
  // Ensure that all coarse cells exist
  Size const num_cc = coarse_cells.size();
  for (auto const & cc_ids_row : cc_ids) {
    for (auto const & id : cc_ids_row) {
      if (id < 0 || id >= num_cc) {
        Log::error("Coarse cell " + std::to_string(id) + " does not exist");
        return -1;
      }
      auto const it = std::find(unique_cc_ids.begin(), unique_cc_ids.end(), id);
      if (it == unique_cc_ids.end()) {
        unique_cc_ids.push_back(id);
        // We know id > 0, so subtracting 1 is safe
        dxdy.push_back(coarse_cells[id].dxdy);
      }
    }
  }
  // For a max pin ID N, the RectilinearGrid constructor needs all dxdy from 0 to N.
  // To get around this requirement, we will renumber the coarse cells to be 0, 1, 2,
  // 3, ..., and then use the renumbered IDs to create the RectilinearGrid.
  std::vector<std::vector<Size>> cc_ids_renumbered(cc_ids.size());
  for (size_t i = 0; i < cc_ids.size(); ++i) {
    cc_ids_renumbered[i].resize(cc_ids[i].size());
    for (size_t j = 0; j < cc_ids[i].size(); ++j) {
      auto const it = std::find(unique_cc_ids.begin(), unique_cc_ids.end(), cc_ids[i][j]);
      assert(it != unique_cc_ids.end());
#if UM2_ENABLE_INT64 == 0
      cc_ids_renumbered[i][j] = static_cast<Size>(it - unique_cc_ids.begin());
#else
      cc_ids_renumbered[i][j] = it - unique_cc_ids.begin();
#endif
    }
  }
  // Create the rectilinear grid
  RectilinearGrid2<Float> grid(dxdy, cc_ids_renumbered);
  // Ensure the grid has the same dxdy as all other RTMs
  if (!rtms.empty()) {
    auto const eps = eps_distance<Float>;
    if (um2::abs(grid.width() - rtms[0].width()) > eps ||
        um2::abs(grid.height() - rtms[0].height()) > eps) {
      Log::error("All RTMs must have the same dxdy");
      return -1;
    }
  }
  // Flatten the coarse cell IDs (rows are reversed)
  size_t const num_rows = cc_ids.size();
  size_t const num_cols = cc_ids[0].size();
  Vector<Int> cc_ids_flat(static_cast<Size>(num_rows * num_cols));
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      cc_ids_flat[static_cast<Size>(i * num_cols + j)] =
          static_cast<Int>(cc_ids[num_rows - 1 - i][j]);
    }
  }
  RTM rtm;
  rtm.grid = um2::move(grid);
  rtm.children = um2::move(cc_ids_flat);
  rtms.push_back(um2::move(rtm));
  return rtm_id;
}

//=============================================================================
// makeLattice
//=============================================================================

auto
SpatialPartition::makeLattice(std::vector<std::vector<Size>> const & rtm_ids) -> Size
{
  Size const lat_id = lattices.size();
  Log::info("Making lattice " + std::to_string(lat_id));
  // Ensure that all RTMs exist
  Size const num_rtm = rtms.size();
  for (auto const & rtm_ids_row : rtm_ids) {
    auto const it =
        std::find_if(rtm_ids_row.begin(), rtm_ids_row.end(),
                     [num_rtm](Size const id) { return id < 0 || id >= num_rtm; });
    if (it != rtm_ids_row.end()) {
      Log::error("RTM " + std::to_string(*it) + " does not exist");
      return -1;
    }
  }
  // Create the lattice
  // Ensure each row has the same number of columns
  Point2<Float> const minima(0, 0);
  Vec2<Float> const spacing(rtms[0].width(), rtms[0].height());
  size_t const num_rows = rtm_ids.size();
  size_t const num_cols = rtm_ids[0].size();
  for (size_t i = 1; i < num_rows; ++i) {
    if (rtm_ids[i].size() != num_cols) {
      Log::error("Each row must have the same number of columns");
      return -1;
    }
  }
  Vec2<Size> const num_cells(num_cols, num_rows);
  RegularGrid2<Float> grid(minima, spacing, num_cells);
  // Flatten the RTM IDs (rows are reversed)
  Vector<Int> rtm_ids_flat(static_cast<Size>(num_rows * num_cols));
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      rtm_ids_flat[static_cast<Size>(i * num_cols + j)] =
          static_cast<Int>(rtm_ids[num_rows - 1 - i][j]);
    }
  }
  Lattice lat;
  lat.grid = um2::move(grid);
  lat.children = um2::move(rtm_ids_flat);
  lattices.push_back(um2::move(lat));
  return lat_id;
}

//=============================================================================
// makeAssembly
//=============================================================================

auto
SpatialPartition::makeAssembly(std::vector<Size> const & lat_ids,
                               std::vector<Float> const & z) -> Size
{
  Size const asy_id = assemblies.size();
  Log::info("Making assembly " + std::to_string(asy_id));
  // Ensure that all lattices exist
  Size const num_lat = lattices.size();
  {
    auto const it =
        std::find_if(lat_ids.cbegin(), lat_ids.cend(),
                     [num_lat](Size const id) { return id < 0 || id >= num_lat; });
    if (it != lat_ids.end()) {
      Log::error("Lattice " + std::to_string(*it) + " does not exist");
      return -1;
    }
  }
  // Ensure the number of lattices is 1 less than the number of z-planes
  if (lat_ids.size() + 1 != z.size()) {
    Log::error("The number of lattices must be 1 less than the number of z-planes");
    return -1;
  }
  // Ensure all z-planes are in ascending order
  if (!std::is_sorted(z.begin(), z.end())) {
    Log::error("The z-planes must be in ascending order");
    return -1;
  }
  // Ensure this assembly is the same height as all other assemblies
  if (!assemblies.empty()) {
    auto const eps = eps_distance<Float>;
    Float const assem_top = assemblies[0].xMax();
    Float const assem_bot = assemblies[0].xMin();
    if (um2::abs(z.back() - assem_top) > eps || um2::abs(z.front() - assem_bot) > eps) {
      Log::error("All assemblies must have the same height");
      return -1;
    }
  }
  // Ensure the lattices all have the same dimensions. Since they are composed of RTMs,
  // it is sufficient to check numXCells and numYCells.
  {
    Size const num_xcells = lattices[lat_ids[0]].numXCells();
    Size const num_ycells = lattices[lat_ids[0]].numYCells();
    auto const it = std::find_if(lat_ids.cbegin(), lat_ids.cend(),
                                 [num_xcells, num_ycells, this](Size const id) {
                                   return this->lattices[id].numXCells() != num_xcells ||
                                          this->lattices[id].numYCells() != num_ycells;
                                 });
    if (it != lat_ids.end()) {
      Log::error("All lattices must have the same xy-dimensions");
      return -1;
    }
  }

  // Clean this up. Too many static_casts.
  Vector<Int> lat_ids_i(static_cast<Size>(lat_ids.size()));
  for (size_t i = 0; i < lat_ids.size(); ++i) {
    lat_ids_i[static_cast<Size>(i)] = static_cast<Int>(lat_ids[i]);
  }

  RectilinearGrid1<Float> grid;
  grid.divs[0].resize(static_cast<Size>(z.size()));
  std::copy(z.cbegin(), z.cend(), grid.divs[0].begin());
  Assembly asy;
  asy.grid = um2::move(grid);
  asy.children = um2::move(lat_ids_i);
  assemblies.push_back(um2::move(asy));
  return asy_id;
}

//=============================================================================
// makeCore
//=============================================================================

auto
SpatialPartition::makeCore(std::vector<std::vector<Size>> const & asy_ids) -> Size
{
  Log::info("Making core");
  // Ensure that all assemblies exist
  Size const num_asy = assemblies.size();
  for (auto const & asy_ids_row : asy_ids) {
    auto const it =
        std::find_if(asy_ids_row.cbegin(), asy_ids_row.cend(),
                     [num_asy](Size const id) { return id < 0 || id >= num_asy; });
    if (it != asy_ids_row.end()) {
      Log::error("Assembly " + std::to_string(*it) + " does not exist");
      return -1;
    }
  }
  std::vector<Vec2<Float>> dxdy(static_cast<size_t>(num_asy));
  for (size_t i = 0; i < static_cast<size_t>(num_asy); ++i) {
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
    auto const lat_id = static_cast<Size>(assemblies[static_cast<Size>(i)].getChild(0));
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
    dxdy[i] = {lattices[lat_id].width(), lattices[lat_id].height()};
  }
  // Create the rectilinear grid
  RectilinearGrid2<Float> grid(dxdy, asy_ids);
  // Flatten the assembly IDs (rows are reversed)
  size_t const num_rows = asy_ids.size();
  size_t const num_cols = asy_ids[0].size();
  Vector<Int> asy_ids_flat(static_cast<Size>(num_rows * num_cols));
  for (size_t i = 0; i < num_rows; ++i) {
    if (asy_ids[i].size() != num_cols) {
      Log::error("Each row must have the same number of columns");
      return -1;
    }
    for (size_t j = 0; j < num_cols; ++j) {
      asy_ids_flat[static_cast<Size>(i * num_cols + j)] =
          static_cast<Int>(asy_ids[num_rows - 1 - i][j]);
    }
  }
  core.grid = um2::move(grid);
  core.children = um2::move(asy_ids_flat);
  return 0;
}

//=============================================================================
// importCoarseCells
//=============================================================================

void
SpatialPartition::importCoarseCells(std::string const & filename)
{
  Log::info("Importing coarse cells from " + filename);
  MeshFile<Float, Int> mesh_file;
  importMesh(filename, mesh_file);

  // Get the materials
  std::vector<std::string> material_names;
  mesh_file.getMaterialNames(material_names);
  materials.resize(static_cast<Size>(material_names.size()));
  for (size_t i = 0; i < material_names.size(); ++i) {
    ShortString & this_name = materials[static_cast<Size>(i)].name;
    this_name = ShortString(material_names[i].substr(9).c_str());
  }

  // For each coarse cell
  std::stringstream ss;
  Size const num_coarse_cells = numCoarseCells();
  for (Size i = 0; i < num_coarse_cells; ++i) {
    // Get the submesh for the coarse cell
    ss.str("");
    ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << i;
    MeshFile<Float, Int> cc_submesh;
    mesh_file.getSubmesh(ss.str(), cc_submesh);

    // Get the mesh type and material IDs
    MeshType const mesh_type = cc_submesh.getMeshType();
    CoarseCell & cc = coarse_cells[i];
    cc.mesh_type = mesh_type;
    std::vector<MaterialID> mat_ids;
    cc_submesh.getMaterialIDs(mat_ids, material_names);
    cc.material_ids.resize(static_cast<Size>(mat_ids.size()));
    std::copy(mat_ids.cbegin(), mat_ids.cend(), cc.material_ids.begin());

    // Create the FaceVertexMesh and shift it from global coordinates to local
    // coordinates, with the bottom left corner of the AABB at the origin
    AxisAlignedBox2<Float> bb;
    Point2<Float> * vertices = nullptr;
    size_t const num_verts = cc_submesh.vertices.size();
    // clang-tidy false positive
    // NOLINTBEGIN(bugprone-branch-clone) justified above
    switch (mesh_type) {
    case MeshType::Tri: {
      cc.mesh_id = tri.size();
      tri.push_back(um2::move(TriMesh<2, Float, Int>(cc_submesh)));
      bb = tri.back().boundingBox();
      vertices = tri.back().vertices.data();
      break;
    }
    case MeshType::Quad: {
      cc.mesh_id = quad.size();
      quad.push_back(um2::move(QuadMesh<2, Float, Int>(cc_submesh)));
      bb = quad.back().boundingBox();
      vertices = quad.back().vertices.data();
      break;
    }
    case MeshType::QuadraticTri: {
      cc.mesh_id = quadratic_tri.size();
      quadratic_tri.push_back(um2::move(QuadraticTriMesh<2, Float, Int>(cc_submesh)));
      bb = quadratic_tri.back().boundingBox();
      vertices = quadratic_tri.back().vertices.data();
      break;
    }
    case MeshType::QuadraticQuad: {
      cc.mesh_id = quadratic_quad.size();
      quadratic_quad.push_back(um2::move(QuadraticQuadMesh<2, Float, Int>(cc_submesh)));
      bb = quadratic_quad.back().boundingBox();
      vertices = quadratic_quad.back().vertices.data();
      break;
    }
    // NOLINTEND(bugprone-branch-clone)
    default:
      Log::error("Mesh type not supported");
    }

    // Shift the points so that the min point is at the origin.
    Point2<Float> const min_point = bb.minima;
    for (size_t ip = 0; ip < num_verts; ++ip) {
      vertices[ip] -= min_point;
    }
#ifndef NDEBUG
    Point2<Float> const dxdy = bb.maxima - bb.minima;
    assert(isApprox(dxdy, cc.dxdy));
#endif
  }
}
} // namespace um2::mpact
