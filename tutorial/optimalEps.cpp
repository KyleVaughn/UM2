#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/modular_rays.hpp>

#include <iostream>

template <Int D>
constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q;
  q[0] = um2::Vec<D, Float>::zero();
  q[1] = um2::Vec<D, Float>::zero();
  q[2] = um2::Vec<D, Float>::zero();
  q[1][0] = castIfNot<Float>(2);
  return q;
}

template <Int D>
constexpr auto
makeSeg2() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(1);
  q[2][1] = castIfNot<Float>(1);
  return q;
}

template <Int D>
constexpr auto
makeSeg3() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(1);
  q[2][1] = castIfNot<Float>(-1);
  return q;
}

template <Int D>
constexpr auto
makeSeg4() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(2);
  q[2][1] = castIfNot<Float>(1);
  return q;
}

template <Int D>
constexpr auto
makeSeg5() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(2);
  q[2][1] = castIfNot<Float>(-1);
  return q;
}

template <Int D>
constexpr auto
makeSeg6() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(0);
  q[2][1] = castIfNot<Float>(1);
  return q;
}

template <Int D>
constexpr auto
makeSeg7() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(0);
  q[2][1] = castIfNot<Float>(-1);
  return q;
}

template <Int D>
constexpr auto
makeSeg8() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[1][0] = castIfNot<Float>(2);
  q[2][0] = castIfNot<Float>(4);
  q[2][1] = castIfNot<Float>(3);
  return q;
}

auto
mySolveCubic(Float a, Float b, Float c, Float d, Float ptol, Float qp_tol) -> um2::Vec3F
{
  auto constexpr invalid = castIfNot<Float>(1e16);

  Float const p = (3 * a * c - b * b) / (3 * a * a);
  Float const q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a);
  Float const q_over_p = q / p;

  um2::Vec3F roots = um2::Vec3F::zero() + invalid;
  if (um2::abs(p) < ptol || um2::abs(q_over_p) > qp_tol) {
    roots[0] = um2::cbrt(-q);
  } else {
    Float const disc = q * q / 4 + p * p * p / 27;
    if (disc > 0) {
      // One real root.
      Float const sqrt_disc = um2::sqrt(disc);
      Float const u = um2::cbrt(-q / 2 + sqrt_disc);
      // v = -p/(3*u)
      roots[0] = u - p / (3 * u);

      Float const qp3 = 3 * q_over_p;
      roots[1] = -qp3 / 2;
      roots[2] = qp3;
    } else {
      ASSERT(p < 0);
      // Three real roots.
      Float const sqrt_p3 = um2::sqrt(-p / 3);
      ASSERT(3 * q / (2 * p * sqrt_p3) <= 1);
      ASSERT(3 * q / (2 * p * sqrt_p3) >= -1);
      Float const theta = um2::acos(3 * q_over_p / (2 * sqrt_p3)) / 3;
      Float constexpr shift = 2 * um2::pi<Float> / 3;
      for (Int i = 0; i < 3; ++i) {
        roots[i] = 2 * sqrt_p3 * um2::cos(theta - i * shift);
      }
    }
  }

  // Convert back from depressed cubic to original cubic.
  for (Int i = 0; i < 3; ++i) {
    roots[i] -= b / (3 * a);
  }
  return roots;
}

auto
pointClosestTo(um2::QuadraticSegment2 const q,
               um2::Point2 const p,
               Float const ptol,
               Float const qp_tol) -> Float
{

  if (um2::isStraight(q)) {
    return um2::LineSegment2(q[0], q[1]).pointClosestTo(p);
  }

  auto const coeffs = q.getPolyCoeffs();
  auto const vc = coeffs[0];
  auto const vb = coeffs[1];
  auto const va = coeffs[2];
  auto const vw = vc - p;
  Float const a = 2 * squaredNorm(va);
  Float const b = 3 * va.dot(vb);
  Float const c = squaredNorm(vb) + 2 * va.dot(vw);
  Float const d = vb.dot(vw);

  // Return the real part of the 3 roots of the cubic equation
  auto const roots = mySolveCubic(a, b, c, d, ptol, qp_tol);

  // Find the root which minimizes the squared distance to the point.
  // If the closest root is less than 0 or greater than 1, it isn't valid.
  // It's not clear that we can simply clamp the root to [0, 1], so we test
  // against v[0] and v[1] explicitly.

  Float r = 0;
  Float sq_dist = p.squaredDistanceTo(q[0]);
  Float const sq_dist1 = p.squaredDistanceTo(q[1]);
  if (sq_dist1 < sq_dist) {
    r = 1;
    sq_dist = sq_dist1;
  }

  for (auto const rr : roots) {
    if (0 <= rr && rr <= 1) {
      auto const p_root = q(rr);
      Float const p_sq_dist = p.squaredDistanceTo(p_root);
      if (p_sq_dist < sq_dist) {
        r = rr;
        sq_dist = p_sq_dist;
      }
    }
  }
  return r;
}

auto
testEdgeForIntersections(um2::QuadraticSegment2 const q,
    Float const ptol, Float const qp_tol
    ) -> Float
{
  Int constexpr num_angles = 128; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  Float max_err = 0;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(num_angles);
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto intersections = q.intersect(ray);
      for (Int j = 0; j < 2; ++j) {
        Float const r = intersections[j];
        if (0 < r) {
          um2::Point2 const p = ray(r);
          Float const s = pointClosestTo(q, p, ptol, qp_tol);
          um2::Point2 const q_closest = q(s);
          Float const d = q_closest.distanceTo(p);
          max_err = um2::max(max_err, d);
        }
      }
    }
  }
  return max_err;
}

auto
main() -> int
{
  um2::Vec<7, Float> best_errs = um2::Vec<7, Float>::zero() + 100;
  Float best_ptol = 10000;
  Float best_qp_tol = 10000;

  Float const qp_tol_start = 1500;
  Float const qp_tol_end = 1700;
  Float const dqp_tol = 10; // +=

  Float const ptol_start = 1e-6;
  Float const ptol_end = 1e-8;
  Float const dptol = 0.5; // *=

  auto qp_tol = qp_tol_start; 
  while (qp_tol < qp_tol_end) { 
    // Vary ptol from 1e-4 to 1e-8
    auto ptol = ptol_start; 
    while (ptol > ptol_end) {
      std::cout << "qp_tol: " << qp_tol << " ptol: " << ptol << std::endl;
      um2::Vec<7, Float> errs;
      errs[0] = testEdgeForIntersections(makeSeg2<2>(), ptol, qp_tol);
      errs[1] = testEdgeForIntersections(makeSeg3<2>(), ptol, qp_tol);
      errs[2] = testEdgeForIntersections(makeSeg4<2>(), ptol, qp_tol);
      errs[3] = testEdgeForIntersections(makeSeg5<2>(), ptol, qp_tol);
      errs[4] = testEdgeForIntersections(makeSeg6<2>(), ptol, qp_tol);
      errs[5] = testEdgeForIntersections(makeSeg7<2>(), ptol, qp_tol);
      errs[6] = testEdgeForIntersections(makeSeg8<2>(), ptol, qp_tol);
      std::cout << "Max errors: ";
      for (auto const e : errs) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
      if (errs.norm() < best_errs.norm()) {
        best_errs = errs;
        best_ptol = ptol;
        best_qp_tol = qp_tol;
      }
      ptol *= dptol; 
    }
    qp_tol += dqp_tol;
  }
  std::cout << "Best errors: ";
  for (auto const e : best_errs) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
  std::cout << "Best ptol: " << best_ptol << " Best qp_tol: " << best_qp_tol << std::endl;
  return 0;
}
