#include <um2/physics/cmfd.hpp>
#include <um2/common/cast_if_not.hpp>
#include <um2/stdlib/numbers.hpp>
#include <um2/stdlib/algorithm/fill.hpp>

#include "../test_macros.hpp"

#include <iostream>

// NOLINTBEGIN(readability-identifier-naming)

TEST_CASE(beta_n)
{
  auto constexpr eps = castIfNot<Float>(1e-6);
  auto constexpr h = castIfNot<Float>(0.25);
  auto constexpr mu_n = castIfNot<Float>(0.339981);
  auto const computed = um2::cmfd::beta_n(h, mu_n);
  auto constexpr expected = castIfNot<Float>(0.12146542875174537);
  ASSERT_NEAR(computed, expected, eps);
}

TEST_CASE(set_An)
{
  auto constexpr w = castIfNot<Float>(1.26);
  Int constexpr p = 4;
  auto constexpr Sigma_t = castIfNot<Float>(2.65038);
  auto constexpr Sigma_s = castIfNot<Float>(2.4807);
  auto constexpr c = Sigma_s / Sigma_t;
  Int constexpr s = 1;
  auto constexpr eta = castIfNot<Float>(0.0);

  auto constexpr mu_n = castIfNot<Float>(-0.9972638618494816);
  auto constexpr alpha = 0.0;

  um2::CMFDCellParams const params(w, p, Sigma_t, c, s, eta);
  using ComplexF = um2::cmfd::ComplexF;
  um2::Matrix<ComplexF> An(p, p);
  An.zero();
  um2::cmfd::set_An(An, params, mu_n, alpha);
  auto constexpr d = castIfNot<Float>(0.526278355167952);
  auto constexpr u = castIfNot<Float>(0.47372164483204804);
  ASSERT_NEAR(An(0, 0).real(), d, 1e-6);
  for (Int i = 1; i < p; ++i) {
    ASSERT_NEAR(An(i - 1, i).real(), u, 1e-6);
    ASSERT_NEAR(An(i, i).real(), d, 1e-6); 
  }
  ASSERT_NEAR(An(p - 1, 0).real(), u, 1e-6);
}

TEST_CASE(set_Bn)
{
  auto constexpr w = castIfNot<Float>(1.26);
  Int constexpr p = 4;
  auto constexpr Sigma_t = castIfNot<Float>(2.65038);
  auto constexpr Sigma_s = castIfNot<Float>(2.4807);
  auto constexpr c = Sigma_s / Sigma_t;
  Int constexpr s = 1;
  auto constexpr eta = castIfNot<Float>(0.0);

  auto constexpr mu_n = castIfNot<Float>(-0.9972638618494816);
  auto constexpr alpha = 0.0;

  um2::CMFDCellParams const params(w, p, Sigma_t, c, s, eta);
  using ComplexF = um2::cmfd::ComplexF;
  um2::Matrix<ComplexF> An(p, p);
  An.zero();
  um2::cmfd::set_An(An, params, mu_n, alpha);

  um2::Matrix<ComplexF> Bn(p, p);
  Bn.zero();
  um2::cmfd::set_Bn(Bn, An, params, mu_n, alpha);

  auto constexpr d = castIfNot<Float>(1.1945143797283355 + 0.526278355167952);
  auto constexpr u = castIfNot<Float>(-1.1945143797283355 + 0.47372164483204804);
  ASSERT_NEAR(Bn(0, 0).real(), d, 1e-6);
  for (Int i = 1; i < p; ++i) {
    ASSERT_NEAR(Bn(i - 1, i).real(), u, 1e-6);
    ASSERT_NEAR(Bn(i, i).real(), d, 1e-6); 
  }
  ASSERT_NEAR(Bn(p - 1, 0).real(), u, 1e-6);
}

TEST_SUITE(cmfd)
{ 
  TEST(beta_n);
  TEST(set_An);
  TEST(set_Bn);
}

auto
main() -> int
{
  RUN_SUITE(cmfd);
  return 0;
}

// NOLINTEND(readability-identifier-naming)
