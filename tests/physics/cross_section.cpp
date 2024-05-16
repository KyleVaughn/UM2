#include <um2/physics/cross_section.hpp>
#include <um2/common/cast_if_not.hpp>

#include "../test_macros.hpp"

//TEST_CASE(collapse)
//{
//  auto constexpr eps = castIfNot<Float>(1e-6);
//  um2::XSec xsec;
//  xsec.t() = {2, 11, 5, 3, 4};
//  xsec.validate();
//  auto const max_1g = xsec.collapse(um2::XSecReduction::Max);
//  ASSERT_NEAR(max_1g.t()[0], 11, eps);
//  auto const mean_1g = xsec.collapse(); // mean by default
//  ASSERT_NEAR(mean_1g.t()[0], 5, eps);
//}

//TEST_SUITE(XSec) { TEST(collapse); }

auto
main() -> int
{
 // RUN_SUITE(XSec);
  return 0;
}
