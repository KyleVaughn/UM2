#include <um2/common/log.hpp>
#include <um2/stdlib/string.hpp>

#include "../test_macros.hpp"

TEST_CASE(log_test)
{
  um2::log::reset();

  // Check defaults
  ASSERT(um2::log::level == um2::log::levels::info);
  ASSERT(um2::log::timestamped);
  ASSERT(um2::log::colorized);
  ASSERT(um2::log::exit_on_error);

  // Test printing a string
  um2::log::exit_on_error = false;
  um2::log::level = um2::log::levels::debug;
  um2::log::debug("debug");
  um2::log::info("info");
  um2::log::warn("warn");
  um2::log::error("error");

  // Test printing non-string types
  um2::log::info(1111);
  um2::log::info(1.0);
  um2::log::info(true);
  um2::log::info(false);

  // Test printing multiple arguments
  um2::log::info("multiple", " arguments");
  um2::log::info("multiple", " arguments");
  um2::log::info("1 + 1 = ", 1 + 1);

  // Test um2::String
  um2::String const s = "um2::String";
  um2::log::info(s);
  um2::log::info(s, " with multiple", " arguments");
}

TEST_SUITE(log) { TEST(log_test); }

auto
main() -> int
{
  RUN_SUITE(log);
  return 0;
}
