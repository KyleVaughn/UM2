#include <um2/common/logger.hpp>
#include <um2/stdlib/string.hpp>

#include "../test_macros.hpp"

TEST_CASE(logger_test)
{
  um2::logger::reset();

  // Check defaults
  ASSERT(um2::logger::level == um2::logger::levels::info);
  ASSERT(um2::logger::timestamped);
  ASSERT(um2::logger::colorized);
  ASSERT(um2::logger::exit_on_error);

  // Test printing a string
  um2::logger::exit_on_error = false;
  um2::logger::level = um2::logger::levels::debug;
  um2::logger::debug("debug");
  um2::logger::info("info");
  um2::logger::warn("warn");
  um2::logger::error("error");

  // Test printing non-string types
  um2::logger::info(1111);
  um2::logger::info(1.0);
  um2::logger::info(true);
  um2::logger::info(false);

  // Test printing multiple arguments
  um2::logger::info("multiple", " arguments");
  um2::logger::info("multiple", " arguments");
  um2::logger::info("1 + 1 = ", 1 + 1);

  // Test um2::String
  um2::String const s = "um2::String";
  um2::logger::info(s);
  um2::logger::info(s, " with multiple", " arguments");
}

TEST_SUITE(logger) { TEST(logger_test); }

auto
main() -> int
{
  RUN_SUITE(logger);
  return 0;
}
