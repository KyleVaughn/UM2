#include <um2/common/log.hpp>

#include "../test_macros.hpp"

TEST_CASE(set_get)
{
  um2::Log::reset();
  // Check defaults and getters
  ASSERT(um2::Log::getMaxLevel() == um2::LogLevel::Info);
  ASSERT(um2::Log::isTimestamped());
  ASSERT(um2::Log::isColorized());
  ASSERT(um2::Log::isExitOnError());
  // Check setters
  um2::Log::setMaxLevel(um2::LogLevel::Debug);
  ASSERT(um2::Log::getMaxLevel() == um2::LogLevel::Debug);
  um2::Log::setTimestamped(/*val=*/false);
  um2::Log::setColorized(/*val=*/false);
  um2::Log::setExitOnError(/*val=*/false);
  ASSERT(!um2::Log::isTimestamped());
  ASSERT(!um2::Log::isColorized());
  ASSERT(!um2::Log::isExitOnError());

  um2::Log::reset();
  um2::Log::setExitOnError(/*val=*/false);
  um2::Log::trace("trace");
  um2::Log::debug("debug");
  um2::Log::info("info");
  um2::Log::warn("warn");
  um2::Log::error("error");
  um2::String msg = "trace";
  um2::Log::trace(msg);
  msg = "debug";
  um2::Log::debug(msg);
  msg = "info";
  um2::Log::info(msg);
  msg = "warn";
  um2::Log::warn(msg);
  msg = "error";
  um2::Log::error(msg);
}

TEST_SUITE(Log) { TEST(set_get); }

auto
main() -> int
{
  RUN_SUITE(Log);
  return 0;
}
