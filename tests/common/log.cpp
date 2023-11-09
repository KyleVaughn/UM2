#include <um2/common/log.hpp>

#include "../test_macros.hpp"

TEST_CASE(set_get)
{
  um2::Log::reset();
  // Check defaults and getters
  ASSERT(um2::Log::getMaxVerbosityLevel() == um2::LogVerbosity::Info);
  ASSERT(!um2::Log::isBuffered());
  ASSERT(um2::Log::isTimestamped());
  ASSERT(um2::Log::isColorized());
  ASSERT(um2::Log::isExitOnError());
  ASSERT(um2::Log::getFlushThreshold() == 20);
  ASSERT(um2::Log::getNumErrors() == 0);
  // Check setters
  um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Debug);
  ASSERT(um2::Log::getMaxVerbosityLevel() == um2::LogVerbosity::Debug);
  um2::Log::setBuffered(/*val=*/true);
  um2::Log::setTimestamped(/*val=*/false);
  um2::Log::setColorized(/*val=*/false);
  um2::Log::setExitOnError(/*val=*/false);
  um2::Log::setFlushThreshold(21);
  ASSERT(um2::Log::isBuffered());
  ASSERT(!um2::Log::isTimestamped());
  ASSERT(!um2::Log::isColorized());
  ASSERT(!um2::Log::isExitOnError());
  ASSERT(um2::Log::getFlushThreshold() == 21);
}

TEST_SUITE(Log) { TEST(set_get); }

auto
main() -> int
{
  RUN_SUITE(Log);
  return 0;
}
