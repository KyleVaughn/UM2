#pragma once

#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

#include <chrono>

//==============================================================================
// LOG
//==============================================================================
// A simple logger class for use in host code.
// The logger can be configured to:
//  - log messages of different verbosity levels
//  - prefix messages with a timestamp
//  - colorize messages based on their verbosity level
//  - exit the program after an error is logged (or not)
//
// The logger can be configured at compile time by defining the LOG_LEVEL macro.
// The logger is not thread-safe.

namespace um2
{

enum class LogLevel {
  Off = 0,   // no messages
  Error = 1, // only errors
  Warn = 2,  // errors and warnings
  Info = 3,  // errors, warnings and info
  Debug = 4, // errors, warnings, info and debug
  Trace = 5, // errors, warnings, info, debug and trace
};

using LogClock = std::chrono::system_clock;
using LogTimePoint = std::chrono::time_point<LogClock>;
using LogDuration = std::chrono::duration<double>;

// We need the global log options to be accessible from anywhere in the code
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables) justified
class Log
{

  // -- Options --

  static LogLevel level;
  static LogTimePoint start_time;
  static bool timestamped;   // messages are prefixed with a timestamp
  static bool colorized;     // messages are colorized based on their verbosity level
  static bool exit_on_error; // the program exits after an error is logged
  static constexpr int buffer_size = 256;

  // -- Message --

  static char buffer[buffer_size];

  // -- Methods --

  static auto
  addTimestamp(char * buffer_begin) -> char *;

  static auto
  addColor(LogLevel msg_level, char * buffer_begin) -> char *;

  static auto
  addLevel(LogLevel msg_level, char * buffer_begin) -> char *;

  static void
  handleMessage(LogLevel msg_level, char const * msg, Size len);

public:
  Log() = delete;

  static void
  reset();

  // -- Setters --

  static void
  setLevel(LogLevel val);

  static void
  setTimestamped(bool val);

  static void
  setColorized(bool val);

  static void
  setExitOnError(bool val);

  // -- Getters --

  PURE static auto
  getLevel() -> LogLevel;

  PURE static auto
  getStartTime() -> LogTimePoint;

  PURE static auto
  isTimestamped() -> bool;

  PURE static auto
  isColorized() -> bool;

  PURE static auto
  isExitOnError() -> bool;

  // -- Methods --

  static void
  error(char const * msg);

  static void
  warn(char const * msg);

  static void
  info(char const * msg);

  static void
  debug(char const * msg);

  static void
  trace(char const * msg);

  static void
  error(String const & msg);

  static void
  warn(String const & msg);

  static void
  info(String const & msg);

  static void
  debug(String const & msg);

  static void
  trace(String const & msg);

}; // class Log
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

#if LOG_LEVEL > 0
#  define LOG_ERROR(msg) um2::Log::error(msg)
#else
#  define LOG_ERROR(msg)
#endif

#if LOG_LEVEL > 1
#  define LOG_WARN(msg) um2::Log::warn(msg)
#else
#  define LOG_WARN(msg)
#endif

#if LOG_LEVEL > 2
#  define LOG_INFO(msg) um2::Log::info(msg)
#else
#  define LOG_INFO(msg)
#endif

#if LOG_LEVEL > 3
#  define LOG_DEBUG(msg) um2::Log::debug(msg)
#else
#  define LOG_DEBUG(msg)
#endif

#if LOG_LEVEL > 4
#  define LOG_TRACE(msg) um2::Log::trace(msg)
#else
#  define LOG_TRACE(msg)
#endif

} // namespace um2
