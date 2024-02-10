#pragma once

#include <um2/common/settings.hpp>

#include <chrono>

//==============================================================================
// LOG
//==============================================================================
// A simple logger for use in host code.
// The logger can be configured to:
//  - log messages of different verbosity levels
//  - prefix messages with a timestamp
//  - colorize messages based on their verbosity level
//  - exit the program after an error is logged (or not)
//
// The logger can be configured at compile time by defining the MIN_LOG_LEVEL macro.
// The logger is not thread-safe.
//
// For developers:
// The logger is just a fixed-size buffer that is filled with the message and then 
// printed. The message arguments are converted using the toBuffer function, which
// is a template that can be specialized for different types.

namespace um2::log
{

using Clock = std::chrono::system_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

namespace levels
{
inline constexpr int32_t off = 0;   // no messages
inline constexpr int32_t error = 1; // only errors
inline constexpr int32_t warn = 2;  // errors and warnings
inline constexpr int32_t info = 3;  // errors, warnings and info
inline constexpr int32_t debug = 4; // errors, warnings, info and debug
} // namespace levels

//==============================================================================
// Global variables
//==============================================================================

// Suppress warnings for non-const global variables, since this is a global logger
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

extern int32_t & level;
extern bool & timestamped;
extern bool & colorized;
extern bool & exit_on_error;
extern TimePoint start_time;

inline constexpr int32_t buffer_size = 256;
extern char buffer[buffer_size];
extern char const * const buffer_end; // 1 past the last valid character in the buffer

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// Functions
//==============================================================================

// Reset the logger to its default state
void
reset() noexcept;

// Write types to the buffer
template <class T>
auto
toBuffer(char * buffer_begin, T const & value) noexcept -> char *;

// Handle fixed-size character arrays by treating them as pointer
template <uint64_t N>
auto
toBuffer(char * buffer_begin, char const (&value)[N]) noexcept -> char *
{
  char const * const p = value;
  return toBuffer(buffer_begin, p);
}

// Add the timestamp to the buffer if the log is timestamped
auto
addTimestamp(char * buffer_begin) noexcept -> char *;

// Add color to the buffer if the log is colorized
auto
addColor(int32_t msg_level, char * buffer_begin) noexcept -> char *;

// Add the log level to the buffer
auto
addLevel(int32_t msg_level, char * buffer_begin) noexcept -> char *;

// Set the preamble of the message
auto
setPreamble(int32_t msg_level) noexcept -> char *;

// Set the postamble of the message
void
setPostamble(char * buffer_begin) noexcept;

// Print the message
template <class... Args>
void
printMessage(int32_t const msg_level, Args const &... args) noexcept
{
  if (msg_level <= level) {
    char * buffer_begin = setPreamble(msg_level);

    // Use fold expression to send each argument to the buffer.
    // We need a lambda function to capture the buffer_begin variable, since it is
    // not a template parameter.
    ([&buffer_begin](auto const & arg) { buffer_begin = toBuffer(buffer_begin, arg); }(
         args),
     ...);

    setPostamble(buffer_begin);

    // Print the message
    int fprintf_result = 0;
    if (msg_level == levels::error) {
      fprintf_result = fprintf(stderr, "%s\n", buffer);
      if (exit_on_error) {
        exit(1);
      }
    } else {
      fprintf_result = fprintf(stdout, "%s\n", buffer);
    }
#if UM2_ENABLE_ASSERTS
    ASSERT(fprintf_result > 0);
#else
    if (fprintf_result == 0) {
      exit(1);
    }
#endif
  } // msg_level <= level
} // printMessage

template <class... Args>
void
error(Args const &... args) noexcept
{
  printMessage(levels::error, args...);
}

template <class... Args>
void
warn(Args const &... args) noexcept
{
  printMessage(levels::warn, args...);
}

template <class... Args>
void
info(Args const &... args) noexcept
{
  printMessage(levels::info, args...);
}

template <class... Args>
void
debug(Args const &... args) noexcept
{
  printMessage(levels::debug, args...);
}

} // namespace um2::log

#if MIN_LOG_LEVEL > 0
#  define LOG_ERROR(...) um2::log::error(__VA_ARGS__)
#else
#  define LOG_ERROR(...)
#endif

#if MIN_LOG_LEVEL > 1
#  define LOG_WARN(...) um2::log::warn(__VA_ARGS__)
#else
#  define LOG_WARN(...)
#endif

#if MIN_LOG_LEVEL > 2
#  define LOG_INFO(...) um2::log::info(__VA_ARGS__)
#else
#  define LOG_INFO(...)
#endif

#if MIN_LOG_LEVEL > 3
#  define LOG_DEBUG(...) um2::log::debug(__VA_ARGS__)
#else
#  define LOG_DEBUG(...)
#endif
