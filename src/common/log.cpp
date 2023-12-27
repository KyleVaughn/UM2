#include <um2/common/log.hpp>

#include <iomanip>  // std::setw, std::setfill
#include <iostream> // std::cout, std::endl, std::cerr
#include <sstream>  // stringstream

namespace um2
{

//==============================================================================
// Default values
//==============================================================================

static constexpr LogLevel log_default_max_level = LogLevel::Info;
static constexpr bool log_default_timestamped = true;
static constexpr bool log_default_colorized = true;
static constexpr bool log_default_exit_on_error = true;

//==============================================================================
// Initialize static members
//==============================================================================

// We need log variables to be global
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables) justified
LogLevel Log::max_level = log_default_max_level;
LogTimePoint Log::start_time = LogClock::now();
bool Log::timestamped = true;
bool Log::colorized = log_default_colorized;
bool Log::exit_on_error = log_default_exit_on_error;
char Log::buffer[Log::buffer_size] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// Member functions
//==============================================================================

void
Log::reset()
{
  // Reset options to default
  max_level = log_default_max_level;
  timestamped = log_default_timestamped;
  colorized = log_default_colorized;
  exit_on_error = log_default_exit_on_error;

  // Reset data
  start_time = LogClock::now();
}

// -- Setters --

void
Log::setMaxLevel(LogLevel val)
{
  max_level = val;
}

void
Log::setTimestamped(bool val)
{
  timestamped = val;
}

void
Log::setColorized(bool val)
{
  colorized = val;
}

void
Log::setExitOnError(bool val)
{
  exit_on_error = val;
}

// -- Getters --

PURE auto
Log::getMaxLevel() -> LogLevel
{
  return max_level;
}

PURE auto
Log::isTimestamped() -> bool
{
  return timestamped;
}

PURE auto
Log::isColorized() -> bool
{
  return colorized;
}

PURE auto
Log::isExitOnError() -> bool
{
  return exit_on_error;
}

PURE auto
Log::getStartTime() -> LogTimePoint
{
  return start_time;
}

// -- Message handling --

//  // Verbosity
//  String verbosity_str;
//  switch (verbosity) {
//  case LogLevel::Error:
//    verbosity_str = "ERROR";
//    break;
//  case LogLevel::Warn:
//    verbosity_str = "WARN";
//    break;
//  case LogLevel::Info:
//    verbosity_str = "INFO";
//    break;
//  case LogLevel::Debug:
//    verbosity_str = "DEBUG";
//    break;
//  case LogLevel::Trace:
//    verbosity_str = "TRACE";
//    break;
//  default:
//    verbosity_str = "UNKNOWN";
//    break;
//  }
//  return c0 + time_str + verbosity_str + ": " + message + c1;
//}
//

auto
Log::addTimestamp(char * buffer_begin) -> char *
{
  if (timestamped) {
    LogDuration const elapsed_seconds = LogClock::now() - start_time;
    Size const hours = static_cast<Size>(elapsed_seconds.count()) / 3600;
    Size const minutes = (static_cast<Size>(elapsed_seconds.count()) / 60) % 60;
    Size const seconds = static_cast<Size>(elapsed_seconds.count()) % 60;
    Size const milliseconds = static_cast<Size>(elapsed_seconds.count() * 1000) % 1000;
    buffer_begin[0] = '[';
    if (hours < 10) {
      buffer_begin[1] = '0';
      buffer_begin[2] = static_cast<char>(hours + '0');
    } else {
      buffer_begin[1] = static_cast<char>(hours / 10 + '0');
      buffer_begin[2] = static_cast<char>(hours % 10 + '0');
    }
    buffer_begin[3] = ':';
    if (minutes < 10) {
      buffer_begin[4] = '0';
      buffer_begin[5] = static_cast<char>(minutes + '0');
    } else {
      buffer_begin[4] = static_cast<char>(minutes / 10 + '0');
      buffer_begin[5] = static_cast<char>(minutes % 10 + '0');
    }
    buffer_begin[6] = ':';
    if (seconds < 10) {
      buffer_begin[7] = '0';
      buffer_begin[8] = static_cast<char>(seconds + '0');
    } else {
      buffer_begin[7] = static_cast<char>(seconds / 10 + '0');
      buffer_begin[8] = static_cast<char>(seconds % 10 + '0');
    }
    buffer_begin[9] = '.';
    if (milliseconds < 10) {
      buffer_begin[10] = '0';
      buffer_begin[11] = '0';
      buffer_begin[12] = static_cast<char>(milliseconds + '0');
    } else if (milliseconds < 100) {
      buffer_begin[10] = '0';
      buffer_begin[11] = static_cast<char>(milliseconds / 10 + '0');
      buffer_begin[12] = static_cast<char>(milliseconds % 10 + '0');
    } else {
      buffer_begin[10] = static_cast<char>(milliseconds / 100 + '0');
      buffer_begin[11] = static_cast<char>((milliseconds % 100) / 10 + '0');
      buffer_begin[12] = static_cast<char>((milliseconds % 100) % 10 + '0');
    }
    buffer_begin[13] = ']';
    buffer_begin[14] = ' ';
    buffer_begin += 15;
  } // timestamped
  return buffer_begin;
}

// Handle the messages from error, warn, etc. and store or print them
// depending on the Log configuration.
void
Log::handleMessage(LogLevel const level, char const * msg, Size len)
{
  ASSERT(msg != nullptr);
  ASSERT(len > 0);
  ASSERT(len < buffer_size);
  if (level <= max_level) {
    char * buffer_begin = addressof(buffer[0]);
    if (colorized) {
        buffer_begin[0] = '\033';
        buffer_begin[1] = '[';
        buffer_begin[2] = '1';
        buffer_begin[3] = ';';
        buffer_begin[4] = '3';
        buffer_begin[6] = 'm';
      switch (level) {
      case LogLevel::Error: // RED
        // \033[1;31m
        buffer_begin[5] = '1';
        break;
      case LogLevel::Warn: // YELLOW
        // \033[1;33m
        buffer_begin[5] = '3';
        break;
      case LogLevel::Debug: // MAGENTA
        // \033[1;35m
        buffer_begin[5] = '5';
        break;
      case LogLevel::Trace: // CYAN
        // \033[1;36m
        buffer_begin[5] = '6';
        break;
      default: // NO COLOR
        buffer_begin -= 7;
        break;
      }
      buffer_begin += 7;
    } // if (colorized)
    char const * begin = addressof(msg[0]);
    char const * end = begin + len;
    buffer_begin = addTimestamp(buffer_begin);
    um2::copy(begin, end, buffer_begin);
    buffer_begin += len;
    if (colorized) {
      buffer_begin[0] = '\033';
      buffer_begin[1] = '[';
      buffer_begin[2] = '0';
      buffer_begin[3] = 'm';
      buffer_begin += 4;
    }
    buffer_begin[0] = '\0';
    int fprintf_result = 0;
    if (level == LogLevel::Error) {
      fprintf_result = fprintf(stderr, "%s\n", buffer);
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
  } // level <= max_level
}

void
Log::error(char const * msg)
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Error, msg, len);
  if (exit_on_error) {
    exit(1);
  }
}

void
Log::warn(char const * msg)
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Warn, msg, len);
}

void
Log::info(char const * msg)
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Info, msg, len);
}

void
Log::debug(char const * msg)
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Debug, msg, len);
}

void
Log::trace(char const * msg)
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Trace, msg, len);
}

void
Log::error(String const & msg)
{
  handleMessage(LogLevel::Error, msg.data(), msg.size());
  if (exit_on_error) {
    exit(1);
  }
}

void
Log::warn(String const & msg)
{
  handleMessage(LogLevel::Warn, msg.data(), msg.size());
}

void
Log::info(String const & msg)
{
  handleMessage(LogLevel::Info, msg.data(), msg.size());
}

void
Log::debug(String const & msg)
{
  handleMessage(LogLevel::Debug, msg.data(), msg.size());
}

void
Log::trace(String const & msg)
{
  handleMessage(LogLevel::Trace, msg.data(), msg.size());
}

} // namespace um2
