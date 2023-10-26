#include <um2/common/Log.hpp>

#include <iomanip>  // std::setw, std::setfill
#include <iostream> // std::cout, std::endl, std::cerr
#include <sstream>  // Stringstream

namespace um2
{

//==============================================================================
// Default values
//==============================================================================

static constexpr LogVerbosity log_default_max_verbosity_level = LogVerbosity::Info;
static constexpr bool log_default_buffered = false;
static constexpr bool log_default_timestamped = true;
static constexpr bool log_default_colorized = true;
static constexpr bool log_default_exit_on_error = true;
static constexpr Size log_default_flush_threshold = 20; // flush after 20 messages

//==============================================================================
// Initialize static members
//==============================================================================

// We need log variables to be global
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables) justified
LogVerbosity Log::max_verbosity_level = log_default_max_verbosity_level;
bool Log::buffered = log_default_buffered;
bool Log::timestamped = log_default_timestamped;
bool Log::colorized = log_default_colorized;
bool Log::exit_on_error = log_default_exit_on_error;
Size Log::flush_threshold = log_default_flush_threshold;

LogTimePoint Log::start_time = LogClock::now();
Size Log::num_errors = 0;
Size Log::num_warnings = 0;
Vector<LogVerbosity> Log::verbosity_levels;
Vector<LogTimePoint> Log::times;
Vector<String> Log::messages;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// Member functions
//==============================================================================

void
Log::reset()
{
  // Reset options to default
  max_verbosity_level = log_default_max_verbosity_level;
  buffered = log_default_buffered;
  timestamped = log_default_timestamped;
  colorized = log_default_colorized;
  exit_on_error = log_default_exit_on_error;
  flush_threshold = log_default_flush_threshold;

  // Reset data
  start_time = LogClock::now();
  num_errors = 0;
  num_warnings = 0;
  verbosity_levels.clear();
  times.clear();
  messages.clear();
}

// -- Setters --

void
Log::setMaxVerbosityLevel(LogVerbosity val)
{
  max_verbosity_level = val;
}
void
Log::setBuffered(bool val)
{
  buffered = val;
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
void
Log::setFlushThreshold(Size val)
{
  flush_threshold = val;
}

// -- Getters --

PURE auto
Log::getMaxVerbosityLevel() -> LogVerbosity
{
  return max_verbosity_level;
}

PURE auto
Log::isBuffered() -> bool
{
  return buffered;
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
Log::getFlushThreshold() -> Size
{
  return flush_threshold;
}
PURE auto
Log::getStartTime() -> LogTimePoint
{
  return start_time;
}
PURE auto
Log::getNumErrors() -> Size
{
  return num_errors;
}
PURE auto
Log::getNumWarnings() -> Size
{
  return num_warnings;
}

// -- Message handling --

namespace
{
auto
logMessageToString(LogVerbosity const verbosity, LogTimePoint const time,
                   String const & message) -> String
{
  // Time
  String time_str;
  if (Log::isTimestamped()) {
    LogTimePoint const start_time = Log::getStartTime();
    LogDuration const elapsed_seconds = time - start_time;
    Size const hours = static_cast<Size>(elapsed_seconds.count()) / 3600;
    Size const minutes = (static_cast<Size>(elapsed_seconds.count()) / 60) % 60;
    Size const seconds = static_cast<Size>(elapsed_seconds.count()) % 60;
    Size const milliseconds = static_cast<Size>(elapsed_seconds.count() * 1000) % 1000;
    std::stringstream ss;
    ss << "[" << std::setw(2) << std::setfill('0') << hours << ":" << std::setw(2)
       << std::setfill('0') << minutes << ":" << std::setw(2) << std::setfill('0')
       << seconds << "." << std::setw(3) << std::setfill('0') << milliseconds << "] ";
    time_str = ss.str();
  }
  // Verbosity
  String verbosity_str;
  switch (verbosity) {
  case LogVerbosity::Error:
    verbosity_str = "ERROR";
    break;
  case LogVerbosity::Warn:
    verbosity_str = "WARN";
    break;
  case LogVerbosity::Info:
    verbosity_str = "INFO";
    break;
  case LogVerbosity::Debug:
    verbosity_str = "DEBUG";
    break;
  case LogVerbosity::Trace:
    verbosity_str = "TRACE";
    break;
  default:
    verbosity_str = "UNKNOWN";
    break;
  }
  // Color
  String c0;
  String c1;
  if (Log::isColorized()) {
    switch (verbosity) {
    case LogVerbosity::Error: // RED
      c0 = "\033[1;31m";
      c1 = "\033[0m";
      break;
    case LogVerbosity::Warn: // YELLOW
      c0 = "\033[1;33m";
      c1 = "\033[0m";
      break;
    case LogVerbosity::Debug: // MAGENTA
      c0 = "\033[1;35m";
      c1 = "\033[0m";
      break;
    case LogVerbosity::Trace: // CYAN
      c0 = "\033[1;36m";
      c1 = "\033[0m";
      break;
    default: // NO COLOR
      break;
    }
  }
  // Message
  std::stringstream ss;
  ss << c0.c_str() << time_str.c_str() << verbosity_str.c_str() << ": " << message.c_str()
     << c1.c_str();
  return String(ss.str().c_str());
}

void
printLogMessage(LogVerbosity const verbosity, LogTimePoint const time,
                String const & message)
{
  String const log_message = logMessageToString(verbosity, time, message);
  if (verbosity == LogVerbosity::Error) {
    std::cerr << log_message.c_str() << std::endl;
  } else {
    std::cout << log_message.c_str() << std::endl;
  }
}
} // namespace

void
Log::flush()
{
  for (Size i = 0; i < messages.size(); ++i) {
    printLogMessage(verbosity_levels[i], times[i], messages[i]);
  }
  verbosity_levels.clear();
  times.clear();
  messages.clear();
}

// Handle the messages from error, warn, etc. and store or print them
// depending on the Log configuration.
void
Log::handleMessage(LogVerbosity const verbosity, String const & msg)
{
  if (verbosity <= max_verbosity_level) {
    if (buffered) {
      verbosity_levels.push_back(verbosity);
      times.push_back(LogClock::now());
      messages.push_back(msg);
      if (messages.size() >= flush_threshold) {
        flush();
      }
    } else {
      printLogMessage(verbosity, LogClock::now(), msg);
    }
  }
}

void
Log::error(String const & msg)
{
  num_errors += 1;
  handleMessage(LogVerbosity::Error, msg);
  if (exit_on_error) {
    exit(1);
  }
}

void
Log::warn(String const & msg)
{
  num_warnings += 1;
  handleMessage(LogVerbosity::Warn, msg);
}

void
Log::info(String const & msg)
{
  handleMessage(LogVerbosity::Info, msg);
}

void
Log::debug(String const & msg)
{
  handleMessage(LogVerbosity::Debug, msg);
}

void
Log::trace(String const & msg)
{
  handleMessage(LogVerbosity::Trace, msg);
}

} // namespace um2
