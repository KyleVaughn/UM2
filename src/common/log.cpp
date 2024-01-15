#include <um2/common/log.hpp>

namespace um2
{

//==============================================================================
// Default values
//==============================================================================

static constexpr LogLevel log_default_level = LogLevel::Info;
static constexpr bool log_default_timestamped = true;
static constexpr bool log_default_colorized = true;
static constexpr bool log_default_exit_on_error = true;

//==============================================================================
// Initialize static members
//==============================================================================

// We need log variables to be global
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables) justified
LogLevel Log::level = log_default_level;
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
Log::reset() noexcept
{
  // Reset options to default
  level = log_default_level;
  timestamped = log_default_timestamped;
  colorized = log_default_colorized;
  exit_on_error = log_default_exit_on_error;

  // Reset data
  start_time = LogClock::now();
}

// -- Setters --

void
Log::setLevel(LogLevel val) noexcept
{
  level = val;
}

void
Log::setTimestamped(bool val) noexcept
{
  timestamped = val;
}

void
Log::setColorized(bool val) noexcept
{
  colorized = val;
}

void
Log::setExitOnError(bool val) noexcept
{
  exit_on_error = val;
}

// -- Getters --

PURE auto
Log::getLevel() noexcept -> LogLevel
{
  return level;
}

PURE auto
Log::isTimestamped() noexcept -> bool
{
  return timestamped;
}

PURE auto
Log::isColorized() noexcept -> bool
{
  return colorized;
}

PURE auto
Log::isExitOnError() noexcept -> bool
{
  return exit_on_error;
}

PURE auto
Log::getStartTime() noexcept -> LogTimePoint
{
  return start_time;
}

// -- Message handling --

auto
Log::addTimestamp(char * buffer_begin) noexcept -> char *
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

auto
Log::addColor(LogLevel const msg_level, char * buffer_begin) noexcept -> char *
{
  if (colorized) {
    buffer_begin[0] = '\033';
    buffer_begin[1] = '[';
    buffer_begin[2] = '1';
    buffer_begin[3] = ';';
    buffer_begin[4] = '3';
    buffer_begin[6] = 'm';
    switch (msg_level) {
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
  return buffer_begin;
}

auto
Log::addLevel(LogLevel msg_level, char * buffer_begin) noexcept -> char *
{
  switch (msg_level) {
  case LogLevel::Error:
    buffer_begin[0] = 'E';
    buffer_begin[1] = 'R';
    buffer_begin[2] = 'R';
    buffer_begin[3] = 'O';
    buffer_begin[4] = 'R';
    buffer_begin += 5;
    break;
  case LogLevel::Warn:
    buffer_begin[0] = 'W';
    buffer_begin[1] = 'A';
    buffer_begin[2] = 'R';
    buffer_begin[3] = 'N';
    buffer_begin += 4;
    break;
  case LogLevel::Info:
    buffer_begin[0] = 'I';
    buffer_begin[1] = 'N';
    buffer_begin[2] = 'F';
    buffer_begin[3] = 'O';
    buffer_begin += 4;
    break;
  case LogLevel::Debug:
    buffer_begin[0] = 'D';
    buffer_begin[1] = 'E';
    buffer_begin[2] = 'B';
    buffer_begin[3] = 'U';
    buffer_begin[4] = 'G';
    buffer_begin += 5;
    break;
  case LogLevel::Trace:
    buffer_begin[0] = 'T';
    buffer_begin[1] = 'R';
    buffer_begin[2] = 'A';
    buffer_begin[3] = 'C';
    buffer_begin[4] = 'E';
    buffer_begin += 5;
    break;
  default:
    buffer_begin[0] = 'U';
    buffer_begin[1] = 'N';
    buffer_begin[2] = 'K';
    buffer_begin[3] = 'N';
    buffer_begin[4] = 'O';
    buffer_begin[5] = 'W';
    buffer_begin[6] = 'N';
    buffer_begin += 7;
    break;
  }
  buffer_begin[0] = ' ';
  buffer_begin[1] = '-';
  buffer_begin[2] = ' ';
  return buffer_begin + 3;
}

// Handle the messages from error, warn, etc. and store or print them
// depending on the Log configuration.
void
Log::handleMessage(LogLevel const msg_level, char const * msg, Size len) noexcept
{
  ASSERT(msg != nullptr);
  ASSERT(len > 0);
  ASSERT(len < buffer_size);
  if (msg_level <= level) {
    char * buffer_begin = addressof(buffer[0]);
    buffer_begin = addColor(msg_level, buffer_begin);
    buffer_begin = addTimestamp(buffer_begin);
    buffer_begin = addLevel(msg_level, buffer_begin);

    // Copy the message to the buffer
    char const * msg_begin = addressof(msg[0]);
    char const * msg_end = msg_begin + len;
    um2::copy(msg_begin, msg_end, buffer_begin);
    buffer_begin += len;

    // Reset color
    if (colorized) {
      buffer_begin[0] = '\033';
      buffer_begin[1] = '[';
      buffer_begin[2] = '0';
      buffer_begin[3] = 'm';
      buffer_begin += 4;
    }

    // Ensure null-terminated string
    buffer_begin[0] = '\0';

    // Print the message
    int fprintf_result = 0;
    if (msg_level == LogLevel::Error) {
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
  } // msg_level <= level
}

void
Log::error(char const * msg) noexcept
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
Log::warn(char const * msg) noexcept
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Warn, msg, len);
}

void
Log::info(char const * msg) noexcept
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Info, msg, len);
}

void
Log::debug(char const * msg) noexcept
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Debug, msg, len);
}

void
Log::trace(char const * msg) noexcept
{
  Size len = 0;
  while (msg[len] != '\0') {
    ++len;
  }
  handleMessage(LogLevel::Trace, msg, len);
}

void
Log::error(String const & msg) noexcept
{
  handleMessage(LogLevel::Error, msg.data(), msg.size());
  if (exit_on_error) {
    exit(1);
  }
}

void
Log::warn(String const & msg) noexcept
{
  handleMessage(LogLevel::Warn, msg.data(), msg.size());
}

void
Log::info(String const & msg) noexcept
{
  handleMessage(LogLevel::Info, msg.data(), msg.size());
}

void
Log::debug(String const & msg) noexcept
{
  handleMessage(LogLevel::Debug, msg.data(), msg.size());
}

void
Log::trace(String const & msg) noexcept
{
  handleMessage(LogLevel::Trace, msg.data(), msg.size());
}

} // namespace um2
