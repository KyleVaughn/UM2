#include <um2/common/log.hpp>

#include <um2/stdlib/string.hpp>

namespace um2::log
{

//==============================================================================
// Initialize global variables
//==============================================================================

// Suppress warnings for non-const global variables, since this is a global logger
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

int32_t & level = um2::settings::log::level;
bool & timestamped = um2::settings::log::timestamped;
bool & colorized = um2::settings::log::colorized;
bool & exit_on_error = um2::settings::log::exit_on_error;

TimePoint start_time = Clock::now();
char buffer[buffer_size] = {0};
char const * const buffer_end = buffer + buffer_size;

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// Functions
//==============================================================================

void
reset() noexcept
{
  // Reset options to default
  level = um2::settings::log::defaults::level;
  timestamped = um2::settings::log::defaults::timestamped;
  colorized = um2::settings::log::defaults::colorized;
  exit_on_error = um2::settings::log::defaults::exit_on_error;

  // Reset data
  start_time = Clock::now();
}

//==============================================================================
// toBuffer functions
//==============================================================================

// string
template <>
auto
toBuffer(char * buffer_begin, char const * const & value) noexcept -> char *
{
  char const * p = value;
  while (*p != '\0') {
    *buffer_begin = *p;
    ++p;
    ++buffer_begin;
  }
  ASSERT(buffer_begin < buffer_end);
  return buffer_begin;
}

// Use snprintf to convert the value to a string and store it in the buffer
template <>
auto
toBuffer(char * buffer_begin, int32_t const & value) noexcept -> char *
{
  int32_t const len = snprintf(nullptr, 0, "%d", value);
  int32_t const written = snprintf(
      buffer_begin, static_cast<uint64_t>(buffer_end - buffer_begin), "%d", value);
  ASSERT(len == written);
  ASSERT(buffer_begin + len < buffer_end);
  return buffer_begin + len;
}

template <>
auto
toBuffer(char * buffer_begin, uint32_t const & value) noexcept -> char *
{
  int32_t const len = snprintf(nullptr, 0, "%u", value);
  int32_t const written = snprintf(
      buffer_begin, static_cast<uint64_t>(buffer_end - buffer_begin), "%u", value);
  ASSERT(len == written);
  ASSERT(buffer_begin + len < buffer_end);
  return buffer_begin + len;
}

template <>
auto
toBuffer(char * buffer_begin, int64_t const & value) noexcept -> char *
{
  int32_t const len = snprintf(nullptr, 0, "%ld", value);
  int32_t const written = snprintf(
      buffer_begin, static_cast<uint64_t>(buffer_end - buffer_begin), "%ld", value);
  ASSERT(len == written);
  ASSERT(buffer_begin + len < buffer_end);
  return buffer_begin + len;
}

template <>
auto
toBuffer(char * buffer_begin, uint64_t const & value) noexcept -> char *
{
  int32_t const len = snprintf(nullptr, 0, "%lu", value);
  int32_t const written = snprintf(
      buffer_begin, static_cast<uint64_t>(buffer_end - buffer_begin), "%lu", value);
  ASSERT(len == written);
  ASSERT(buffer_begin + len < buffer_end);
  return buffer_begin + len;
}

template <>
auto
toBuffer(char * buffer_begin, double const & value) noexcept -> char *
{
  int32_t const len = snprintf(nullptr, 0, "%f", value);
  int32_t const written = snprintf(
      buffer_begin, static_cast<uint64_t>(buffer_end - buffer_begin), "%f", value);
  ASSERT(len == written);
  ASSERT(buffer_begin + len < buffer_end);
  return buffer_begin + len;
}

template <>
auto
toBuffer(char * buffer_begin, float const & value) noexcept -> char *
{
  // snprintf does not support float, so we cast to double
  auto const dvalue = static_cast<double>(value);
  return toBuffer(buffer_begin, dvalue);
}

template <>
auto
toBuffer(char * buffer_begin, bool const & value) noexcept -> char *
{
  if (value) {
    buffer_begin[0] = 't';
    buffer_begin[1] = 'r';
    buffer_begin[2] = 'u';
    buffer_begin[3] = 'e';
    buffer_begin += 4;
  } else {
    buffer_begin[0] = 'f';
    buffer_begin[1] = 'a';
    buffer_begin[2] = 'l';
    buffer_begin[3] = 's';
    buffer_begin[4] = 'e';
    buffer_begin += 5;
  }
  ASSERT(buffer_begin < buffer_end);
  return buffer_begin;
}

template <>
auto
toBuffer(char * buffer_begin, String const & value) noexcept -> char *
{
  char const * p = value.c_str();
  while (*p != '\0') {
    *buffer_begin = *p;
    ++p;
    ++buffer_begin;
  }
  ASSERT(buffer_begin < buffer_end);
  return buffer_begin;
}

//==============================================================================

// Add the timestamp to the buffer if the log is timestamped
auto
addTimestamp(char * buffer_begin) noexcept -> char *
{
  if (um2::settings::log::timestamped) {
    Duration const elapsed_seconds = Clock::now() - start_time;
    I const hours = static_cast<I>(elapsed_seconds.count()) / 3600;
    I const minutes = (static_cast<I>(elapsed_seconds.count()) / 60) % 60;
    I const seconds = static_cast<I>(elapsed_seconds.count()) % 60;
    I const milliseconds = static_cast<I>(elapsed_seconds.count() * 1000) % 1000;
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
} // addTimestamp

// Add the color to the buffer if the log is colorized
auto
addColor(int32_t const msg_level, char * buffer_begin) noexcept -> char *
{
  if (colorized) {
    buffer_begin[0] = '\033';
    buffer_begin[1] = '[';
    buffer_begin[2] = '1';
    buffer_begin[3] = ';';
    buffer_begin[4] = '3';
    buffer_begin[6] = 'm';
    switch (msg_level) {
    case levels::error: // RED
      // \033[1;31m
      buffer_begin[5] = '1';
      break;
    case levels::warn: // YELLOW
      // \033[1;33m
      buffer_begin[5] = '3';
      break;
    case levels::debug: // MAGENTA
      // \033[1;35m
      buffer_begin[5] = '5';
      break;
    case levels::trace: // CYAN
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
} // addColor

// Add the level to the buffer
auto
addLevel(int32_t const msg_level, char * buffer_begin) noexcept -> char *
{
  switch (msg_level) {
  case levels::error:
    buffer_begin[0] = 'E';
    buffer_begin[1] = 'R';
    buffer_begin[2] = 'R';
    buffer_begin[3] = 'O';
    buffer_begin[4] = 'R';
    buffer_begin += 5;
    break;
  case levels::warn:
    buffer_begin[0] = 'W';
    buffer_begin[1] = 'A';
    buffer_begin[2] = 'R';
    buffer_begin[3] = 'N';
    buffer_begin += 4;
    break;
  case levels::info:
    buffer_begin[0] = 'I';
    buffer_begin[1] = 'N';
    buffer_begin[2] = 'F';
    buffer_begin[3] = 'O';
    buffer_begin += 4;
    break;
  case levels::debug:
    buffer_begin[0] = 'D';
    buffer_begin[1] = 'E';
    buffer_begin[2] = 'B';
    buffer_begin[3] = 'U';
    buffer_begin[4] = 'G';
    buffer_begin += 5;
    break;
  case levels::trace:
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
} // addLevel

// Set the preamble of the message
auto
setPreamble(int32_t const msg_level) noexcept -> char *
{
  char * buffer_begin = addressof(buffer[0]);
  buffer_begin = addColor(msg_level, buffer_begin);
  buffer_begin = addTimestamp(buffer_begin);
  buffer_begin = addLevel(msg_level, buffer_begin);
  return buffer_begin;
}

// Set the postamble of the message
void
setPostamble(char * buffer_begin) noexcept
{
  // Reset color
  if (colorized) {
    buffer_begin[0] = '\033';
    buffer_begin[1] = '[';
    buffer_begin[2] = '0';
    buffer_begin[3] = 'm';
    buffer_begin += 4;
  }
  ASSERT(buffer_begin < buffer_end);

  // Ensure null-terminated string
  buffer_begin[0] = '\0';
}

} // namespace um2::log
