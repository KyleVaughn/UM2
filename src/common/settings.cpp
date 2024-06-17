#include <um2/common/settings.hpp>

// #include <um2/stdlib/string.hpp>

#include <cstdint>

// Suppress warnings for non-const global variables, since these are global
// settings that are intended to be modified by the user.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// LOG
//==============================================================================

namespace um2::settings::logger
{
int32_t level = defaults::level;
bool timestamped = defaults::timestamped;
bool colorized = defaults::colorized;
bool exit_on_error = defaults::exit_on_error;
} // namespace um2::settings::logger

//==============================================================================
// CROSS SECTION LIBRARY (XS)
//==============================================================================

// namespace um2::settings::xs
//{
// String library_path = defaults::LIBRARY_PATH;
// String library_name = defaults::LIBRARY_NAME;
// } // namespace um2::settings::xs

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
