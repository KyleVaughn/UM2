#include <um2/common/settings.hpp>

// Suppress warnings for non-const global variables, since these are global
// settings that are intended to be modified by the user.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// Log
//==============================================================================

namespace um2::settings::log
{
int32_t level = defaults::level;
bool timestamped = defaults::timestamped;
bool colorized = defaults::colorized;
bool exit_on_error = defaults::exit_on_error;
} // namespace um2::settings::log

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
