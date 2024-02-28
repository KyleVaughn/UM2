#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/string.hpp>

//==============================================================================
// SETTINGS
//==============================================================================
// A collection of settings that can be used to configure the behavior of the
// library.

// Suppress warnings for non-const global variables, since these are global
// settings that are intended to be modified by the user.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

//==============================================================================
// LOG
//==============================================================================

namespace um2::settings::log
{

namespace defaults
{
inline constexpr int32_t level = 3; // 3 == info
inline constexpr bool timestamped = true;
inline constexpr bool colorized = true;
inline constexpr bool exit_on_error = true;
} // namespace defaults

// Global settings
extern int32_t level;
extern bool timestamped;
extern bool colorized;
extern bool exit_on_error;

} // namespace um2::settings::log

//==============================================================================
// CROSS SECTION LIBRARY (XS)
//==============================================================================

namespace um2::settings::xs
{

namespace defaults
{
String const LIBRARY_PATH = MPACT_DATA_DIR;
String const LIBRARY_NAME = "mpact51g_71_v4.2m5_12062016_sph.fmt";
} // namespace defaults

// Global settings
extern String library_path;
extern String library_name;

} // namespace um2::settings::xs

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
