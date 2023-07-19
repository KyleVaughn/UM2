#pragma once

#include <um2/common/config.hpp>
#if UM2_ENABLE_VIS

#  include <glad/gl.h>

#  include <spdlog/spdlog.h>

namespace um2
{

// -----------------------------------------------------------------------------
// SHADER
// -----------------------------------------------------------------------------
// A struct for managing shader programs.

struct Shader {

  uint32_t id; // The OpenGL shader program ID

  // ---------------------------------------------------------------------------
  // Constructor
  // ---------------------------------------------------------------------------

  Shader(char const * vertex_shader_source, char const * fragment_shader_source);

  // ---------------------------------------------------------------------------
  // Methods
  // ---------------------------------------------------------------------------

  inline void
  use() const
  {
    glUseProgram(id);
  }
  inline void
  destroy() const
  {
    glDeleteProgram(id);
  }
};

} // namespace um2
#endif // UM2_ENABLE_VIS
