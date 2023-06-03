#pragma once

#include <um2/common/config.hpp>

#include <glad/gl.h>

#include <spdlog/spdlog.h>

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

  Shader( char const * vertex_shader_source, 
          char const * fragment_shader_source );

  // ---------------------------------------------------------------------------
  // Methods
  // ---------------------------------------------------------------------------

  void use() const { glUseProgram( id ); }
};

} // namespace um2
