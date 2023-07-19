#pragma once

#include <um2/common/config.hpp>
#if UM2_ENABLE_VIS

#  include <glad/gl.h>

namespace um2
{

// -----------------------------------------------------------------------------
// VERTEX BUFFER
// -----------------------------------------------------------------------------

struct VertexBuffer {
  uint32_t id;

  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  VertexBuffer(float const * data, int32_t n)
  {
    glGenBuffers(1, &id);
    glBindBuffer(GL_ARRAY_BUFFER, id);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(n) * 4, data, GL_STATIC_DRAW);
  }

  // ---------------------------------------------------------------------------
  // Methods
  // ---------------------------------------------------------------------------

  inline void
  bind() const
  {
    glBindBuffer(GL_ARRAY_BUFFER, id);
  }
  static inline void
  unbind()
  {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  inline void
  destroy() const
  {
    glDeleteBuffers(1, &id);
  }
};

} // namespace um2
#endif // UM2_ENABLE_VIS
