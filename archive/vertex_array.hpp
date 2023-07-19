#pragma once

#include <um2/common/config.hpp>
#if UM2_ENABLE_VIS

#  include <glad/gl.h>

namespace um2
{

// -----------------------------------------------------------------------------
// VERTEX ARRAY
// -----------------------------------------------------------------------------

struct VertexArray {
  uint32_t id;

  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  VertexArray() { glGenVertexArrays(1, &id); }

  // ---------------------------------------------------------------------------
  // Methods
  // ---------------------------------------------------------------------------

  inline void
  bind() const
  {
    glBindVertexArray(id);
  }
  static inline void
  unbind()
  {
    glBindVertexArray(0);
  }
  inline void
  destroy() const
  {
    glDeleteVertexArrays(1, &id);
  }
  static inline void
  set_vertex_dimension(int32_t n)
  {
    glVertexAttribPointer(0, n, GL_FLOAT, GL_FALSE, 4 * n, nullptr);
    glEnableVertexAttribArray(0);
  }
};

} // namespace um2
#endif // UM2_ENABLE_VIS
