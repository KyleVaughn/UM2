#include <um2/visualization/shader.hpp>

namespace um2
{

static void checkShaderCompilation(uint32_t shader)
{
  int32_t success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (success == 0)
  {
    char info_log[1024];
    glGetShaderInfoLog(shader, 1024, nullptr, info_log);
    spdlog::error("ERROR::SHADER::COMPILATION_FAILED\n{}", info_log);
  }
}

static void checkShaderLinking(uint32_t shader)
{
  int32_t success = 0;
  glGetProgramiv(shader, GL_LINK_STATUS, &success);
  if (success == 0)
  {
    char info_log[1024];
    glGetProgramInfoLog(shader, 1024, nullptr, info_log);
    spdlog::error("ERROR::SHADER::LINKING_FAILED\n{}", info_log);
  }
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

Shader::Shader(char const * vertex_shader_source,
               char const * fragment_shader_source)
  : id(static_cast<uint32_t>(-1))
{
  // Compile the vertex shader
  uint32_t vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
  glCompileShader(vertex_shader);
  checkShaderCompilation(vertex_shader);

  // Compile the fragment shader
  uint32_t fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
  glCompileShader(fragment_shader);
  checkShaderCompilation(fragment_shader);
  
  // Link the shaders
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  id = glCreateProgram();
  glAttachShader(id, vertex_shader);
  glAttachShader(id, fragment_shader);
  glLinkProgram(id);
  checkShaderLinking(id);

  // Delete the shaders
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
}

} // namespace um2
