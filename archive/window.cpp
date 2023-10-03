#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

void
errorCallback(int error, const char * description)
{
  fprintf(stderr, "Error %d: %s\n", error, description);
}

void
keyCallback(GLFWwindow * window, int key, int /*scancode*/, int action, int /*mods*/)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

void
framebufferSizeCallback(GLFWwindow * /*window*/, int width, int height)
{
  glViewport(0, 0, width, height);
}

int const WIDTH = 1024;
int const HEIGHT = 768;

auto
main() -> int
{

  glfwSetErrorCallback(errorCallback);

  if (glfwInit() == GL_FALSE) {
    fprintf(stderr, "ERROR: could not start GLFW3\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "TestWindow", nullptr, nullptr);
  if (window == nullptr) {
    fprintf(stderr, "ERROR: could not open window with GLFW3\n");
    glfwTerminate();
    return 1;
  }
  glfwSetKeyCallback(window, keyCallback);
  glfwMakeContextCurrent(window);
  if (gladLoadGL(glfwGetProcAddress) == 0) {
    printf("Failed to initialize OpenGL context\n");
    return -1;
  }
  // get version info
  GLubyte const * renderer = glGetString(GL_RENDERER);
  GLubyte const * version = glGetString(GL_VERSION);
  printf("Renderer: %s\n", renderer);
  printf("OpenGL version supported %s\n", version);

  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  uint32_t vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  GLfloat points[9] = {0.0F, 0.5F, 0.0F, 0.5F, -0.5F, 0.0F, -0.5F, -0.5F, 0.0F};
  glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(GLfloat), points, GL_STATIC_DRAW);

  uint32_t vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  uint32_t const vert_shader = glCreateShader(GL_VERTEX_SHADER);
  char const * vertex_shader = "#version 460\n"
                               "in vec3 vp;"
                               "void main () {"
                               "  gl_Position = vec4(vp, 1.0);"
                               "}";
  glShaderSource(vert_shader, 1, &vertex_shader, nullptr);
  glCompileShader(vert_shader);
  uint32_t const frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

  char const * fragment_shader = "#version 460\n"
                                 "out vec4 frag_colour;"
                                 "void main () {"
                                 "  frag_colour = vec4(0.5, 0.0, 0.5, 1.0);"
                                 "}";

  glShaderSource(frag_shader, 1, &fragment_shader, nullptr);
  glCompileShader(frag_shader);
  uint32_t const shader_programme = glCreateProgram();
  glAttachShader(shader_programme, frag_shader);
  glAttachShader(shader_programme, vert_shader);
  glLinkProgram(shader_programme);

  while (glfwWindowShouldClose(window) == 0) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shader_programme);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glfwPollEvents();
    glfwSwapBuffers(window);
  }

  glfwTerminate();
  return 0;
}
