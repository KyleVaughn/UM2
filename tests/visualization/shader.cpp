#include <um2/common/config.hpp>

#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include "../test_framework.hpp"
#include <um2/visualization/element_buffer.hpp>
#include <um2/visualization/shader.hpp>
#include <um2/visualization/vertex_array.hpp>
#include <um2/visualization/vertex_buffer.hpp>

#if UM2_ENABLE_VIS
static void errorCallback(int error, const char * description)
{
  fprintf(stderr, "Error %d: %s\n", error, description);
}

static void keyCallback(GLFWwindow * window, int key, int /*scancode*/, int action,
                        int /*mods*/)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

static void framebufferSizeCallback(GLFWwindow * /*window*/, int width, int height)
{
  glViewport(0, 0, width, height);
}

int const WIDTH = 1024;
int const HEIGHT = 768;

UM2_HOSTDEV TEST_CASE(construct_from_source)
{

  glfwSetErrorCallback(errorCallback);

  if (glfwInit() == GL_FALSE) {
    fprintf(stderr, "ERROR: could not start GLFW3\n");
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "TestWindow", nullptr, nullptr);
  if (window == nullptr) {
    fprintf(stderr, "ERROR: could not open window with GLFW3\n");
    glfwTerminate();
  }
  glfwSetKeyCallback(window, keyCallback);
  glfwMakeContextCurrent(window);
  if (gladLoadGL(glfwGetProcAddress) == 0) {
    printf("Failed to initialize OpenGL context\n");
  }
  // get version info
  GLubyte const * renderer = glGetString(GL_RENDERER);
  GLubyte const * version = glGetString(GL_VERSION);
  printf("Renderer: %s\n", renderer);
  printf("OpenGL version supported %s\n", version);

  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  float vertices[] = {
      -0.5F, -0.5F, 0.0F, // bottom left
      0.5F,  -0.5F, 0.0F, // bottom right
      0.5F,  0.5F,  0.0F, // top right
      -0.5F, 0.5F,  0.0F  // top left
  };
  int32_t indices[] = {
      0, 1, 2, // first triangle
      2, 3, 0  // second triangle
  };
  um2::VertexArray vao;
  EXPECT_EQ(vao.id, 1);
  vao.bind();
  um2::VertexBuffer vbo(vertices, 12);
  EXPECT_EQ(vbo.id, 1);
  um2::ElementBuffer ebo(indices, 6);
  EXPECT_EQ(ebo.id, 2);
  um2::VertexArray::set_vertex_dimension(3);
  um2::VertexBuffer::unbind();
  um2::VertexArray::unbind();

  char const * vertex_shader_source = "#version 460\n"
                                      "in vec3 pos;\n"
                                      "void main() {\n"
                                      "  gl_Position = vec4(pos, 1.0);\n"
                                      "}\n";
  char const * fragment_shader_source = "#version 460\n"
                                        "out vec4 frag_color;\n"
                                        "void main() {\n"
                                        "  frag_color = vec4(0.5, 0.0, 0.5, 1.0);\n"
                                        "}\n";
  um2::Shader shdr(vertex_shader_source, fragment_shader_source);
  EXPECT_EQ(shdr.id, 3);

  while (glfwWindowShouldClose(window) == 0) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    shdr.use();
    vao.bind();
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glfwPollEvents();
    glfwSwapBuffers(window);
  }

  vbo.destroy();
  ebo.destroy();
  shdr.destroy();
  glfwTerminate();
}

TEST_SUITE(shader) { TEST(construct_from_source) }
#endif

auto main() -> int
{
#if UM2_ENABLE_VIS
  RUN_TESTS(shader);
#endif
  return 0;
}
