// Don't reorder headers, glad must be included before glfw3
// clang-format off
#include <glad/gl.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <cstdint>
#include <cstdio>

uint32_t const WIDTH = 1920;
uint32_t const HEIGHT = 1080;

void errorCallback(int error, const char * description)
{
  fprintf(stderr, "Error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow * window, int key, int /*scancode*/, int action, int /*mods*/)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

void framebufferSizeCallback(GLFWwindow * /*window*/, int width, int height)
{
  glViewport(0, 0, width, height);
}

auto main() -> int
{

  glfwSetErrorCallback(errorCallback);

  if (glfwInit() == GLFW_FALSE) {
    printf("Failed to initialize GLFW\n");
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "TestWindow", nullptr, nullptr);
  if (window == nullptr) {
    printf("Failed to create GLFW window\n");
    glfwTerminate();
    return -1;
  }

  glfwSetKeyCallback(window, keyCallback);

  glfwMakeContextCurrent(window);
  if (gladLoadGL(glfwGetProcAddress) == 0) {
    printf("Failed to initialize OpenGL context\n");
    return -1;
  }

  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  //  while (glfwWindowShouldClose(window) == GLFW_FALSE) {
  //    glfwSwapBuffers(window);
  //    glfwPollEvents();
  //  }

  glfwTerminate();
  return 0;
}
