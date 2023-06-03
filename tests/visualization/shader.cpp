#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "../test_framework.hpp"
#include <um2/visualization/shader.hpp>

static void errorCallback(int error, const char * description)
{
  fprintf(stderr, "Error %d: %s\n", error, description);
}

static void keyCallback(GLFWwindow * window, int key, int /*scancode*/, int action, int /*mods*/)
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

  if ( glfwInit() == GL_FALSE) {
    fprintf( stderr, "ERROR: could not start GLFW3\n" );
  }

  glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
  glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 6 );
  glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
  glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

  GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "TestWindow", nullptr, nullptr);
  if ( window == nullptr ) { 
    fprintf( stderr, "ERROR: could not open window with GLFW3\n" );
    glfwTerminate();
  }
  glfwSetKeyCallback(window, keyCallback);
  glfwMakeContextCurrent(window);
  if (gladLoadGL(glfwGetProcAddress) == 0) {
    printf("Failed to initialize OpenGL context\n");
  }
  // get version info 
  GLubyte const * renderer = glGetString( GL_RENDERER );
  GLubyte const * version = glGetString( GL_VERSION );
  printf( "Renderer: %s\n", renderer );
  printf( "OpenGL version supported %s\n", version );

  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  glEnable( GL_DEPTH_TEST );
  glDepthFunc( GL_LESS );

  uint32_t vbo = 0;
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  GLfloat points[9] = { 0.0F, 0.5F, 0.0F, 0.5F, -0.5F, 0.0F, -0.5F, -0.5F, 0.0F };
  glBufferData( GL_ARRAY_BUFFER, 9 * sizeof( GLfloat ), points, GL_STATIC_DRAW );

  uint32_t vao = 0;
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );
  glEnableVertexAttribArray( 0 );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

  char const * vertex_shader_source =
    "#version 460\n"
    "in vec3 pos;\n"
    "void main() {\n"
    "  gl_Position = vec4(pos, 1.0);\n"
    "}\n";
  char const * fragment_shader_source =
    "#version 460\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "  frag_color = vec4(0.5, 0.0, 0.5, 1.0);\n"
    "}\n";
  um2::Shader shdr(vertex_shader_source, fragment_shader_source);
  EXPECT_EQ(shdr.id, 3);

  while ( glfwWindowShouldClose( window ) == 0) {
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    shdr.use();
    glBindVertexArray( vao );
    glDrawArrays( GL_TRIANGLES, 0, 3 );
    glfwPollEvents();
    glfwSwapBuffers( window );
  }
  
  glfwTerminate();

}

TEST_SUITE(shader)
{
  TEST(construct_from_source)
}

auto main() -> int
{
  RUN_TESTS(shader);
  return 0;
}
