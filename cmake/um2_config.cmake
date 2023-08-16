### visualization #################################
###################################################
#if (UM2_ENABLE_VIS)
#  set(UM2_VIS_LIBRARIES
#    "OpenGL::GL"
#    "glfw"
#    "glad"
#    CACHE STRING "Visualization libraries")
#  # OpenGL
#  find_package(OpenGL REQUIRED)
#  # GLFW
#  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
#  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
#  add_subdirectory("${PROJECT_SOURCE_DIR}/dependencies/glfw" SYSTEM)
#  # GLAD
#  add_subdirectory("${PROJECT_SOURCE_DIR}/dependencies/glad" SYSTEM)
#  target_link_libraries(um2 PRIVATE ${UM2_VIS_LIBRARIES}) 
#endif()
