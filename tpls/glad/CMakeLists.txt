cmake_minimum_required(VERSION 3.25 FATAL_ERROR)    
    
project(    
  Glad 
  HOMEPAGE_URL "https://github.com/Dav1dde/glad"
  LANGUAGES C)

add_library(glad 
  src/gl.c
  include/glad/gl.h 
  include/KHR/khrplatform.h
  )
target_include_directories(glad PUBLIC include)
