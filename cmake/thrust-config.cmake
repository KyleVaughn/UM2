set(Thrust_DIR "${PROJECT_SOURCE_DIR}/tpls/thrust/thrust/cmake")
find_package(Thrust REQUIRED CONFIG)        
# Host backend
# Prefer OpenMP if available
if (NOT UM2_THRUST_HOST)
  if (UM2_ENABLE_OPENMP)
    set(UM2_THRUST_HOST "OMP")
  else()
    set(UM2_THRUST_HOST "CPP")
  endif()
endif()
# Device backend
# Prefer CUDA > OpenMP > CPP
if (NOT UM2_THRUST_DEVICE)
  if (UM2_ENABLE_CUDA)    
    set(UM2_THRUST_DEVICE "CUDA")
  elseif (UM2_ENABLE_OPENMP)
    set(UM2_THRUST_DEVICE "OMP")
  else()
    set(UM2_THRUST_DEVICE "CPP")
  endif()
endif()

thrust_create_target(Thrust HOST ${UM2_THRUST_HOST} DEVICE ${UM2_THRUST_DEVICE})  
