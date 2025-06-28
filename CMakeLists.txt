cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fa1_forward LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_ARCHITECTURES 80) 
find_package(CUDAToolkit REQUIRED)


set(CUDAToolkit_ROOT /usr/local/cuda-12.2)

find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_ROOT}/include)

link_directories(${CUDAToolkit_ROOT}/lib64)

add_executable(fa1_forward fa1_forward.cpp fa1_forward.cu reductionStep.cu qkt_computation.cu)
target_link_libraries(fa1_forward CUDA::cudart)



set_property(TARGET fa1_forward PROPERTY CXX_STANDARD 17)

# Force override any remaining problematic flags
if(APPLE)
    set_target_properties(fa1_forward PROPERTIES
        LINK_FLAGS "-Wl,-undefined,dynamic_lookup"
    )
endif()


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET fa1_forward
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:fa1_forward>)
endif (MSVC)

