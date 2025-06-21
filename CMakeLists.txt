cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fa1_forward Languages CXX CUDA)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(fa1_forward fa1_forward.cpp fa1_forward.cu)
target_link_libraries(fa1_forward)
set_property(TARGET fa1_forward PROPERTY CXX_STANDARD 17)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.2)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(fa1_forward cudart)
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

