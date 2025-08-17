cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fa1_forward LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_ARCHITECTURES 80) 


set(CUDAToolkit_ROOT /usr/local/cuda-12.2)

find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_ROOT}/include)

link_directories(${CUDAToolkit_ROOT}/lib64)


option (BUILD_MAIN "build main tester executable" ON) 
option (BUILD_NAIVE_QKT "build naive qkt tester executable" OFF)
option(BUILD_REDUCTION_STEP "build reduction step tester executable" OFF)


if (BUILD_MAIN)#full test
    add_executable(fa1_forward fa1_forward/fa1_forward.cu fa1_forward/naive_attention.cpp)
    target_link_libraries(fa1_forward CUDA::cudart)
    include_directories(qktStep)
    include_directories(reductionStep)
    set_property(TARGET fa1_forward PROPERTY CXX_STANDARD 17)
endif (BUILD_MAIN)

if (BUILD_NAIVE_QKT)#just test the qkt
    add_executable(qkt_runner qktStep/qkt_runner.cu qktStep/naive_qkt.cpp) 
    target_link_libraries(qkt_runner CUDA::cudart)
    include_directories(qktStep) 
    set_property(TARGET qkt_runner PROPERTY CXX_STANDARD 17)
endif (BUILD_NAIVE_QKT)

if (BUILD_REDUCTION_STEP) 
    //TODO
endif (BUILD_REDUCTION_STEP)




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

