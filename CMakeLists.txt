cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fa1_forward LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_ARCHITECTURES 80)#sm80 is a100


set(CUDAToolkit_ROOT /usr/local/cuda-12.2)

find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_ROOT}/include)

link_directories(${CUDAToolkit_ROOT}/lib64)


option (BUILD_MAIN "build main tester executable" ON) 
option (BUILD_NAIVE_QKT "build naive qkt tester executable" OFF)
option(BUILD_REDUCTION_STEP "build reduction step tester executable" OFF)


if (BUILD_MAIN)#full test
    add_executable(fa1Forward fa1Forward/fa1Forward.cu fa1Forward/naiveAttention.cpp)
    target_link_libraries(fa1Forward CUDA::cudart)
    include_directories(qktStep)
    include_directories(reductionStep)
    set_property(TARGET fa1Forward PROPERTY CXX_STANDARD 17)
endif (BUILD_MAIN)

if (BUILD_NAIVE_QKT)#just test the qkt
    add_executable(qktRunner qktStep/qktRunner.cu) 
    target_link_libraries(qktRunner CUDA::cudart)
    include_directories(qktStep) 
    set_property(TARGET qktRunner PROPERTY CXX_STANDARD 17)
endif (BUILD_NAIVE_QKT)

if (BUILD_REDUCTION_STEP)
    add_executable(reductionStepRunner reductionStep/reductionStep.cuh reductionStep/reductionStepRunner.cuh)
    target_link_libraries(reductionStepRunner CUDA::cudart)
    include_directories(reductionStep)
    set_property(TARGET reductionStepRunner PROPERTY CXX_STANDARD 17)

endif (BUILD_REDUCTION_STEP)