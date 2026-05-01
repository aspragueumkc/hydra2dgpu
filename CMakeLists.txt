cmake_minimum_required(VERSION 3.16)
project(backwater_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Disable LTO globally if CUDA will be used (to avoid version conflicts)
if(BACKWATER_USE_CUDA)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

include(FetchContent)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  message(STATUS "pybind11 config not found; fetching pybind11 with FetchContent")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ── 1D native backend ────────────────────────────────────────────────────────
pybind11_add_module(backwater_native cpp/src/backwater_native.cpp)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(backwater_native PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(backwater_native PRIVATE BACKWATER_HAS_OPENMP=1)
endif()

if(MSVC)
  target_compile_options(backwater_native PRIVATE /W4)
else()
  target_compile_options(backwater_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── 2D SWE hybrid GPU/CPU solver ─────────────────────────────────────────────
option(BACKWATER_USE_CUDA "Enable CUDA GPU path for 2D SWE solver" ON)

set(SWE2D_CPU_SOURCES
    cpp/src/swe2d_mesh.cpp
    cpp/src/swe2d_numerics.cpp
    cpp/src/swe2d_solver.cpp
    cpp/src/swe2d_bindings.cpp
)

set(SWE2D_GPU_SOURCES "")
set(SWE2D_CUDA_FOUND FALSE)

if(BACKWATER_USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)  # Volta, Ampere, Hopper; allows fallback to native
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu.cu)
    set(SWE2D_CUDA_FOUND TRUE)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — 2D solver GPU path enabled")
  else()
    message(STATUS "CUDA not found — 2D solver will use CPU (OpenMP) path only")
  endif()
endif()

pybind11_add_module(backwater_swe2d
    ${SWE2D_CPU_SOURCES}
    ${SWE2D_GPU_SOURCES}
)

target_include_directories(backwater_swe2d PRIVATE cpp/src)

# Disable LTO when CUDA is involved to avoid LTO version conflicts
if(SWE2D_CUDA_FOUND)
  set_property(TARGET backwater_swe2d PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

if(SWE2D_CUDA_FOUND)
  target_link_libraries(backwater_swe2d PRIVATE CUDA::cudart)
  target_compile_definitions(backwater_swe2d PRIVATE BACKWATER_HAS_CUDA=1)
  set_property(TARGET backwater_swe2d PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  # Allow CUDA files to find shared headers
  set_source_files_properties(cpp/src/swe2d_gpu.cu PROPERTIES LANGUAGE CUDA)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(backwater_swe2d PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(backwater_swe2d PRIVATE BACKWATER_HAS_OPENMP=1)
endif()

if(MSVC)
  target_compile_options(backwater_swe2d PRIVATE /W4)
else()
  target_compile_options(backwater_swe2d PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── TQMesh mesh generator ─────────────────────────────────────────────────────
set(TQMESH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/third_party/TQMesh)

# Generate TQMeshConfig.h (only defines version + source path macros)
configure_file(
    ${TQMESH_DIR}/auxiliary/TQMeshConfig.h.in
    ${CMAKE_BINARY_DIR}/TQMeshConfig.h
)

pybind11_add_module(backwater_tqmesh
    cpp/src/tqmesh_bindings.cpp
)

target_include_directories(backwater_tqmesh PRIVATE
    ${TQMESH_DIR}/src/algorithm
    ${TQMESH_DIR}/src/utils
    ${CMAKE_BINARY_DIR}          # for generated TQMeshConfig.h
)

# TQMesh uses heavy template / inline code — suppress noisy warnings
if(MSVC)
  target_compile_options(backwater_tqmesh PRIVATE /W1)
else()
  target_compile_options(backwater_tqmesh PRIVATE -w)
endif()

set_property(TARGET backwater_tqmesh PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
