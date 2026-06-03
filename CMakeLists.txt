cmake_minimum_required(VERSION 3.16)
project(hydra_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Disable LTO globally if CUDA will be used (to avoid version conflicts)
if(BACKWATER_USE_CUDA)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

include(FetchContent)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Force pybind11 to use modern FindPython mode so it reuses Python3_* from
# above instead of classic FindPythonInterp/FindPythonLibs (which can pick up
# unintended virtualenv interpreters/headers).
set(PYBIND11_FINDPYTHON ON)

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
pybind11_add_module(hydra_native cpp/src/backwater_native.cpp)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hydra_native PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(hydra_native PRIVATE HYDRA_HAS_OPENMP=1)
endif()

if(MSVC)
  target_compile_options(hydra_native PRIVATE /W4)
else()
  target_compile_options(hydra_native PRIVATE -Wall -Wextra -Wpedantic)
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
    # CUDA 13+ drops support for some older virtual architectures (e.g. compute_70).
    # Pick an architecture list compatible with the detected toolkit version.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 90)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
    endif()
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu.cu)
    set(SWE2D_CUDA_FOUND TRUE)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — 2D solver GPU path enabled")
  else()
    message(STATUS "CUDA not found — 2D solver will use CPU (OpenMP) path only")
  endif()
endif()

function(_configure_swe2d_module target_name py_module_name enable_openmp)
  pybind11_add_module(${target_name}
      ${SWE2D_CPU_SOURCES}
      ${SWE2D_GPU_SOURCES}
  )

  target_include_directories(${target_name} PRIVATE cpp/src)
  target_compile_definitions(${target_name} PRIVATE HYDRA_SWE2D_PY_MODULE_NAME=${py_module_name})

  # Disable LTO when CUDA is involved to avoid LTO version conflicts
  if(SWE2D_CUDA_FOUND)
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    target_compile_definitions(${target_name} PRIVATE HYDRA_HAS_CUDA=1)
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    # Allow CUDA files to find shared headers
    set_source_files_properties(cpp/src/swe2d_gpu.cu PROPERTIES LANGUAGE CUDA)
    # Relaxed constexpr and extended lambda support for device-side std:: functions
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )
    if(NOT MSVC)
      target_compile_options(${target_name} PRIVATE -fno-lto)
      target_link_options(${target_name} PRIVATE -fno-lto)
    endif()
  endif()

  if(enable_openmp AND OpenMP_CXX_FOUND)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${target_name} PRIVATE HYDRA_HAS_OPENMP=1)
  endif()

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# OpenMP-enabled default module (existing import path).
_configure_swe2d_module(hydra_swe2d hydra_swe2d TRUE)

# Serial module variant for true non-OpenMP runs from the same build tree.
_configure_swe2d_module(hydra_swe2d_serial hydra_swe2d_serial FALSE)

# ── TQMesh mesh generator ─────────────────────────────────────────────────────
set(TQMESH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/third_party/TQMesh)

# Generate TQMeshConfig.h (only defines version + source path macros)
configure_file(
    ${TQMESH_DIR}/auxiliary/TQMeshConfig.h.in
    ${CMAKE_BINARY_DIR}/TQMeshConfig.h
)

pybind11_add_module(hydra_tqmesh
    cpp/src/tqmesh_bindings.cpp
)

target_include_directories(hydra_tqmesh PRIVATE
    ${TQMESH_DIR}/src/algorithm
    ${TQMESH_DIR}/src/utils
    ${CMAKE_BINARY_DIR}          # for generated TQMeshConfig.h
)

# TQMesh uses heavy template / inline code — suppress noisy warnings
if(MSVC)
  target_compile_options(hydra_tqmesh PRIVATE /W1)
else()
  target_compile_options(hydra_tqmesh PRIVATE -w)
endif()

set_property(TARGET hydra_tqmesh PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)

# ── Custom hybrid topology mesher (C++ backend) ────────────────────────────
pybind11_add_module(hydra_hybridmesh
    cpp/src/hybrid_mesh_bindings.cpp
)

if(MSVC)
  target_compile_options(hydra_hybridmesh PRIVATE /W1)
else()
  target_compile_options(hydra_hybridmesh PRIVATE -w)
endif()

set_property(TARGET hydra_hybridmesh PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)

# ── MFEM beta mesh optimization bridge ──────────────────────────────────────
option(HYDRA_ENABLE_MFEM "Enable MFEM beta mesh optimization module" ON)

if(HYDRA_ENABLE_MFEM)
  pybind11_add_module(hydra_mfem_meshopt
      cpp/src/mfem_meshopt_bindings.cpp
  )

  if(MSVC)
    target_compile_options(hydra_mfem_meshopt PRIVATE /W1)
  else()
    target_compile_options(hydra_mfem_meshopt PRIVATE -w)
  endif()

  set_property(TARGET hydra_mfem_meshopt PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

# ── Meshing prebuild native kernels ────────────────────────────────────────
pybind11_add_module(hydra_meshing_native
    cpp/src/meshing_native_bindings.cpp
)

if(MSVC)
  target_compile_options(hydra_meshing_native PRIVATE /W4)
else()
  target_compile_options(hydra_meshing_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_property(TARGET hydra_meshing_native PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)

# ── High-performance overlay backend (compiled raster core) ────────────────
pybind11_add_module(hydra_overlay
    cpp/src/overlay_backend.cpp
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(hydra_overlay PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(hydra_overlay PRIVATE HYDRA_HAS_OPENMP=1)
endif()

if(SWE2D_CUDA_FOUND)
  target_compile_definitions(hydra_overlay PRIVATE HYDRA_HAS_CUDA=1)
endif()

if(MSVC)
  target_compile_options(hydra_overlay PRIVATE /W4)
else()
  target_compile_options(hydra_overlay PRIVATE -Wall -Wextra -Wpedantic)
endif()
