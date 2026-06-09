cmake_minimum_required(VERSION 3.16)
project(hydra_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to release-with-debug-info (optimized + profileable)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

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

# GPU-only: no OpenMP (CPU path removed)

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
    # CUDA 13.x is incompatible with GCC 14+ host headers.  When the
    # default host compiler is too new, fall back to gcc-13 if available.
    if(NOT CMAKE_CUDA_HOST_COMPILER)
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
        OUTPUT_VARIABLE _host_gcc_ver OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_host_gcc_ver VERSION_GREATER_EQUAL 14)
        find_program(_gcc13 gcc-13)
        if(_gcc13)
          set(CMAKE_CUDA_HOST_COMPILER "${_gcc13}" CACHE FILEPATH "CUDA host compiler (gcc-13)")
          message(STATUS "CUDA host compiler: ${_gcc13} (gcc-13 for GCC 14+ compat)")
        endif()
        unset(_gcc13)
      endif()
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # CUDA 13+ drops support for some older virtual architectures (e.g. compute_70).
    # Pick an architecture list compatible with the detected toolkit version.
    # Includes sm_86 (RTX 3080/3090) and sm_89 (RTX 4060/4070/4080/4090) for
    # optimal kernel generation on modern consumer GPUs.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu_redistribute.cu)
    set(SWE2D_CUDA_FOUND TRUE)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — 2D solver GPU path enabled")
  else()
    message(FATAL_ERROR "CUDA not found — GPU-only 2D solver cannot build")
  endif()
endif()

function(_configure_swe2d_module target_name py_module_name)
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
    set_source_files_properties(cpp/src/swe2d_gpu_redistribute.cu PROPERTIES LANGUAGE CUDA)
    # Relaxed constexpr and extended lambda support for device-side std:: functions
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )
    # Enable fast math (relaxed IEEE precision) for GPU kernels.
    # Adds --use_fast_math to nvcc which enables:
    #   - faster div/rsqrt/sqrt/log/exp
    #   - Flush-To-Zero / Denormals-Are-Zero
    #   - relaxed signed-zero and rounding comparisons
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    )
    # High optimization level for GPU device code
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    if(NOT MSVC)
      target_compile_options(${target_name} PRIVATE -fno-lto)
      target_link_options(${target_name} PRIVATE -fno-lto)
    endif()
  endif()

  # Host compiler: optimize for the build machine's CPU and enable fast math
  if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo|MinSizeRel")
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3 -mtune=native -march=native -ffast-math>
    )
  endif()

  # GPU-only: no OpenMP

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# GPU-only SWE2D module.
_configure_swe2d_module(hydra_swe2d hydra_swe2d)

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

# GPU-only: no OpenMP for hydra_overlay

if(SWE2D_CUDA_FOUND)
  target_compile_definitions(hydra_overlay PRIVATE HYDRA_HAS_CUDA=1)
endif()

if(MSVC)
  target_compile_options(hydra_overlay PRIVATE /W4)
else()
  target_compile_options(hydra_overlay PRIVATE -Wall -Wextra -Wpedantic)
endif()
