cmake_minimum_required(VERSION 3.16)
project(hydra_swe2d LANGUAGES CXX)

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

# ── MSVC + CUDA: prevent /Fd and /FS from reaching nvcc ──────────────────
# CMake's MSVC platform module injects /Fd (PDB path) and /FS (force sync
# PDB writes) into compile commands.  nvcc's -Xcompiler splits on commas,
# so -Xcompiler=-FdFoo,-FS becomes two args and nvcc sees -FS as an input
# file → "single input file required" fatal error.
#
# Fix: force /Z7 (embedded debug info) so no PDB file is generated.
if(MSVC AND POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded" CACHE STRING "" FORCE)
endif()

include(FetchContent)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Phase 3 — CUDA-OpenGL interop for zero-D2H GUI rendering.
find_package(OpenGL REQUIRED)

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

# ── 2D SWE hybrid GPU/CPU solver ─────────────────────────────────────────────
option(BACKWATER_USE_CUDA "Enable CUDA GPU path for 2D SWE solver" ON)
option(CULVERT_DIAG "Enable device-side printf debug in culvert kernel (slow)" OFF)

# Mixed-precision strategy:
#   The codebase stores state arrays as `double` (FP64) throughout. Selective FP32
#   is applied inside `swe2d_flux_kernel` only where it is numerically safe
#   (see reference/MIXED_PRECISION_GPU_PLAN.md). A previous `SWE2D_STATE_FP32`
#   option that switched the global `State` typedef to `float` was removed: it
#   bypassed the conservation guarantees and broke the higher-order RK schemes.

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
    list(APPEND SWE2D_GPU_SOURCES cpp/src/pipe1d.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_reconstruct.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu_viewer.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu_viewer_interop.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu_diag_ring.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu_viewer_hud.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_nv_utils.cu)
    list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_gpu_viewer_nvenc.cu)
    # Phase 5: TS muxer (plain C, no CUDA) — added to the host sources.
    list(APPEND SWE2D_CPU_SOURCES cpp/src/swe2d_ts_muxer.c)
    set(SWE2D_CUDA_FOUND TRUE)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — 2D solver GPU path enabled")
  else()
    message(FATAL_ERROR "CUDA not found — GPU-only 2D solver cannot build")
  endif()
endif()

function(_configure_swe2d_module target_name py_module_name)
  # Phase 5: TS muxer (.c, no CUDA) — compile as plain C explicitly.
  set(_ts_muxer_obj "")
  if(EXISTS "${CMAKE_SOURCE_DIR}/cpp/src/swe2d_ts_muxer.c")
    if(NOT CMAKE_C_COMPILER)
    # Use the C compiler available in the conda env.  The system gcc-13
    # defaults to C++ for .c files (Ubuntu's gcc-13 is multi-language),
    # which breaks the TS muxer compile.  Fall back to whichever gcc
    # is in the mamba env PATH.
    find_program(_conda_gcc NAMES gcc gcc-13 gcc-12 REQUIRED)
    set(CMAKE_C_COMPILER "${_conda_gcc}")
  endif()
    set(_ts_muxer_obj "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_ts_muxer.c.o")
    add_custom_command(
      OUTPUT "${_ts_muxer_obj}"
      COMMAND ${CMAKE_C_COMPILER} -x c -c -O2 -fPIC
              -I${CMAKE_SOURCE_DIR}/cpp/src
              "${CMAKE_SOURCE_DIR}/cpp/src/swe2d_ts_muxer.c"
              -o "${_ts_muxer_obj}"
      DEPENDS "${CMAKE_SOURCE_DIR}/cpp/src/swe2d_ts_muxer.c"
               "${CMAKE_SOURCE_DIR}/cpp/src/swe2d_ts_muxer.h"
      COMMENT "Compiling TS muxer (Phase 5) for ${target_name}"
    )
    add_custom_target(${target_name}_ts_muxer_obj DEPENDS "${_ts_muxer_obj}")
  endif()
  pybind11_add_module(${target_name}
      ${SWE2D_CPU_SOURCES}
      ${SWE2D_GPU_SOURCES}
  )
  if(_ts_muxer_obj)
    target_sources(${target_name} PRIVATE "${_ts_muxer_obj}")
    add_dependencies(${target_name} ${target_name}_ts_muxer_obj)
  endif()

  target_include_directories(${target_name} PRIVATE cpp/src)
  target_compile_definitions(${target_name} PRIVATE HYDRA_SWE2D_PY_MODULE_NAME=${py_module_name})

  if(SWE2D_CUDA_FOUND)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    # CUDA-OpenGL interop (cuGraphicsGLRegisterImage etc.) lives in libcuda.so
    # (NVIDIA driver).  No CMake target — find and link directly.
    find_library(CUDA_DRIVER_LIB cuda)
    if(NOT CUDA_DRIVER_LIB)
      # Build environments (manylinux containers, wheel CI runners) have no
      # NVIDIA driver.  Fall back to the CUDA toolkit's link-time stub
      # (${CUDAToolkit_LIBRARY_DIR}/stubs works on Linux and Windows) —
      # runtime symbol resolution happens against the user's driver.
      find_library(CUDA_DRIVER_LIB cuda
        PATHS "${CUDAToolkit_LIBRARY_DIR}/stubs"
        NO_DEFAULT_PATH)
    endif()
    if(NOT CUDA_DRIVER_LIB)
      message(FATAL_ERROR "CUDA driver library (libcuda) not found; "
        "install the NVIDIA driver or add the toolkit's stubs dir to the search path")
    endif()
    message(STATUS "Linking CUDA driver lib: ${CUDA_DRIVER_LIB}")
    target_link_libraries(${target_name} PRIVATE ${CUDA_DRIVER_LIB})
    target_link_libraries(${target_name} PRIVATE OpenGL::GL)
    # Phase 5: NVENC API header path.  libnvidia-encode.so is dlopen()'d
    # at runtime (no hard link) — it's in the NVIDIA driver package.
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
    target_compile_definitions(${target_name} PRIVATE HYDRA_HAS_CUDA=1)
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    # Allow CUDA files to find shared headers
    set_source_files_properties(cpp/src/swe2d_gpu.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_gpu_redistribute.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/pipe1d.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_reconstruct.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_gpu_viewer.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_gpu_viewer_interop.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_gpu_diag_ring.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_gpu_viewer_hud.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_nv_utils.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(cpp/src/swe2d_gpu_viewer_nvenc.cu PROPERTIES LANGUAGE CUDA)
    # Note: cpp/src/swe2d_ts_muxer.c is plain C — no LANGUAGE_CUDA needed.
    # Relaxed constexpr and extended lambda support for device-side std:: functions
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )
    # Enable CULVERT_DIAG printf in swe2d_compute_structure_flows_kernel.
    # Set via -DCULVERT_DIAG=ON at cmake-configure time.  Keep off in
    # production (device printf is slow).
    if(DEFINED CULVERT_DIAG AND CULVERT_DIAG)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-DCULVERT_DIAG=1>
      )
    endif()
    # Enable fast math (relaxed IEEE precision) for GPU kernels.
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    )
    # High optimization level for GPU device code
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
  endif()

  # Host compiler: optimize for the build machine's CPU and enable fast math
  if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo|MinSizeRel")
    if(MSVC)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:fast>
      )
    else()
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -mtune=native -march=native -ffast-math>
      )
    endif()
  endif()

  # GPU-only: no OpenMP

  if(MSVC)
    target_compile_definitions(${target_name} PRIVATE _USE_MATH_DEFINES)
    target_compile_definitions(${target_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
    # Only apply /W4 to CXX, not CUDA — nvcc doesn't understand MSVC flags
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# GPU-only SWE2D module.
_configure_swe2d_module(hydra_swe2d hydra_swe2d)

# ── Meshing prebuild native kernels ────────────────────────────────────────
pybind11_add_module(hydra_meshing_native
    cpp/src/meshing_native_bindings.cpp
)

if(MSVC)
  target_compile_options(hydra_meshing_native PRIVATE /W4)
else()
  target_compile_options(hydra_meshing_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

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

# ── scikit-build-core install rules ────────────────────────────────────────
# scikit-build-core reads install(TARGETS ...) to know which pybind11 modules
# to bundle under wheel.install-dir in the resulting wheel.
install(TARGETS hydra_swe2d hydra_meshing_native hydra_overlay
        DESTINATION hydra_swe2d
        COMPONENT hydra_swe2d_runtime)
install(FILES   cpp/src/swe2d_solver.hpp
        DESTINATION hydra_swe2d/include
        COMPONENT  hydra_swe2d_development
        OPTIONAL)
# Python shim that wires the C++ extension symbols into `import hydra_swe2d`.
# Without this, the wheel has the .so files but no Python wrapper, so
# `from hydra_swe2d import swe2d_deserialize_mesh` fails with
# "(unknown location)". Dev workflow never noticed because the source
# tree's hydra_swe2d/__init__.py was on sys.path via realpath() math.
install(FILES   hydra_swe2d/__init__.py
        DESTINATION hydra_swe2d
        COMPONENT  hydra_swe2d_runtime
        OPTIONAL)

# Bundle the swe2d/ Python sources into the wheel. scikit-build-core
# packages whatever CMake installs; without this, the wheel only has the
# compiled C++ extension and the QGIS plugin crashes on `from swe2d.*`.
install(DIRECTORY swe2d
        DESTINATION .
        COMPONENT  swe2d_sources
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.pyo" EXCLUDE)
