cmake_minimum_required(VERSION 3.16)
project(backwater_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
