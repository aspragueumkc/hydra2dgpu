// swe2d_numerics.cpp
// unit tests and standalone implementations for swe2d_numerics.hpp.
// The header provides all inline logic; this file exists to satisfy the
// CMake source list and can house non-inline helpers if needed in future.

#include "swe2d_numerics.hpp"

// All kernel logic lives in the header as inline functions.
// This translation unit intentionally left minimal so the compiler
// can inline everything into the solver hot loops.
