#pragma once
// Mixed precision: state arrays stored as float, compute in double.
#ifdef SWE2D_STATE_FP32
using State = float;
#else
using State = double;
#endif
