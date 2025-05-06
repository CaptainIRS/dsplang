#ifndef CONVERSION_ARITHTOHLHVX_PASSES_H
#define CONVERSION_ARITHTOHLHVX_PASSES_H

#include "dsplang/Conversion/ArithToHLHVX/ArithToHLHVX.h"

namespace dsplang {

#define GEN_PASS_REGISTRATION
#include "dsplang/Conversion/ArithToHLHVX/Passes.h.inc"

} // namespace dsplang

#endif // CONVERSION_ARITHTOHLHVX_PASSES_H
