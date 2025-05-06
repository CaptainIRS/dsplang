#ifndef CONVERSION_HLHVXTOHLHVX_PASSES_H
#define CONVERSION_HLHVXTOHLHVX_PASSES_H

#include "dsplang/Conversion/HLHVXToLLHVX/HLHVXToLLHVX.h"

namespace dsplang {

#define GEN_PASS_REGISTRATION
#include "dsplang/Conversion/HLHVXToLLHVX/Passes.h.inc"

} // namespace dsplang

#endif // CONVERSION_HLHVXTOHLHVX_PASSES_H
