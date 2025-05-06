#ifndef CONVERSION_HLHVXTOHLHVX_HLHVXTOHLHVX_H
#define CONVERSION_HLHVXTOHLHVX_HLHVXTOHLHVX_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

namespace dsplang {

#define GEN_PASS_DECL_HLHVXTOLLHVXPASS
#include "dsplang/Conversion/HLHVXToLLHVX/Passes.h.inc"

} // namespace dsplang

#endif // CONVERSION_HLHVXTOHLHVX_HLHVXTOHLHVX_H
