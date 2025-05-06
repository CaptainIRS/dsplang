#ifndef CONVERSION_ARITHTOHLHVX_ARITHTOHLHVX_H
#define CONVERSION_ARITHTOHLHVX_ARITHTOHLHVX_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

namespace dsplang {

#define GEN_PASS_DECL_ARITHTOHLHVXPASS
#include "dsplang/Conversion/ArithToHLHVX/Passes.h.inc"

} // namespace dsplang

#endif // CONVERSION_ARITHTOHLHVX_ARITHTOHLHVX_H
