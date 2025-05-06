#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "dsplang/Dialect/HLHVX/HLHVXDialect.h"
#include "dsplang/Dialect/LLHVX/LLHVXDialect.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "dsplang/Conversion/ArithToHLHVX/ArithToHLHVX.h"

namespace dsplang {

#define GEN_PASS_DEF_ARITHTOHLHVXPASS
#include "dsplang/Conversion/ArithToHLHVX/Passes.h.inc"

#include "dsplang/Conversion/ArithToHLHVX/ArithToHLHVX.h.inc"

struct LowerArithToHLHVXOps : impl::ArithToHLHVXPassBase<LowerArithToHLHVXOps> {
  using ArithToHLHVXPassBase::ArithToHLHVXPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace dsplang
