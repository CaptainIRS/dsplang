// Adapted from LLVM MLIR Toy example

#include "dsplang/Dialect/Dsplang/DsplangDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "dsplang/Dialect/Dsplang/IR/DsplangCombine.inc"
} // namespace

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void dsplang::ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results
      .add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
           FoldConstantSI8ReshapeOptPattern, FoldConstantUI8ReshapeOptPattern,
           FoldConstantSI16ReshapeOptPattern, FoldConstantUI16ReshapeOptPattern,
           FoldConstantSI32ReshapeOptPattern, FoldConstantUI32ReshapeOptPattern,
           FoldConstantF16ReshapeOptPattern, FoldConstantF32ReshapeOptPattern>(
          context);
}
