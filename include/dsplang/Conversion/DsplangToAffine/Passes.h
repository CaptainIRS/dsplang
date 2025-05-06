// Adapted from LLVM MLIR Dsplang example

#ifndef DSPLANG_TO_AFFINE_PASSES_H
#define DSPLANG_TO_AFFINE_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace dsplang {
std::unique_ptr<mlir::Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Dsplang IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

std::unique_ptr<mlir::Pass> createAffineSelectiveVectorizePass();

} // namespace dsplang

#endif // DSPLANG_TO_AFFINE_PASSES_H
